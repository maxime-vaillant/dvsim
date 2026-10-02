"""Vectorised event-camera (DVS) model: linear luminance frames -> sorted event stream."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from backend.config import EventConfig
from backend.types import EVENT_DTYPE, empty_events

_F = np.float32                 # dtype of the per-pixel state (log-intensity, reference)
_TOL = 1e-4                     # n = floor(|d| / C + _TOL): an exact multiple of C fires despite float noise
_MIN_C_FACTOR = 0.1             # floor of the per-pixel threshold factor (1 + sigma_c * N(0, 1))
_NEVER = -(1 << 60)             # last-event time of a pixel that never fired
_RADIX_MAX_DT = 0xFFFF          # largest step (us) whose event times fit a 16-bit radix sort


class EventSimulator:
    """Ideal DVS on log-intensity plus optional non-idealities; pure numpy, no per-pixel Python loops.

    `step(lum, t_prev_us, t_now_us)` takes the linear luminance (H, W) in [0, 1] observed at `t_now_us`
    and returns the EVENT_DTYPE events in (t_prev_us, t_now_us], sorted by t (ties: raster order), so
    concatenating consecutive steps gives a globally sorted stream. The first step after `reset()` only
    initialises the reference and returns no events.

    Model per pixel, with L = log(lum + log_eps) (filtered first if lpf_cutoff_hz > 0) and reference r:
        d = L - r;  n = floor(|d| / C),  C = c_pos if d > 0 (ON, p=1) else c_neg (OFF, p=0)
        event k = 1..n at t_prev + k * C / |d| * (t_now - t_prev)   (linear interpolation, rounded to whole
        microseconds and kept in (t_prev, t_now]);  r += sign(d) * n * C
    All n events are emitted. n is computed as floor(|d| / C + 1e-4), so a change that is an exact multiple
    of C fires despite float32 rounding.

    Options (all off by default; a config value of 0 disables it):
      sigma_c         per-pixel threshold C_pix = C * max(1 + sigma_c * N(0, 1), 0.1), drawn independently for
                      ON and OFF at reset() from default_rng(seed).
      refractory_us   a pixel emits event k only if its time is >= refractory_us after that pixel's last emitted
                      event (across steps as well). Per pixel and step the candidate events are scanned in time
                      order and the first allowed one is taken, then the next one that is >= refractory_us later,
                      and so on. Dropped candidates are lost: they do not advance the reference r and do not
                      update the last-event time, so a persisting change fires again in a later step (the pixel
                      is rate-limited to one event per refractory period). Applies to signal events only.
      shot_noise_hz   background activity: Poisson(shot_noise_hz * dt * H * W) events per step (the sum of the
                      per-pixel Poisson counts), pixel uniform, time uniform in (t_prev, t_now], polarity
                      uniform. They leave the reference and the refractory state untouched.
      lpf_cutoff_hz   first-order IIR on L: y += (L - y) * (1 - exp(-2 pi fc dt)), y initialised with the first
                      L; events are generated from y.
    Everything is deterministic given `config.seed` (`reset()` re-seeds).
    """

    def __init__(self, shape: Tuple[int, int], config: Optional[EventConfig] = None):
        try:
            h, w = (int(v) for v in shape)
        except (TypeError, ValueError):
            raise ValueError(f"shape must be (H, W), got {shape!r}") from None
        if not (0 < h <= 1 << 16 and 0 < w <= 1 << 16):
            raise ValueError(f"shape must be within 1..65536 per axis, got {(h, w)}")
        cfg = config if config is not None else EventConfig()
        for name in ("c_pos", "c_neg", "log_eps"):
            if not getattr(cfg, name) > 0:
                raise ValueError(f"EventConfig.{name} must be > 0, got {getattr(cfg, name)!r}")
        for name in ("sigma_c", "refractory_us", "shot_noise_hz", "lpf_cutoff_hz"):
            if not getattr(cfg, name) >= 0:
                raise ValueError(f"EventConfig.{name} must be >= 0, got {getattr(cfg, name)!r}")
        self.shape = (h, w)
        self.config = cfg
        self._c_pos, self._c_neg = float(cfg.c_pos), float(cfg.c_neg)
        self._sigma_c = float(cfg.sigma_c)
        self._refr = int(cfg.refractory_us)
        self._noise_hz = float(cfg.shot_noise_hz)
        self._fc = float(cfg.lpf_cutoff_hz)
        self._eps = float(cfg.log_eps)
        self._seed = cfg.seed
        n = h * w
        self._xb = max(1, (w - 1).bit_length())
        self._yb = max(1, (h - 1).bit_length())
        self._low_bits = 1 + self._xb + self._yb                # sort key = t_rel | y | x | p
        self._max_dt = (1 << (62 - self._low_bits)) - 1
        pix = np.arange(n)
        self._xs = (pix % w).astype(np.uint16)                  # pixel coordinates by flat index
        self._ys = (pix // w).astype(np.uint16)
        self._x = np.empty(n, _F)                               # work buffers
        self._d = np.empty(n, _F)
        self._a = np.empty(n, _F)
        self._mask = np.empty(n, bool)
        self.reset()

    def reset(self):
        """Forget the reference / filter / refractory state and re-seed (thresholds are redrawn)."""
        n = self.shape[0] * self.shape[1]
        self._ref: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._last = np.full(n, _NEVER, np.int64) if self._refr > 0 else None
        self._rng = np.random.default_rng(self._seed)
        if self._sigma_c > 0:
            f = np.maximum(1.0 + self._sigma_c * self._rng.standard_normal((2, n)), _MIN_C_FACTOR)
            self._cp, self._cn = self._c_pos * f[0], self._c_neg * f[1]
            c_min = min(self._cp.min(), self._cn.min())
        else:
            self._cp, self._cn = self._c_pos, self._c_neg
            c_min = min(self._c_pos, self._c_neg)
        self._c_lo = _F(c_min * (1.0 - 2.0 * _TOL))             # |d| below this can never make an event

    def step(self, lum: np.ndarray, t_prev_us: int, t_now_us: int) -> np.ndarray:
        lum = self._check_lum(lum)
        first = self._ref is None
        if not first:
            t0, dt = self._interval(t_prev_us, t_now_us)
        sig = self._log_lum(lum)
        if first:
            self._ref = sig.copy()
            if self._fc > 0:
                self._y = sig.copy()
            return empty_events()

        if self._fc > 0:
            sig -= self._y
            sig *= _F(-np.expm1(-2.0 * np.pi * self._fc * dt * 1e-6))
            self._y += sig
            sig = self._y
        d = np.subtract(sig, self._ref, out=self._d)
        np.abs(d, out=self._a)
        np.greater_equal(self._a, self._c_lo, out=self._mask)
        idx = np.flatnonzero(self._mask)

        parts = self._signal(idx, d, t0, dt) if idx.size else None
        noise = self._noise(dt) if self._noise_hz > 0 else None
        if parts is None and noise is None:
            return empty_events()
        if noise is None and dt <= _RADIX_MAX_DT:
            return self._emit_radix(*parts, t0)
        return self._emit_keysort(parts, noise, t0)

    def _check_lum(self, lum) -> np.ndarray:
        lum = np.asarray(lum)
        if lum.shape != self.shape:
            raise ValueError(f"lum has shape {lum.shape}, expected {self.shape}")
        if lum.dtype.kind not in "fiub":
            raise ValueError(f"lum must be a real numeric array, got dtype {lum.dtype}")
        if not np.isfinite(lum).all():
            raise ValueError("lum contains non-finite values")
        return lum

    def _log_lum(self, lum: np.ndarray) -> np.ndarray:
        x = self._x
        np.maximum(lum.reshape(-1), 0, out=x, dtype=_F, casting="same_kind")
        x += self._eps
        np.log(x, out=x)
        return x

    def _interval(self, t_prev_us, t_now_us) -> Tuple[int, int]:
        t0, t1 = _us(t_prev_us, "t_prev_us"), _us(t_now_us, "t_now_us")
        if t1 <= t0:
            raise ValueError(f"t_now_us must be > t_prev_us, got {t1} <= {t0}")
        if t1 - t0 > self._max_dt or abs(t0) >= 1 << 62 or abs(t1) >= 1 << 62:
            raise ValueError(f"time step {t1 - t0} us out of range (max {self._max_dt} us for {self.shape})")
        return t0, t1 - t0

    def _signal(self, idx: np.ndarray, d: np.ndarray, t0: int, dt: int):
        """Events of the changed pixels `idx`; returns (ci, t_rel, x, y, pol) or None.

        ci: index into idx of each event, in raster (pixel-major) order; t_rel in [1, dt] (us after t_prev);
        x, y (uint16) and pol (uint8, 1 = ON) are per changed pixel.
        """
        dv = d[idx].astype(np.float64)
        pos = dv > 0
        c = np.where(pos, self._cp if np.ndim(self._cp) == 0 else self._cp[idx],
                     self._cn if np.ndim(self._cn) == 0 else self._cn[idx])
        ad = np.abs(dv)
        n = np.floor(ad / c + _TOL).astype(np.int64)             # rows with n == 0 (float slack) just emit nothing
        step_us = c * (dt / ad)                                  # spacing of the events inside the step

        if self._refr > 0:
            ci, t_rel, kept = _refractory(n, step_us, self._last[idx] - t0, dt, self._refr)
            if ci.size == 0:
                return None
            e = kept > 0
            last = t_rel[np.cumsum(kept)[e] - 1] + t0            # kept events are contiguous per pixel
            self._last[idx[e]] = last
        else:
            kept = n
            ci = np.repeat(np.arange(idx.size), n)
            if ci.size == 0:
                return None
            k = np.arange(1, ci.size + 1) - (np.cumsum(n) - n)[ci]     # rank 1..n of each event inside its pixel
            t_rel = _quantise(k * step_us[ci], dt)

        self._ref[idx] += np.copysign(kept * c, dv)
        return ci, t_rel, self._xs[idx], self._ys[idx], pos.astype(np.uint8)

    def _noise(self, dt: int):
        """Background-activity events: (t_rel, x, y, pol) or None."""
        h, w = self.shape
        n = int(self._rng.poisson(self._noise_hz * dt * 1e-6 * h * w))
        if n == 0:
            return None
        t_rel = self._rng.integers(1, dt + 1, n, dtype=np.int64)
        pix = self._rng.integers(0, h * w, n, dtype=np.int64)
        pol = self._rng.integers(0, 2, n, dtype=np.int64)
        return t_rel, self._xs[pix], self._ys[pix], pol

    def _emit_radix(self, ci, t_rel, xs, ys, pol, t0: int) -> np.ndarray:
        """Fast path: events are already in raster order, a stable 16-bit radix sort on t finishes."""
        order = np.argsort(t_rel.astype(np.uint16), kind="stable")
        cs = ci[order]
        ev = np.empty(ci.size, EVENT_DTYPE)
        np.add(t_rel[order], t0, out=ev["t"])
        ev["x"] = xs[cs]
        ev["y"] = ys[cs]
        ev["p"] = pol[cs]
        return ev

    def _emit_keysort(self, parts, noise, t0: int) -> np.ndarray:
        """General path: sort packed int64 keys (t_rel | y | x | p) => order by (t, raster, p)."""
        sh, xb = self._low_bits, self._xb
        keys = []
        if parts is not None:
            ci, t_rel, xs, ys, pol = parts
            low = (((ys.astype(np.int64) << xb) | xs) << 1) | pol
            keys.append((t_rel << sh) | low[ci])
        if noise is not None:
            t_rel, xs, ys, pol = noise
            keys.append((t_rel << sh) | (((ys.astype(np.int64) << xb) | xs) << 1) | pol)
        key = keys[0] if len(keys) == 1 else np.concatenate(keys)
        key.sort()
        low = key & ((1 << sh) - 1)
        ev = np.empty(key.size, EVENT_DTYPE)
        ev["t"] = key >> sh
        ev["t"] += t0
        ev["x"] = (low >> 1) & ((1 << xb) - 1)
        ev["y"] = low >> (1 + xb)
        ev["p"] = low & 1
        return ev


def _us(v, name: str) -> int:
    try:
        i = int(v)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer number of microseconds, got {v!r}") from None
    if i != v:
        raise ValueError(f"{name} must be an integer number of microseconds, got {v!r}")
    return i


def _quantise(t: np.ndarray, dt: int) -> np.ndarray:
    """Round event times (us after t_prev, float) to whole microseconds and clamp into [1, dt]."""
    t += 0.5
    out = t.astype(np.int64)                                     # floor for positive values
    return np.clip(out, 1, dt, out=out)


def _t_rel(k: np.ndarray, step_us: np.ndarray, dt: int) -> np.ndarray:
    """Time of candidate k of a pixel, as float (same rounding as _quantise)."""
    return np.clip(np.floor(k * step_us + 0.5), 1.0, float(dt))


def _refractory(n, step_us, last_rel, dt: int, refr: int):
    """Refractory scan of the candidate events k*step_us (k = 1..n) of every changed pixel.

    last_rel: last emitted time of each pixel relative to t_prev (very negative if none). Scans in passes: pass p
    finds, for the pixels still active, the first candidate whose time is >= (previous emitted time + refr).
    Returns (ci, t_rel, kept): events in pixel-major order (each pixel's events contiguous, time ascending),
    their times, and the number emitted per pixel.
    """
    m = n.size
    act = np.arange(m)
    sa, na = step_us, n
    ka = np.zeros(m)                                             # rank of the last emitted candidate
    ta = (last_rel + refr).astype(np.float64)                    # earliest allowed time of the next event
    passes = []
    while act.size:
        k = np.maximum(np.ceil((ta - 0.5) / sa), ka + 1)
        k += _t_rel(k, sa, dt) < ta                              # the division estimate can be off by one
        km = k - 1
        k = np.where((km > ka) & (_t_rel(km, sa, dt) >= ta), km, k)
        t = _t_rel(k, sa, dt)
        ok = (k <= na) & (t >= ta)
        if not ok.any():
            break
        passes.append((act[ok], t[ok].astype(np.int64)))
        ok &= k < na                                             # nothing left to emit after rank n
        act, sa, na, ka, ta = act[ok], sa[ok], na[ok], k[ok], t[ok] + refr
    if not passes:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.zeros(m, np.int64)
    kept = np.bincount(np.concatenate([a for a, _ in passes]), minlength=m)
    start = np.cumsum(kept) - kept
    ci = np.empty(int(kept.sum()), np.int64)
    t_rel = np.empty_like(ci)
    for p, (a, t) in enumerate(passes):                          # the p-th event of pixel a lands at start[a] + p
        pos = start[a] + p
        ci[pos] = a
        t_rel[pos] = t
    return ci, t_rel, kept
