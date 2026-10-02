"""Event -> tensor representations for spiking / frame-based networks (pure numpy, no torch).

Conventions: an event window is ``(t0_us, t1_us]`` (the simulator's step convention: an event exactly on a bin edge
belongs to the earlier bin, an event at ``t0_us`` belongs to the previous window and is ignored). ``num_bins`` bins
split the window evenly; the last bin ends at ``t1_us``. Polarity ``p`` is 0 (OFF) / 1 (ON).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from backend.types import EVENT_DTYPE

__all__ = ["bin_end_times_us", "events_to_count_frames", "events_to_voxel_grid", "rebase_events"]


def _window(num_bins: int, t0_us: int, t1_us: int, height: int, width: int) -> Tuple[int, int]:
    if num_bins < 1 or height < 1 or width < 1:
        raise ValueError("num_bins, height and width must be positive")
    span = int(t1_us) - int(t0_us)
    if span <= 0:
        raise ValueError(f"empty time window ({t0_us}, {t1_us}]")
    return int(num_bins), span


def _in_window(events: np.ndarray, t0_us: int, t1_us: int, height: int, width: int) -> np.ndarray:
    """Events with ``t0_us < t <= t1_us``; raises if a pixel lies outside the sensor."""
    events = np.asarray(events)
    if len(events) == 0:
        return events
    if int(events["x"].max()) >= width or int(events["y"].max()) >= height:
        raise ValueError(f"event outside the {height}x{width} sensor")
    t = events["t"]
    if int(t.min()) > t0_us and int(t.max()) <= t1_us:      # the usual case: skip the mask and the copy
        return events
    return events[(t > t0_us) & (t <= t1_us)]


def bin_end_times_us(num_bins: int, t0_us: int, t1_us: int) -> np.ndarray:
    """(num_bins,) int64 end time of each bin relative to ``t0_us``: ``floor((k + 1) * (t1 - t0) / num_bins)``."""
    num_bins, span = _window(num_bins, t0_us, t1_us, 1, 1)
    return (np.arange(1, num_bins + 1, dtype=np.int64) * span) // num_bins


def rebase_events(events: np.ndarray, t0_us: int) -> np.ndarray:
    """Copy of ``events`` with ``t`` shifted so that ``t0_us`` becomes 0 (int64 microseconds)."""
    out = np.array(events, dtype=EVENT_DTYPE, copy=True)
    out["t"] -= int(t0_us)
    return out


def events_to_count_frames(
    events: np.ndarray, num_bins: int, t0_us: int, t1_us: int, height: int, width: int,
    split_polarity: bool = True, binary: bool = False,
) -> np.ndarray:
    """Per-bin, per-pixel event counts: float32 ``(T, 2, H, W)`` (channel = polarity: 0 OFF, 1 ON).

    ``split_polarity=False`` sums the signed polarity (+1 / -1) into ``(T, 1, H, W)``.
    ``binary=True`` gives spikes: ``count > 0`` per channel, or ``sign(net count)`` when polarity is not split.
    """
    num_bins, span = _window(num_bins, t0_us, t1_us, height, width)
    ev = _in_window(events, t0_us, t1_us, height, width)
    channels = 2 if split_polarity else 1
    size = num_bins * channels * height * width
    if len(ev) == 0:
        return np.zeros((num_bins, channels, height, width), dtype=np.float32)
    flat = ev["t"] - np.int64(t0_us)                                 # int64, in place from here on
    flat *= num_bins
    flat += span - 1
    flat //= span                                                    # ceil(dt * T / span) = bin + 1
    flat -= 1
    if split_polarity:
        flat *= 2
        flat += ev["p"] != 0
    flat *= height
    flat += ev["y"]
    flat *= width
    flat += ev["x"]
    if split_polarity:
        counts = np.bincount(flat, minlength=size)
    else:
        counts = np.bincount(flat, weights=np.where(ev["p"] != 0, 1.0, -1.0), minlength=size)
    if binary:
        counts = np.sign(counts) if not split_polarity else (counts > 0)
    return counts.astype(np.float32).reshape(num_bins, channels, height, width)


def events_to_voxel_grid(
    events: np.ndarray, num_bins: int, t0_us: int, t1_us: int, height: int, width: int
) -> np.ndarray:
    """E2VID voxel grid: float32 ``(T, H, W)``, signed polarity, bilinear in time.

    An event at ``t`` is placed at ``t* = (T - 1) * (t - t0) / (t1 - t0)`` and adds ``+-(1 - |k - t*|)`` to the bins
    ``floor(t*)`` and ``floor(t*) + 1`` (so bin ``k`` is centred on ``t0 + k * (t1 - t0) / (T - 1)``). The weights of an
    event sum to its polarity, hence ``grid.sum() = #ON - #OFF``.
    """
    num_bins, span = _window(num_bins, t0_us, t1_us, height, width)
    ev = _in_window(events, t0_us, t1_us, height, width)
    size = num_bins * height * width
    if len(ev) == 0:
        return np.zeros((num_bins, height, width), dtype=np.float32)
    ts = (num_bins - 1) * (ev["t"].astype(np.float64) - t0_us) / span
    b0 = np.minimum(np.floor(ts).astype(np.int64), num_bins - 1)
    frac = ts - b0
    sign = np.where(ev["p"] != 0, 1.0, -1.0)
    pix = ev["y"].astype(np.int64) * width + ev["x"]
    grid = np.bincount(b0 * height * width + pix, weights=sign * (1.0 - frac), minlength=size)
    up = b0 + 1 < num_bins
    grid += np.bincount((b0[up] + 1) * height * width + pix[up], weights=(sign * frac)[up], minlength=size)
    return grid.astype(np.float32).reshape(num_bins, height, width)
