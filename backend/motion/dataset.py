"""Sequences of (residual, conditioning), the sequence-level split, normalisation and window sampling. numpy only.

Windows are trained on only where the person walks: a window whose mean speed is below ``MIN_WINDOW_SPEED`` or that
contains a sample whose heading is held (the path stands still, see ``decompose.PathFrame.held``) is skipped."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence as Seq, Tuple

import numpy as np

from backend.motion.decompose import N_COND, N_RESIDUAL, conditioning, split
from backend.motion.tracks import Track

WINDOW = 200     # 4 s at 50 Hz
CONTEXT = 50     # 1 s shared with the previous window
HOP = WINDOW - CONTEXT
MIN_WINDOW_SPEED = 0.2   # m/s: slower windows have no defined heading and are not trained on


@dataclass(frozen=True)
class Sequence:
    source: str
    name: str
    residual: np.ndarray   # (N, 6)
    cond: np.ndarray       # (N, 3) speed, yaw rate, curvature
    held: Optional[np.ndarray] = None   # (N,) bool: heading held (path standing still); None: no such sample

    def __len__(self) -> int:
        return len(self.residual)


def build_sequences(tracks: Iterable[Track]) -> List[Sequence]:
    out = []
    for tr in tracks:
        path, residual = split(tr.positions, tr.quats)
        out.append(Sequence(tr.source, tr.name, residual, conditioning(path), path.held))
    return out


def recording_key(seq: Sequence) -> str:
    """Segments of one recording ("walk#0", "walk#1") share a key, so they never straddle two splits."""
    return f"{seq.source}/{seq.name.split('#')[0]}"


def assign_splits(seqs: Seq[Sequence], val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0,
                  holdout_sources: Iterable[str] = ()) -> Dict[str, str]:
    """recording key -> "train" | "val" | "test", decided by a hash of (seed, key): stable when sequences are added.
    Every recording of ``holdout_sources`` goes to "test"."""
    if not (0 <= val_frac and 0 <= test_frac and val_frac + test_frac < 1):
        raise ValueError(f"val_frac + test_frac must be in [0, 1), got {val_frac} + {test_frac}")
    holdout = set(holdout_sources)
    out: Dict[str, str] = {}
    for s in seqs:
        key = recording_key(s)
        if s.source in holdout:
            out[key] = "test"
            continue
        u = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0x100000000
        out[key] = "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"
    return out


def select(seqs: Seq[Sequence], splits: Dict[str, str], which: str) -> List[Sequence]:
    return [s for s in seqs if splits[recording_key(s)] == which]


@dataclass(frozen=True)
class Stats:
    """Per-channel normalisation and the speed range seen in training."""

    res_mean: np.ndarray
    res_std: np.ndarray
    cond_mean: np.ndarray
    cond_std: np.ndarray
    speed_range: Tuple[float, float]

    @classmethod
    def fit(cls, seqs: Seq[Sequence]) -> "Stats":
        if not seqs:
            raise ValueError("no training sequence")
        res = np.concatenate([s.residual for s in seqs])
        cond = np.concatenate([s.cond for s in seqs])
        moving = cond[cond[:, 0] > MIN_WINDOW_SPEED, 0]
        lo, hi = np.percentile(moving if len(moving) else cond[:, 0], [0.5, 99.5])
        return cls(res.mean(0), np.maximum(res.std(0), 1e-6), cond.mean(0), np.maximum(cond.std(0), 1e-6),
                   (float(lo), float(hi)))

    def norm_res(self, x):
        return (x - self.res_mean) / self.res_std

    def denorm_res(self, x):
        return x * self.res_std + self.res_mean

    def norm_cond(self, c):
        return (c - self.cond_mean) / self.cond_std

    def to_dict(self) -> dict:
        return {"res_mean": self.res_mean.tolist(), "res_std": self.res_std.tolist(),
                "cond_mean": self.cond_mean.tolist(), "cond_std": self.cond_std.tolist(),
                "speed_range": list(self.speed_range)}

    @classmethod
    def from_dict(cls, d: dict) -> "Stats":
        return cls(np.array(d["res_mean"]), np.array(d["res_std"]), np.array(d["cond_mean"]),
                   np.array(d["cond_std"]), (float(d["speed_range"][0]), float(d["speed_range"][1])))


class WindowSampler:
    """Random normalised windows, sources weighted equally, then recordings by length. Only windows with a mean speed
    of at least ``min_speed`` and no held-heading sample are drawn."""

    def __init__(self, seqs: Seq[Sequence], stats: Stats, hop: int = 25, min_speed: float = MIN_WINDOW_SPEED):
        self.stats = stats
        self._res = [stats.norm_res(s.residual).astype(np.float32) for s in seqs]
        self._cond = [stats.norm_cond(s.cond).astype(np.float32) for s in seqs]
        by_source: Dict[str, List[Tuple[int, int]]] = {}
        for i, s in enumerate(seqs):
            for start in range(0, len(s) - WINDOW + 1, hop):
                if s.held is not None and s.held[start:start + WINDOW].any():
                    continue
                if s.cond[start:start + WINDOW, 0].mean() >= min_speed:
                    by_source.setdefault(s.source, []).append((i, start))
        if not by_source:
            raise ValueError(f"no window of {WINDOW} samples with a mean speed >= {min_speed} m/s")
        self.sources = sorted(by_source)
        self._windows = [np.array(by_source[k]) for k in self.sources]
        self.n_windows = sum(len(w) for w in self._windows)

    def batch(self, rng: np.random.Generator, size: int) -> Dict[str, np.ndarray]:
        res = np.empty((size, WINDOW, N_RESIDUAL), np.float32)
        cond = np.empty((size, WINDOW, N_COND), np.float32)
        for b, k in enumerate(rng.integers(len(self.sources), size=size)):
            i, start = self._windows[k][rng.integers(len(self._windows[k]))]
            res[b] = self._res[i][start:start + WINDOW]
            cond[b] = self._cond[i][start:start + WINDOW]
        return {"residual": res, "cond": cond}
