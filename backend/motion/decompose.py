"""Trajectory <-> (smooth path, residual). The only place that defines the split (training and sampling share it).

The path is the low-passed position with a heading taken from its horizontal tangent (no pitch, no roll). The
residual is what the person does on top of it, expressed in the path's own frame (axes: x right, y up, z back):
the position offset (3) and the camera rotation relative to the path frame as a rotation vector (3).
Where the path stands still the heading is held (``PathFrame.held``); the yaw rate is zero there and next to it, so
turning round during a stop is not a spike of the conditioning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from backend.motion.tracks import RATE_HZ

CUTOFF_HZ = 0.5      # below the step frequency (about 1.8 Hz)
MIN_SPEED = 0.1      # m/s: slower than this the heading is undefined and is held
MIN_CURVATURE_SPEED = 0.3
N_RESIDUAL = 6
N_COND = 3           # speed (m/s), yaw rate (rad/s), curvature (1/m)


@dataclass(frozen=True)
class PathFrame:
    positions: np.ndarray   # (N, 3) smooth path
    rot: Rotation           # N rotations, path frame -> world (forward = -Z, up = +Y)
    speed: np.ndarray       # (N,) horizontal speed, m/s
    yaw_rate: np.ndarray    # (N,) rad/s, counter-clockwise seen from above
    held: np.ndarray        # (N,) bool: the heading is held or back-filled (speed <= MIN_SPEED)

    def __len__(self) -> int:
        return len(self.positions)


def _smooth(p: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass whose ends continue the motion: the signal is extended by point reflection about its first
    and last sample (a straight walk stays straight), not held constant (which would slow the path down at the ends)."""
    pad = min(int(4 * sigma) + 1, len(p) - 1)
    head = 2 * p[0] - p[pad:0:-1]
    tail = 2 * p[-1] - p[-2:-pad - 2:-1]
    return ndimage.gaussian_filter1d(np.concatenate([head, p, tail]), sigma, axis=0, mode="nearest")[pad:-pad]


def extract_path(positions, rate_hz: float = RATE_HZ, cutoff_hz: float = CUTOFF_HZ) -> PathFrame:
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 5:
        raise ValueError(f"expected (N>=5, 3) positions, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("positions contain non-finite values")
    sigma = 0.1325 / cutoff_hz * rate_hz      # Gaussian whose -3 dB point is cutoff_hz
    path = _smooth(p, sigma)
    v = np.gradient(path, 1.0 / rate_hz, axis=0)[:, [0, 2]]
    speed = np.linalg.norm(v, axis=1)
    valid = speed > MIN_SPEED
    h = np.zeros_like(v)
    h[:] = [0.0, -1.0]                        # nothing moves: look along -Z
    if valid.any():
        idx = np.where(valid, np.arange(len(v)), -1)
        idx = np.maximum.accumulate(idx)      # hold the last valid heading ...
        first = int(np.argmax(valid))
        idx[idx < 0] = first                  # ... and back-fill the start
        h = v[idx] / speed[idx][:, None]
    psi = np.unwrap(np.arctan2(-h[:, 0], -h[:, 1]))   # forward = (-sin psi, 0, -cos psi)
    rot = Rotation.from_euler("y", psi)
    yaw_rate = np.gradient(psi, 1.0 / rate_hz)
    held = ~valid
    near = held.copy()                        # the held span and one sample each side: the heading jumps there
    near[1:] |= held[:-1]
    near[:-1] |= held[1:]
    yaw_rate[near] = 0.0
    return PathFrame(path, rot, speed, yaw_rate, held)


def conditioning(path: PathFrame) -> np.ndarray:
    """(N, 3): speed, yaw rate, curvature."""
    curvature = path.yaw_rate / np.maximum(path.speed, MIN_CURVATURE_SPEED)
    return np.stack([path.speed, path.yaw_rate, curvature], axis=1)


def split(positions, quats, rate_hz: float = RATE_HZ) -> Tuple[PathFrame, np.ndarray]:
    """(path, residual (N, 6)) of a track."""
    path = extract_path(positions, rate_hz)
    inv = path.rot.inv()
    offset = inv.apply(np.asarray(positions, dtype=np.float64) - path.positions)
    rvec = (inv * Rotation.from_quat(np.asarray(quats, dtype=np.float64))).as_rotvec()
    return path, np.concatenate([offset, rvec], axis=1)


def compose(path: PathFrame, residual) -> Tuple[np.ndarray, np.ndarray]:
    """(positions (N, 3), quats (N, 4)) of a path plus a residual (N, 6): the inverse of ``split``."""
    r = np.asarray(residual, dtype=np.float64)
    if r.shape != (len(path), N_RESIDUAL):
        raise ValueError(f"residual must have shape ({len(path)}, {N_RESIDUAL}), got {r.shape}")
    positions = path.positions + path.rot.apply(r[:, :3])
    quats = (path.rot * Rotation.from_rotvec(r[:, 3:])).as_quat()
    return positions, quats
