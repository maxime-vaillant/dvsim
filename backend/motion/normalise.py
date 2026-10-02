"""RawTrack -> Track segments: clean, convert to DVSim's frame, split at gaps, resample at RATE_HZ."""
from __future__ import annotations

from typing import List

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation, Slerp

from backend.motion.tracks import DT_US, RATE_HZ, RawTrack, Track

MAX_GAP_S = 0.2
MIN_DURATION_S = 4.0
_OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0])  # the same rotation maps either camera frame to the other


def rotation_to_y(up) -> Rotation:
    """Smallest rotation taking the direction ``up`` to +Y."""
    u = np.asarray(up, dtype=np.float64)
    u = u / np.linalg.norm(u)
    y = np.array([0.0, 1.0, 0.0])
    axis = np.cross(u, y)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return Rotation.identity() if u @ y > 0 else Rotation.from_rotvec([np.pi, 0.0, 0.0])
    return Rotation.from_rotvec(axis / s * np.arctan2(s, u @ y))


def normalise(raw: RawTrack, source: str, max_gap_s: float = MAX_GAP_S,
              min_duration_s: float = MIN_DURATION_S) -> List[Track]:
    """Segments of ``raw`` in DVSim's frame at RATE_HZ. Gaps longer than ``max_gap_s`` split the track; segments
    shorter than ``min_duration_s`` are dropped. Rows that are not finite or not strictly later than the previous
    kept row are dropped."""
    t = np.asarray(raw.times_us, dtype=np.int64)
    p = np.asarray(raw.positions, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(raw.quats, dtype=np.float64).reshape(-1, 4)
    if not (len(t) == len(p) == len(q)):
        raise ValueError(f"{raw.name}: times, positions and quats differ in length ({len(t)}, {len(p)}, {len(q)})")
    ok = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 1e-9)
    t, p, q = t[ok], p[ok], q[ok]
    if len(t):  # strictly increasing: drop rows that do not move time forward
        later = t > np.concatenate([[t[0] - 1], np.maximum.accumulate(t)[:-1]])
        t, p, q = t[later], p[later], q[later]
    if len(t) < 2:
        return []

    world = rotation_to_y(raw.frame.up)
    rot = world * Rotation.from_quat(q)
    if raw.frame.camera == "opencv":
        rot = rot * Rotation.from_matrix(_OPENCV_TO_OPENGL)
    p = world.apply(p)

    cuts = np.flatnonzero(np.diff(t) > max_gap_s * 1e6) + 1
    bounds = np.concatenate([[0], cuts, [len(t)]])
    pieces = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]
    out: List[Track] = []
    for k, (a, b) in enumerate(pieces):
        ts, ps, rs = t[a:b], p[a:b], rot[a:b]
        n = int((ts[-1] - ts[0]) // DT_US) + 1
        if (n - 1) / RATE_HZ < min_duration_s:
            continue
        grid = ts[0] + np.arange(n, dtype=np.int64) * DT_US
        pos = np.stack([np.interp(grid, ts, ps[:, i]) for i in range(3)], axis=1)
        quat = Slerp(ts.astype(np.float64), rs)(grid.astype(np.float64)).as_quat()
        name = raw.name if len(pieces) == 1 else f"{raw.name}#{k}"
        out.append(Track(source, name, grid, pos, quat))
    return out


def heading_agreement(track: Track, min_speed: float = 0.3, smooth_s: float = 1.0) -> float:
    """Mean cosine between the horizontal direction of travel and the horizontal camera direction, over the moving
    samples (NaN when there are none). A walking person with a head-mounted camera gives a value near 1; a wrong
    frame convention gives something near 0 or negative."""
    from backend.trajectories.trajectory import quats_to_fronts

    v = ndimage.gaussian_filter1d(np.gradient(track.positions, 1.0 / RATE_HZ, axis=0), smooth_s * RATE_HZ / 3.0,
                                  axis=0, mode="nearest")
    vh = v[:, [0, 2]]
    speed = np.linalg.norm(vh, axis=1)
    f = quats_to_fronts(track.quats)[:, [0, 2]]
    fn = np.linalg.norm(f, axis=1)
    moving = (speed > min_speed) & (fn > 1e-6)
    if not moving.any():
        return float("nan")
    cos = np.sum(vh[moving] * f[moving], axis=1) / (speed[moving] * fn[moving])
    return float(cos.mean())
