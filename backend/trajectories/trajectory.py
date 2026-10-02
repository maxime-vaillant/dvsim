"""Camera paths as pure functions of time (see docs/architecture.md)."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Sequence

import numpy as np

from backend.pose import Pose

_MS_TO_US = 1000.0


class Trajectory(ABC):
    """ONE continuous camera path. ``pose_at`` is a pure function of time, clamped to [start_us, end_us]."""

    @property
    @abstractmethod
    def start_us(self) -> int: ...

    @property
    @abstractmethod
    def end_us(self) -> int: ...

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us

    @abstractmethod
    def pose_at(self, t_us: float) -> Pose: ...

    def poses_at(self, times_us) -> np.ndarray:
        """(M,7) poses (x y z qx qy qz qw); subclasses override with a vectorised version."""
        return np.stack([self.pose_at(t).as_array() for t in np.asarray(times_us, dtype=np.float64).reshape(-1)])


def slerp_many(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Vectorised slerp of (M,4) unit quaternions (short way round); exact at alpha = 0 and 1."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)[:, None]
    q1 = np.where(np.sum(q0 * q1, axis=1, keepdims=True) < 0.0, -q1, q1)
    theta = 2.0 * np.arctan2(np.linalg.norm(q1 - q0, axis=1), np.linalg.norm(q1 + q0, axis=1))[:, None]
    small = theta < 1e-8
    s = np.where(small, 1.0, np.sin(theta))
    w0 = np.where(small, 1.0 - a, np.sin((1.0 - a) * theta) / s)
    w1 = np.where(small, a, np.sin(a * theta) / s)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def fronts_to_quats(fronts) -> np.ndarray:
    """(N,3) forward vectors -> (N,4) quaternions without roll (``Pose.from_look``)."""
    f = np.asarray(fronts, dtype=np.float64).reshape(-1, 3)
    bad = np.flatnonzero(~np.all(np.isfinite(f), axis=1) | (np.linalg.norm(f, axis=1) < 1e-9))
    if len(bad):
        raise ValueError(f"forward vector at index {int(bad[0])} is zero or not finite: {f[bad[0]].tolist()}")
    return np.stack([Pose.from_look(np.zeros(3), v).quat for v in f])


def quats_to_fronts(quats) -> np.ndarray:
    """(N,4) quaternions -> (N,3) unit forward vectors (camera -Z axis in the world frame)."""
    x, y, z, w = np.asarray(quats, dtype=np.float64).reshape(-1, 4).T
    return -np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=1)


class WaypointTrajectory(Trajectory):
    """Waypoints (times_us (N,), positions (N,3), quats (N,4)); lerp position, slerp orientation.

    Quaternions are normalised and sign-aligned so consecutive ones have a non-negative dot product.
    """

    def __init__(self, times_us, positions, quats):
        t = np.asarray(times_us)
        if t.ndim != 1:
            raise ValueError(f"times_us must be 1-D, got shape {t.shape}")
        if t.dtype.kind == "f":
            if not np.all(np.isfinite(t)):
                raise ValueError("times_us contains non-finite values")
            t = np.rint(t)
        t = t.astype(np.int64)
        p = np.array(positions, dtype=np.float64)  # copies: the arrays below are frozen
        q = np.array(quats, dtype=np.float64)
        n = len(t)
        if n < 2:
            raise ValueError(f"a trajectory needs at least 2 waypoints, got {n}")
        if p.shape != (n, 3) or q.shape != (n, 4):
            raise ValueError(f"expected positions (N,3) and quats (N,4) for N={n}, got {p.shape} and {q.shape}")
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            raise ValueError("positions / orientations contain non-finite values")
        bad = np.flatnonzero(np.diff(t) <= 0)
        if len(bad):
            i = int(bad[0]) + 1
            raise ValueError(
                f"timestamps must be strictly increasing: waypoint {i} has t={int(t[i])} us <= t={int(t[i - 1])} us "
                f"of waypoint {i - 1} ({len(bad)} violation(s))"
            )
        norms = np.linalg.norm(q, axis=1)
        if np.any(norms < 1e-9):
            raise ValueError(f"zero quaternion at waypoint {int(np.argmax(norms < 1e-9))}")
        q = q / norms[:, None]
        flips = np.where(np.sum(q[:-1] * q[1:], axis=1) < 0.0, -1.0, 1.0)
        q = q * np.concatenate([[1.0], np.cumprod(flips)])[:, None]
        for a in (t, p, q):
            a.flags.writeable = False
        self.times_us, self.positions, self.quats = t, p, q
        self._rel = (t - t[0]).astype(np.float64)

    def __len__(self) -> int:
        return len(self.times_us)

    @property
    def start_us(self) -> int:
        return int(self.times_us[0])

    @property
    def end_us(self) -> int:
        return int(self.times_us[-1])

    def pose_at(self, t_us: float) -> Pose:
        return Pose.from_array(self.poses_at([t_us])[0])

    def poses_at(self, times_us) -> np.ndarray:
        t = np.asarray(times_us, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(t)):
            raise ValueError("query times must be finite")
        rel = np.clip(t - self.start_us, 0.0, self._rel[-1])
        i = np.clip(np.searchsorted(self._rel, rel, side="right") - 1, 0, len(self._rel) - 2)
        alpha = np.clip((rel - self._rel[i]) / (self._rel[i + 1] - self._rel[i]), 0.0, 1.0)
        pos = self.positions[i] * (1.0 - alpha)[:, None] + self.positions[i + 1] * alpha[:, None]
        return np.concatenate([pos, slerp_many(self.quats[i], self.quats[i + 1], alpha)], axis=1)

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_waypoints(cls, times_us, positions, quats=None, fronts=None) -> "WaypointTrajectory":
        """``fronts`` (forward vectors) give orientations without roll; neither given -> looking along -Z."""
        if quats is not None and fronts is not None:
            raise ValueError("give either quats or fronts, not both")
        p = np.asarray(positions, dtype=np.float64)
        if quats is None:
            quats = fronts_to_quats(fronts) if fronts is not None else np.tile([0.0, 0.0, 0.0, 1.0], (len(p), 1))
        return WaypointTrajectory(times_us, p, quats)

    @classmethod
    def from_json(cls, path_or_records, origin=(0.0, 0.0, 0.0), on_duplicate: str = "raise") -> "WaypointTrajectory":
        """Load legacy records ``{timestamp (ms), x, y, z, rx, ry, rz | qx, qy, qz, qw}`` (list, ``{"points": [...]}`` or a file).

        (rx, ry, rz) is a forward vector; quaternions win when both are present. ``origin`` is added to the positions.
        Timestamps keep their absolute value. Equal timestamps raise, or are dropped (first kept) with ``on_duplicate="drop"``.
        """
        return WaypointTrajectory(*_parse_records(path_or_records, origin, on_duplicate))

    # ---- export ---------------------------------------------------------
    def to_records(self) -> List[Dict[str, float]]:
        """Legacy JSON records (timestamp ms, x y z, rx ry rz forward vector) plus qx qy qz qw."""
        fw = quats_to_fronts(self.quats)
        out = []
        for t, p, f, q in zip(self.times_us, self.positions, fw, self.quats):
            out.append({
                "timestamp": int(t) / _MS_TO_US,
                "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                "rx": float(f[0]), "ry": float(f[1]), "rz": float(f[2]),
                "qx": float(q[0]), "qy": float(q[1]), "qz": float(q[2]), "qw": float(q[3]),
            })
        return out

    def save_json(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_records(), fh)


def _parse_records(source, origin, on_duplicate):
    if on_duplicate not in ("raise", "drop"):
        raise ValueError(f"on_duplicate must be 'raise' or 'drop', got {on_duplicate!r}")
    origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError(f"origin must be 3 finite numbers, got {origin.tolist()}")
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r") as fh:
            source = json.load(fh)
    if isinstance(source, Mapping) and "points" in source:
        source = source["points"]
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise ValueError("expected a list of trajectory records (or a path to a JSON file holding one)")
    if len(source) < 2:
        raise ValueError(f"a trajectory needs at least 2 records, got {len(source)}")

    ts, pos, quats, fronts, use_q = [], [], [], [], []
    for i, r in enumerate(source):
        if not isinstance(r, Mapping):
            raise ValueError(f"record {i}: expected an object, got {type(r).__name__}")
        missing = [k for k in ("timestamp", "x", "y", "z") if k not in r]
        has_q = all(k in r for k in ("qx", "qy", "qz", "qw"))
        has_f = all(k in r for k in ("rx", "ry", "rz"))
        if missing or not (has_q or has_f):
            need = missing + ([] if has_q or has_f else ["rx,ry,rz (forward vector) or qx,qy,qz,qw"])
            raise ValueError(f"record {i}: missing {', '.join(need)}")
        try:
            ts.append(float(r["timestamp"]))
            pos.append([float(r["x"]), float(r["y"]), float(r["z"])])
            quats.append([float(r[k]) for k in ("qx", "qy", "qz", "qw")] if has_q else [0.0, 0.0, 0.0, 1.0])
            fronts.append([0.0, 0.0, -1.0] if has_q else [float(r[k]) for k in ("rx", "ry", "rz")])
        except (TypeError, ValueError):
            raise ValueError(f"record {i}: non-numeric value in {dict(r)}") from None
        use_q.append(has_q)

    ts = np.asarray(ts)
    pos = np.asarray(pos)
    quats = np.asarray(quats)
    fronts = np.asarray(fronts)
    use_q = np.asarray(use_q)
    finite = np.isfinite(ts) & np.isfinite(pos).all(axis=1) & np.where(use_q, np.isfinite(quats).all(axis=1), np.isfinite(fronts).all(axis=1))
    if not finite.all():
        raise ValueError(f"record {int(np.argmin(finite))}: non-finite value (NaN or inf)")
    if (~use_q).any():
        idx = np.flatnonzero(~use_q)
        zero = np.linalg.norm(fronts[idx], axis=1) < 1e-9
        if zero.any():
            raise ValueError(f"record {int(idx[np.argmax(zero)])}: forward vector (rx, ry, rz) has zero length")
        quats[idx] = fronts_to_quats(fronts[idx])

    t_us = np.rint(ts * _MS_TO_US).astype(np.int64)
    if np.any(np.diff(t_us) < 0):
        i = int(np.argmax(np.diff(t_us) < 0)) + 1
        raise ValueError(f"record {i}: timestamp {ts[i]} ms goes backwards (previous record: {ts[i - 1]} ms)")
    dup = np.diff(t_us) == 0
    if dup.any():
        if on_duplicate == "raise":
            i = int(np.argmax(dup)) + 1
            raise ValueError(
                f"record {i}: duplicate timestamp {ts[i]} ms ({int(dup.sum())} duplicate(s)); "
                "pass on_duplicate='drop' to keep the first record of each"
            )
        keep = np.concatenate([[True], ~dup])
        t_us, pos, quats = t_us[keep], pos[keep], quats[keep]
    return t_us, pos + origin, quats
