from __future__ import annotations

import numpy as np
from scipy import interpolate

from backend.trajectories.generated import TrajectoryGenerator, as_rng
from backend.trajectories.trajectory import WaypointTrajectory, fronts_to_quats, slerp_many


class SplineTrajectoryGenerator(TrajectoryGenerator):
    """Random walk of ``n_key_points`` steps of length ``step`` through free space, joined by a smooth spline.

    The camera looks along the horizontal component of the motion (no roll, pitch 0); orientation is slerped between
    key points. The spline is resampled at constant speed with ``num_samples_per_point * n_key_points`` waypoints and
    timestamps are uniform over ``duration_us``. Every waypoint and every segment between consecutive waypoints is
    checked against the grid; walks whose smooth path clips an obstacle are redrawn.
    ``max_turn_deg`` (default None = any direction, as before) limits the angle between consecutive steps: a
    reversing random walk makes near-cusps in the path.
    """

    def __init__(self, mesh_or_grid, min_clearance: float = 0.3, step: float = 1.0, max_attempts: int = 50,
                 max_turn_deg=None):
        super().__init__(mesh_or_grid, min_clearance)
        self.step = float(step)
        self.max_attempts = int(max_attempts)
        self.max_turn_deg = max_turn_deg

    def generate(self, duration_us: int, n_key_points: int = 10, rng=None, initial_point=None,
                 num_samples_per_point: int = 20) -> WaypointTrajectory:
        duration_us = self._check_duration(duration_us)
        if isinstance(n_key_points, (bool, np.bool_)) or not isinstance(n_key_points, (int, np.integer)) or n_key_points < 2:
            raise ValueError(f"n_key_points must be an integer >= 2, got {n_key_points!r}")
        n_samples = int(n_key_points) * int(num_samples_per_point)
        if n_samples < 2 or duration_us < n_samples:
            raise ValueError(f"duration_us={duration_us} is too short for {n_samples} waypoints")
        rng = as_rng(rng)
        start = None if initial_point is None else self._start_point(initial_point)
        for _ in range(self.max_attempts):
            keys = self._key_points(int(n_key_points), start, rng)
            if keys is None:
                continue
            traj = self._smooth(keys, n_samples, duration_us, rng)
            if traj is not None:
                return traj
        raise RuntimeError(
            f"no collision-free spline found in {self.max_attempts} attempts with clearance {self.min_clearance} "
            f"and step {self.step}; the free space may be too narrow (try a smaller step or min_clearance)")

    def _key_points(self, n: int, start, rng, tries: int = 20, batch: int = 64):
        """Random walk in free space; every step is drawn from ``batch`` candidate directions at once."""
        c = self.min_clearance
        keys = np.empty((n, 3))
        keys[0] = self.grid.sample_free(1, c, rng)[0] if start is None else start
        cos_turn = -2.0 if self.max_turn_deg is None else np.cos(np.radians(self.max_turn_deg))
        for i in range(1, n):
            cur = np.broadcast_to(keys[i - 1], (batch, 3))
            prev = (keys[i - 1] - keys[i - 2]) / self.step if i > 1 else None
            for _ in range(tries):
                d = rng.normal(size=(batch, 3))
                d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                cand = cur + self.step * d
                ok = self.grid.segments_free(cur, cand, c)
                if prev is not None:
                    ok &= d @ prev >= cos_turn
                if ok.any():
                    keys[i] = cand[np.argmax(ok)]
                    break
            else:
                return None
        return keys

    def _key_fronts(self, keys: np.ndarray, rng) -> np.ndarray:
        """Horizontal direction of the step leaving each key point (the last key keeps its arrival direction)."""
        h = np.diff(keys, axis=0)
        h[:, 1] = 0.0
        norm = np.linalg.norm(h, axis=1)
        valid = norm > 1e-6 * self.step
        if valid.any():  # vertical steps keep the previous heading (the first ones take the next valid one)
            last = np.maximum.accumulate(np.where(valid, np.arange(len(h)), -1))
            last[last < 0] = np.argmax(valid)
            fronts = (h[last] / norm[last, None])
        else:
            a = rng.uniform(0.0, 2.0 * np.pi)
            fronts = np.tile([np.cos(a), 0.0, np.sin(a)], (len(h), 1))
        return np.concatenate([fronts, fronts[-1:]])

    def _smooth(self, keys: np.ndarray, n_samples: int, duration_us: int, rng):
        n = len(keys)
        tck, u_key = interpolate.splprep(keys.T, s=0, k=min(2, n - 1))
        u_dense = np.linspace(0.0, 1.0, 8 * n_samples + 1)
        dense = np.stack(interpolate.splev(u_dense, tck), axis=1)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
        s_t = np.linspace(0.0, s[-1], n_samples)  # constant speed
        pos = np.stack(interpolate.splev(np.interp(s_t, s, u_dense), tck), axis=1)
        if not self.grid.segments_free(pos[:-1], pos[1:], self.min_clearance).all():
            return None
        quats = fronts_to_quats(self._key_fronts(keys, rng))
        s_key = np.interp(u_key, u_dense, s)  # orientation is slerped along arc length between key points
        i = np.clip(np.searchsorted(s_key, s_t, side="right") - 1, 0, n - 2)
        alpha = np.clip((s_t - s_key[i]) / (s_key[i + 1] - s_key[i]), 0.0, 1.0)
        times = np.rint(np.linspace(0.0, duration_us, n_samples)).astype(np.int64)
        return WaypointTrajectory(times, pos, slerp_many(quats[i], quats[i + 1], alpha))
