from __future__ import annotations

import numpy as np

from backend.trajectories.generated import TrajectoryGenerator, as_rng
from backend.trajectories.trajectory import WaypointTrajectory


class RectilinearTrajectoryGenerator(TrajectoryGenerator):
    """Random horizontal (constant y) straight segment through free space, constant forward direction.

    Each round evaluates ``max_attempts`` random candidate segments at once and keeps the first longer than
    ``valid_distance`` (else the longest); rounds repeat until a valid segment exists. ``fixed_height`` forces y
    (ignored for the start when ``initial_point`` is given, which then also sets y).
    """

    def __init__(self, mesh_or_grid, min_clearance: float = 0.3, valid_distance: float = 5.0,
                 max_attempts: int = 300, fixed_height=None, max_rounds: int = 50):
        super().__init__(mesh_or_grid, min_clearance)
        self.valid_distance = float(valid_distance)
        self.max_attempts = int(max_attempts)
        self.fixed_height = fixed_height
        self.max_rounds = int(max_rounds)

    def generate(self, duration_us: int, rng=None, initial_point=None) -> WaypointTrajectory:
        duration_us = self._check_duration(duration_us)
        rng = as_rng(rng)
        c, n = self.min_clearance, self.max_attempts
        start = None if initial_point is None else self._start_point(initial_point)
        for _ in range(self.max_rounds):
            p1 = np.tile(start, (n, 1)) if start is not None else self.grid.sample_free(n, c, rng)
            p2 = self.grid.sample_free(n, c, rng)
            if self.fixed_height is not None and start is None:
                p1[:, 1] = self.fixed_height
            p2[:, 1] = p1[:, 1]
            length = np.linalg.norm(p2 - p1, axis=1)
            ok = (length > 1e-6) & self.grid.segments_free(p1, p2, c)
            if ok.any():
                idx = np.flatnonzero(ok)
                long = idx[length[idx] > self.valid_distance]
                best = long[0] if len(long) else idx[np.argmax(length[idx])]
                front = p2[best] - p1[best]
                return WaypointTrajectory.from_waypoints(
                    [0, duration_us], np.stack([p1[best], p2[best]]), fronts=np.stack([front, front]))
        raise RuntimeError(
            f"no free horizontal segment found in {self.max_rounds * n} attempts with clearance {c}; "
            "the free space may be too small (try a smaller min_clearance)")

