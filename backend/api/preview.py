"""POST /api/trajectories/generate: run a trajectory generator on an uploaded scene without simulating it.

The interface draws the returned paths and plays them in its camera views. The seed used is always returned (a
random one is drawn when the request has none): a job with the same spec and seed simulates exactly these paths.
The last scene and its free-space maps stay loaded, so asking again with another seed only re-plans.
"""
from __future__ import annotations

import secrets
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import HTTPException

from backend.api.schemas import TrajectoryGenerate
from backend.api.stores import SceneStore


class TrajectoryPreviewer:
    def __init__(self) -> None:
        self._lock = threading.Lock()   # one scene in memory, one generation at a time
        self._scene_id: Optional[str] = None
        self._scene = None
        self._cache: Dict[Any, Any] = {}

    def generate(self, body: TrajectoryGenerate, scenes: SceneStore) -> Dict[str, Any]:
        from backend.pipeline import build_trajectories
        from backend.scene import load_scene

        spec_json = scenes.spec(body.scene_id)  # 404 before any work
        spec = body.trajectory.model_dump(exclude_none=True)
        spec.setdefault("seed", secrets.randbelow(2 ** 31))
        with self._lock:
            if self._scene_id != body.scene_id:
                self._scene_id, self._scene, self._cache = None, None, {}
                try:
                    self._scene = load_scene(spec_json, root=scenes.path(body.scene_id))
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=f"cannot load the scene: {exc}") from None
                self._scene_id = body.scene_id
            try:
                trajs = build_trajectories(spec, self._scene, self._cache)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=422, detail=f"cannot generate the trajectory: {exc}") from None
        return {
            "seed": spec["seed"],
            "trajectory": spec,
            "trajectories": [_points(t, body.max_points) for t in trajs],
        }


def _points(traj, max_points: int) -> Dict[str, Any]:
    """Legacy records (ms, x y z, forward vector), thinned evenly to at most ``max_points`` (first and last kept)."""
    records: List[Dict[str, float]] = traj.to_records()
    if len(records) > max_points:
        keep = np.unique(np.rint(np.linspace(0, len(records) - 1, max_points)).astype(int))
        records = [records[i] for i in keep]
    keys = ("timestamp", "x", "y", "z", "rx", "ry", "rz")
    return {
        "duration_s": (traj.end_us - traj.start_us) / 1e6,
        "points": [{k: round(r[k], 6) for k in keys} for r in records],
    }
