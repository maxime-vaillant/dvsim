"""POST /api/jobs body -> worker request. Everything that can be checked before spawning is checked here (404 / 422)."""
from __future__ import annotations

import dataclasses
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import HTTPException

from backend.api.schemas import ImportedSpec, InlineSpec, JobCreate
from backend.api.settings import MAX_RESOLUTION, MAX_STEPS
from backend.api.stores import SceneStore, TrajectoryStore, clean_points
from backend.api.util import NotFound
from backend.config import SimConfig
from backend.motion.registry import resolve_model


@dataclass
class JobPlan:
    scene_id: str
    scene: Dict[str, Any]           # SceneSpec for load_scene
    scene_root: str                 # directory holding the scene's files (the only place the loader may read)
    trajectory: Dict[str, Any]      # spec for pipeline.build_trajectories
    summary: Dict[str, Any]         # trajectory as shown in the job JSON (no points)
    config: Dict[str, Any]          # full SimConfig dict
    visualize: bool
    num_sequences: int


def _invalid(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=message)


def _is(value: Any, kind: type) -> bool:
    if kind is bool:
        return isinstance(value, bool)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) if kind is int else math.isfinite(value)


def _check_types(obj: Any, where: str = "") -> None:
    """Scalar fields must have the type of their default (SimConfig does not check types itself)."""
    defaults = type(obj)()
    for f in dataclasses.fields(obj):
        value, default = getattr(obj, f.name), getattr(defaults, f.name)
        if dataclasses.is_dataclass(value):
            _check_types(value, f"{where}{f.name}.")
        elif f.name == "gt" or (value is None and default is None):
            continue
        else:
            kind = float if default is None else type(default)  # camera near / far default to None: floats
            if not _is(value, kind):
                raise _invalid(f"invalid config: {where}{f.name} must be a finite {kind.__name__}, got {value!r}")


def parse_config(raw: Dict[str, Any]) -> SimConfig:
    try:
        config = SimConfig.from_dict(raw)
    except (ValueError, TypeError, AttributeError) as exc:
        raise _invalid(f"invalid config: {exc}") from None
    _check_types(config)
    cam = config.camera
    if cam.width > MAX_RESOLUTION or cam.height > MAX_RESOLUTION:
        raise _invalid(f"invalid config: camera size is limited to {MAX_RESOLUTION}x{MAX_RESOLUTION} px")
    return config


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def motion_file(model_id: str, data_dir: Path) -> Path:
    """The checkpoint of a motion model id: 404 when there is none, 501 when this server cannot run it (no torch)."""
    try:
        path = resolve_model({"model": model_id}, data_dir)
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from None
    if not _torch_available():
        raise HTTPException(status_code=501, detail="human motion needs PyTorch on the server: pip install -e .[torch]")
    return path


def _trajectory(
    spec: Any, trajectories: TrajectoryStore, data_dir: Path
) -> Tuple[Dict[str, Any], Dict[str, Any], float, int]:
    """(worker spec, public summary, duration of one sequence in s, number of sequences)."""
    if isinstance(spec, ImportedSpec):
        duration = float(trajectories.meta(spec.id)["duration_s"])
        worker = {"type": "file", "path": str(trajectories.file(spec.id)), "origin": spec.origin}
        return worker, {"type": "imported", "id": spec.id, "origin": spec.origin}, duration, 1
    if isinstance(spec, InlineSpec):
        from backend.trajectories import WaypointTrajectory

        points = clean_points(spec.points)
        try:
            traj = WaypointTrajectory.from_json(points, origin=spec.origin or (0.0, 0.0, 0.0))
        except ValueError as exc:
            raise _invalid(f"invalid trajectory: {exc}") from None
        worker = {"type": "inline", "points": points, "origin": spec.origin}
        summary = {"type": "inline", "num_points": len(points), "origin": spec.origin}
        return worker, summary, (traj.end_us - traj.start_us) / 1e6, 1
    worker = spec.model_dump(exclude_none=True)
    summary = dict(worker)
    if "motion" in worker:  # the worker has no data directory: it gets the file, the job JSON keeps the id
        worker["motion"] = {"path": str(motion_file(worker["motion"]["model"], data_dir))}
    return worker, summary, spec.duration_s, spec.count


def plan_job(body: JobCreate, scenes: SceneStore, trajectories: TrajectoryStore, data_dir: Path) -> JobPlan:
    scene = scenes.spec(body.scene_id)
    scene_root = str(scenes.path(body.scene_id))
    config = parse_config(body.config)
    worker, summary, duration_s, count = _trajectory(body.trajectory, trajectories, data_dir)
    steps = int(duration_s * 1e6 // config.dt_us) + 1
    if steps > MAX_STEPS:
        raise _invalid(f"too many simulation steps per sequence ({steps} > {MAX_STEPS}): raise dt_us or shorten it")
    return JobPlan(
        scene_id=body.scene_id, scene=scene, scene_root=scene_root, trajectory=worker, summary=summary,
        config=config.to_dict(), visualize=body.visualize, num_sequences=count,
    )

