"""Run a simulation to disk: trajectory specs -> sequences -> HDF5 / mp4. Shared by the CLI and the API worker."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from backend.config import SimConfig
from backend.scene import Scene, load_scene
from backend.simulator import Simulator
from backend.trajectories import (
    FreeSpaceGrid,
    RectilinearTrajectoryGenerator,
    SplineTrajectoryGenerator,
    StreetTrajectoryGenerator,
    Trajectory,
    WaypointTrajectory,
)

# progress(sequence_index, num_sequences, step, total_steps)
ProgressFn = Callable[[int, int, int, int], None]


class Cancelled(Exception):
    """Raised inside run_to_disk when the cancel callback returns True."""


def build_trajectories(
    spec: Union[Trajectory, Dict[str, Any], Sequence[Any]], scene: Optional[Scene] = None,
    cache: Optional[Dict[Any, Any]] = None,
) -> List[Trajectory]:
    """Resolve trajectory specs into Trajectory objects.

    Accepted specs (dicts, JSON-friendly) — or ready-made Trajectory objects, or a list of either:
      {"type": "file", "path": str, "origin": [x, y, z] | None}
      {"type": "inline", "points": [records...], "origin": [x, y, z] | None}
      {"type": "rectilinear", "duration_s": float, "count": int, "seed": int | None, "min_clearance": float}
      {"type": "spline", "duration_s": float, "n_key_points": int, "count": int, "seed": int | None,
       "min_clearance": float}
      {"type": "street", "duration_s": float, "count": int, "seed": int | None, "speed": float (m/s, 1.4),
       "eye_height": float (m, 1.6), "min_clearance": float (m to obstacles, 1.0),
       "edge_margin": float (m to the scene edge, 5.0)}  -- eye-height walks, +Y up
    Generated types need ``scene`` (their free space is computed from its geometry, once per call). ``cache``: a dict
    kept by the caller to reuse that free space across calls on the same scene (generation is deterministic given the
    seed, so cached and fresh results are identical).
    """
    cache = {} if cache is None else cache
    if isinstance(spec, (list, tuple)):
        out: List[Trajectory] = []
        for item in spec:
            if isinstance(item, dict):
                out.extend(_build_one(item, scene, cache))
            else:
                out.extend(build_trajectories(item, scene, cache))
        return out
    if isinstance(spec, Trajectory):
        return [spec]
    return _build_one(spec, scene, cache)


def _build_one(spec: Dict[str, Any], scene: Optional[Scene], cache: Dict[Any, Any]) -> List[Trajectory]:
    kind = spec.get("type")
    origin = spec.get("origin") or (0.0, 0.0, 0.0)
    if kind in ("file", "imported"):
        return [WaypointTrajectory.from_json(spec["path"], origin=origin)]
    if kind == "inline":
        return [WaypointTrajectory.from_json(spec["points"], origin=origin)]
    if kind in ("rectilinear", "spline"):
        if scene is None:
            raise ValueError(f"'{kind}' trajectories need a scene to compute free space")
        clearance = float(spec.get("min_clearance", 0.3))
        grid = cache.get(("grid", clearance))
        if grid is None:
            grid = cache[("grid", clearance)] = FreeSpaceGrid(scene.to_trimesh(), min_clearance=clearance)
        rng = np.random.default_rng(spec.get("seed"))
        duration_us = int(round(float(spec["duration_s"]) * 1e6))
        count = int(spec.get("count", 1))
        if kind == "rectilinear":
            gen = RectilinearTrajectoryGenerator(grid, min_clearance=clearance)
            return [gen.generate(duration_us, rng=rng) for _ in range(count)]
        gen = SplineTrajectoryGenerator(grid, min_clearance=clearance)
        n_key = int(spec.get("n_key_points", 10))
        return [gen.generate(duration_us, n_key_points=n_key, rng=rng) for _ in range(count)]
    if kind == "street":
        if scene is None:
            raise ValueError("'street' trajectories need a scene to find walkable ground")
        params = {k: float(spec[k]) for k in ("speed", "eye_height", "min_clearance", "edge_margin")
                  if spec.get(k) is not None}
        key = ("street",) + tuple(sorted(params.items()))
        gen = cache.get(key)
        if gen is None:
            gen = cache[key] = StreetTrajectoryGenerator(scene.to_trimesh(), **params)
        rng = np.random.default_rng(spec.get("seed"))
        duration_us = int(round(float(spec["duration_s"]) * 1e6))
        return gen.generate_many(int(spec.get("count", 1)), duration_us, rng=rng)
    raise ValueError(f"unknown trajectory type: {kind!r}")


def run_to_disk(
    scene: Union[Scene, str, Path, Dict[str, Any]],
    trajectories: Union[Trajectory, Dict[str, Any], Sequence[Any]],
    config: Optional[SimConfig],
    output_dir: Union[str, Path],
    *,
    visualize: bool = False,
    compression: str = "zstd",
    prefix: str = "seq",
    progress: Optional[ProgressFn] = None,
    cancel: Optional[Callable[[], bool]] = None,
    show_bar: bool = False,
    scene_root: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Simulate every trajectory and write ``<output_dir>/<prefix>_NNN/`` sequence directories. Returns them.

    ``scene_root`` confines scene loading to that directory (use it for uploaded, untrusted scenes).
    Output timestamps are relative to each trajectory's start; the offset is recorded in the metadata.
    """
    from backend.saving import SequenceOutput

    config = (config or SimConfig()).validate()
    if isinstance(scene, Scene):
        scene_obj = scene
    else:
        scene_obj = load_scene(scene) if scene_root is None else load_scene(scene, root=Path(scene_root))
    trajs = build_trajectories(trajectories, scene_obj)
    if not trajs:
        raise ValueError("no trajectory to simulate")

    out_root = Path(output_dir)
    dirs: List[Path] = []
    with Simulator(scene_obj, config) as sim:
        for i, traj in enumerate(trajs):
            seq_dir = out_root / f"{prefix}_{i:03d}"
            total = sim.total_steps(traj)
            meta = {
                "sequence": i, "num_sequences": len(trajs),
                "trajectory_start_us": int(traj.start_us), "trajectory_end_us": int(traj.end_us),
            }
            bar = _bar(total, f"{seq_dir.name}") if show_bar else None
            try:
                with SequenceOutput(
                    seq_dir, config, visualize=visualize, compression=compression, meta=meta,
                    t_offset_us=int(traj.start_us),
                ) as out:
                    if _is_waypoint(traj):  # the path that was simulated, in the source's own (absolute) time
                        traj.save_json(seq_dir / "trajectory.json")
                    for step in sim.run(traj):
                        out.write_step(step)
                        if bar is not None:
                            bar.update(1)
                        if progress is not None:
                            progress(i, len(trajs), step.step + 1, total)
                        if cancel is not None and cancel():
                            raise Cancelled()
            finally:
                if bar is not None:
                    bar.close()
            dirs.append(seq_dir)
    return dirs


def _is_waypoint(traj: Trajectory) -> bool:
    return hasattr(traj, "save_json")


def _bar(total: int, desc: str):
    import sys

    from tqdm import tqdm

    # a plain \r progress bar floods a redirected log (piped output, CI) with one line per update
    return tqdm(total=total, desc=desc, unit="step", disable=not sys.stderr.isatty())
