"""Informational comparison of the event streams a plain street walk and a human-motion walk produce on a scene.

Not a pass / fail test: there is no real event recording to compare against. It shows whether the added motion
changes the event rate and the optical flow in the expected direction (more events, more and faster flow jitter).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from backend.config import CameraConfig, SimConfig
from backend.pipeline import build_trajectories
from backend.simulator import Simulator
from backend.trajectories import Trajectory

DEFAULT_CONFIG = SimConfig(camera=CameraConfig(width=128, height=96), dt_us=10_000, gt_every=5, gt=("flow",))


def event_statistics(scene, trajectory: Trajectory, config: Optional[SimConfig] = None) -> Dict[str, float]:
    """Events per second, events per step (median, 95th percentile) and optical-flow speed in px/s (mean, 95th
    percentile over the valid pixels) of one simulated trajectory."""
    config = config or DEFAULT_CONFIG
    counts, speeds = [], []
    with Simulator(scene, config) as sim:
        for step in sim.run(trajectory):
            counts.append(len(step.events_left))
            gt = step.gt
            if gt is not None and gt.flow is not None and gt.flow_dt_us:
                valid = gt.flow_valid.astype(bool)
                if valid.any():
                    speeds.append(np.linalg.norm(gt.flow[valid], axis=1) * 1e6 / gt.flow_dt_us)
    seconds = (trajectory.end_us - trajectory.start_us) / 1e6
    flow = np.concatenate(speeds) if speeds else np.zeros(1)
    return {
        "events_per_s": float(np.sum(counts) / seconds),
        "events_per_step_p50": float(np.percentile(counts, 50)),
        "events_per_step_p95": float(np.percentile(counts, 95)),
        "flow_px_s_mean": float(flow.mean()),
        "flow_px_s_p95": float(np.percentile(flow, 95)),
    }


def compare_event_statistics(scene, motion: Dict[str, Any], street: Dict[str, Any],
                             config: Optional[SimConfig] = None) -> Dict[str, Dict[str, float]]:
    """``{"plain": ..., "human": ...}``: the same planned street walk (same ``street`` spec and seed), without and
    with the ``motion`` model reference (``{"model": id}`` or ``{"path": file}``).
    """
    spec = {"type": "street", "count": 1, "seed": 0, **street}
    cache: Dict[Any, Any] = {}
    (plain,) = build_trajectories(spec, scene, cache)
    (human,) = build_trajectories({**spec, "motion": motion}, scene, cache)
    return {"plain": event_statistics(scene, plain, config), "human": event_statistics(scene, human, config)}


def format_event_report(report: Dict[str, Dict[str, float]]) -> str:
    rows = (("events_per_s", "events / s"), ("events_per_step_p50", "events / step (median)"),
            ("events_per_step_p95", "events / step (95th pct)"), ("flow_px_s_mean", "flow speed, px/s (mean)"),
            ("flow_px_s_p95", "flow speed, px/s (95th pct)"))
    lines = [f"{'':<28}{'plain':>12}{'human':>12}"]
    lines += [f"{label:<28}{report['plain'][key]:>12.1f}{report['human'][key]:>12.1f}" for key, label in rows]
    return "\n".join(lines)
