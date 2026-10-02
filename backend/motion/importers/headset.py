"""DVSim headset recordings and any legacy trajectory JSON (timestamps in ms): already in DVSim's frame."""
from __future__ import annotations

from pathlib import Path

from backend.motion.tracks import RawTrack
from backend.trajectories.trajectory import WaypointTrajectory


def read_headset(path: Path, name: str) -> RawTrack:
    traj = WaypointTrajectory.from_json(path, on_duplicate="drop")
    return RawTrack(name, traj.times_us, traj.positions, traj.quats)
