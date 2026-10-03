"""Synthetic walks for the motion tests: a smooth path plus bob, sway and roll that depend on the speed."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from backend.motion.tracks import DT_US, Track
from backend.trajectories import WaypointTrajectory


def synthetic_walk(seed: int, n: int = 1500, source: str = "syn") -> Track:
    """n samples at 50 Hz of a walk with a varying speed (0.7-1.7 m/s) and a gently weaving heading."""
    r = np.random.default_rng(seed)
    t = np.arange(n) / 50.0
    speed = 1.2 + 0.5 * np.sin(2 * np.pi * t / r.uniform(8, 15) + r.uniform(0, 6))
    yaw_rate = 0.25 * np.sin(2 * np.pi * t / r.uniform(10, 20) + r.uniform(0, 6))
    psi = np.cumsum(yaw_rate) / 50.0
    x = np.cumsum(-np.sin(psi) * speed) / 50.0
    z = np.cumsum(-np.cos(psi) * speed) / 50.0
    phase = np.cumsum(1.6 + 0.2 * speed) / 50.0 * 2 * np.pi      # step frequency grows with the speed
    y = 1.6 + 0.02 * speed * np.sin(2 * phase)                   # bob at twice the step frequency
    sway = 0.02 * np.sin(phase)
    pos = np.stack([x, y, z], axis=1) + np.stack([-sway * np.cos(psi), 0 * t, sway * np.sin(psi)], axis=1)
    wobble = np.stack([0.02 * np.sin(2 * phase), 0.05 * np.sin(phase), 0.03 * np.sin(phase)], axis=1)
    rot = Rotation.from_euler("y", psi) * Rotation.from_rotvec(wobble)
    return Track(source, f"walk{seed}", (np.arange(n) * DT_US).astype(np.int64), pos, rot.as_quat())


def planner_line(seconds: float, speed: float = 1.4, start_us: int = 0, height: float = 1.6) -> WaypointTrajectory:
    """What the street planner returns, in miniature: a straight walk along -Z, a waypoint per second."""
    n = max(2, int(np.ceil(seconds)) + 1)
    times = np.linspace(0.0, seconds * 1e6, n).round().astype(np.int64) + start_us
    s = np.linspace(0.0, seconds, n) * speed
    pos = np.stack([np.zeros(n), np.full(n, height), -s], axis=1)
    return WaypointTrajectory.from_waypoints(times, pos, fronts=np.tile([0.0, 0.0, -1.0], (n, 1)))


def write_tum_walk(path, seconds: float = 40.0, rate_hz: float = 30.0) -> None:
    """A TUM RGB-D groundtruth file: a walk along +x in a z-up world, optical camera looking where it walks."""
    t = np.arange(0.0, seconds, 1.0 / rate_hz)
    quat = Rotation.from_matrix(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])).as_quat()
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    lines += [f"{1000 + ti:.6f} {1.3 * ti} 0 1.6 {quat[0]} {quat[1]} {quat[2]} {quat[3]}" for ti in t]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def stop_and_turn(speed: float = 1.4) -> Track:
    """Walk 8 s along -Z, stand still 2 s, walk 8 s back along +Z (50 Hz, camera looking where it walks)."""
    v = np.concatenate([np.full(400, -speed), np.zeros(100), np.full(400, speed)])
    z = np.cumsum(v) / 50.0
    n = len(v)
    pos = np.stack([np.zeros(n), np.full(n, 1.6), z], axis=1)
    psi = np.where(np.arange(n) < 450, 0.0, np.pi)
    quats = Rotation.from_euler("y", psi).as_quat()
    return Track("syn", "turn", (np.arange(n) * DT_US).astype(np.int64), pos, quats)
