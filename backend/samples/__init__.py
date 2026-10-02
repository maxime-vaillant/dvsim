"""Procedural demo scene + trajectory (`python -m backend.samples <out_dir>`)."""
from __future__ import annotations

from .demo import DEMO_SEED, ROOM_MAX, ROOM_MIN, DemoScene, create_demo_scene, make_trajectory_records

__all__ = ["DEMO_SEED", "ROOM_MAX", "ROOM_MIN", "DemoScene", "create_demo_scene", "make_trajectory_records"]
