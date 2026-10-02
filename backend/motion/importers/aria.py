"""Project Aria / Aria Digital Twin trajectories (MPS ``closed_loop_trajectory.csv``): header-named columns."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from backend.motion.tracks import FrameSpec, RawTrack

_T = "tracking_timestamp_us"
_P = ("tx_world_device", "ty_world_device", "tz_world_device")
_Q = ("qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device")
_G = ("gravity_x_world", "gravity_y_world", "gravity_z_world")


def read_aria(path: Path, name: str) -> RawTrack:
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in (_T,) + _P + _Q if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        has_g = all(c in reader.fieldnames for c in _G)
        rows, grav = [], []
        for no, rec in enumerate(reader, 2):
            try:
                rows.append([float(rec[c]) for c in (_T,) + _P + _Q])
                if has_g:
                    grav.append([float(rec[c]) for c in _G])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{no}: non-numeric value in a pose or gravity column") from None
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than 2 poses")
    a = np.asarray(rows)
    g = -np.mean(grav, axis=0) if has_g else np.array([0.0, 0.0, 1.0])   # gravity points down
    up = tuple(float(v) for v in g / np.linalg.norm(g))
    # the device frame is assumed to be optical-style (+Z forward, +Y down); the import report checks the assumption
    return RawTrack(name, np.rint(a[:, 0]).astype(np.int64), a[:, 1:4], a[:, 4:8], FrameSpec(up=up, camera="opencv"))
