"""TUM RGB-D text format: ``timestamp[s] tx ty tz qx qy qz qw`` per line, ``#`` comments (also written by DVSim)."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from backend.motion.tracks import FrameSpec, RawTrack

TUM_FRAME = FrameSpec(up=(0.0, 0.0, 1.0), camera="opencv")   # motion-capture world (z up), optical camera frame


def read_tum(path: Path, name: str) -> RawTrack:
    rows = []
    with open(path, "r") as fh:
        for no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                if len(parts) != 8:
                    raise ValueError
                rows.append([float(v) for v in parts])
            except ValueError:
                raise ValueError(
                    f"{path}:{no}: expected 8 numbers 't tx ty tz qx qy qz qw', got {line[:80]!r}") from None
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than 2 poses")
    a = np.asarray(rows)
    return RawTrack(name, np.rint(a[:, 0] * 1e6).astype(np.int64), a[:, 1:4], a[:, 4:8], TUM_FRAME)
