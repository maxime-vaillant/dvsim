"""Generic CSV poses: a layout names the columns, the time unit and the number of header lines."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from backend.motion.tracks import FrameSpec, RawTrack

_UNIT_TO_US = {"s": 1e6, "ms": 1e3, "us": 1.0, "ns": 1e-3}


@dataclass(frozen=True)
class CsvLayout:
    """Column indices of t, x, y, z and the quaternion (qw first or last) in a comma separated file."""

    t: int
    x: int
    y: int
    z: int
    qx: int
    qy: int
    qz: int
    qw: int
    unit: str = "s"
    header_lines: int = 1
    frame: FrameSpec = FrameSpec()


CSV_LAYOUTS: Dict[str, CsvLayout] = {
    # TUM-VI ground truth (EuRoC layout): timestamp [ns], p x y z, q w x y z. The frame is an assumption: the
    # import report prints the heading agreement; override with --up / --camera when it is not close to 1.
    "tumvi": CsvLayout(t=0, x=1, y=2, z=3, qw=4, qx=5, qy=6, qz=7, unit="ns",
                       frame=FrameSpec(up=(0.0, 0.0, 1.0), camera="opencv")),
    # t [s], x y z, qx qy qz qw, in DVSim's own frame
    "dvsim": CsvLayout(t=0, x=1, y=2, z=3, qx=4, qy=5, qz=6, qw=7),
}


def read_csv(path: Path, name: str, layout: str = "tumvi") -> RawTrack:
    if layout not in CSV_LAYOUTS:
        raise ValueError(f"unknown csv layout {layout!r}: choose one of {', '.join(CSV_LAYOUTS)}")
    lay = CSV_LAYOUTS[layout]
    cols = [lay.t, lay.x, lay.y, lay.z, lay.qx, lay.qy, lay.qz, lay.qw]
    rows = []
    with open(path, "r", newline="") as fh:
        for no, rec in enumerate(csv.reader(fh), 1):
            if no <= lay.header_lines or not rec or rec[0].lstrip().startswith("#"):
                continue
            try:
                rows.append([float(rec[c]) for c in cols])
            except (ValueError, IndexError):
                shown = ",".join(rec)[:80]
                raise ValueError(f"{path}:{no}: cannot read columns {cols} as numbers from {shown!r}") from None
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than 2 poses")
    a = np.asarray(rows)
    t = np.rint(a[:, 0] * _UNIT_TO_US[lay.unit]).astype(np.int64)
    return RawTrack(name, t, a[:, 1:4], a[:, 4:8], lay.frame)
