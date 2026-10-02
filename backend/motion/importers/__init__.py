"""Readers for trajectory files. ``read(path, fmt)`` -> RawTrack list; a directory is searched recursively."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional

from backend.motion.importers.aria import read_aria
from backend.motion.importers.csv_ import read_csv
from backend.motion.importers.headset import read_headset
from backend.motion.importers.tum import read_tum
from backend.motion.tracks import FrameSpec, RawTrack

_PATTERNS = {
    "tum": ("groundtruth.txt", "*.txt"),
    "aria": ("closed_loop_trajectory.csv", "*trajectory*.csv"),
    "headset": ("*.json",),
    "csv": ("*.csv",),
}
FORMATS = tuple(_PATTERNS)


def _files(path: Path, fmt: str) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"{path}: no such file or directory")
    for pattern in _PATTERNS[fmt]:   # the specific name first, the generic one if nothing matched
        found = sorted(path.rglob(pattern))
        if found:
            return found
    return []


def read(path, fmt: str, layout: str = "tumvi", up=None, camera: Optional[str] = None) -> List[RawTrack]:
    """All tracks of ``path`` (a file, or a directory searched recursively). Each format has a default frame
    (``aria`` takes its up direction from the gravity column); ``up`` (X, Y, Z in the dataset's world frame) and
    ``camera`` ("opengl" | "opencv") replace what the format says. ``layout`` selects the columns of ``csv``
    (see ``importers.csv_.CSV_LAYOUTS``)."""
    if fmt not in _PATTERNS:
        raise ValueError(f"unknown format {fmt!r}: choose one of {', '.join(FORMATS)}")
    root = Path(path)
    files = _files(root, fmt)
    if not files:
        raise FileNotFoundError(f"{root}: no {fmt} trajectory file found")
    tracks: List[RawTrack] = []
    for f in files:
        name = f.stem if root.is_file() else str(f.relative_to(root).with_suffix(""))
        if fmt == "tum":
            raw = read_tum(f, name)
        elif fmt == "aria":
            raw = read_aria(f, name)
        elif fmt == "headset":
            raw = read_headset(f, name)
        else:
            raw = read_csv(f, name, layout)
        if up is not None or camera is not None:
            raw = dataclasses.replace(raw, frame=FrameSpec(tuple(up) if up is not None else raw.frame.up,
                                                           camera or raw.frame.camera))
        tracks.append(raw)
    return tracks
