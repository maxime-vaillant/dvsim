"""A corpus is the normalised tracks of every source, saved as one .npz so training does not reparse the files."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from backend.motion.importers import read
from backend.motion.normalise import heading_agreement, normalise
from backend.motion.tracks import RATE_HZ, Track

MIN_AGREEMENT = 0.5


@dataclass
class SourceReport:
    source: str
    files: int = 0
    segments: int = 0
    hours: float = 0.0
    agreement: float = float("nan")     # mean over segments of heading_agreement
    warnings: List[str] = field(default_factory=list)


def import_source(source: str, path, fmt: str, layout: str = "tumvi", up=None, camera: Optional[str] = None):
    """(tracks, report) of one dataset; ``up`` and ``camera`` override the format's frame (see ``importers.read``)."""
    raws = read(path, fmt, layout, up, camera)
    tracks: List[Track] = []
    rep = SourceReport(source, files=len(raws))
    for raw in raws:
        got = normalise(raw, source)
        if not got:
            rep.warnings.append(f"{raw.name}: no segment of at least 4 s")
        tracks.extend(got)
    rep.segments = len(tracks)
    rep.hours = sum(len(t) for t in tracks) / RATE_HZ / 3600.0
    agree = [a for a in (heading_agreement(t) for t in tracks) if np.isfinite(a)]
    if agree:
        rep.agreement = float(np.mean(agree))
        if rep.agreement < MIN_AGREEMENT:
            rep.warnings.append(
                f"heading agreement {rep.agreement:.2f} < {MIN_AGREEMENT}: the camera does not look where the person "
                "walks; the frame is probably wrong (try --camera opencv|opengl and --up X Y Z)")
    return tracks, rep


def npz_path(path) -> Path:
    """Where a corpus named ``path`` is stored: numpy appends ``.npz`` to a name without that suffix."""
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def save_corpus(path, tracks: List[Track]) -> Path:
    """Write the corpus and return the file written (``path``, with ``.npz`` appended if it lacked it)."""
    path = npz_path(path)
    arrays = {"index": np.array(json.dumps([[t.source, t.name] for t in tracks]))}
    for i, t in enumerate(tracks):
        arrays[f"t{i}"], arrays[f"p{i}"], arrays[f"q{i}"] = t.times_us, t.positions, t.quats
    np.savez_compressed(path, **arrays)
    return path


def load_corpus(path) -> List[Track]:
    """The tracks of a corpus file; ``path`` may omit the ``.npz`` that ``save_corpus`` appends."""
    path = Path(path)
    if not path.exists() and npz_path(path).exists():
        path = npz_path(path)
    with np.load(path, allow_pickle=False) as z:
        index = json.loads(str(z["index"]))
        return [Track(s, n, z[f"t{i}"], z[f"p{i}"], z[f"q{i}"]) for i, (s, n) in enumerate(index)]
