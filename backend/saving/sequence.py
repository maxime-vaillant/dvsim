"""Everything written for one sequence directory."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from backend.config import SimConfig
from backend.saving.common import CONVENTIONS, normalize_compression
from backend.saving.events import EventWriter
from backend.saving.ground_truth import GroundTruthWriter
from backend.saving.tum import write_tum
from backend.types import StepOutput


def _json_default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def package_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__}
    for name in ("hdf5plugin", "cv2"):
        mod = sys.modules.get(name)
        if mod is not None:
            versions["opencv" if name == "cv2" else name] = str(getattr(mod, "__version__", None) or getattr(mod, "version", ""))
    return versions


class SequenceOutput:
    """Sink for the `StepOutput` stream of one sequence.

    Files in `sequence_dir`: `left_events.h5` (+ `right_events.h5` if stereo), `ground_truth.h5` if `config.gt` is
    non-empty, `trajectory_tum.txt` if "pose" is in `config.gt`, `viz_*.mp4` if `visualize`, and `meta.json` on close
    (its `files` lists everything in the directory at that moment, including files the caller added).

    Steps carry times relative to the sequence start; `t_offset_us` (the absolute start) is recorded in the HDF5
    attributes, `meta.json` and the TUM header, and timestamps are written unchanged.
    """

    def __init__(self, sequence_dir, config: SimConfig, visualize: bool = False, compression: str = "zstd",
                 meta: Optional[Dict[str, Any]] = None, viz_options: Optional[Dict[str, Any]] = None,
                 t_offset_us: int = 0):
        self.dir = Path(sequence_dir)
        self.config = config.validate()
        self.compression = normalize_compression(compression)
        self.meta = dict(meta or {})
        self.t_offset_us = int(t_offset_us)
        self.dir.mkdir(parents=True, exist_ok=True)
        cam = config.camera
        self._left: Optional[EventWriter] = None
        self._right: Optional[EventWriter] = None
        self._gt: Optional[GroundTruthWriter] = None
        self._viz = None
        self._tum_t: List[int] = []
        self._tum_pose: List[np.ndarray] = []
        self._n_steps = 0
        self._t_first: Optional[int] = None
        self._t_last: Optional[int] = None
        self._closed = False
        self._failed = False
        try:
            self._left = EventWriter(self.dir / "left_events.h5", cam.width, cam.height, self.compression,
                                     t_offset_us=self.t_offset_us)
            if cam.stereo:
                self._right = EventWriter(self.dir / "right_events.h5", cam.width, cam.height, self.compression,
                                          t_offset_us=self.t_offset_us)
            if config.gt:
                self._gt = GroundTruthWriter(self.dir / "ground_truth.h5", config, self.compression,
                                             t_offset_us=self.t_offset_us)
            if visualize:
                from backend.vizualisation import Visualizer
                self._viz = Visualizer(self.dir, config, **(viz_options or {}))
        except BaseException:
            self._close_sinks()
            raise

    def write_step(self, step: StepOutput) -> None:
        if self._closed:
            raise ValueError("SequenceOutput is closed")
        try:
            self._write_step(step)
        except BaseException:
            self._failed = True
            raise

    def _write_step(self, step: StepOutput) -> None:
        if self._right is not None and step.events_right is None:
            raise ValueError(f"stereo config but step {step.step} has no events_right")
        self._left.write(step.events_left)
        if self._right is not None:
            self._right.write(step.events_right)
        if self._gt is not None and step.gt is not None:
            self._gt.write(step.gt)
            if "pose" in self.config.gt:
                self._tum_t.append(int(step.gt.t_us))
                self._tum_pose.append(np.asarray(step.gt.pose, dtype=np.float64))
        if self._viz is not None:
            self._viz.add(step)
        self._n_steps += 1
        self._t_first = step.t_us if self._t_first is None else self._t_first
        self._t_last = step.t_us

    def _close_sinks(self) -> List[Exception]:
        errors: List[Exception] = []
        for sink in (self._left, self._right, self._gt, self._viz):
            if sink is not None:
                try:
                    sink.close()
                except Exception as e:
                    errors.append(e)
        return errors

    def close(self, complete: bool = True) -> None:
        """Finalise every file (all sinks are closed even if one fails), then write meta.json; re-raises the first error."""
        if self._closed:
            return
        self._closed = True
        errors = self._close_sinks()
        try:
            if "pose" in self.config.gt:
                write_tum(self.dir / "trajectory_tum.txt", self._tum_t, np.array(self._tum_pose).reshape(-1, 7),
                          t_offset_us=self.t_offset_us)
            self._write_meta(complete and not self._failed and not errors)
        except Exception as e:
            errors.append(e)
        if errors:
            raise errors[0]

    def _write_meta(self, complete: bool) -> None:
        span = 0 if self._t_first is None else int(self._t_last - self._t_first)
        files = sorted(p.name for p in self.dir.iterdir() if p.is_file() and p.name != "meta.json")
        meta = {
            "complete": bool(complete),
            "config": self.config.to_dict(),
            "conventions": CONVENTIONS,
            "compression": self.compression,
            "t_offset_us": self.t_offset_us,
            "versions": package_versions(),
            "n_steps": self._n_steps,
            "n_events_left": self._left.num_events,
            "n_events_right": None if self._right is None else self._right.num_events,
            "n_gt_frames": None if self._gt is None else self._gt.num_frames,
            "start_us": self._t_first,
            "end_us": self._t_last,
            "duration_us": span,
            "files": files,
            "meta": self.meta,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default) + "\n")

    def __enter__(self) -> "SequenceOutput":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close(complete=exc_type is None)
        except Exception:
            if exc_type is None:
                raise
