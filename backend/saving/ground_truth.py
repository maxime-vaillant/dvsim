"""Streaming HDF5 ground-truth writer."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, NamedTuple, Tuple

import h5py
import numpy as np

from backend.config import SimConfig
from backend.saving.common import CONVENTIONS, check_uint32, dataset_filters, normalize_compression
from backend.types import GTFrame

_SMALL_CHUNK_ROWS = 1024  # scalars and poses: rows per chunk (left uncompressed)
_ATTR_KEYS = ("time", "pose", "depth", "optical_flow", "flow_dt_us", "rgb", "compression")


class _Spec(NamedTuple):
    dtype: type
    tail: Tuple[int, ...]   # shape of one row
    attr: str               # GTFrame attribute feeding the dataset


class GroundTruthWriter:
    """Appends one frame at a time to `/ground_truth/*`; datasets exist only for the fields enabled in `config.gt`.

    Image datasets use one frame per chunk. Datasets are created up front, so a file with zero frames is valid.
    Timestamps are stored as given (uint32, relative to the sequence start); `t_offset_us` is recorded as an int64
    attribute so that absolute time = stored timestamp + t_offset_us.
    """

    def __init__(self, path, config: SimConfig, compression: str = "zstd", t_offset_us: int = 0):
        cam = config.camera
        self.path = Path(path)
        self.t_offset_us = int(t_offset_us)
        self.compression = normalize_compression(compression)
        self.height, self.width = int(cam.height), int(cam.width)
        hw = (self.height, self.width)
        gt = set(config.gt)
        specs: Dict[str, _Spec] = {"timestamp": _Spec(np.uint32, (), "t_us")}
        if "rgb" in gt:
            specs["rgb_left"] = _Spec(np.uint8, hw + (3,), "rgb_left")
            if cam.stereo:
                specs["rgb_right"] = _Spec(np.uint8, hw + (3,), "rgb_right")
        if "depth" in gt:
            specs["depth"] = _Spec(np.float32, hw, "depth")
        if "flow" in gt:
            specs["optical_flow"] = _Spec(np.float32, hw + (2,), "flow")
            specs["flow_valid"] = _Spec(np.uint8, hw, "flow_valid")
            specs["flow_dt_us"] = _Spec(np.uint32, (), "flow_dt_us")
        if "pose" in gt:
            specs["pose"] = _Spec(np.float64, (7,), "pose")
        self._specs = specs
        self._n = 0
        self._file = h5py.File(self.path, "w")
        try:
            group = self._file.create_group("ground_truth")
            self._ds = {}
            for name, spec in specs.items():
                image = len(spec.tail) >= 2
                filters = dataset_filters(self.compression, np.dtype(spec.dtype).itemsize) if image else {}
                self._ds[name] = group.create_dataset(
                    name, shape=(0,) + spec.tail, maxshape=(None,) + spec.tail, dtype=spec.dtype,
                    chunks=((1,) if image else (_SMALL_CHUNK_ROWS,)) + spec.tail, **filters)
            group.attrs.update(width=self.width, height=self.height, stereo=bool(cam.stereo), dt_us=int(config.dt_us),
                               gt_every=int(config.gt_every), fields=",".join(config.gt), time_unit="us",
                               compression=self.compression, pose_order="x y z qx qy qz qw", num_frames=0,
                               t_offset_us=np.int64(self.t_offset_us))
            for key in _ATTR_KEYS:
                group.attrs[f"convention_{key}"] = CONVENTIONS[key]
        except BaseException:
            self._file.close()
            raise

    @property
    def num_frames(self) -> int:
        return self._n

    def write(self, gt: GTFrame) -> None:
        if self._file is None:
            raise ValueError(f"GroundTruthWriter({self.path.name}) is closed")
        missing = [s.attr for s in self._specs.values() if getattr(gt, s.attr) is None]
        if missing:
            raise ValueError(f"GTFrame at t={gt.t_us} us is missing enabled field(s) {missing}")
        rows = {name: self._convert(name, spec, getattr(gt, spec.attr)) for name, spec in self._specs.items()}
        i = self._n
        for name, row in rows.items():
            ds = self._ds[name]
            ds.resize(i + 1, axis=0)
            ds[i] = row
        self._n += 1

    @staticmethod
    def _convert(name: str, spec: _Spec, value):
        if not spec.tail:
            check_uint32(int(value), int(value), name)
            return int(value)
        arr = np.asarray(value)
        if name.startswith("rgb") and arr.dtype != np.uint8:
            raise TypeError(f"{spec.attr} must be uint8, got {arr.dtype}")
        if arr.shape != spec.tail:
            raise ValueError(f"{spec.attr} has shape {arr.shape}, expected {spec.tail}")
        return arr.astype(spec.dtype, copy=False)

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._file["ground_truth"].attrs["num_frames"] = self._n
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "GroundTruthWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
