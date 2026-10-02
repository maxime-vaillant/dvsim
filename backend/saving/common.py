"""Compression settings, integer-range checks and convention strings shared by the writers."""
from __future__ import annotations

from typing import Any, Dict

COMPRESSIONS = ("zstd", "gzip", "none")
U32_MAX = 2 ** 32 - 1

CONVENTIONS: Dict[str, str] = {
    "time": "integer microseconds, stored as uint32 RELATIVE to the sequence start (at most 4294967295 us = 71.6 min "
            "of relative time per sequence); absolute time = stored value + t_offset_us (attribute / meta.json)",
    "events": "t[us], x (column, 0 = left), y (row, 0 = top), p (1 = ON / brighter, 0 = OFF); sorted by t",
    "pose": "x y z qx qy qz qw (TUM order), camera -> world, left camera; world frame = scene frame, "
            "right-handed, +Y up; camera frame = OpenGL (+X right, +Y up, looks along -Z)",
    "depth": "float32 z-depth along the optical axis in scene units; 0 = no surface",
    "optical_flow": "float32 forward flow in pixels, (dx right, dy down), from frame t to t + flow_dt_us, "
                    "sampled on the pixel grid of frame t; meaningful where flow_valid == 1",
    "flow_dt_us": "uint32 per frame: time span of the flow (gt_every * dt_us, clamped at the trajectory end)",
    "rgb": "uint8, RGB channel order, row 0 = top",
    "compression": "zstd files need `import hdf5plugin` before h5py can read them",
}


def normalize_compression(name) -> str:
    key = "none" if name is None else str(name).lower()
    if key not in COMPRESSIONS:
        raise ValueError(f"unknown compression {name!r}; choose one of {COMPRESSIONS}")
    return key


def dataset_filters(compression, itemsize: int = 1) -> Dict[str, Any]:
    """`create_dataset` kwargs: zstd (hdf5plugin, level 3) | gzip (level 4) | none, plus byte-shuffle for multi-byte types."""
    key = normalize_compression(compression)
    if key == "none":
        return {}
    kw: Dict[str, Any] = {}
    if key == "zstd":
        try:
            import hdf5plugin
        except ImportError as e:
            raise ImportError("compression='zstd' needs the hdf5plugin package; "
                              "install it or use compression='gzip' / 'none'") from e
        kw.update(hdf5plugin.Zstd(clevel=3))
    else:
        kw.update(compression="gzip", compression_opts=4)
    if itemsize > 1:
        kw["shuffle"] = True
    return kw


def check_uint32(lo: int, hi: int, what: str) -> None:
    if lo < 0 or hi > U32_MAX:
        raise ValueError(f"{what} range [{lo}, {hi}] us does not fit uint32 (0..{U32_MAX} us = {U32_MAX / 6e7:.1f} min). "
                         "Timestamps are stored relative to the sequence start (record the start as t_offset_us); "
                         "only a sequence longer than 71.6 min itself has to be split")
