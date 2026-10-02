"""Camera tracks sampled at a fixed rate in DVSim's frame: the common currency of the motion package."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RATE_HZ = 50
DT_US = 1_000_000 // RATE_HZ


@dataclass(frozen=True)
class FrameSpec:
    """How a dataset's poses map to DVSim's frame (+Y up, camera looks along its own -Z, X right).

    ``up``: the direction of "up" in the dataset's world frame. ``camera``: "opengl" (X right, Y up, looks -Z, as
    DVSim) or "opencv" (X right, Y down, looks +Z).
    """

    up: tuple = (0.0, 1.0, 0.0)
    camera: str = "opengl"

    def __post_init__(self):
        if self.camera not in ("opengl", "opencv"):
            raise ValueError(f"camera must be 'opengl' or 'opencv', got {self.camera!r}")
        if not (len(self.up) == 3 and np.all(np.isfinite(self.up)) and np.linalg.norm(self.up) > 1e-9):
            raise ValueError(f"up must be a non-zero 3-vector, got {self.up!r}")


@dataclass(frozen=True)
class RawTrack:
    """A recording as read from a file: poses (camera -> world, quaternions x y z w) in the dataset's own frame."""

    name: str
    times_us: np.ndarray    # (N,) int64
    positions: np.ndarray   # (N, 3) metres
    quats: np.ndarray       # (N, 4)
    frame: FrameSpec = FrameSpec()


@dataclass(frozen=True)
class Track:
    """A segment resampled at RATE_HZ in DVSim's frame: sample k is at ``times_us[0] + k * DT_US``."""

    source: str
    name: str
    times_us: np.ndarray    # (N,) int64
    positions: np.ndarray   # (N, 3)
    quats: np.ndarray       # (N, 4)

    def __len__(self) -> int:
        return len(self.times_us)

    @property
    def duration_s(self) -> float:
        return (len(self) - 1) / RATE_HZ
