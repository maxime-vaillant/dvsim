"""Plain data containers shared by the simulator, writers, API and dataloader."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# One event = (timestamp in microseconds, pixel x, pixel y, polarity 1=ON (brighter) / 0=OFF (darker)).
EVENT_DTYPE = np.dtype([("t", "<i8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])


def empty_events() -> np.ndarray:
    return np.empty(0, dtype=EVENT_DTYPE)


@dataclass
class GTFrame:
    """Ground truth for one instant. Every array is optional: only requested fields are filled.

    Image arrays have row 0 at the top. See docs/architecture.md for exact conventions.
    """

    t_us: int
    pose: Optional[np.ndarray] = None         # (7,) float64: x y z qx qy qz qw, camera->world, LEFT camera
    rgb_left: Optional[np.ndarray] = None     # (H, W, 3) uint8, RGB order
    rgb_right: Optional[np.ndarray] = None    # (H, W, 3) uint8 (stereo only)
    depth: Optional[np.ndarray] = None        # (H, W) float32, z-depth in scene units, 0 = no surface
    flow: Optional[np.ndarray] = None         # (H, W, 2) float32 px (dx right, dy down): t_us -> t_us + flow_dt_us
    flow_valid: Optional[np.ndarray] = None   # (H, W) uint8, 1 where flow is meaningful
    flow_dt_us: Optional[int] = None


@dataclass
class StepOutput:
    """Everything produced by one simulation step (= one rendered frame at t_us)."""

    step: int
    t_us: int
    events_left: np.ndarray                   # EVENT_DTYPE, sorted by t, events in (t_us - dt, t_us]
    events_right: Optional[np.ndarray] = None
    gt: Optional[GTFrame] = None              # present on ground-truth steps only
