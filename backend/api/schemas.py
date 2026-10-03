"""Request bodies (pydantic v2). Unknown keys are rejected so typos surface as 422 instead of being ignored."""
from __future__ import annotations

import math
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.api.settings import MAX_DURATION_S, MAX_INLINE_POINTS, MAX_SEQUENCES
from backend.motion.registry import MODEL_ID

POINT_KEYS = ("timestamp", "x", "y", "z", "rx", "ry", "rz", "qx", "qy", "qz", "qw")

Points = Annotated[List[Dict[str, float]], Field(min_length=2, max_length=MAX_INLINE_POINTS)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _WithOrigin(_Strict):
    origin: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)

    @field_validator("origin")
    @classmethod
    def _finite(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is not None and not all(math.isfinite(v) for v in value):
            raise ValueError("origin must be 3 finite numbers")
        return value


class TrajectoryCreate(_Strict):
    name: str = Field(default="trajectory", max_length=200)
    points: Points


class ImportedSpec(_WithOrigin):
    type: Literal["imported"]
    id: str


class InlineSpec(_WithOrigin):
    type: Literal["inline"]
    points: Points


class RectilinearSpec(_Strict):
    type: Literal["rectilinear"]
    duration_s: float = Field(gt=0, le=MAX_DURATION_S)
    count: int = Field(default=1, ge=1, le=MAX_SEQUENCES)
    seed: Optional[int] = Field(default=None, ge=0, lt=2 ** 63)
    min_clearance: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)


class SplineSpec(_Strict):
    type: Literal["spline"]
    duration_s: float = Field(gt=0, le=MAX_DURATION_S)
    n_key_points: int = Field(default=10, ge=2, le=1000)
    count: int = Field(default=1, ge=1, le=MAX_SEQUENCES)
    seed: Optional[int] = Field(default=None, ge=0, lt=2 ** 63)
    min_clearance: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)


class MotionRef(_Strict):
    """A trained human-motion model of ``$DVSIM_DATA_DIR/motion`` (GET /api/motion-models lists them)."""

    model: str = Field(pattern=MODEL_ID.pattern)


class StreetSpec(_Strict):
    type: Literal["street"]
    duration_s: float = Field(gt=0, le=MAX_DURATION_S)
    count: int = Field(default=1, ge=1, le=MAX_SEQUENCES)
    seed: Optional[int] = Field(default=None, ge=0, lt=2 ** 63)
    speed: Optional[float] = Field(default=None, gt=0, le=50, allow_inf_nan=False)
    eye_height: Optional[float] = Field(default=None, gt=0, le=100, allow_inf_nan=False)
    edge_margin: Optional[float] = Field(default=None, ge=0, le=1000, allow_inf_nan=False)
    min_clearance: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    motion: Optional[MotionRef] = None


TrajectorySpec = Annotated[
    Union[ImportedSpec, InlineSpec, RectilinearSpec, SplineSpec, StreetSpec], Field(discriminator="type")
]


GeneratedSpec = Annotated[Union[RectilinearSpec, SplineSpec, StreetSpec], Field(discriminator="type")]


class TrajectoryGenerate(_Strict):
    """Generate trajectories on a scene without simulating them (preview)."""

    scene_id: str
    trajectory: GeneratedSpec
    max_points: int = Field(default=2000, ge=2, le=MAX_INLINE_POINTS)


class JobCreate(_Strict):
    scene_id: str
    trajectory: TrajectorySpec
    config: Dict[str, Any] = Field(default_factory=dict)
    visualize: bool = False


class LiveStart(_Strict):
    """First message of /api/live: which scene to render, how, and from which origin (see backend/live.py)."""

    type: Literal["start"]
    scene_id: str
    width: int = Field(default=346, ge=16, le=1920)
    height: int = Field(default=260, ge=16, le=1080)
    fov: float = Field(default=60.0, gt=1, lt=170, allow_inf_nan=False)
    c_pos: float = Field(default=0.2, gt=0, le=10, allow_inf_nan=False)
    c_neg: float = Field(default=0.2, gt=0, le=10, allow_inf_nan=False)
    origin: Literal["centre", "floor", "shared"] = "centre"      # shared: the point picked in the interface
    view: Literal["rgb", "events", "depth", "flow", "grid"] = "rgb"
    fps: float = Field(default=30.0, ge=1, le=60, allow_inf_nan=False)


class LiveOrigin(_Strict):
    """Where the live camera's origin is in the scene (world coordinates) and which way its forward points."""

    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)
    heading_deg: float = Field(default=0.0, ge=-360, le=360, allow_inf_nan=False)


class LivePose(_Strict):
    """A camera pose relative to the origin, with the sender's sequence number (echoed with the frame)."""

    type: Literal["pose"]
    seq: int = Field(ge=0, lt=2 ** 53)
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)
    rx: float = Field(allow_inf_nan=False)
    ry: float = Field(allow_inf_nan=False)
    rz: float = Field(allow_inf_nan=False)


class LiveView(_Strict):
    type: Literal["view"]
    view: Literal["rgb", "events", "depth", "flow", "grid"]


class PositionData(BaseModel):
    """`data` of a headset message: one camera sample (timestamp in ms, position, forward vector)."""

    timestamp: float = Field(allow_inf_nan=False)
    x: Optional[float] = Field(default=None, allow_inf_nan=False)
    y: Optional[float] = Field(default=None, allow_inf_nan=False)
    z: Optional[float] = Field(default=None, allow_inf_nan=False)
    rx: Optional[float] = Field(default=None, allow_inf_nan=False)
    ry: Optional[float] = Field(default=None, allow_inf_nan=False)
    rz: Optional[float] = Field(default=None, allow_inf_nan=False)


class HeadsetMessage(BaseModel):
    action: str
    data: PositionData
