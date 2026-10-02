"""Simulation configuration (JSON-serializable so the CLI, the API and the dataloader share it)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

GT_FIELDS = ("rgb", "depth", "flow", "pose")


def data_dir() -> Path:
    """Root for scenes / trajectories / job outputs. Override with $DVSIM_DATA_DIR (default: ./data)."""
    return Path(os.environ.get("DVSIM_DATA_DIR", "data")).resolve()


@dataclass
class CameraConfig:
    width: int = 346
    height: int = 260
    fov_y_deg: float = 60.0
    near: Optional[float] = None       # None -> derived from the scene bounds
    far: Optional[float] = None        # None -> derived from the scene bounds
    stereo: bool = False
    stereo_baseline: float = 0.1       # scene units, right camera = left camera moved along camera +X


@dataclass
class EventConfig:
    """Ideal DVS by default; the noise / filtering terms are opt-in (0 = off)."""

    c_pos: float = 0.2                 # ON  contrast threshold, in natural-log intensity units
    c_neg: float = 0.2                 # OFF contrast threshold
    sigma_c: float = 0.0               # relative per-pixel threshold mismatch (std of the multiplicative factor)
    refractory_us: int = 0             # minimum time between two events of one pixel
    shot_noise_hz: float = 0.0         # background-activity rate per pixel (ON and OFF combined)
    lpf_cutoff_hz: float = 0.0         # first-order low-pass on log-intensity (photoreceptor bandwidth); 0 = off
    log_eps: float = 1e-3              # log(L + log_eps) with L the linear luminance in [0, 1]
    seed: int = 0


@dataclass
class SimConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    events: EventConfig = field(default_factory=EventConfig)
    events_right: Optional[EventConfig] = None   # None -> same as `events`
    dt_us: int = 10_000                # simulation step: one frame is rendered and one event update done per step
    gt_every: int = 1                  # a ground-truth frame every N steps (flow spans N steps)
    gt: Tuple[str, ...] = GT_FIELDS    # subset of GT_FIELDS; events are always produced

    def validate(self) -> "SimConfig":
        """Raise ValueError on anything the renderer / event model could not handle. Returns self."""
        if self.dt_us <= 0 or self.gt_every <= 0:
            raise ValueError("dt_us and gt_every must be positive")
        if isinstance(self.gt, str):
            raise ValueError(f"gt must be a list of fields from {GT_FIELDS}, got the string {self.gt!r}")
        bad = set(self.gt) - set(GT_FIELDS)
        if bad:
            raise ValueError(f"unknown ground-truth fields {sorted(bad)}; allowed: {GT_FIELDS}")
        cam = self.camera
        if cam.width <= 0 or cam.height <= 0:
            raise ValueError("camera width/height must be positive")
        if not 0.0 < cam.fov_y_deg < 180.0:
            raise ValueError(f"camera.fov_y_deg must be in (0, 180), got {cam.fov_y_deg}")
        for name in ("near", "far"):
            value = getattr(cam, name)
            if value is not None and not value > 0.0:
                raise ValueError(f"camera.{name} must be positive, got {value}")
        if cam.near is not None and cam.far is not None and cam.near >= cam.far:
            raise ValueError(f"camera.near ({cam.near}) must be smaller than camera.far ({cam.far})")
        if not cam.stereo_baseline >= 0.0:
            raise ValueError(f"camera.stereo_baseline must be >= 0, got {cam.stereo_baseline}")
        for label, ev in (("events", self.events), ("events_right", self.events_right)):
            if ev is not None:
                _validate_events(label, ev)
        return self

    def right_event_config(self) -> EventConfig:
        return self.events_right or self.events

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gt"] = list(self.gt)
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SimConfig":
        """Build from a (possibly partial) plain dict; unknown keys raise ValueError."""
        d = dict(d or {})
        camera = _make(CameraConfig, d.pop("camera", None))
        events = _make(EventConfig, d.pop("events", None))
        right = d.pop("events_right", None)
        events_right = _make(EventConfig, right) if right is not None else None
        if "gt" in d:
            if isinstance(d["gt"], str):
                raise ValueError(f"gt must be a list of fields from {GT_FIELDS}, got the string {d['gt']!r}")
            d["gt"] = tuple(d["gt"])
        _check_keys(cls, d, exclude=("camera", "events", "events_right"))
        return cls(camera=camera, events=events, events_right=events_right, **d).validate()


def _validate_events(label: str, ev: EventConfig) -> None:
    for name in ("c_pos", "c_neg", "log_eps"):
        if not getattr(ev, name) > 0.0:
            raise ValueError(f"{label}.{name} must be positive, got {getattr(ev, name)}")
    for name in ("sigma_c", "refractory_us", "shot_noise_hz", "lpf_cutoff_hz"):
        if not getattr(ev, name) >= 0.0:
            raise ValueError(f"{label}.{name} must be >= 0, got {getattr(ev, name)}")


def _check_keys(cls, d: Dict[str, Any], exclude=()) -> None:
    unknown = set(d) - {f.name for f in fields(cls)} - set(exclude)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")


def _make(cls, d: Optional[Dict[str, Any]]):
    d = dict(d or {})
    _check_keys(cls, d)
    return cls(**d)
