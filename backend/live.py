"""Live rendering: the simulator's own renderer and event model driven by poses that arrive one at a time.

``Simulator.run`` follows a known trajectory. A live camera (a phone, a headset) only gives the current pose, so
:class:`LiveRenderer` renders it as it comes: the same ``Renderer`` (exact RGB, depth, flow) and the same
``EventSimulator`` as a simulation. One call = one frame.

Poses are relative to an origin in the scene, like the interface's Live mode: the centre of the scene's bounding box
(``origin="centre"``) or the middle of its floor, ``FLOOR_ORIGIN_HEIGHT`` above the lowest point (``"floor"``).
``set_origin`` puts it anywhere instead, with a heading: a pose (p, f) relative to the origin is the world pose
(origin + R(p), R(f)) with R the rotation about +Y by the heading (the same formula as the interface's Live mode).
Time is whatever the caller says (microseconds): events are generated between consecutive frames with the real
elapsed time, so they follow the speed of the camera. Flow needs the *next* pose, which does not exist yet: the
pose ``config.dt_us`` later is extrapolated at constant velocity from the last two poses, so the flow has the units
of a simulation's flow (pixels of the simulated image per simulation step) and equals it for a steady motion.

The OpenGL context belongs to the thread that creates the LiveRenderer.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from backend.config import SimConfig
from backend.event_processing import EventSimulator
from backend.graphical_computation.utils.renderer import Renderer
from backend.pose import Pose, interpolate
from backend.scene import Scene
from backend.vizualisation.visualizer import depth_frame, events_frame, flow_frame

VIEWS = ("rgb", "events", "depth", "flow", "grid")
ORIGINS = ("centre", "floor")
FLOOR_ORIGIN_HEIGHT = 1.5      # m above the floor, where a hand-held phone starts
MAX_GAP_US = 200_000           # a longer pause between frames restarts the event model instead of one huge step


class LiveRenderer:
    """One frame per ``render`` call. ``view`` is one of ``VIEWS`` (``grid`` = RGB, events / depth, flow in 2 x 2).

    ``event_window_us``: events of the last window are accumulated into the events image (one frame interval would
    show too few). ``flow_max_px``: the flow colour scale, in pixels per ``config.dt_us``.
    """

    def __init__(self, scene: Scene, config: Optional[SimConfig] = None, *, origin: str = "centre",
                 event_window_us: int = 50_000, flow_max_px: float = 2.0, jpeg_quality: int = 80, context=None):
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")
        if event_window_us <= 0 or flow_max_px <= 0 or not 1 <= jpeg_quality <= 100:
            raise ValueError("event_window_us and flow_max_px must be positive, jpeg_quality in [1, 100]")
        self.config = (config or SimConfig()).validate()
        self.scene = scene
        self.event_window_us, self.flow_max_px = int(event_window_us), float(flow_max_px)
        self.jpeg_quality = int(jpeg_quality)
        cam = self.config.camera
        self.events = EventSimulator((cam.height, cam.width), self.config.events)
        self.renderer = Renderer(scene, cam, context)
        lo, hi = scene.bounds
        centre = (lo + hi) / 2.0
        self.origin = (np.array([centre[0], centre[1], centre[2]]) if origin == "centre"
                       else np.array([centre[0], lo[1] + FLOOR_ORIGIN_HEIGHT, centre[2]]))
        self.heading_deg = 0.0
        self._default_origin = self.origin.copy()
        self._depth_range = min(self.renderer.far, max(float(scene.diagonal), 1.0))
        self._prev_pose: Optional[Pose] = None
        self._prev_t: Optional[int] = None
        self._recent: Deque[Tuple[int, np.ndarray]] = deque()
        # raw flow of the last flow / grid frame, (H, W, 2) px per config.dt_us, and its validity (for tests, tools)
        self.last_flow: Optional[np.ndarray] = None
        self.last_flow_valid: Optional[np.ndarray] = None

    # ---- poses ---------------------------------------------------------------------------------------------
    def set_origin(self, position=None, heading_deg: float = 0.0) -> None:
        """Move the origin to ``position`` (world coordinates) with a heading in degrees; None: back to the one chosen
        at construction. The motion history is dropped (the next frame has no flow, the events start over): the
        jump of a recalibration is not a camera motion."""
        self.origin = (self._default_origin.copy() if position is None
                       else np.asarray(position, dtype=np.float64).reshape(3))
        self.heading_deg = 0.0 if position is None else float(heading_deg)
        self.events.reset()
        self._prev_pose = self._prev_t = None
        self._recent.clear()

    def _rotated(self, vector) -> np.ndarray:
        angle = np.radians(self.heading_deg)
        c, s = np.cos(angle), np.sin(angle)
        x, y, z = (float(v) for v in vector)
        return np.array([x * c + z * s, y, -x * s + z * c])

    def world_pose(self, position, forward) -> Pose:
        """Pose from a position relative to the origin and a forward vector (no roll)."""
        return Pose.from_look(self.origin + self._rotated(position), self._rotated(forward))

    def _extrapolated(self, pose: Pose, frame_dt_us: int) -> Pose:
        """The pose ``config.dt_us`` later at constant velocity (the pose itself when there is no history)."""
        if self._prev_pose is None or frame_dt_us <= 0:
            return pose
        return interpolate(self._prev_pose, pose, 1.0 + self.config.dt_us / frame_dt_us)

    # ---- frames --------------------------------------------------------------------------------------------
    def render(self, position, forward, t_us: int, view: str = "rgb") -> np.ndarray:
        """(H, W, 3) uint8 RGB image of ``view`` for the camera at ``position`` (relative to the origin) looking
        along ``forward``, at time ``t_us`` (microseconds, increasing)."""
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
        pose = self.world_pose(position, forward)
        t = int(t_us)
        if self._prev_t is not None and t - self._prev_t > MAX_GAP_US:
            self.events.reset()                      # the camera paused (or the link stalled): start over
            self._prev_pose = self._prev_t = None
            self._recent.clear()
        t_prev = self._prev_t if self._prev_t is not None else t
        t = max(t, t_prev + 1) if self._prev_t is not None else t

        frame_dt = t - self._prev_t if self._prev_t is not None else 0
        want_rgb, want_depth, want_flow = (view in ("rgb", "grid"), view in ("depth", "grid"), view in ("flow", "grid"))
        out = self.renderer.render(pose, self._extrapolated(pose, frame_dt) if want_flow else None,
                                   rgb=want_rgb, depth=want_depth, flow=want_flow)
        events = self.events.step(out.lum, t_prev, t)
        self._remember(t, events)

        flow = None
        if want_flow:
            self.last_flow, self.last_flow_valid = out.flow, out.flow_valid
            flow = flow_frame(out.flow, out.flow_valid, self.flow_max_px)
        images = {
            "rgb": out.rgb if want_rgb else None,
            "events": self._events_image() if view in ("events", "grid") else None,
            "depth": depth_frame(out.depth, 0.0, self._depth_range) if want_depth else None,
            "flow": flow,
        }
        self._prev_pose, self._prev_t = pose, t
        if view == "grid":
            top = np.hstack([images["rgb"], images["events"]])
            return np.vstack([top, np.hstack([images["depth"], images["flow"]])])
        return images[view]

    def _remember(self, t: int, events: np.ndarray) -> None:
        self._recent.append((t, events))
        while self._recent and self._recent[0][0] < t - self.event_window_us:
            self._recent.popleft()

    def _events_image(self) -> np.ndarray:
        cam = self.config.camera
        acc = np.zeros((cam.height, cam.width), np.int32)
        for _, events in self._recent:
            if events.size:
                np.add.at(acc, (events["y"], events["x"]), np.where(events["p"] == 1, 1, -1))
        return events_frame(acc)

    def encode(self, image: np.ndarray) -> bytes:
        """JPEG bytes of an RGB image."""
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                  [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buffer.tobytes()

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "LiveRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["LiveRenderer", "VIEWS", "ORIGINS", "FLOOR_ORIGIN_HEIGHT"]
