"""Streaming video sinks: frames are written as they arrive and never kept in RAM.

All frames are RGB uint8 until `VideoSink.write`, the only place converting to BGR for OpenCV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import SimConfig
from backend.types import StepOutput



def _make_colorwheel() -> np.ndarray:
    """Middlebury colour wheel (Baker et al. 2007): 55 RGB hues, red -> yellow -> green -> cyan -> blue -> magenta."""
    segments = ((15, (255, None, 0)), (6, (None, 255, 0)), (4, (0, 255, None)),
                (11, (0, None, 255)), (13, (None, 0, 255)), (6, (255, 0, None)))
    rising = (True, False, True, False, True, False)  # the varying channel ramps up or down in each segment
    rows = []
    for (n, template), up in zip(segments, rising):
        ramp = np.floor(255.0 * np.arange(n) / n)
        seg = np.empty((n, 3))
        for c, value in enumerate(template):
            seg[:, c] = (ramp if up else 255.0 - ramp) if value is None else value
        rows.append(seg)
    return np.concatenate(rows)


_COLORWHEEL = _make_colorwheel()  # (55, 3) float64, 0..255


class VideoSink:
    """mp4v writer opened lazily on the first frame.

    mp4v crops odd sizes silently (33x17 becomes 32x16), so frames of odd width / height are padded with black on the
    right / bottom: the video size is `(width + width % 2, height + height % 2)`.
    """

    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.path, self.fps, self.frame_size = Path(path), float(fps), (int(width), int(height))
        self.pad = (self.frame_size[1] % 2, self.frame_size[0] % 2)  # (bottom, right)
        self.size = (self.frame_size[0] + self.pad[1], self.frame_size[1] + self.pad[0])
        self.frames = 0
        self._writer: Optional[cv2.VideoWriter] = None

    def write(self, rgb: np.ndarray) -> None:
        shape = (self.frame_size[1], self.frame_size[0], 3)
        if rgb.shape != shape or rgb.dtype != np.uint8:
            raise ValueError(f"{self.path.name}: expected uint8 frame of shape {shape}, got {rgb.dtype} {rgb.shape}")
        if self._writer is None:
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self._writer.isOpened():
                self._writer = None
                raise RuntimeError(f"OpenCV could not open {self.path} for writing (mp4v)")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if any(self.pad):
            bgr = cv2.copyMakeBorder(bgr, 0, self.pad[0], 0, self.pad[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self._writer.write(bgr)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def events_frame(acc: np.ndarray) -> np.ndarray:
    """Net event counts (H, W) -> RGB on white; ON red, OFF blue, saturation = log-normalised |count|."""
    mag = np.abs(acc)
    out = np.full(acc.shape + (3,), 255, np.uint8)
    peak = int(mag.max()) if mag.size else 0
    if peak == 0:
        return out
    fade = (255.0 * (1.0 - np.log1p(mag) / np.log1p(peak))).astype(np.uint8)
    on, off = acc > 0, acc < 0
    out[on, 1] = out[on, 2] = fade[on]      # keep R
    out[off, 0] = out[off, 1] = fade[off]   # keep B
    return out


def depth_frame(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Gray depth on a fixed range: near bright (255), far dark (32), invalid (<= 0 or NaN) black."""
    valid = depth > 0
    t = np.clip((depth - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    gray = np.where(valid, 32.0 + 223.0 * (1.0 - t), 0.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def flow_to_rgb(flow: np.ndarray, scale: float) -> np.ndarray:
    """Middlebury / Baker et al. 2007 flow colours, the convention of the optical-flow literature (PWC-Net, RAFT):
    hue = direction, saturation = |flow| / `scale`, value fixed. Zero flow is white; beyond `scale` the fully
    saturated hue is dimmed by 0.75 to flag "off the calibrated scale". `flow` is (H, W, 2) = (dx right, dy down).
    Returns (H, W, 3) uint8 RGB."""
    u = flow[..., 0].astype(np.float64) / scale
    v = flow[..., 1].astype(np.float64) / scale
    rad = np.sqrt(u * u + v * v)
    angle = np.arctan2(-v, -u) / np.pi                   # (-1, 1]
    ncols = len(_COLORWHEEL)
    fk = (angle + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int64)
    k1 = (k0 + 1) % ncols
    f = (fk - k0)[..., None]
    col = ((1.0 - f) * _COLORWHEEL[k0] + f * _COLORWHEEL[k1]) / 255.0
    in_range = (rad <= 1.0)[..., None]
    radc = np.minimum(rad, 1.0)[..., None]
    col = np.where(in_range, 1.0 - radc * (1.0 - col), col * 0.75)
    return np.clip(np.round(col * 255.0), 0, 255).astype(np.uint8)


def flow_frame(flow: np.ndarray, valid: Optional[np.ndarray], max_px: float) -> np.ndarray:
    """`flow_to_rgb` on a fixed scale `max_px`; invalid pixels (flow_valid == 0 or non-finite flow) are black, so
    they stay distinct from near-zero flow, which is white."""
    ok = np.isfinite(flow).all(axis=-1)
    if valid is not None:
        ok &= valid.astype(bool)
    out = flow_to_rgb(np.where(ok[..., None], flow, 0.0), max_px)
    out[~ok] = 0
    return out


def _flow_scale(flow: np.ndarray, valid: Optional[np.ndarray]) -> float:
    """Colour scale for a flow video: 99th percentile of |flow| over the valid pixels of the first frame (>= 1 px)."""
    ok = np.isfinite(flow).all(axis=-1)
    if valid is not None:
        ok &= valid.astype(bool)
    mag = np.hypot(flow[..., 0], flow[..., 1])[ok]
    return max(float(np.percentile(mag, 99)), 1.0) if mag.size else 1.0


class _EventVideo:
    """Accumulates events over fixed windows (t0 + k*W, t0 + (k+1)*W] and writes one frame per completed window."""

    def __init__(self, sink: VideoSink, width: int, height: int, window_us: int):
        self.sink, self.width, self.height, self.window = sink, width, height, int(window_us)
        self.acc = np.zeros(width * height, np.int64)
        self.t0: Optional[int] = None
        self.k = 0
        self.last_t = 0

    def add(self, events: np.ndarray, t_now: int) -> None:
        if self.t0 is None:
            self.t0 = int(t_now)
        self.last_t = int(t_now)
        t = events["t"]
        while True:
            end = self.t0 + (self.k + 1) * self.window
            n = int(np.searchsorted(t, end, side="right")) if len(t) else 0
            if n:
                self._accumulate(events[:n])
                events, t = events[n:], t[n:]
            if t_now < end:
                return
            self._emit()

    def _accumulate(self, events: np.ndarray) -> None:
        idx = events["y"].astype(np.int64) * self.width + events["x"]
        on = events["p"] > 0
        size = self.acc.size
        self.acc += np.bincount(idx[on], minlength=size)
        self.acc -= np.bincount(idx[~on], minlength=size)

    def _emit(self) -> None:
        self.sink.write(events_frame(self.acc.reshape(self.height, self.width)))
        self.acc[:] = 0
        self.k += 1

    def close(self) -> None:
        try:
            if self.t0 is not None and self.last_t > self.t0 + self.k * self.window:
                self._emit()  # partial last window
        finally:
            self.sink.close()


class Visualizer:
    """Turns the `StepOutput` stream into mp4 files in `output_dir` (`viz_<name>.mp4`), opened on first use.

    Event videos accumulate `time_accumulation_us` (default `dt_us * gt_every`) per frame; ground-truth videos get one
    frame per ground-truth step (fps = 1e6 / (dt_us * gt_every)). Colour scales are fixed for the whole video:
    depth on `depth_range` (default: min/max of the valid pixels of the first frame), flow on `flow_max_px`
    (Middlebury colours, see `flow_to_rgb`; default: 99th percentile of |flow| in the first frame, >= 1 px).
    Odd frame sizes are padded with black to even ones in the video files (see `VideoSink`).
    """

    def __init__(self, output_dir, config: SimConfig, time_accumulation_us: Optional[int] = None,
                 depth_range: Optional[Tuple[float, float]] = None, flow_max_px: Optional[float] = None,
                 prefix: str = "viz_"):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        cam = config.camera
        self.width, self.height = int(cam.width), int(cam.height)
        self.gt_period_us = int(config.dt_us) * int(config.gt_every)
        self.time_accumulation_us = int(time_accumulation_us or self.gt_period_us)
        self.flow_max_px = None if flow_max_px is None else float(flow_max_px)
        self.depth_range = None if depth_range is None else (float(depth_range[0]), float(depth_range[1]))
        self._prefix = prefix
        self._sinks: Dict[str, VideoSink] = {}
        self._events: Dict[str, _EventVideo] = {}

    @property
    def files(self) -> List[Path]:
        return [s.path for s in self._sinks.values() if s.frames > 0]

    def _sink(self, name: str, fps: float) -> VideoSink:
        if name not in self._sinks:
            self._sinks[name] = VideoSink(self.dir / f"{self._prefix}{name}.mp4", fps, self.width, self.height)
        return self._sinks[name]

    def _gt_sink(self, name: str) -> VideoSink:
        return self._sink(name, 1e6 / self.gt_period_us)

    def _add_events(self, side: str, events: np.ndarray, t_us: int) -> None:
        if side not in self._events:
            sink = self._sink(f"{side}_events", 1e6 / self.time_accumulation_us)
            self._events[side] = _EventVideo(sink, self.width, self.height, self.time_accumulation_us)
        self._events[side].add(events, t_us)

    def add(self, step: StepOutput) -> None:
        self._add_events("left", step.events_left, step.t_us)
        if step.events_right is not None:
            self._add_events("right", step.events_right, step.t_us)
        gt = step.gt
        if gt is None:
            return
        if gt.rgb_left is not None:
            self._gt_sink("left_rgb").write(gt.rgb_left)
        if gt.rgb_right is not None:
            self._gt_sink("right_rgb").write(gt.rgb_right)
        if gt.depth is not None:
            self._add_depth(gt.depth)
        if gt.flow is not None:
            if self.flow_max_px is None:
                self.flow_max_px = _flow_scale(gt.flow, gt.flow_valid)
            self._gt_sink("optical_flow").write(flow_frame(gt.flow, gt.flow_valid, self.flow_max_px))

    def _add_depth(self, depth: np.ndarray) -> None:
        if self.depth_range is None:
            valid = depth[depth > 0]
            if valid.size:
                self.depth_range = (float(valid.min()), float(valid.max()))
        lo, hi = self.depth_range or (0.0, 1.0)
        frame = depth_frame(depth, lo, hi) if self.depth_range else np.zeros(depth.shape + (3,), np.uint8)
        self._gt_sink("depth").write(frame)

    def close(self) -> None:
        errors = []
        for ev in self._events.values():
            try:
                ev.close()
            except Exception as e:
                errors.append(e)
        for sink in self._sinks.values():
            sink.close()
        if errors:
            raise errors[0]

    def __enter__(self) -> "Visualizer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
