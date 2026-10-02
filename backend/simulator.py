"""Simulator: scene + config + camera trajectory -> stream of events and ground truth."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterator, Optional, Tuple

import numpy as np

from backend.config import SimConfig
from backend.event_processing import EventSimulator
from backend.graphical_computation.utils.renderer import Renderer
from backend.scene import Scene, load_scene
from backend.trajectories import Trajectory
from backend.types import GTFrame, StepOutput

MAX_DURATION_US = 2**32 - 1  # timestamps are stored as uint32 microseconds
_PATH_SAMPLES = 256          # camera positions sampled from the trajectory to size the clip planes


class Simulator:
    """Renders the scene along a trajectory and turns the luminance stream into events.

    One step = one rendered frame (left eye, plus right eye if stereo) followed by one event-model update.
    Ground truth is produced every ``gt_every`` steps. Nothing is accumulated: consume the generator
    returned by :meth:`run`.

    Time base: every output timestamp (``StepOutput.t_us``, ``GTFrame.t_us``, event ``t``) is RELATIVE to the
    trajectory start, so step ``k`` is at ``k * dt_us`` whatever clock the trajectory was recorded with
    (a headset's ``performance.now()`` can be hours in). The offset is ``trajectory.start_us``.

    The OpenGL context belongs to the thread that creates the Simulator: create it and consume ``run()``
    in the same thread (or process).
    """

    def __init__(self, scene, config: Optional[SimConfig] = None, context=None):
        self.config = (config or SimConfig()).validate()
        # event models first: they validate their config and hold no GL resources
        self._events_left, self._events_right = self._make_event_models()
        self.scene: Scene = scene if isinstance(scene, Scene) else load_scene(scene)
        self.renderer = Renderer(self.scene, self.config.camera, context)

    def total_steps(self, trajectory: Trajectory) -> int:
        return int((trajectory.end_us - trajectory.start_us) // self.config.dt_us) + 1

    def _make_event_models(self, seed: Optional[int] = None) -> Tuple[EventSimulator, Optional[EventSimulator]]:
        cam = self.config.camera
        shape = (cam.height, cam.width)
        left_cfg, right_cfg = self.config.events, self.config.right_event_config()
        if seed is not None:
            left_cfg, right_cfg = replace(left_cfg, seed=seed), replace(right_cfg, seed=seed + 1)
        return EventSimulator(shape, left_cfg), EventSimulator(shape, right_cfg) if cam.stereo else None

    def run(self, trajectory: Trajectory, *, event_seed: Optional[int] = None) -> Iterator[StepOutput]:
        """Yield one StepOutput per step. ``event_seed`` overrides the event models' seed for this run
        (left eye ``event_seed``, right eye ``event_seed + 1``): it only matters when a noise / mismatch
        option is enabled."""
        cfg, cam = self.config, self.config.camera
        if event_seed is not None:
            self._events_left, self._events_right = self._make_event_models(event_seed)
        dt, gt = cfg.dt_us, set(cfg.gt)
        if trajectory.end_us - trajectory.start_us > MAX_DURATION_US:
            raise ValueError(
                f"trajectory lasts {(trajectory.end_us - trajectory.start_us) / 6e7:.1f} min, more than the "
                f"{MAX_DURATION_US / 6e7:.1f} min that uint32 microsecond timestamps can hold; split it"
            )
        self._fit_clip_planes(trajectory)
        self._events_left.reset()
        if self._events_right is not None:
            self._events_right.reset()

        t0 = trajectory.start_us
        prev_t: Optional[int] = None
        for step in range(self.total_steps(trajectory)):
            t = step * dt                               # output time, relative to the trajectory start
            pose = trajectory.pose_at(t0 + t)
            is_gt = bool(gt) and step % cfg.gt_every == 0

            want_flow = is_gt and "flow" in gt
            next_t = min(t + cfg.gt_every * dt, trajectory.end_us - t0)
            next_pose = trajectory.pose_at(t0 + next_t) if want_flow else None
            left = self.renderer.render(
                pose, next_pose, rgb=is_gt and "rgb" in gt, depth=is_gt and "depth" in gt, flow=want_flow
            )
            if want_flow and next_t == t:  # last frame of the trajectory: nothing to flow towards
                left.flow, left.flow_valid = np.zeros_like(left.flow), np.zeros_like(left.flow_valid)
            events_left = self._events_left.step(left.lum, t if prev_t is None else prev_t, t)

            events_right = right = None
            if self._events_right is not None:
                right = self.renderer.render(pose.translated_local(cam.stereo_baseline), rgb=is_gt and "rgb" in gt)
                events_right = self._events_right.step(right.lum, t if prev_t is None else prev_t, t)

            frame = None
            if is_gt:
                frame = GTFrame(
                    t_us=t,
                    pose=pose.as_array() if "pose" in gt else None,
                    rgb_left=left.rgb,
                    rgb_right=right.rgb if right is not None else None,
                    depth=left.depth,
                    flow=left.flow,
                    flow_valid=left.flow_valid,
                    flow_dt_us=int(next_t - t) if want_flow else None,
                )
            yield StepOutput(step=step, t_us=t, events_left=events_left, events_right=events_right, gt=frame)
            prev_t = t

    def _fit_clip_planes(self, trajectory: Trajectory) -> None:
        """Let the renderer size near/far for the actual camera path (auto planes only)."""
        times = np.linspace(trajectory.start_us, trajectory.end_us, _PATH_SAMPLES)
        self.renderer.set_camera_positions(np.array([trajectory.pose_at(t).position for t in times]))

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "Simulator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
