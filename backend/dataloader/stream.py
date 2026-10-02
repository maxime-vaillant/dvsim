"""ClipStream: an endless iterator of training clips (events + ground truth) simulated on the fly, numpy only.

Each trajectory is simulated as ONE continuous ``Simulator.run`` and cut into consecutive clips of ``clip_steps`` steps.
Clip ``j`` of a sequence covers the window ``(t_j, t_j + clip_steps * dt_us]`` with ``t_j = j * clip_steps * dt_us``
(the simulator's time base: microseconds since the start of the trajectory): the events of steps
``j * S + 1 .. (j + 1) * S`` and the ground-truth frames of those steps, so bin ``b`` ends exactly at ground-truth
frame ``b``. The first simulator step only initialises the event reference and belongs to no clip; a trailing partial
clip is dropped.
"""
from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from backend.config import GT_FIELDS, SimConfig
from backend.dataloader.representations import (
    bin_end_times_us,
    events_to_count_frames,
    events_to_voxel_grid,
    rebase_events,
)
from backend.pipeline import build_trajectories
from backend.scene import Scene, load_scene
from backend.simulator import Simulator
from backend.trajectories import (
    FreeSpaceGrid,
    RectilinearTrajectoryGenerator,
    SplineTrajectoryGenerator,
    StreetTrajectoryGenerator,
    Trajectory,
)

__all__ = ["ClipStream", "REPRESENTATIONS"]

REPRESENTATIONS = ("count_frames", "voxel", "raw")
_GENERATORS = ("spline", "rectilinear", "street")


class _SceneSlot:
    """Everything kept alive for one scene: the loaded scene, its Simulator (GL objects) and its free-space grid."""

    def __init__(self, index: int, spec: Any):
        self.index, self.spec = index, spec
        self.scene: Optional[Scene] = None
        self.sim: Optional[Simulator] = None
        self.grid: Optional[FreeSpaceGrid] = None
        self.street: Optional[StreetTrajectoryGenerator] = None

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
        self.scene = self.sim = self.grid = self.street = None


class ClipStream:
    """Iterator of clip dicts of numpy arrays (see the module docstring for the clip timing).

    ``scenes``: a scene spec or a list of them (model path, SceneSpec dict / ``.json`` path, ``Scene``); each sequence
    picks one uniformly. ``trajectories``: None to generate one random trajectory per sequence inside the scene's free
    space (``generator``, ``trajectory_duration_s``, ``n_key_points``, ``min_clearance``), or trajectory spec(s)
    accepted by ``pipeline.build_trajectories`` (cycled). ``generator="street"`` walks at eye height on the open
    ground instead (+Y up, metres); ``street_options`` are passed to ``StreetTrajectoryGenerator`` (``speed``,
    ``eye_height``, ``min_clearance``, ``edge_margin``, ...), whose defaults apply otherwise. ``num_sequences`` bounds
    the stream (default: endless).

    Clip keys: ``events`` (``count_frames``: float32 (T, 2, H, W) with channel = polarity, or (T, 1, H, W) signed when
    ``split_polarity=False``; ``voxel``: (T, H, W); ``raw``: EVENT_DTYPE array with ``t`` in microseconds from the clip
    start), ``events_right`` (stereo), ``t_us`` (T,) bin end times from the clip start, ``gt`` (channels-first arrays of
    the ground-truth frames inside the clip: ``depth`` (G,1,H,W) f32, ``flow`` (G,2,H,W) f32, ``flow_valid`` (G,1,H,W)
    u8, ``rgb`` / ``rgb_right`` (G,3,H,W) u8, ``pose`` (G,7) f64), ``gt_t_us`` (G,) and ``meta`` (``scene_index``,
    ``sequence_id``, ``clip_index``, ``start_us`` since the sequence start, ``is_first_clip``).

    Everything is deterministic given ``seed``; the event-model seed of each sequence is derived from it. One
    Simulator (and FreeSpaceGrid) per scene stays alive in an LRU of ``max_cached_scenes``; ``close()`` (also via
    ``with`` or garbage collection) releases them and the GL context this stream created.
    """

    def __init__(
        self,
        scenes,
        config: Optional[SimConfig] = None,
        trajectories=None,
        *,
        generator: str = "spline",
        trajectory_duration_s: float = 2.0,
        n_key_points: int = 8,
        min_clearance: float = 0.3,
        street_options: Optional[Dict[str, Any]] = None,
        clip_steps: int = 10,
        num_bins: Optional[int] = None,
        representation: str = "count_frames",
        binary: bool = False,
        split_polarity: bool = True,
        gt: Sequence[str] = ("depth", "flow"),
        seed: Optional[int] = 0,
        num_sequences: Optional[int] = None,
        max_cached_scenes: int = 2,
        context=None,
    ):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}, got {representation!r}")
        if generator not in _GENERATORS:
            raise ValueError(f"generator must be one of {_GENERATORS}, got {generator!r}")
        bad = set(gt) - set(GT_FIELDS)
        if bad:
            raise ValueError(f"unknown ground-truth fields {sorted(bad)}; allowed: {GT_FIELDS}")
        if clip_steps < 1 or max_cached_scenes < 1:
            raise ValueError("clip_steps and max_cached_scenes must be positive")
        if isinstance(config, dict):
            config = SimConfig.from_dict(config)
        base = (config or SimConfig()).validate()
        self.config = dataclasses.replace(base, gt=tuple(gt)).validate()
        self.scenes: List[Any] = list(scenes) if isinstance(scenes, (list, tuple)) else [scenes]
        if not self.scenes:
            raise ValueError("no scene given")
        self.trajectories = trajectories
        self.generator, self.n_key_points, self.min_clearance = generator, int(n_key_points), float(min_clearance)
        self.street_options = dict(street_options or {})
        self.duration_us = int(round(float(trajectory_duration_s) * 1e6))
        if trajectories is None and self.duration_us < clip_steps * self.config.dt_us:
            raise ValueError("trajectory_duration_s is shorter than one clip (clip_steps * dt_us)")
        self.clip_steps = int(clip_steps)
        self.num_bins = int(num_bins) if num_bins is not None else self.clip_steps
        self.representation, self.binary, self.split_polarity = representation, bool(binary), bool(split_polarity)
        self.num_sequences = num_sequences
        self.max_cached_scenes = int(max_cached_scenes)
        self._entropy = seed if seed is not None else np.random.SeedSequence().entropy
        self._slots: "OrderedDict[int, _SceneSlot]" = OrderedDict()
        self._resolved: Dict[int, List[Trajectory]] = {}
        self._ctx = context
        self._owns_ctx = False
        self._gen: Optional[Iterator[Dict[str, Any]]] = None
        self._closed = False

    # ---- iteration ----------------------------------------------------------------
    def __iter__(self) -> "ClipStream":
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._closed:
            raise StopIteration
        if self._gen is None:
            self._gen = self._sequences()
        return next(self._gen)

    def _sequences(self) -> Iterator[Dict[str, Any]]:
        k = 0
        while self.num_sequences is None or k < self.num_sequences:
            seq_ss = np.random.SeedSequence([self._entropy, k])
            traj_ss, event_ss = seq_ss.spawn(2)
            rng = np.random.default_rng(traj_ss)
            scene_index = int(rng.integers(len(self.scenes)))
            slot = self._slot(scene_index)
            sim = self._simulator(slot)
            traj = self._trajectory(slot, k, rng)
            if sim.total_steps(traj) - 1 < self.clip_steps:
                raise ValueError(f"trajectory {k} ({traj.end_us - traj.start_us} us) is shorter than one clip")
            event_seed = int(event_ss.generate_state(2)[0])
            yield from self._clips(sim, traj, k, scene_index, event_seed)
            k += 1

    def _clips(
        self, sim: Simulator, traj: Trajectory, sequence_id: int, scene_index: int, event_seed: int
    ) -> Iterator[Dict[str, Any]]:
        span = self.clip_steps * self.config.dt_us
        steps = sim.run(traj, event_seed=event_seed)
        try:
            first = next(steps)                                  # initialises the event reference, in no clip
            buffer, j = [], 0
            for step in steps:
                buffer.append(step)
                if len(buffer) == self.clip_steps:
                    meta = {"scene_index": scene_index, "sequence_id": sequence_id, "clip_index": j,
                            "start_us": first.t_us + j * span, "is_first_clip": j == 0}
                    yield self._make_clip(buffer, meta)
                    buffer, j = [], j + 1
        finally:
            steps.close()

    # ---- clip assembly ------------------------------------------------------------
    def _represent(self, events: np.ndarray, t0: int, t1: int) -> np.ndarray:
        cam = self.config.camera
        if self.representation == "raw":
            return rebase_events(events, t0)
        if self.representation == "voxel":
            return events_to_voxel_grid(events, self.num_bins, t0, t1, cam.height, cam.width)
        return events_to_count_frames(events, self.num_bins, t0, t1, cam.height, cam.width,
                                      self.split_polarity, self.binary)

    def _make_clip(self, steps, meta: Dict[str, Any]) -> Dict[str, Any]:
        t0 = meta["start_us"]
        t1 = t0 + self.clip_steps * self.config.dt_us
        clip: Dict[str, Any] = {"events": self._represent(np.concatenate([s.events_left for s in steps]), t0, t1)}
        if self.config.camera.stereo:
            clip["events_right"] = self._represent(np.concatenate([s.events_right for s in steps]), t0, t1)
        clip["t_us"] = bin_end_times_us(self.num_bins, t0, t1)
        frames = [s.gt for s in steps if s.gt is not None]
        clip["gt"] = _stack_gt(frames, self.config.gt, self.config.camera.height, self.config.camera.width)
        clip["gt_t_us"] = np.array([f.t_us - t0 for f in frames], dtype=np.int64)
        clip["meta"] = meta
        return clip

    # ---- scenes, trajectories, simulators -------------------------------------------
    def _context(self):
        if self._ctx is None:
            from backend.graphical_computation.utils.gl_context import create_gl_context

            self._ctx, self._owns_ctx = create_gl_context(), True
        return self._ctx

    def _slot(self, index: int) -> _SceneSlot:
        slot = self._slots.get(index)
        if slot is None:
            slot = self._slots[index] = _SceneSlot(index, self.scenes[index])
            while len(self._slots) > self.max_cached_scenes:
                self._slots.popitem(last=False)[1].close()
        self._slots.move_to_end(index)
        return slot

    def _simulator(self, slot: _SceneSlot) -> Simulator:
        if slot.sim is None:
            slot.scene = load_scene(slot.spec)
            slot.sim = Simulator(slot.scene, self.config, self._context())
        return slot.sim

    def _trajectory(self, slot: _SceneSlot, sequence_id: int, rng: np.random.Generator) -> Trajectory:
        if self.trajectories is not None:
            trajs = self._resolved.get(slot.index)
            if trajs is None:
                trajs = self._resolved[slot.index] = build_trajectories(
                    self._seeded_specs(self.trajectories, slot.index), slot.scene)
                if not trajs:
                    raise ValueError("no trajectory to simulate")
            return trajs[sequence_id % len(trajs)]
        if self.generator == "street":
            if slot.street is None:
                slot.street = StreetTrajectoryGenerator(slot.scene, **self.street_options)
            return slot.street.generate(self.duration_us, rng=rng)
        if slot.grid is None:
            slot.grid = FreeSpaceGrid(slot.scene, min_clearance=self.min_clearance)
        if self.generator == "rectilinear":
            return RectilinearTrajectoryGenerator(slot.grid, min_clearance=self.min_clearance).generate(
                self.duration_us, rng=rng)
        return SplineTrajectoryGenerator(slot.grid, min_clearance=self.min_clearance).generate(
            self.duration_us, n_key_points=self.n_key_points, rng=rng)

    def _seeded_specs(self, spec, scene_index: int):
        """Generated-trajectory specs without a seed get one derived from the stream seed (reproducible)."""
        items = list(spec) if isinstance(spec, (list, tuple)) else [spec]
        out = []
        for i, item in enumerate(items):
            if isinstance(item, dict) and item.get("type") in _GENERATORS and item.get("seed") is None:
                seed = int(np.random.SeedSequence([self._entropy, scene_index, i, 1]).generate_state(1)[0])
                item = {**item, "seed": seed}
            out.append(item)
        return out

    # ---- lifetime -----------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Restart from sequence 0 with a new ``seed``; cached scenes, simulators and grids are kept."""
        if self._gen is not None:
            self._gen.close()
            self._gen = None
        self._entropy = seed if seed is not None else np.random.SeedSequence().entropy
        self._resolved.clear()
        self._closed = False

    def close(self) -> None:
        """Release the simulators (GL objects) and the GL context this stream created. Idempotent."""
        self._closed = True
        if self._gen is not None:
            self._gen.close()
            self._gen = None
        while self._slots:
            self._slots.popitem()[1].close()
        if self._ctx is not None and self._owns_ctx:
            self._ctx.release()
        self._ctx, self._owns_ctx = None, False

    def __enter__(self) -> "ClipStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown: the GL library may already be gone
            pass


def _stack_gt(frames, fields: Tuple[str, ...], height: int, width: int) -> Dict[str, np.ndarray]:
    """GTFrames -> channels-first arrays stacked along a leading G axis (only the requested fields; G may be 0)."""
    n = len(frames)
    out: Dict[str, np.ndarray] = {}

    def images(name: str, channels: int, dtype, last: bool = False) -> np.ndarray:
        if not n:
            return np.zeros((0, channels, height, width), dtype=dtype)
        stacked = np.stack([getattr(f, name) for f in frames])
        return np.ascontiguousarray(np.moveaxis(stacked, -1, 1)) if last else stacked[:, None]

    if "depth" in fields:
        out["depth"] = images("depth", 1, np.float32)
    if "flow" in fields:
        out["flow"] = images("flow", 2, np.float32, last=True)
        out["flow_valid"] = images("flow_valid", 1, np.uint8)
    if "rgb" in fields:
        out["rgb"] = images("rgb_left", 3, np.uint8, last=True)
        if n and frames[0].rgb_right is not None:
            out["rgb_right"] = images("rgb_right", 3, np.uint8, last=True)
    if "pose" in fields:
        out["pose"] = np.stack([f.pose for f in frames]) if n else np.zeros((0, 7), dtype=np.float64)
    return out
