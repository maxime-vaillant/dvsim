# Human-like camera motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import recorded human trajectories, learn how a person moves a head-mounted camera with a small conditional diffusion model, and replay that motion on the paths of the existing `street` planner (DVSim v0.3.0).

**Architecture:** A new `backend/motion/` package. Importers turn TUM / Aria / headset / CSV recordings into 50 Hz tracks in DVSim's frame; `decompose.py` is the single definition of "smooth path + human residual"; a 1D U-Net learns the residual (conditioned on the path's speed, yaw rate and curvature) with x0-prediction diffusion; `style.py` samples it window by window (each window is given the last second of the previous one) and composes it back onto the planner's path as a normal `WaypointTrajectory`. The simulator, dataloader and outputs do not change. Only `model.py`, `train.py` and `style.py` import torch, and they do it lazily.

**Tech Stack:** Python 3.9, numpy, scipy (`scipy.spatial.transform`, `signal`, `optimize`), PyTorch >= 2.0 (optional `torch` extra), FastAPI/pydantic (existing), lil-gui front end (existing), pytest, node's test runner.

**Spec:** `docs/superpowers/specs/2026-10-02-human-motion-design.md`

## Global Constraints

- Python 3.9 compatible: every new module starts with `from __future__ import annotations`; no `X | Y` types evaluated at runtime, no `match`.
- DVSim frame everywhere: world +Y up, camera looks along its own -Z, X right, quaternions `(x, y, z, w)` camera -> world, metres, integer microsecond timestamps.
- Fixed rate: `RATE_HZ = 50`, `DT_US = 20_000`. Windows: `WINDOW = 200` samples (4 s), `CONTEXT = 50` (1 s shared with the previous window), `HOP = 150`.
- torch is imported only in `backend/motion/model.py`, `train.py`, `style.py` (and the code that imports them lazily). `import backend.motion`, `registry`, `importers`, `normalise`, `decompose`, `dataset`, `evaluate`, `corpus` must work without torch. Without torch, using `motion` raises `ImportError("learned human motion needs PyTorch: pip install -e .[torch]")` (HTTP 501 through the API).
- Checkpoints: `version = "dvsim-motion-1"`, tensors + JSON strings only, always loaded with `torch.load(..., weights_only=True)`.
- Generation is deterministic given the seed (same device).
- The existing test suites (Python and frontend) stay unchanged and passing; ruff must report nothing new (`python -m ruff check backend tests` has 20 findings in untouched files before this work).
- EuRoC is not a supported source (drone motion). `motion` is accepted on `street` trajectory specs only.
- Commit style of the repo: `feat(motion): ...`, `docs: ...`. Add the commit attribution lines your session requires (`Co-Authored-By`, `Claude-Session`) to every commit message.

## Environment

Run everything from the repository root with the project venv: `.claude/scratch/venv39/bin/python` (Python 3.9.12, numpy 2.0.2, scipy 1.13.1, trimesh, moderngl, fastapi, httpx). It has torch 2.8.0 with Apple MPS. If `import torch` fails there: `.claude/scratch/venv39/bin/pip install "torch>=2.0"`. On this machine OpenGL tests run (no skips). The default `python3` on PATH lacks trimesh: do not use it.

Training speed (measured): the default model (4.1M parameters, batch 64) takes 132 ms/step on MPS and 2 s/step on CPU. Tests therefore use a tiny model (`base=16, mults=(1, 2)`) on the CPU.

## Spec notes (deviations and additions found while planning)

These are small changes to the approved spec, found by building and testing every file of this plan in a scratch copy of the repository. Task 14 appends them to the spec file.

1. The residual's position axes are the path frame's own (x right, y up, z back), not (forward, lateral, up).
2. `dvsim motion-import` writes a **corpus file** (`.npz`) that `motion-train` and `motion-eval` read; importing a source name again replaces that source. The frame of a dataset can be overridden with `--up X Y Z` and `--camera opengl|opencv`.
3. Every import reports a **heading agreement** (mean cosine between the direction of travel and the camera's horizontal direction). It catches wrong frame conventions (a correct walk gives 1.0, a flipped camera -1.0) and warns below 0.5.
4. Training windows whose mean path speed is below 0.2 m/s are skipped (the heading of a path that does not move is undefined).
5. The path's low-pass extends the signal by point reflection at both ends. With constant padding the measured speed ramped down to 0.3 m/s over the first and last second of every clip.
6. `TrainConfig` records the split settings (`val_frac`, `test_frac`, `split_seed`, `holdout_sources`) so `motion-eval` recomputes exactly the held-out recordings.
7. The "real vs real" reference of `motion-eval` compares even-numbered with odd-numbered held-out recordings (a noise floor); pooling all recordings would compare a set with itself.
8. The API answers 404 for an unknown model id, 501 when the server has no torch, and 422 for a motion preview longer than 300 s in total (duration x count): the preview runs the model synchronously.
9. Command line: `dvsim run --generate street --motion MODEL`, `dvsim motion-import|motion-train|motion-eval`; `motion-eval --scene` adds the event-rate / optical-flow comparison; the motion commands print `error: ...` and exit 1 on bad input.

## Review Focus

The spec says what the software must do; these are the inputs it does not mention and a person is most likely to meet. Each has a test in the task that owns the code.

1. **Mocap dropouts and bad timestamps** (NaN poses, repeated or backwards timestamps, holes of half a second): the track is split or the rows dropped, never crashes or interpolates across the hole. Task 1.
2. **A person who stops or stands still** (heading undefined): the heading is held, no NaN, such windows are not trained on. Tasks 2 and 4.
3. **A dataset with the wrong frame convention** (flipped camera, y-up vs z-up): reported by the heading agreement at import time, not discovered after training. Task 3.
4. **A planned speed the model never saw** (a 5 m/s run, a 0.2 m/s shuffle): clamped with a warning, finite output. Task 7.
5. **A corrupt or foreign checkpoint** (a pickle, a truncated file, another version): a clear error, never executed. Task 6.
6. **No torch on the server / unknown model id / a 10-minute motion preview**: 501 / 404 / 422 before any work is queued or the server is blocked. Task 12.

## File Structure

New, all under `backend/motion/` unless noted:

| File | Responsibility |
|---|---|
| `__init__.py` | `require_torch()` (no torch import at package import) |
| `tracks.py` | `RATE_HZ`, `DT_US`, `FrameSpec`, `RawTrack`, `Track` |
| `normalise.py` | `RawTrack -> [Track]`: frame conversion, gap splitting, 50 Hz resampling; `heading_agreement` |
| `decompose.py` | `split` / `compose` / `conditioning`: the one definition of path and residual |
| `importers/` | `tum.py`, `aria.py`, `headset.py`, `csv_.py` and `read()` (format dispatch, directory search) |
| `corpus.py` | `import_source` (+ report), `save_corpus` / `load_corpus` |
| `registry.py` | model ids and files under `$DVSIM_DATA_DIR/motion`; no torch |
| `dataset.py` | sequences, split by recording, normalisation stats, window sampler |
| `model.py` | `MotionUNet`, `Diffusion` (torch) |
| `train.py` | training loop, checkpoints (torch) |
| `style.py` | `DiffusionMotionStyle.apply(trajectory)` (torch) |
| `evaluate.py` | realism metrics, `evaluate_model` |
| `events_report.py` | plain vs human event statistics on a scene |

Modified: `backend/pipeline.py`, `backend/cli.py`, `backend/api/{settings,schemas,planner,preview,routes}.py`, `frontend/src/{utils/simRequest.js,api/client.js,components/trajectoryOptions.js}`, `README.md`, `docs/architecture.md`, `CHANGELOG.md`, `pyproject.toml`, and the tests. New tests: `tests/motion_helpers.py`, `tests/test_motion_*.py`.

---

### Task 1: Package root, track types and normalisation

**Files:**
- Create: `backend/motion/__init__.py`, `backend/motion/tracks.py`, `backend/motion/normalise.py`
- Test: `tests/test_motion_normalise.py`

**Interfaces:**
- Produces: `backend.motion.require_torch() -> module`; `tracks.RATE_HZ`, `tracks.DT_US`; `tracks.FrameSpec(up=(0,1,0), camera="opengl"|"opencv")`; `tracks.RawTrack(name, times_us, positions, quats, frame=FrameSpec())`; `tracks.Track(source, name, times_us, positions, quats)` with `len()` and `duration_s`; `normalise.rotation_to_y(up) -> Rotation`; `normalise.normalise(raw, source, max_gap_s=0.2, min_duration_s=4.0) -> List[Track]`; `normalise.heading_agreement(track, min_speed=0.3, smooth_s=1.0) -> float`.

**Domain note.** Datasets disagree on frames. TUM RGB-D poses are in a z-up motion-capture world with an *optical* camera (X right, Y down, looking along +Z); DVSim is y-up with an *OpenGL* camera (X right, Y up, looking along -Z). `normalise` rotates the world so `FrameSpec.up` becomes +Y, and for `camera="opencv"` right-multiplies the orientation by a 180 degree turn about X. A walking person looks where they walk, so `heading_agreement` (cosine of travel direction and camera direction) is 1.0 for a correct conversion and -1.0 for a flipped camera.

- [ ] **Step 1: Check the environment and record the baseline**


Run: `.claude/scratch/venv39/bin/python -c "import torch, scipy, trimesh; print(torch.__version__, scipy.__version__)"`
Expected: prints the versions (torch 2.8.0 or any >= 2.0). If torch is missing: `.claude/scratch/venv39/bin/pip install "torch>=2.0"`.

Run: `.claude/scratch/venv39/bin/python -m pytest -q -p no:cacheprovider`
Expected: `444 passed` (the baseline before this work, about 95 s); the count must only grow.


- [ ] **Step 2: Write the failing test**


Create `tests/test_motion_normalise.py`:

```python
"""motion: frame conversion, gap handling and resampling of recorded tracks."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.motion.normalise import heading_agreement, normalise, rotation_to_y
from backend.motion.tracks import DT_US, FrameSpec, RawTrack
from backend.trajectories.trajectory import quats_to_fronts

# an optical (OpenCV) camera looking along world +x in a z-up world: x_cam = -y_world, y_cam = -z_world, z_cam = +x_world
OPTICAL_LOOKING_PLUS_X = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])


def zup_walk(n=800, step_us=10_000, speed=1.4, frame=None) -> RawTrack:
    t = np.arange(n, dtype=np.int64) * step_us + 1_000_000
    pos = np.c_[np.arange(n) * speed * step_us / 1e6, np.zeros(n), np.full(n, 1.6)]
    q = np.tile(Rotation.from_matrix(OPTICAL_LOOKING_PLUS_X).as_quat(), (n, 1))
    return RawTrack("walk", t, pos, q, frame or FrameSpec(up=(0, 0, 1), camera="opencv"))


@pytest.mark.parametrize("up", [(0, 0, 1), (0, 1, 0), (0, -1, 0), (1, 2, 3)])
def test_rotation_to_y_takes_up_to_plus_y(up):
    u = np.asarray(up, dtype=float)
    assert np.allclose(rotation_to_y(u).apply(u / np.linalg.norm(u)), [0, 1, 0], atol=1e-12)


def test_zup_optical_walk_becomes_a_dvsim_walk():
    (track,) = normalise(zup_walk(), "tum")
    assert len(track) == 400 and track.source == "tum"
    assert np.all(np.diff(track.times_us) == DT_US)
    assert np.allclose(track.positions[:, 1], 1.6)                          # the old z is up now
    assert np.allclose(quats_to_fronts(track.quats), [1, 0, 0], atol=1e-9)  # the camera looks where it walks
    assert heading_agreement(track) == pytest.approx(1.0)


def test_a_wrong_camera_convention_is_visible_in_the_heading_agreement():
    (track,) = normalise(zup_walk(frame=FrameSpec(up=(0, 0, 1), camera="opengl")), "tum")
    assert heading_agreement(track) == pytest.approx(-1.0)


def test_heading_agreement_is_nan_when_nothing_moves():
    raw = zup_walk(speed=0.0)
    (track,) = normalise(raw, "tum")
    assert np.isnan(heading_agreement(track))


def test_gaps_split_and_bad_rows_are_dropped():
    raw = zup_walk()
    t, p = raw.times_us.copy(), raw.positions.copy()
    t[400:] += 500_000       # a 0.5 s hole in the middle
    t[10] = t[9]             # a repeated timestamp
    p[20] = np.nan           # a lost frame
    tracks = normalise(RawTrack("g", t, p, raw.quats, raw.frame), "x", min_duration_s=1.0)
    assert [(tr.name, len(tr)) for tr in tracks] == [("g#0", 200), ("g#1", 200)]
    for tr in tracks:
        assert np.all(np.diff(tr.times_us) == DT_US) and np.isfinite(tr.positions).all()


def test_short_segments_are_dropped():
    assert normalise(zup_walk(n=300), "x") == []                   # 3 s < 4 s
    assert len(normalise(zup_walk(n=300), "x", min_duration_s=2.0)) == 1


def test_resampling_interpolates_position_linearly_and_orientation_by_slerp():
    t = np.array([0, 10_000_000], dtype=np.int64)
    pos = np.array([[0.0, 1.0, 0.0], [14.0, 1.0, 0.0]])
    q = Rotation.from_euler("y", [0.0, 90.0], degrees=True).as_quat()
    (track,) = normalise(RawTrack("r", t, pos, q), "x", max_gap_s=20.0)
    assert len(track) == 501
    mid = len(track) // 2
    assert np.allclose(track.positions[:, 0], np.linspace(0, 14, 501))
    assert Rotation.from_quat(track.quats[mid]).magnitude() == pytest.approx(np.radians(45.0), abs=1e-9)


def test_unusable_input():
    raw = zup_walk()
    with pytest.raises(ValueError, match="differ in length"):
        normalise(RawTrack("x", raw.times_us[:-1], raw.positions, raw.quats, raw.frame), "x")
    nan = RawTrack("x", raw.times_us, np.full_like(raw.positions, np.nan), raw.quats, raw.frame)
    assert normalise(nan, "x") == []


def test_frame_spec_validates():
    with pytest.raises(ValueError, match="camera"):
        FrameSpec(camera="fisheye")
    with pytest.raises(ValueError, match="up"):
        FrameSpec(up=(0, 0, 0))
```


- [ ] **Step 3: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_normalise.py`
Expected: FAIL / collection error: `ModuleNotFoundError: No module named 'backend.motion'`.


- [ ] **Step 4: Write the implementation**


Create `backend/motion/__init__.py`:

```python
"""Learned human-like camera motion (see docs/superpowers/specs/2026-10-02-human-motion-design.md).

Importing this package never imports torch: only ``model``, ``train`` and ``style`` do, through ``require_torch``.
"""


def require_torch():
    """The torch module, or an ImportError that says how to get it."""
    try:
        import torch
    except ImportError:
        raise ImportError("learned human motion needs PyTorch: pip install -e .[torch]") from None
    return torch
```

Create `backend/motion/tracks.py`:

```python
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
```

Create `backend/motion/normalise.py`:

```python
"""RawTrack -> Track segments: clean, convert to DVSim's frame, split at gaps, resample at RATE_HZ."""
from __future__ import annotations

from typing import List

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation, Slerp

from backend.motion.tracks import DT_US, RATE_HZ, RawTrack, Track

MAX_GAP_S = 0.2
MIN_DURATION_S = 4.0
_OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0])  # the same rotation maps either camera frame to the other


def rotation_to_y(up) -> Rotation:
    """Smallest rotation taking the direction ``up`` to +Y."""
    u = np.asarray(up, dtype=np.float64)
    u = u / np.linalg.norm(u)
    y = np.array([0.0, 1.0, 0.0])
    axis = np.cross(u, y)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return Rotation.identity() if u @ y > 0 else Rotation.from_rotvec([np.pi, 0.0, 0.0])
    return Rotation.from_rotvec(axis / s * np.arctan2(s, u @ y))


def normalise(raw: RawTrack, source: str, max_gap_s: float = MAX_GAP_S,
              min_duration_s: float = MIN_DURATION_S) -> List[Track]:
    """Segments of ``raw`` in DVSim's frame at RATE_HZ. Gaps longer than ``max_gap_s`` split the track; segments
    shorter than ``min_duration_s`` are dropped. Rows that are not finite or not strictly later than the previous
    kept row are dropped."""
    t = np.asarray(raw.times_us, dtype=np.int64)
    p = np.asarray(raw.positions, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(raw.quats, dtype=np.float64).reshape(-1, 4)
    if not (len(t) == len(p) == len(q)):
        raise ValueError(f"{raw.name}: times, positions and quats differ in length ({len(t)}, {len(p)}, {len(q)})")
    ok = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 1e-9)
    t, p, q = t[ok], p[ok], q[ok]
    if len(t):  # strictly increasing: drop rows that do not move time forward
        later = t > np.concatenate([[t[0] - 1], np.maximum.accumulate(t)[:-1]])
        t, p, q = t[later], p[later], q[later]
    if len(t) < 2:
        return []

    world = rotation_to_y(raw.frame.up)
    rot = world * Rotation.from_quat(q)
    if raw.frame.camera == "opencv":
        rot = rot * Rotation.from_matrix(_OPENCV_TO_OPENGL)
    p = world.apply(p)

    cuts = np.flatnonzero(np.diff(t) > max_gap_s * 1e6) + 1
    bounds = np.concatenate([[0], cuts, [len(t)]])
    pieces = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]
    out: List[Track] = []
    for k, (a, b) in enumerate(pieces):
        ts, ps, rs = t[a:b], p[a:b], rot[a:b]
        n = int((ts[-1] - ts[0]) // DT_US) + 1
        if (n - 1) / RATE_HZ < min_duration_s:
            continue
        grid = ts[0] + np.arange(n, dtype=np.int64) * DT_US
        pos = np.stack([np.interp(grid, ts, ps[:, i]) for i in range(3)], axis=1)
        quat = Slerp(ts.astype(np.float64), rs)(grid.astype(np.float64)).as_quat()
        name = raw.name if len(pieces) == 1 else f"{raw.name}#{k}"
        out.append(Track(source, name, grid, pos, quat))
    return out


def heading_agreement(track: Track, min_speed: float = 0.3, smooth_s: float = 1.0) -> float:
    """Mean cosine between the horizontal direction of travel and the horizontal camera direction, over the moving
    samples (NaN when there are none). A walking person with a head-mounted camera gives a value near 1; a wrong
    frame convention gives something near 0 or negative."""
    from backend.trajectories.trajectory import quats_to_fronts

    v = ndimage.gaussian_filter1d(np.gradient(track.positions, 1.0 / RATE_HZ, axis=0), smooth_s * RATE_HZ / 3.0,
                                  axis=0, mode="nearest")
    vh = v[:, [0, 2]]
    speed = np.linalg.norm(vh, axis=1)
    f = quats_to_fronts(track.quats)[:, [0, 2]]
    fn = np.linalg.norm(f, axis=1)
    moving = (speed > min_speed) & (fn > 1e-6)
    if not moving.any():
        return float("nan")
    cos = np.sum(vh[moving] * f[moving], axis=1) / (speed[moving] * fn[moving])
    return float(cos.mean())
```


- [ ] **Step 5: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_normalise.py`
Expected: 12 passed.

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/motion tests/test_motion_normalise.py`
Expected: All checks passed.


- [ ] **Step 6: Commit**

```bash
git add backend/motion tests/test_motion_normalise.py
git commit -m "feat(motion): track types and normalisation of recorded trajectories"
```

---

### Task 2: Path / residual decomposition

**Files:**
- Create: `backend/motion/decompose.py`
- Test: `tests/test_motion_decompose.py`

**Interfaces:**
- Consumes: `tracks.RATE_HZ`.
- Produces: `decompose.N_RESIDUAL = 6`, `N_COND = 3`; `decompose.PathFrame(positions (N,3), rot: scipy Rotation[N], speed (N,), yaw_rate (N,))`; `extract_path(positions, rate_hz=50, cutoff_hz=0.5) -> PathFrame`; `conditioning(path) -> (N,3)` columns speed m/s, yaw rate rad/s, curvature 1/m; `split(positions, quats, rate_hz=50) -> (PathFrame, residual (N,6))`; `compose(path, residual) -> (positions (N,3), quats (N,4))`.

**Domain note.** This is the one place that defines what is "the path" and what is "the person". Training splits real recordings with it; generation splits the planner's path with it (to get the conditioning) and composes the sampled residual back. The path is the Gaussian low-pass (-3 dB at 0.5 Hz, below the ~1.8 Hz step frequency) of the positions, extended at both ends by point reflection so a straight walk stays straight; its heading is the horizontal tangent (held when slower than 0.1 m/s) with no pitch and no roll. The residual is `[offset in the path frame (x right, y up, z back), rotation vector of path^-1 * camera]`. `split` then `compose` is exact.

- [ ] **Step 1: Write the failing test**


Create `tests/test_motion_decompose.py`:

```python
"""motion: the path / residual split is exact, invariant, and finds what was put in."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.motion.decompose import N_COND, N_RESIDUAL, PathFrame, compose, conditioning, extract_path, split


def walk(n=1000, speed=1.4, bob=0.03, roll=0.05, turn=0.0):
    """Straight (turn=0) or circling walk at ``speed`` with a 2 Hz vertical bob and a 1 Hz roll about the view axis."""
    t = np.arange(n) / 50.0
    psi = turn * t
    pos = np.stack([np.cumsum(-np.sin(psi) * speed) / 50.0,
                    1.6 + bob * np.sin(2 * np.pi * 2.0 * t),
                    np.cumsum(-np.cos(psi) * speed) / 50.0], axis=1)
    rolls = np.stack([0 * t, 0 * t, roll * np.sin(2 * np.pi * 1.0 * t)], axis=1)
    return pos, (Rotation.from_euler("y", psi) * Rotation.from_rotvec(rolls)).as_quat()


def test_split_then_compose_gives_the_track_back():
    pos, quat = walk(turn=0.2)
    path, residual = split(pos, quat)
    pos2, quat2 = compose(path, residual)
    assert residual.shape == (1000, N_RESIDUAL)
    assert np.abs(pos2 - pos).max() < 1e-9
    assert np.abs(np.abs((quat2 * quat).sum(axis=1)) - 1).max() < 1e-9      # same rotations (q and -q are the same)


def test_the_residual_holds_the_bob_and_the_roll():
    pos, quat = walk()
    _, residual = split(pos, quat)
    mid = slice(150, -150)                                                  # away from the filter's edges
    assert residual[mid, 1].std() * np.sqrt(2) == pytest.approx(0.03, rel=0.05)    # y of the path frame is up
    assert residual[mid, 5].std() * np.sqrt(2) == pytest.approx(0.05, rel=0.05)    # roll = rotation about z
    assert np.abs(residual[mid, 3:5]).max() < 1e-9                                 # no pitch or yaw was added
    assert np.abs(residual[mid, [0, 2]]).max() < 1e-6                              # nor a sideways / forward shift


def test_conditioning_of_a_straight_walk_and_a_turn():
    pos, quat = walk()
    cond = conditioning(split(pos, quat)[0])
    assert cond.shape == (1000, N_COND)
    assert cond[500] == pytest.approx([1.4, 0.0, 0.0], abs=1e-6)
    pos, quat = walk(turn=0.2)
    cond = conditioning(split(pos, quat)[0])
    assert cond[500, 1] == pytest.approx(0.2, abs=1e-3)                     # counter-clockwise seen from above
    assert cond[500, 2] == pytest.approx(0.2 / 1.4, rel=0.03)               # curvature = yaw rate / speed


def test_the_path_does_not_slow_down_at_the_ends_of_a_walk():
    pos, quat = walk(bob=0.0, roll=0.0)
    path = split(pos, quat)[0]
    assert path.speed == pytest.approx(np.full(1000, 1.4), rel=1e-6)           # also in the first and last second
    assert np.abs(path.positions - pos).max() < 1e-9                           # a straight walk is its own path
    pos, quat = walk(turn=0.2, bob=0.0, roll=0.0)
    speed = split(pos, quat)[0].speed
    assert speed.min() > 1.3 and speed.max() < 1.5                              # a gentle turn: no edge ramp either


def test_residual_and_conditioning_do_not_depend_on_where_or_which_way_the_walk_is():
    pos, quat = walk(turn=0.2)
    path, residual = split(pos, quat)
    turn = Rotation.from_euler("y", 37, degrees=True)
    path2, residual2 = split(turn.apply(pos) + [5.0, 0.0, -3.0], (turn * Rotation.from_quat(quat)).as_quat())
    assert np.abs(residual2 - residual).max() < 1e-9
    assert np.abs(conditioning(path2) - conditioning(path)).max() < 1e-9


def test_a_stationary_camera_has_a_defined_heading_and_no_residual():
    pos, quat = np.tile([0.0, 1.6, 0.0], (100, 1)), np.tile([0.0, 0.0, 0.0, 1.0], (100, 1))
    path, residual = split(pos, quat)
    assert np.abs(residual).max() < 1e-12 and np.abs(path.speed).max() < 1e-12
    assert isinstance(path, PathFrame) and len(path) == 100


def test_heading_is_held_when_the_walk_pauses():
    pos, quat = walk(n=600)
    pos[300:] = pos[300]                                                     # stops at sample 300
    path, _ = split(pos, quat)
    forward = path.rot.apply([0, 0, -1])
    assert np.allclose(forward[400:], forward[299], atol=1e-6)               # keeps looking where it was going


def test_invalid_input():
    with pytest.raises(ValueError, match="N>=5"):
        extract_path(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        extract_path(np.full((20, 3), np.nan))
    path, _ = split(*walk(n=100))
    with pytest.raises(ValueError, match="residual must have shape"):
        compose(path, np.zeros((99, 6)))
```


- [ ] **Step 2: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_decompose.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'backend.motion.decompose'`.


- [ ] **Step 3: Write the implementation**


Create `backend/motion/decompose.py`:

```python
"""Trajectory <-> (smooth path, residual). The only place that defines the split (training and sampling share it).

The path is the low-passed position with a heading taken from its horizontal tangent (no pitch, no roll). The
residual is what the person does on top of it, expressed in the path's own frame (axes: x right, y up, z back):
the position offset (3) and the camera rotation relative to the path frame as a rotation vector (3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from backend.motion.tracks import RATE_HZ

CUTOFF_HZ = 0.5      # below the step frequency (about 1.8 Hz)
MIN_SPEED = 0.1      # m/s: slower than this the heading is undefined and is held
MIN_CURVATURE_SPEED = 0.3
N_RESIDUAL = 6
N_COND = 3           # speed (m/s), yaw rate (rad/s), curvature (1/m)


@dataclass(frozen=True)
class PathFrame:
    positions: np.ndarray   # (N, 3) smooth path
    rot: Rotation           # N rotations, path frame -> world (forward = -Z, up = +Y)
    speed: np.ndarray       # (N,) horizontal speed, m/s
    yaw_rate: np.ndarray    # (N,) rad/s, counter-clockwise seen from above

    def __len__(self) -> int:
        return len(self.positions)


def _smooth(p: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass whose ends continue the motion: the signal is extended by point reflection about its first
    and last sample (a straight walk stays straight), not held constant (which would slow the path down at the ends)."""
    pad = min(int(4 * sigma) + 1, len(p) - 1)
    head = 2 * p[0] - p[pad:0:-1]
    tail = 2 * p[-1] - p[-2:-pad - 2:-1]
    return ndimage.gaussian_filter1d(np.concatenate([head, p, tail]), sigma, axis=0, mode="nearest")[pad:-pad]


def extract_path(positions, rate_hz: float = RATE_HZ, cutoff_hz: float = CUTOFF_HZ) -> PathFrame:
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 5:
        raise ValueError(f"expected (N>=5, 3) positions, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("positions contain non-finite values")
    sigma = 0.1325 / cutoff_hz * rate_hz      # Gaussian whose -3 dB point is cutoff_hz
    path = _smooth(p, sigma)
    v = np.gradient(path, 1.0 / rate_hz, axis=0)[:, [0, 2]]
    speed = np.linalg.norm(v, axis=1)
    valid = speed > MIN_SPEED
    h = np.zeros_like(v)
    h[:] = [0.0, -1.0]                        # nothing moves: look along -Z
    if valid.any():
        idx = np.where(valid, np.arange(len(v)), -1)
        idx = np.maximum.accumulate(idx)      # hold the last valid heading ...
        first = int(np.argmax(valid))
        idx[idx < 0] = first                  # ... and back-fill the start
        h = v[idx] / speed[idx][:, None]
    psi = np.unwrap(np.arctan2(-h[:, 0], -h[:, 1]))   # forward = (-sin psi, 0, -cos psi)
    rot = Rotation.from_euler("y", psi)
    yaw_rate = np.gradient(psi, 1.0 / rate_hz)
    return PathFrame(path, rot, speed, yaw_rate)


def conditioning(path: PathFrame) -> np.ndarray:
    """(N, 3): speed, yaw rate, curvature."""
    curvature = path.yaw_rate / np.maximum(path.speed, MIN_CURVATURE_SPEED)
    return np.stack([path.speed, path.yaw_rate, curvature], axis=1)


def split(positions, quats, rate_hz: float = RATE_HZ) -> Tuple[PathFrame, np.ndarray]:
    """(path, residual (N, 6)) of a track."""
    path = extract_path(positions, rate_hz)
    inv = path.rot.inv()
    offset = inv.apply(np.asarray(positions, dtype=np.float64) - path.positions)
    rvec = (inv * Rotation.from_quat(np.asarray(quats, dtype=np.float64))).as_rotvec()
    return path, np.concatenate([offset, rvec], axis=1)


def compose(path: PathFrame, residual) -> Tuple[np.ndarray, np.ndarray]:
    """(positions (N, 3), quats (N, 4)) of a path plus a residual (N, 6): the inverse of ``split``."""
    r = np.asarray(residual, dtype=np.float64)
    if r.shape != (len(path), N_RESIDUAL):
        raise ValueError(f"residual must have shape ({len(path)}, {N_RESIDUAL}), got {r.shape}")
    positions = path.positions + path.rot.apply(r[:, :3])
    quats = (path.rot * Rotation.from_rotvec(r[:, 3:])).as_quat()
    return positions, quats
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_decompose.py`
Expected: 8 passed.

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/motion tests/test_motion_decompose.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/motion/decompose.py tests/test_motion_decompose.py
git commit -m "feat(motion): split a trajectory into a smooth path and a human-motion residual"
```

---

### Task 3: Importers, corpus and model registry

**Files:**
- Create: `backend/motion/importers/__init__.py`, `importers/tum.py`, `importers/aria.py`, `importers/headset.py`, `importers/csv_.py`, `backend/motion/corpus.py`, `backend/motion/registry.py`
- Test: `tests/test_motion_importers.py`

**Interfaces:**
- Consumes: `tracks.FrameSpec/RawTrack/Track`, `normalise.normalise`, `normalise.heading_agreement`, `trajectories.WaypointTrajectory.from_json`, `backend.config.data_dir`.
- Produces: `importers.read(path, fmt, layout="tumvi", up=None, camera=None) -> List[RawTrack]` and `importers.FORMATS = ("tum", "aria", "headset", "csv")`; `corpus.import_source(source, path, fmt, layout="tumvi", up=None, camera=None) -> (List[Track], SourceReport)` (`SourceReport`: `source, files, segments, hours, agreement, warnings`); `corpus.save_corpus(path, tracks)`, `corpus.load_corpus(path) -> List[Track]`, `corpus.MIN_AGREEMENT = 0.5`; `registry.MODEL_ID` (regex), `registry.models_dir(data_dir=None)`, `registry.resolve_model(ref, data_dir=None) -> Path` (`ref` = `{"model": id}` or `{"path": file}`; ValueError for a bad id, FileNotFoundError for a missing file), `registry.list_models(data_dir=None) -> [{"id", "size_bytes"}]`.

**Formats.** `tum`: `t[s] x y z qx qy qz qw` lines (also what DVSim writes); frame z-up / optical camera. `csv` with layout `tumvi`: header line, `t[ns], x, y, z, qw, qx, qy, qz` (EuRoC/TUM-VI order, w first); the z-up / optical frame is an assumption the heading agreement checks. `aria`: MPS `closed_loop_trajectory.csv` (named columns); up = minus the mean gravity vector; device frame assumed optical. `headset`: DVSim JSON (ms timestamps), already in DVSim's frame.

- [ ] **Step 1: Write the failing test**


Create `tests/test_motion_importers.py`:

```python
"""motion: reading trajectory files of each format, directory discovery, and the import report."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.motion.corpus import MIN_AGREEMENT, import_source, load_corpus, save_corpus
from backend.motion.importers import FORMATS, read
from backend.motion.registry import MODEL_ID, list_models, resolve_model
from backend.motion.tracks import FrameSpec

N = 1200
T = np.arange(N) * 0.0333                                                    # 30 Hz, 40 s
POS = np.c_[T * 1.3, np.zeros(N), np.full(N, 1.6)]                           # walks along +x in a z-up world
QUAT = Rotation.from_matrix(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])).as_quat()   # optical camera looks +x


def write_tum(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    lines += [f"{t + 1000:.6f} {p[0]} {p[1]} {p[2]} {QUAT[0]} {QUAT[1]} {QUAT[2]} {QUAT[3]}" for t, p in zip(T, POS)]
    path.write_text("\n".join(lines) + "\n")


def test_tum_file(tmp_path):
    write_tum(tmp_path / "groundtruth.txt")
    (raw,) = read(tmp_path / "groundtruth.txt", "tum")
    assert raw.name == "groundtruth" and len(raw.times_us) == N
    assert raw.times_us[0] == 1_000_000_000 and raw.frame == FrameSpec(up=(0, 0, 1), camera="opencv")
    (forced,) = read(tmp_path, "tum", up=(0, 1, 0), camera="opengl")
    assert forced.frame == FrameSpec(up=(0, 1, 0), camera="opengl")
    tracks, report = import_source("tumrgbd", tmp_path, "tum")
    assert report.files == 1 and report.segments == 1 and report.agreement == pytest.approx(1.0)
    assert report.hours == pytest.approx(39.96 / 3600, rel=0.01) and report.warnings == []


def test_tumvi_csv_has_nanoseconds_and_w_first(tmp_path):
    lines = ["#timestamp [ns],px,py,pz,qw,qx,qy,qz"]
    lines += [f"{int(t * 1e9)},{p[0]},{p[1]},{p[2]},{QUAT[3]},{QUAT[0]},{QUAT[1]},{QUAT[2]}" for t, p in zip(T, POS)]
    (tmp_path / "data.csv").write_text("\n".join(lines))
    _, report = import_source("tumvi", tmp_path / "data.csv", "csv", layout="tumvi")
    assert report.agreement == pytest.approx(1.0)


def test_aria_takes_up_from_gravity(tmp_path):
    head = ("graph_uid,tracking_timestamp_us,tx_world_device,ty_world_device,tz_world_device,qx_world_device,"
            "qy_world_device,qz_world_device,qw_world_device,gravity_x_world,gravity_y_world,gravity_z_world")
    rows = [f"g,{int(t * 1e6)},{p[0]},{p[1]},{p[2]},{QUAT[0]},{QUAT[1]},{QUAT[2]},{QUAT[3]},0,0,-9.81"
            for t, p in zip(T, POS)]
    (tmp_path / "closed_loop_trajectory.csv").write_text("\n".join([head] + rows))
    (raw,) = read(tmp_path, "aria")
    assert raw.frame.up == pytest.approx((0, 0, 1))
    assert import_source("aria", tmp_path, "aria")[1].agreement == pytest.approx(1.0)


def test_headset_json_is_already_in_dvsim_frame(tmp_path):
    records = [{"timestamp": t * 1000, "x": -1.3 * t, "y": 1.6, "z": 0.0, "rx": -1.0, "ry": 0.0, "rz": 0.0} for t in T]
    (tmp_path / "walk.json").write_text(json.dumps(records))
    assert import_source("headset", tmp_path / "walk.json", "headset")[1].agreement == pytest.approx(1.0)


def test_directories_are_searched_recursively(tmp_path):
    write_tum(tmp_path / "a" / "groundtruth.txt")
    write_tum(tmp_path / "b" / "groundtruth.txt")
    assert sorted(r.name for r in read(tmp_path, "tum")) == ["a/groundtruth", "b/groundtruth"]
    write_tum(tmp_path / "loose" / "run1.txt")                               # no groundtruth.txt: any .txt
    assert [r.name for r in read(tmp_path / "loose", "tum")] == ["run1"]


def test_a_wrong_frame_is_reported(tmp_path):
    write_tum(tmp_path / "groundtruth.txt")
    _, report = import_source("bad", tmp_path, "tum", camera="opengl")
    assert report.agreement == pytest.approx(-1.0)
    assert len(report.warnings) == 1 and "frame is probably wrong" in report.warnings[0] and MIN_AGREEMENT == 0.5


def test_a_recording_without_a_long_enough_segment_is_reported(tmp_path):
    lines = [f"{t:.3f} {t} 0 1.6 {QUAT[0]} {QUAT[1]} {QUAT[2]} {QUAT[3]}" for t in np.arange(0, 2, 0.05)]
    (tmp_path / "short.txt").write_text("\n".join(lines))
    tracks, report = import_source("s", tmp_path / "short.txt", "tum")
    assert tracks == [] and report.segments == 0 and "no segment of at least 4 s" in report.warnings[0]


def test_errors_name_the_file_and_line(tmp_path):
    (tmp_path / "bad.txt").write_text("# header\n1 2 3\n")
    with pytest.raises(ValueError, match=r"bad\.txt:2: expected 8 numbers"):
        read(tmp_path / "bad.txt", "tum")
    (tmp_path / "bad.csv").write_text("h\n1,2,x,4,5,6,7,8\n")
    with pytest.raises(ValueError, match=r"bad\.csv:2"):
        read(tmp_path / "bad.csv", "csv")
    (tmp_path / "aria.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing column"):
        read(tmp_path / "aria.csv", "aria")
    with pytest.raises(ValueError, match="unknown format"):
        read(tmp_path, "rosbag")
    with pytest.raises(ValueError, match="unknown csv layout"):
        read(tmp_path / "bad.csv", "csv", "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no tum trajectory file"):
        read(empty, "tum")
    with pytest.raises(FileNotFoundError):
        read(tmp_path / "missing", "tum")
    assert FORMATS == ("tum", "aria", "headset", "csv")


def test_a_corpus_survives_saving(tmp_path):
    write_tum(tmp_path / "groundtruth.txt")
    tracks, _ = import_source("tumrgbd", tmp_path, "tum")
    save_corpus(tmp_path / "corpus.npz", tracks)
    (back,) = load_corpus(tmp_path / "corpus.npz")
    assert (back.source, back.name) == (tracks[0].source, tracks[0].name)
    assert np.array_equal(back.times_us, tracks[0].times_us) and np.allclose(back.positions, tracks[0].positions)


def test_registry(tmp_path):
    folder = tmp_path / "motion"
    folder.mkdir()
    (folder / "walk-v1.pt").write_bytes(b"x")
    (folder / "notes.txt").write_bytes(b"x")
    (folder / ".hidden.pt").write_bytes(b"x")
    assert resolve_model({"model": "walk-v1"}, tmp_path) == folder / "walk-v1.pt"
    assert [m["id"] for m in list_models(tmp_path)] == ["walk-v1"]
    assert list_models(tmp_path / "none") == []
    assert resolve_model({"path": str(folder / "walk-v1.pt")}) == folder / "walk-v1.pt"
    for bad in ("../walk-v1", "a/b", "", ".x", "x" * 65, None, 3):
        with pytest.raises(ValueError, match="invalid motion model id"):
            resolve_model({"model": bad}, tmp_path)
    with pytest.raises(FileNotFoundError, match="'other' not found"):
        resolve_model({"model": "other"}, tmp_path)
    with pytest.raises(FileNotFoundError, match="file not found"):
        resolve_model({"path": str(folder / "gone.pt")})
    assert MODEL_ID.match("walk_v1.2")
```


- [ ] **Step 2: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_importers.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'backend.motion.corpus'`.


- [ ] **Step 3: Write the implementation**


Create `backend/motion/importers/tum.py`:

```python
"""TUM RGB-D text format: ``timestamp[s] tx ty tz qx qy qz qw`` per line, ``#`` comments (also written by DVSim)."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from backend.motion.tracks import FrameSpec, RawTrack

TUM_FRAME = FrameSpec(up=(0.0, 0.0, 1.0), camera="opencv")   # motion-capture world (z up), optical camera frame


def read_tum(path: Path, name: str) -> RawTrack:
    rows = []
    with open(path, "r") as fh:
        for no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                if len(parts) != 8:
                    raise ValueError
                rows.append([float(v) for v in parts])
            except ValueError:
                raise ValueError(
                    f"{path}:{no}: expected 8 numbers 't tx ty tz qx qy qz qw', got {line[:80]!r}") from None
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than 2 poses")
    a = np.asarray(rows)
    return RawTrack(name, np.rint(a[:, 0] * 1e6).astype(np.int64), a[:, 1:4], a[:, 4:8], TUM_FRAME)
```

Create `backend/motion/importers/csv_.py`:

```python
"""Generic CSV poses: a layout names the columns, the time unit and the number of header lines."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from backend.motion.tracks import FrameSpec, RawTrack

_UNIT_TO_US = {"s": 1e6, "ms": 1e3, "us": 1.0, "ns": 1e-3}


@dataclass(frozen=True)
class CsvLayout:
    """Column indices of t, x, y, z and the quaternion (qw first or last) in a comma separated file."""

    t: int
    x: int
    y: int
    z: int
    qx: int
    qy: int
    qz: int
    qw: int
    unit: str = "s"
    header_lines: int = 1
    frame: FrameSpec = FrameSpec()


CSV_LAYOUTS: Dict[str, CsvLayout] = {
    # TUM-VI ground truth (EuRoC layout): timestamp [ns], p x y z, q w x y z. The frame is an assumption: the
    # import report prints the heading agreement; override with --up / --camera when it is not close to 1.
    "tumvi": CsvLayout(t=0, x=1, y=2, z=3, qw=4, qx=5, qy=6, qz=7, unit="ns",
                       frame=FrameSpec(up=(0.0, 0.0, 1.0), camera="opencv")),
    # t [s], x y z, qx qy qz qw, in DVSim's own frame
    "dvsim": CsvLayout(t=0, x=1, y=2, z=3, qx=4, qy=5, qz=6, qw=7),
}


def read_csv(path: Path, name: str, layout: str = "tumvi") -> RawTrack:
    if layout not in CSV_LAYOUTS:
        raise ValueError(f"unknown csv layout {layout!r}: choose one of {', '.join(CSV_LAYOUTS)}")
    lay = CSV_LAYOUTS[layout]
    cols = [lay.t, lay.x, lay.y, lay.z, lay.qx, lay.qy, lay.qz, lay.qw]
    rows = []
    with open(path, "r", newline="") as fh:
        for no, rec in enumerate(csv.reader(fh), 1):
            if no <= lay.header_lines or not rec or rec[0].lstrip().startswith("#"):
                continue
            try:
                rows.append([float(rec[c]) for c in cols])
            except (ValueError, IndexError):
                shown = ",".join(rec)[:80]
                raise ValueError(f"{path}:{no}: cannot read columns {cols} as numbers from {shown!r}") from None
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than 2 poses")
    a = np.asarray(rows)
    t = np.rint(a[:, 0] * _UNIT_TO_US[lay.unit]).astype(np.int64)
    return RawTrack(name, t, a[:, 1:4], a[:, 4:8], lay.frame)
```

Create `backend/motion/importers/aria.py`:

```python
"""Project Aria / Aria Digital Twin trajectories (MPS ``closed_loop_trajectory.csv``): header-named columns."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from backend.motion.tracks import FrameSpec, RawTrack

_T = "tracking_timestamp_us"
_P = ("tx_world_device", "ty_world_device", "tz_world_device")
_Q = ("qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device")
_G = ("gravity_x_world", "gravity_y_world", "gravity_z_world")


def read_aria(path: Path, name: str) -> RawTrack:
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in (_T,) + _P + _Q if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        has_g = all(c in reader.fieldnames for c in _G)
        rows, grav = [], []
        for no, rec in enumerate(reader, 2):
            try:
                rows.append([float(rec[c]) for c in (_T,) + _P + _Q])
                if has_g:
                    grav.append([float(rec[c]) for c in _G])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{no}: non-numeric value in a pose or gravity column") from None
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than 2 poses")
    a = np.asarray(rows)
    g = -np.mean(grav, axis=0) if has_g else np.array([0.0, 0.0, 1.0])   # gravity points down
    up = tuple(float(v) for v in g / np.linalg.norm(g))
    # the device frame is assumed to be optical-style (+Z forward, +Y down); the import report checks the assumption
    return RawTrack(name, np.rint(a[:, 0]).astype(np.int64), a[:, 1:4], a[:, 4:8], FrameSpec(up=up, camera="opencv"))
```

Create `backend/motion/importers/headset.py`:

```python
"""DVSim headset recordings and any legacy trajectory JSON (timestamps in ms): already in DVSim's frame."""
from __future__ import annotations

from pathlib import Path

from backend.motion.tracks import RawTrack
from backend.trajectories.trajectory import WaypointTrajectory


def read_headset(path: Path, name: str) -> RawTrack:
    traj = WaypointTrajectory.from_json(path, on_duplicate="drop")
    return RawTrack(name, traj.times_us, traj.positions, traj.quats)
```

Create `backend/motion/importers/__init__.py`:

```python
"""Readers for trajectory files. ``read(path, fmt)`` -> RawTrack list; a directory is searched recursively."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional

from backend.motion.importers.aria import read_aria
from backend.motion.importers.csv_ import read_csv
from backend.motion.importers.headset import read_headset
from backend.motion.importers.tum import read_tum
from backend.motion.tracks import FrameSpec, RawTrack

_PATTERNS = {
    "tum": ("groundtruth.txt", "*.txt"),
    "aria": ("closed_loop_trajectory.csv", "*trajectory*.csv"),
    "headset": ("*.json",),
    "csv": ("*.csv",),
}
FORMATS = tuple(_PATTERNS)


def _files(path: Path, fmt: str) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"{path}: no such file or directory")
    for pattern in _PATTERNS[fmt]:   # the specific name first, the generic one if nothing matched
        found = sorted(path.rglob(pattern))
        if found:
            return found
    return []


def read(path, fmt: str, layout: str = "tumvi", up=None, camera: Optional[str] = None) -> List[RawTrack]:
    """All tracks of ``path`` (a file, or a directory searched recursively). Each format has a default frame
    (``aria`` takes its up direction from the gravity column); ``up`` (X, Y, Z in the dataset's world frame) and
    ``camera`` ("opengl" | "opencv") replace what the format says. ``layout`` selects the columns of ``csv``
    (see ``importers.csv_.CSV_LAYOUTS``)."""
    if fmt not in _PATTERNS:
        raise ValueError(f"unknown format {fmt!r}: choose one of {', '.join(FORMATS)}")
    root = Path(path)
    files = _files(root, fmt)
    if not files:
        raise FileNotFoundError(f"{root}: no {fmt} trajectory file found")
    tracks: List[RawTrack] = []
    for f in files:
        name = f.stem if root.is_file() else str(f.relative_to(root).with_suffix(""))
        if fmt == "tum":
            raw = read_tum(f, name)
        elif fmt == "aria":
            raw = read_aria(f, name)
        elif fmt == "headset":
            raw = read_headset(f, name)
        else:
            raw = read_csv(f, name, layout)
        if up is not None or camera is not None:
            raw = dataclasses.replace(raw, frame=FrameSpec(tuple(up) if up is not None else raw.frame.up,
                                                           camera or raw.frame.camera))
        tracks.append(raw)
    return tracks
```

Create `backend/motion/corpus.py`:

```python
"""A corpus is the normalised tracks of every source, saved as one .npz so training does not reparse the files."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from backend.motion.importers import read
from backend.motion.normalise import heading_agreement, normalise
from backend.motion.tracks import RATE_HZ, Track

MIN_AGREEMENT = 0.5


@dataclass
class SourceReport:
    source: str
    files: int = 0
    segments: int = 0
    hours: float = 0.0
    agreement: float = float("nan")     # mean over segments of heading_agreement
    warnings: List[str] = field(default_factory=list)


def import_source(source: str, path, fmt: str, layout: str = "tumvi", up=None, camera: Optional[str] = None):
    """(tracks, report) of one dataset; ``up`` and ``camera`` override the format's frame (see ``importers.read``)."""
    raws = read(path, fmt, layout, up, camera)
    tracks: List[Track] = []
    rep = SourceReport(source, files=len(raws))
    for raw in raws:
        got = normalise(raw, source)
        if not got:
            rep.warnings.append(f"{raw.name}: no segment of at least 4 s")
        tracks.extend(got)
    rep.segments = len(tracks)
    rep.hours = sum(len(t) for t in tracks) / RATE_HZ / 3600.0
    agree = [a for a in (heading_agreement(t) for t in tracks) if np.isfinite(a)]
    if agree:
        rep.agreement = float(np.mean(agree))
        if rep.agreement < MIN_AGREEMENT:
            rep.warnings.append(
                f"heading agreement {rep.agreement:.2f} < {MIN_AGREEMENT}: the camera does not look where the person "
                "walks; the frame is probably wrong (try --camera opencv|opengl and --up X,Y,Z)")
    return tracks, rep


def save_corpus(path, tracks: List[Track]) -> None:
    arrays = {"index": np.array(json.dumps([[t.source, t.name] for t in tracks]))}
    for i, t in enumerate(tracks):
        arrays[f"t{i}"], arrays[f"p{i}"], arrays[f"q{i}"] = t.times_us, t.positions, t.quats
    np.savez_compressed(path, **arrays)


def load_corpus(path) -> List[Track]:
    with np.load(path, allow_pickle=False) as z:
        index = json.loads(str(z["index"]))
        return [Track(s, n, z[f"t{i}"], z[f"p{i}"], z[f"q{i}"]) for i, (s, n) in enumerate(index)]
```

Create `backend/motion/registry.py`:

```python
"""Where trained motion models live and how a spec names one. No torch: the API uses it to validate requests."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from backend.config import data_dir as default_data_dir

MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SUFFIX = ".pt"


def models_dir(data_dir: Optional[Path] = None) -> Path:
    return Path(data_dir or default_data_dir()) / "motion"


def resolve_model(ref: Mapping[str, Any], data_dir: Optional[Path] = None) -> Path:
    """The checkpoint file a ``motion`` spec points at: ``{"model": id}`` (a file of the models directory) or
    ``{"path": file}`` (CLI and trusted callers). Raises ValueError for a malformed reference and
    FileNotFoundError for a missing file."""
    if "path" in ref:
        path = Path(str(ref["path"])).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"motion model file not found: {path}")
        return path
    model = ref.get("model")
    if not isinstance(model, str) or not MODEL_ID.match(model):
        raise ValueError(f"invalid motion model id {model!r}: use 1-64 letters, digits, '.', '_' or '-'")
    path = models_dir(data_dir) / f"{model}{SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(f"motion model {model!r} not found in {path.parent}")
    return path


def list_models(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    folder = models_dir(data_dir)
    if not folder.is_dir():
        return []
    return [{"id": p.stem, "size_bytes": p.stat().st_size}
            for p in sorted(folder.glob(f"*{SUFFIX}")) if MODEL_ID.match(p.stem)]
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_importers.py`
Expected: 10 passed.

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/motion tests/test_motion_importers.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/motion tests/test_motion_importers.py
git commit -m "feat(motion): importers (TUM, TUM-VI csv, Aria, headset), corpus files and the model registry"
```

---

### Task 4: Sequences, split by recording, normalisation, window sampling (and the test helpers)

**Files:**
- Create: `backend/motion/dataset.py`, `tests/motion_helpers.py`
- Test: `tests/test_motion_dataset.py`

**Interfaces:**
- Consumes: `decompose.split/conditioning/N_RESIDUAL/N_COND`, `tracks.Track`.
- Produces: `dataset.WINDOW=200, CONTEXT=50, HOP=150, MIN_WINDOW_SPEED=0.2`; `Sequence(source, name, residual (N,6), cond (N,3))`; `build_sequences(tracks) -> List[Sequence]`; `recording_key(seq) -> str` (`"source/name-before-#"`); `assign_splits(seqs, val_frac=0.1, test_frac=0.1, seed=0, holdout_sources=()) -> {key: "train"|"val"|"test"}`; `select(seqs, splits, which)`; `Stats.fit(seqs)`, `.norm_res/.denorm_res/.norm_cond`, `.to_dict()/.from_dict()`, `.speed_range`; `WindowSampler(seqs, stats, hop=25, min_speed=0.2).batch(rng, size) -> {"residual": (B,200,6) f32, "cond": (B,200,3) f32}`. `tests/motion_helpers.py`: `synthetic_walk(seed, n=1500, source="syn") -> Track`, `planner_line(seconds, speed=1.4, start_us=0, height=1.6) -> WaypointTrajectory`, `write_tum_walk(path, seconds=40, rate_hz=30)`.

**Domain note.** Segments of one recording (`walk#0`, `walk#1`) share a split, decided by a hash, so no window of a held-out recording leaks into training. Sources are sampled equally whatever their size.

- [ ] **Step 1: Write the helpers and the failing test**


Create `tests/motion_helpers.py`:

```python
"""Synthetic walks for the motion tests: a smooth path plus bob, sway and roll that depend on the speed."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from backend.motion.tracks import DT_US, Track
from backend.trajectories import WaypointTrajectory


def synthetic_walk(seed: int, n: int = 1500, source: str = "syn") -> Track:
    """n samples at 50 Hz of a walk with a varying speed (0.7-1.7 m/s) and a gently weaving heading."""
    r = np.random.default_rng(seed)
    t = np.arange(n) / 50.0
    speed = 1.2 + 0.5 * np.sin(2 * np.pi * t / r.uniform(8, 15) + r.uniform(0, 6))
    yaw_rate = 0.25 * np.sin(2 * np.pi * t / r.uniform(10, 20) + r.uniform(0, 6))
    psi = np.cumsum(yaw_rate) / 50.0
    x = np.cumsum(-np.sin(psi) * speed) / 50.0
    z = np.cumsum(-np.cos(psi) * speed) / 50.0
    phase = np.cumsum(1.6 + 0.2 * speed) / 50.0 * 2 * np.pi      # step frequency grows with the speed
    y = 1.6 + 0.02 * speed * np.sin(2 * phase)                   # bob at twice the step frequency
    sway = 0.02 * np.sin(phase)
    pos = np.stack([x, y, z], axis=1) + np.stack([-sway * np.cos(psi), 0 * t, sway * np.sin(psi)], axis=1)
    wobble = np.stack([0.02 * np.sin(2 * phase), 0.05 * np.sin(phase), 0.03 * np.sin(phase)], axis=1)
    rot = Rotation.from_euler("y", psi) * Rotation.from_rotvec(wobble)
    return Track(source, f"walk{seed}", (np.arange(n) * DT_US).astype(np.int64), pos, rot.as_quat())


def planner_line(seconds: float, speed: float = 1.4, start_us: int = 0, height: float = 1.6) -> WaypointTrajectory:
    """What the street planner returns, in miniature: a straight walk along -Z, a waypoint per second."""
    n = max(2, int(np.ceil(seconds)) + 1)
    times = np.linspace(0.0, seconds * 1e6, n).round().astype(np.int64) + start_us
    s = np.linspace(0.0, seconds, n) * speed
    pos = np.stack([np.zeros(n), np.full(n, height), -s], axis=1)
    return WaypointTrajectory.from_waypoints(times, pos, fronts=np.tile([0.0, 0.0, -1.0], (n, 1)))


def write_tum_walk(path, seconds: float = 40.0, rate_hz: float = 30.0) -> None:
    """A TUM RGB-D groundtruth file: a walk along +x in a z-up world, optical camera looking where it walks."""
    t = np.arange(0.0, seconds, 1.0 / rate_hz)
    quat = Rotation.from_matrix(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])).as_quat()
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    lines += [f"{1000 + ti:.6f} {1.3 * ti} 0 1.6 {quat[0]} {quat[1]} {quat[2]} {quat[3]}" for ti in t]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
```

Create `tests/test_motion_dataset.py`:

```python
"""motion: sequences, the split by recording, normalisation and window sampling."""
from __future__ import annotations

import numpy as np
import pytest
from motion_helpers import synthetic_walk

from backend.motion.dataset import (
    CONTEXT, HOP, WINDOW, Sequence, Stats, WindowSampler, assign_splits, build_sequences, recording_key, select,
)


@pytest.fixture(scope="module")
def seqs():
    return build_sequences([synthetic_walk(i, source="a" if i < 6 else "b") for i in range(10)])


def test_constants_describe_a_four_second_window_with_one_second_of_context():
    assert (WINDOW, CONTEXT, HOP) == (200, 50, 150)


def test_sequences_hold_the_residual_and_the_conditioning(seqs):
    s = seqs[0]
    assert s.residual.shape == (1500, 6) and s.cond.shape == (1500, 3) and len(s) == 1500
    assert 0.6 < s.cond[200:-200, 0].min() and s.cond[200:-200, 0].max() < 1.9      # the walk's speed range


def test_the_split_is_stable_and_by_recording(seqs):
    a = assign_splits(seqs, 0.2, 0.2, seed=1)
    assert a == assign_splits(list(reversed(seqs)), 0.2, 0.2, seed=1)             # does not depend on the order
    assert set(a.values()) <= {"train", "val", "test"} and len(a) == 10
    assert a != assign_splits(seqs, 0.2, 0.2, seed=2)
    parts = [Sequence("a", "walk#0", seqs[0].residual, seqs[0].cond), Sequence("a", "walk#1", seqs[0].residual, seqs[0].cond)]
    assert recording_key(parts[0]) == recording_key(parts[1]) == "a/walk"
    split = assign_splits(parts, 0.3, 0.3)
    assert len(split) == 1                                                         # both segments, one decision


def test_holdout_sources_are_test_only(seqs):
    splits = assign_splits(seqs, 0.2, 0.2, holdout_sources=["b"])
    assert {splits[recording_key(s)] for s in seqs if s.source == "b"} == {"test"}
    assert len(select(seqs, splits, "test")) >= 4
    with pytest.raises(ValueError, match="val_frac"):
        assign_splits(seqs, 0.6, 0.5)


def test_stats_normalise_and_survive_json(seqs):
    stats = Stats.fit(seqs)
    z = stats.norm_res(np.concatenate([s.residual for s in seqs]))
    assert np.allclose(z.mean(axis=0), 0, atol=1e-9) and np.allclose(z.std(axis=0), 1, atol=1e-9)
    assert np.allclose(stats.denorm_res(stats.norm_res(seqs[0].residual)), seqs[0].residual)
    lo, hi = stats.speed_range
    assert 0.6 < lo < 1.0 < hi < 1.9
    back = Stats.from_dict(stats.to_dict())
    assert np.allclose(back.res_std, stats.res_std) and back.speed_range == stats.speed_range
    with pytest.raises(ValueError, match="no training sequence"):
        Stats.fit([])


def test_sampler_returns_normalised_float32_windows(seqs):
    batch = WindowSampler(seqs, Stats.fit(seqs)).batch(np.random.default_rng(0), 16)
    assert batch["residual"].shape == (16, WINDOW, 6) and batch["cond"].shape == (16, WINDOW, 3)
    assert batch["residual"].dtype == np.float32 and np.isfinite(batch["residual"]).all()


def test_sources_are_drawn_equally_however_much_data_each_has():
    def constant(source, name, speed):
        return Sequence(source, name, np.zeros((600, 6)), np.tile([speed, 0.0, 0.0], (600, 1)))

    many_slow = [constant("slow", f"s{i}", 1.0) for i in range(8)]
    one_fast = [constant("fast", "f0", 2.0)]
    sampler = WindowSampler(many_slow + one_fast, Stats.fit(many_slow + one_fast))
    speeds = sampler.batch(np.random.default_rng(0), 400)["cond"][:, 0, 0]
    assert 0.4 < np.mean(speeds > 0) < 0.6                  # "fast" is one recording in nine but half of the draws


def test_slow_and_short_material_gives_no_windows():
    stats = Stats.fit(build_sequences([synthetic_walk(0)]))
    still = Sequence("s", "still", np.zeros((400, 6)), np.zeros((400, 3)))
    with pytest.raises(ValueError, match="no window of 200 samples"):
        WindowSampler([still], stats)
    short = Sequence("s", "short", np.zeros((100, 6)), np.ones((100, 3)))
    with pytest.raises(ValueError, match="no window"):
        WindowSampler([short], stats)
```


- [ ] **Step 2: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_dataset.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'backend.motion.dataset'`.


- [ ] **Step 3: Write the implementation**


Create `backend/motion/dataset.py`:

```python
"""Sequences of (residual, conditioning), the sequence-level split, normalisation and window sampling. numpy only."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence as Seq, Tuple

import numpy as np

from backend.motion.decompose import N_COND, N_RESIDUAL, conditioning, split
from backend.motion.tracks import Track

WINDOW = 200     # 4 s at 50 Hz
CONTEXT = 50     # 1 s shared with the previous window
HOP = WINDOW - CONTEXT
MIN_WINDOW_SPEED = 0.2   # m/s: slower windows have no defined heading and are not trained on


@dataclass(frozen=True)
class Sequence:
    source: str
    name: str
    residual: np.ndarray   # (N, 6)
    cond: np.ndarray       # (N, 3) speed, yaw rate, curvature

    def __len__(self) -> int:
        return len(self.residual)


def build_sequences(tracks: Iterable[Track]) -> List[Sequence]:
    out = []
    for tr in tracks:
        path, residual = split(tr.positions, tr.quats)
        out.append(Sequence(tr.source, tr.name, residual, conditioning(path)))
    return out


def recording_key(seq: Sequence) -> str:
    """Segments of one recording ("walk#0", "walk#1") share a key, so they never straddle two splits."""
    return f"{seq.source}/{seq.name.split('#')[0]}"


def assign_splits(seqs: Seq[Sequence], val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0,
                  holdout_sources: Iterable[str] = ()) -> Dict[str, str]:
    """recording key -> "train" | "val" | "test", decided by a hash of (seed, key): stable when sequences are added.
    Every recording of ``holdout_sources`` goes to "test"."""
    if not (0 <= val_frac and 0 <= test_frac and val_frac + test_frac < 1):
        raise ValueError(f"val_frac + test_frac must be in [0, 1), got {val_frac} + {test_frac}")
    holdout = set(holdout_sources)
    out: Dict[str, str] = {}
    for s in seqs:
        key = recording_key(s)
        if s.source in holdout:
            out[key] = "test"
            continue
        u = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0x100000000
        out[key] = "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"
    return out


def select(seqs: Seq[Sequence], splits: Dict[str, str], which: str) -> List[Sequence]:
    return [s for s in seqs if splits[recording_key(s)] == which]


@dataclass(frozen=True)
class Stats:
    """Per-channel normalisation and the speed range seen in training."""

    res_mean: np.ndarray
    res_std: np.ndarray
    cond_mean: np.ndarray
    cond_std: np.ndarray
    speed_range: Tuple[float, float]

    @classmethod
    def fit(cls, seqs: Seq[Sequence]) -> "Stats":
        if not seqs:
            raise ValueError("no training sequence")
        res = np.concatenate([s.residual for s in seqs])
        cond = np.concatenate([s.cond for s in seqs])
        moving = cond[cond[:, 0] > MIN_WINDOW_SPEED, 0]
        lo, hi = np.percentile(moving if len(moving) else cond[:, 0], [0.5, 99.5])
        return cls(res.mean(0), np.maximum(res.std(0), 1e-6), cond.mean(0), np.maximum(cond.std(0), 1e-6),
                   (float(lo), float(hi)))

    def norm_res(self, x):
        return (x - self.res_mean) / self.res_std

    def denorm_res(self, x):
        return x * self.res_std + self.res_mean

    def norm_cond(self, c):
        return (c - self.cond_mean) / self.cond_std

    def to_dict(self) -> dict:
        return {"res_mean": self.res_mean.tolist(), "res_std": self.res_std.tolist(),
                "cond_mean": self.cond_mean.tolist(), "cond_std": self.cond_std.tolist(),
                "speed_range": list(self.speed_range)}

    @classmethod
    def from_dict(cls, d: dict) -> "Stats":
        return cls(np.array(d["res_mean"]), np.array(d["res_std"]), np.array(d["cond_mean"]),
                   np.array(d["cond_std"]), (float(d["speed_range"][0]), float(d["speed_range"][1])))


class WindowSampler:
    """Random normalised windows, sources weighted equally, then recordings by length."""

    def __init__(self, seqs: Seq[Sequence], stats: Stats, hop: int = 25, min_speed: float = MIN_WINDOW_SPEED):
        self.stats = stats
        self._res = [stats.norm_res(s.residual).astype(np.float32) for s in seqs]
        self._cond = [stats.norm_cond(s.cond).astype(np.float32) for s in seqs]
        by_source: Dict[str, List[Tuple[int, int]]] = {}
        for i, s in enumerate(seqs):
            for start in range(0, len(s) - WINDOW + 1, hop):
                if s.cond[start:start + WINDOW, 0].mean() >= min_speed:
                    by_source.setdefault(s.source, []).append((i, start))
        if not by_source:
            raise ValueError(f"no window of {WINDOW} samples with a mean speed >= {min_speed} m/s")
        self.sources = sorted(by_source)
        self._windows = [np.array(by_source[k]) for k in self.sources]
        self.n_windows = sum(len(w) for w in self._windows)

    def batch(self, rng: np.random.Generator, size: int) -> Dict[str, np.ndarray]:
        res = np.empty((size, WINDOW, N_RESIDUAL), np.float32)
        cond = np.empty((size, WINDOW, N_COND), np.float32)
        for b, k in enumerate(rng.integers(len(self.sources), size=size)):
            i, start = self._windows[k][rng.integers(len(self._windows[k]))]
            res[b] = self._res[i][start:start + WINDOW]
            cond[b] = self._cond[i][start:start + WINDOW]
        return {"residual": res, "cond": cond}
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_dataset.py`
Expected: 8 passed.

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/motion tests/motion_helpers.py tests/test_motion_dataset.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/motion/dataset.py tests/motion_helpers.py tests/test_motion_dataset.py
git commit -m "feat(motion): training sequences, split by recording and window sampling"
```

---

### Task 5: The diffusion network and sampler

**Files:**
- Create: `backend/motion/model.py`
- Test: `tests/test_motion_model.py`

**Interfaces:**
- Consumes: `backend.motion.require_torch`, `dataset.CONTEXT`, `decompose.N_RESIDUAL/N_COND`.
- Produces: `model.torch` (the imported module), `model.MotionUNet(base=64, mults=(1,2,4), emb=128)` with `forward(x_t (B,W,6), t (B,) long, cond (B,W,3), ctx (B,W,6), mask (B,W)) -> x0 (B,W,6)` (`base` must be a multiple of 8: GroupNorm(8)); `model.Diffusion(steps=1000)` with `.abar`, `.loss(model, x0, cond, use_ctx (B,) bool, generator=None) -> scalar tensor`, `.sample(model, cond, ctx=None, ctx_len=0, steps=50, generator=None) -> (B,W,6)` (deterministic DDIM; the first `ctx_len` samples of `ctx` are kept as given).

**Domain note.** Per time step the network sees 16 channels: noisy residual (6), conditioning (3), context (6, zeroed where not given) and a context mask (1). The loss is the MSE of the predicted clean residual over the part of the window that is *generated* (everything when no context is given, samples 50..200 otherwise). `generator` is a CPU `torch.Generator`; the noise is drawn on the CPU and moved, so a seed gives the same noise on every device.

- [ ] **Step 1: Write the failing test**


Create `tests/test_motion_model.py`:

```python
"""motion: the diffusion network and sampler (needs torch)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from backend.motion import require_torch
from backend.motion.dataset import CONTEXT, WINDOW

torch = pytest.importorskip("torch")
from backend.motion.model import Diffusion, MotionUNet  # noqa: E402


def tiny():
    return MotionUNet(base=16, mults=(1, 2))


def test_torch_is_required_with_a_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)             # `import torch` now fails
    with pytest.raises(ImportError, match=r"pip install -e \.\[torch\]"):
        require_torch()


def test_network_shapes_and_size():
    assert 2e6 < sum(p.numel() for p in MotionUNet().parameters()) < 6e6          # the default is a few million parameters
    out = tiny()(torch.randn(3, WINDOW, 6), torch.zeros(3, dtype=torch.long), torch.randn(3, WINDOW, 3),
                 torch.randn(3, WINDOW, 6), torch.zeros(3, WINDOW))
    assert out.shape == (3, WINDOW, 6)


def test_noise_schedule_goes_from_clean_to_noise():
    abar = Diffusion(200).abar.numpy()
    assert abar[0] > 0.99 and abar[-1] < 1e-3 and np.all(np.diff(abar) <= 0)


def test_loss_ignores_the_context_part_of_the_window():
    class Zero(torch.nn.Module):
        def forward(self, x_t, t, cond, ctx, mask):
            return torch.zeros_like(x_t)

    x0 = torch.ones(2, WINDOW, 6)
    x0[:, :CONTEXT] = 100.0
    cond = torch.zeros(2, WINDOW, 3)
    diffusion = Diffusion(50)
    assert diffusion.loss(Zero(), x0, cond, torch.tensor([True, True])).item() == pytest.approx(1.0)
    assert diffusion.loss(Zero(), x0, cond, torch.tensor([False, False])).item() > 1000.0


def test_sampling_is_reproducible_and_keeps_the_context():
    model, diffusion = tiny().eval(), Diffusion(100)
    cond, ctx = torch.randn(2, WINDOW, 3), torch.randn(2, WINDOW, 6)

    def sample(seed):
        return diffusion.sample(model, cond, ctx, CONTEXT, steps=4, generator=torch.Generator().manual_seed(seed))

    a = sample(5)
    assert torch.isfinite(a).all() and a.shape == (2, WINDOW, 6)
    assert torch.equal(a, sample(5)) and not torch.equal(a, sample(6))
    assert torch.equal(a[:, :CONTEXT], ctx[:, :CONTEXT])                      # given samples are kept as they are
    assert not torch.equal(a[:, CONTEXT:], ctx[:, CONTEXT:])
    free = diffusion.sample(model, cond, steps=4, generator=torch.Generator().manual_seed(5))   # no context at all
    assert free.shape == a.shape


def test_the_network_can_learn_a_fixed_batch():
    torch.manual_seed(0)
    model, diffusion = tiny(), Diffusion(100)
    x0 = torch.sin(torch.linspace(0, 20, WINDOW))[None, :, None].repeat(4, 1, 6)
    cond, no_ctx = torch.zeros(4, WINDOW, 3), torch.zeros(4, dtype=torch.bool)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(80):
        loss = diffusion.loss(model, x0, cond, no_ctx, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])
```


- [ ] **Step 2: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_model.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'backend.motion.model'`.


- [ ] **Step 3: Write the implementation**


Create `backend/motion/model.py`:

```python
"""Conditional 1D diffusion over residual windows (torch). Imported lazily: see ``require_torch``."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from backend.motion import require_torch
from backend.motion.dataset import CONTEXT
from backend.motion.decompose import N_COND, N_RESIDUAL


torch = require_torch()
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

IN_CH = N_RESIDUAL + N_COND + N_RESIDUAL + 1   # noisy residual, conditioning, context, context mask


def timestep_embedding(t, dim: int):
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t.float()[:, None] * freq[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, 5, padding=2)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, 5, padding=2)
        self.film = nn.Linear(emb, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(emb).chunk(2, dim=1)
        h = self.n2(h) * (1 + scale[:, :, None]) + shift[:, :, None]
        return self.skip(x) + self.c2(F.silu(h))


class MotionUNet(nn.Module):
    """x0-prediction network: (noisy residual, step, conditioning, context) -> clean residual, all (B, W, C)."""

    def __init__(self, base: int = 64, mults=(1, 2, 4), emb: int = 128):
        super().__init__()
        self.emb = emb
        chans = [base * m for m in mults]
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv1d(IN_CH, chans[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = chans[0]
        for i, c in enumerate(chans):
            self.down.append(ResBlock(prev, c, emb))
            self.pool.append(nn.Conv1d(c, c, 3, stride=2, padding=1) if i < len(chans) - 1 else nn.Identity())
            prev = c
        self.mid = nn.ModuleList([ResBlock(prev, prev, emb), ResBlock(prev, prev, emb)])
        self.up = nn.ModuleList()
        for i in reversed(range(len(chans))):
            self.up.append(ResBlock(prev + chans[i], chans[i], emb))
            prev = chans[i]
        self.out = nn.Sequential(nn.GroupNorm(8, prev), nn.SiLU(), nn.Conv1d(prev, N_RESIDUAL, 3, padding=1))

    def forward(self, x_t, t, cond, ctx, mask):
        e = self.t_mlp(timestep_embedding(t, self.emb))
        h = self.inp(torch.cat([x_t, cond, ctx * mask[..., None], mask[..., None]], dim=-1).transpose(1, 2))
        skips = []
        for block, pool in zip(self.down, self.pool):
            h = block(h, e)
            skips.append(h)
            h = pool(h)
        for block in self.mid:
            h = block(h, e)
        for i, block in enumerate(self.up):
            skip = skips[-1 - i]
            if i > 0:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), e)
        return self.out(h).transpose(1, 2)


class Diffusion:
    """Cosine-schedule DDPM training and deterministic DDIM sampling, x0-parameterised."""

    def __init__(self, steps: int = 1000):
        s = 0.008
        x = np.linspace(0, steps, steps + 1)
        f = np.cos((x / steps + s) / (1 + s) * np.pi / 2) ** 2
        abar = np.clip(f[1:] / f[0], 1e-5, 1.0)
        self.steps = steps
        self.abar = torch.tensor(abar, dtype=torch.float32)

    def loss(self, model, x0, cond, use_ctx, generator: Optional["torch.Generator"] = None):
        """MSE on the part of the window the model has to generate. ``use_ctx`` (B,) bool: the first CONTEXT samples
        are given (and then not part of the loss)."""
        b, w, _ = x0.shape
        dev = x0.device
        t = torch.randint(0, self.steps, (b,), device="cpu", generator=generator).to(dev)
        noise = torch.randn(x0.shape, generator=generator).to(dev)
        a = self.abar.to(dev)[t][:, None, None]
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
        mask = (use_ctx[:, None] & (torch.arange(w, device=dev)[None] < CONTEXT)).float()
        pred = model(x_t, t, cond, x0, mask)
        err = ((pred - x0) ** 2).mean(-1) * (1 - mask)
        return err.sum() / (1 - mask).sum()

    @torch.no_grad()
    def sample(self, model, cond, ctx=None, ctx_len: int = 0, steps: int = 50,
               generator: Optional["torch.Generator"] = None):
        """(B, W, 6) residual windows. ``ctx`` (B, W, 6): its first ``ctx_len`` samples are given and kept."""
        b, w, _ = cond.shape
        dev = cond.device
        ctx = torch.zeros(b, w, N_RESIDUAL, device=dev) if ctx is None else ctx
        mask = torch.zeros(b, w, device=dev)
        mask[:, :ctx_len] = 1.0
        x = torch.randn((b, w, N_RESIDUAL), generator=generator).to(dev)
        abar = self.abar.to(dev)
        ts = np.linspace(self.steps - 1, 0, steps).round().astype(int)
        for k, ti in enumerate(ts):
            t = torch.full((b,), int(ti), device=dev, dtype=torch.long)
            x0 = model(x, t, cond, ctx, mask).clamp(-6, 6)
            a = abar[ti]
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            a_prev = abar[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0, device=dev)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        if ctx_len:
            x[:, :ctx_len] = ctx[:, :ctx_len]
        return x
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_model.py`
Expected: 6 passed (a few seconds).

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/motion tests/test_motion_model.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/motion/model.py tests/test_motion_model.py
git commit -m "feat(motion): conditional 1D diffusion model with context for seamless long clips"
```

---

### Task 6: Training and checkpoints

**Files:**
- Create: `backend/motion/train.py`
- Modify: `tests/conftest.py` (the shared `motion_checkpoint` fixture: a tiny model trained in a few seconds)
- Test: `tests/test_motion_train.py`

**Interfaces:**
- Consumes: `dataset.Sequence/Stats/WindowSampler`, `model.Diffusion/MotionUNet/torch`.
- Produces: `train.CHECKPOINT_VERSION = "dvsim-motion-1"`; `train.TrainConfig(steps=20000, batch=64, lr=2e-4, ema=0.999, base=64, mults=(1,2,4), diffusion_steps=1000, ctx_prob=0.5, hop=25, seed=0, val_every=1000, val_batches=16, val_frac=0.1, test_frac=0.1, split_seed=0, holdout_sources=())`; `pick_device(name=None) -> torch.device` (cuda, then mps, then cpu); `save_checkpoint(path, model, cfg, stats, sources, step)`; `load_checkpoint(path, device=None) -> (model in eval mode, TrainConfig, Stats, sources)` (ValueError for anything that is not a `dvsim-motion-1` checkpoint); `train(train_seqs, val_seqs, stats, cfg, out_path, device=None, init=None, log=print) -> [{"step", "val_loss"}]` (saves the EMA weights; `init` continues from a checkpoint of the same architecture). Fixture `motion_checkpoint` (session scope) -> Path of a tiny checkpoint (`base=16, mults=(1,2)`, 120 steps, `test_frac=0.5`).

**Domain note.** The checkpoint is a dict of tensors, numbers and JSON strings so `torch.load(weights_only=True)` can read it: a checkpoint from someone else cannot run code. The split settings are stored so `motion-eval` can recompute the held-out recordings.

- [ ] **Step 1: Write the failing test and the shared fixture**


Apply this change to `tests/conftest.py` (a unified diff against the current files: `git apply` takes it as is, or make the same edits by hand):

```diff
--- a/tests/conftest.py
+++ b/tests/conftest.py
@@ -43,3 +43,20 @@
 def demo_scene_dir(demo_scene):
     """Directory (Path) holding demo_room.obj/.png, demo_objects.glb, demo_scene.json, demo_trajectory.json."""
     return demo_scene.root
+
+
+@pytest.fixture(scope="session")
+def motion_checkpoint(tmp_path_factory):
+    """A tiny trained motion model (a few seconds on the CPU): enough to exercise loading and sampling."""
+    pytest.importorskip("torch")
+    from motion_helpers import synthetic_walk
+
+    from backend.motion.dataset import Stats, build_sequences
+    from backend.motion.train import TrainConfig, train
+
+    seqs = build_sequences([synthetic_walk(i) for i in range(6)])
+    path = tmp_path_factory.mktemp("motion") / "tiny.pt"
+    cfg = TrainConfig(steps=120, batch=8, base=16, mults=(1, 2), diffusion_steps=200, val_every=60, val_batches=2,
+                      val_frac=0.0, test_frac=0.5)
+    train(seqs, [], Stats.fit(seqs), cfg, path, device="cpu", log=lambda message: None)
+    return path
```

Create `tests/test_motion_train.py`:

```python
"""motion: training, checkpoints and what loading one accepts (needs torch)."""
from __future__ import annotations

import json

import pytest
from motion_helpers import synthetic_walk

torch = pytest.importorskip("torch")
from backend.motion.dataset import Stats, build_sequences  # noqa: E402
from backend.motion.train import CHECKPOINT_VERSION, TrainConfig, load_checkpoint, train  # noqa: E402

SEQS = build_sequences([synthetic_walk(i) for i in range(4)])
TINY = dict(batch=8, base=16, mults=(1, 2), diffusion_steps=100, val_batches=1)


def test_the_fixture_checkpoint_holds_only_plain_data(motion_checkpoint):
    ckpt = torch.load(motion_checkpoint, weights_only=True)                  # would raise on anything executable
    assert ckpt["version"] == CHECKPOINT_VERSION and ckpt["step"] == 120
    assert json.loads(ckpt["sources"]) == ["syn"] and json.loads(ckpt["config"])["base"] == 16
    assert all(isinstance(v, torch.Tensor) for v in ckpt["state_dict"].values())


def test_load_gives_an_eval_model_and_its_training_setup(motion_checkpoint):
    model, cfg, stats, sources = load_checkpoint(motion_checkpoint, "cpu")
    assert not model.training and (cfg.base, cfg.mults) == (16, (1, 2)) and sources == ["syn"]
    assert 0.5 < stats.speed_range[0] < stats.speed_range[1] < 2.0


def test_training_reports_the_validation_loss(tmp_path):
    cfg = TrainConfig(steps=20, val_every=10, **TINY)
    lines = []
    history = train(SEQS[:3], SEQS[3:], Stats.fit(SEQS[:3]), cfg, tmp_path / "m.pt", device="cpu", log=lines.append)
    assert [h["step"] for h in history] == [10, 20] and all(h["val_loss"] > 0 for h in history)
    assert len(lines) == 2 and "val" in lines[0] and (tmp_path / "m.pt").is_file()


def test_the_same_seed_trains_the_same_model(tmp_path):
    cfg = TrainConfig(steps=10, val_every=10, **TINY)
    for name in ("a.pt", "b.pt"):
        train(SEQS, [], Stats.fit(SEQS), cfg, tmp_path / name, device="cpu", log=lambda m: None)
    a, b = (torch.load(tmp_path / n, weights_only=True)["state_dict"] for n in ("a.pt", "b.pt"))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_training_can_continue_from_a_checkpoint(motion_checkpoint, tmp_path):
    cfg = TrainConfig(steps=5, val_every=5, **TINY)
    train(SEQS, [], Stats.fit(SEQS), cfg, tmp_path / "next.pt", device="cpu", init=str(motion_checkpoint), log=lambda m: None)
    assert load_checkpoint(tmp_path / "next.pt", "cpu")[1].steps == 5
    other = TrainConfig(steps=5, **{**TINY, "base": 32})
    with pytest.raises(ValueError, match="--init model has base=16"):
        train(SEQS, [], Stats.fit(SEQS), other, tmp_path / "x.pt", device="cpu", init=str(motion_checkpoint))


class _NotATensor:
    pass


def test_bad_files_are_refused_cleanly(tmp_path):
    torch.save({"version": "other"}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="not a dvsim-motion-1 checkpoint"):
        load_checkpoint(tmp_path / "old.pt", "cpu")
    torch.save(_NotATensor(), tmp_path / "pickle.pt")                         # arbitrary objects are never unpickled
    with pytest.raises(ValueError, match="cannot read this file"):
        load_checkpoint(tmp_path / "pickle.pt", "cpu")
    (tmp_path / "junk.pt").write_bytes(b"not a torch file")
    with pytest.raises(ValueError, match="cannot read this file"):
        load_checkpoint(tmp_path / "junk.pt", "cpu")


def test_the_split_settings_are_stored_for_the_evaluation(tmp_path):
    cfg = TrainConfig(steps=2, val_every=2, val_frac=0.2, test_frac=0.3, split_seed=7, holdout_sources=("a", "b"), **TINY)
    train(SEQS, [], Stats.fit(SEQS), cfg, tmp_path / "m.pt", device="cpu", log=lambda m: None)
    loaded = load_checkpoint(tmp_path / "m.pt", "cpu")[1]
    assert (loaded.val_frac, loaded.test_frac, loaded.split_seed, loaded.holdout_sources) == (0.2, 0.3, 7, ("a", "b"))
```


- [ ] **Step 2: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_train.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'backend.motion.train'`.


- [ ] **Step 3: Write the implementation**


Create `backend/motion/train.py`:

```python
"""Training loop and checkpoints. Checkpoints hold tensors, numbers and strings only, so ``torch.load`` can run with
``weights_only=True`` (a checkpoint from someone else cannot execute code)."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence as Seq

import numpy as np

from backend.motion.dataset import Sequence, Stats, WindowSampler
from backend.motion.model import Diffusion, MotionUNet, torch

CHECKPOINT_VERSION = "dvsim-motion-1"


@dataclass
class TrainConfig:
    steps: int = 20000
    batch: int = 64
    lr: float = 2e-4
    ema: float = 0.999
    base: int = 64
    mults: tuple = (1, 2, 4)
    diffusion_steps: int = 1000
    ctx_prob: float = 0.5
    hop: int = 25
    seed: int = 0
    val_every: int = 1000
    val_batches: int = 16
    # how the corpus was split (``motion-eval`` recomputes the held-out recordings from these)
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_seed: int = 0
    holdout_sources: tuple = ()


def pick_device(name: Optional[str] = None) -> "torch.device":
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ema_update(ema, model, decay: float) -> None:
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(pm.detach(), alpha=1.0 - decay)


def _batch(sampler: WindowSampler, rng, size: int, device, ctx_prob: float):
    b = sampler.batch(rng, size)
    x0 = torch.from_numpy(b["residual"]).to(device)
    cond = torch.from_numpy(b["cond"]).to(device)
    use_ctx = torch.from_numpy(rng.random(size) < ctx_prob).to(device)
    return x0, cond, use_ctx


def save_checkpoint(path, model, cfg: TrainConfig, stats: Stats, sources: Seq[str], step: int) -> None:
    torch.save({
        "version": CHECKPOINT_VERSION,
        "config": json.dumps(asdict(cfg)),
        "stats": json.dumps(stats.to_dict()),
        "sources": json.dumps(sorted(set(sources))),
        "step": int(step),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }, path)


def load_checkpoint(path, device=None):
    """(model in eval mode, TrainConfig, Stats, sources)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # unpickling errors, truncated files, not a torch file
        raise ValueError(f"{path}: cannot read this file as a motion checkpoint ({exc})") from None
    if not isinstance(ckpt, dict) or ckpt.get("version") != CHECKPOINT_VERSION:
        found = ckpt.get("version") if isinstance(ckpt, dict) else type(ckpt).__name__
        raise ValueError(f"{path}: not a {CHECKPOINT_VERSION} checkpoint (found {found!r})")
    raw = json.loads(ckpt["config"])
    cfg = TrainConfig(**{**raw, "mults": tuple(raw["mults"]), "holdout_sources": tuple(raw.get("holdout_sources", ()))})
    model = MotionUNet(cfg.base, cfg.mults)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device or pick_device()).eval()
    return model, cfg, Stats.from_dict(json.loads(ckpt["stats"])), json.loads(ckpt["sources"])


def train(train_seqs: Seq[Sequence], val_seqs: Seq[Sequence], stats: Stats, cfg: TrainConfig, out_path,
          device=None, init: Optional[str] = None, log: Callable[[str], None] = print) -> List[Dict[str, float]]:
    """Train and save the EMA weights to ``out_path``. Returns the validation history [{step, val_loss}]."""
    device = pick_device(device) if not isinstance(device, torch.device) else device
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    sampler = WindowSampler(train_seqs, stats, hop=cfg.hop)
    val_sampler = WindowSampler(val_seqs, stats, hop=cfg.hop) if val_seqs else None
    model = MotionUNet(cfg.base, cfg.mults).to(device)
    if init:
        prev, prev_cfg, _, _ = load_checkpoint(init, device)
        if (prev_cfg.base, prev_cfg.mults) != (cfg.base, cfg.mults):
            raise ValueError(f"--init model has base={prev_cfg.base} mults={prev_cfg.mults}, "
                             f"expected {cfg.base} {cfg.mults}")
        model.load_state_dict(prev.state_dict())
    ema = copy.deepcopy(model).eval()
    diff = Diffusion(cfg.diffusion_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    history: List[Dict[str, float]] = []
    run = 0.0
    for step in range(1, cfg.steps + 1):
        x0, cond, use_ctx = _batch(sampler, rng, cfg.batch, device, cfg.ctx_prob)
        loss = diff.loss(model, x0, cond, use_ctx, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        _ema_update(ema, model, min(cfg.ema, (1 + step) / (10 + step)))
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % cfg.val_every == 0 or step == cfg.steps:
            msg = f"step {step}/{cfg.steps} train {run:.4f}"
            if val_sampler is not None:
                vrng, vgen = np.random.default_rng(1), torch.Generator().manual_seed(1)
                with torch.no_grad():
                    v = np.mean([diff.loss(ema, *_batch(val_sampler, vrng, cfg.batch, device, 0.5), vgen).item()
                                 for _ in range(cfg.val_batches)])
                history.append({"step": step, "val_loss": float(v)})
                msg += f" val {v:.4f}"
            log(msg)
    save_checkpoint(out_path, ema, cfg, stats, [s.source for s in train_seqs], cfg.steps)
    return history
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_train.py`
Expected: 7 passed (the fixture trains once, ~2 s).

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/motion tests/test_motion_train.py tests/conftest.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/motion/train.py tests/conftest.py tests/test_motion_train.py
git commit -m "feat(motion): training loop and safe checkpoints"
```

---

### Task 7: Turning a planned path into a human walk

**Files:**
- Create: `backend/motion/style.py`
- Test: `tests/test_motion_style.py`

**Interfaces:**
- Consumes: `decompose.extract_path/conditioning/compose`, `model.Diffusion/torch`, `train.load_checkpoint/pick_device`, `dataset.CONTEXT/HOP/WINDOW`, `tracks.DT_US`, `trajectories.WaypointTrajectory`.
- Produces: `style.DiffusionMotionStyle(model, cfg, stats, device, sample_steps=50)` with attributes `.model .cfg .stats .device .diffusion .sample_steps`; `DiffusionMotionStyle.load(path, device=None, sample_steps=50)`; `.apply(traj: WaypointTrajectory, rng=None) -> WaypointTrajectory` (same start; sampled at 50 Hz, so `floor(duration/20 ms) + 1` waypoints; `rng` is a seed or `np.random.Generator`; ValueError when shorter than 5 samples; UserWarning "...outside the ... m/s the model was trained on: clamped" for speeds outside the trained range).

**Domain note.** Window `i` starts at sample `150 i`. Window 0 is generated from nothing; window `i > 0` is given the previous output's last 50 samples as context and generates the other 150, so the windows join without seams. The planner's own orientations are discarded: only its positions define the path.

- [ ] **Step 1: Write the failing test**


Create `tests/test_motion_style.py`:

```python
"""motion: turning a planned path into a human walk (needs torch)."""
from __future__ import annotations

import numpy as np
import pytest
from motion_helpers import planner_line

torch = pytest.importorskip("torch")
from backend.motion.dataset import CONTEXT, WINDOW  # noqa: E402
from backend.motion.style import DiffusionMotionStyle  # noqa: E402
from backend.motion.tracks import DT_US  # noqa: E402


@pytest.fixture(scope="module")
def style(motion_checkpoint):
    return DiffusionMotionStyle.load(motion_checkpoint, device="cpu", sample_steps=6)


def test_a_seed_gives_the_same_walk_and_another_seed_a_different_one(style):
    line = planner_line(6)
    a, b, c = style.apply(line, rng=3), style.apply(line, rng=3), style.apply(line, rng=4)
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.quats, b.quats)
    assert not np.allclose(a.positions, c.positions)


def test_the_walk_keeps_the_time_base_and_stays_near_the_planned_path(style):
    line = planner_line(7.03, start_us=5_000_000)
    walk = style.apply(line, rng=0)
    assert walk.start_us == 5_000_000 and 0 <= line.end_us - walk.end_us < DT_US
    assert np.all(np.diff(walk.times_us) == DT_US)
    planned = line.poses_at(walk.times_us)[:, :3]
    assert np.abs(walk.positions - planned).max() < 0.5                      # a person's wobble, not a detour
    assert np.std(walk.quats, axis=0).max() > 1e-3                            # the orientation moves: it is not the planner's


def test_each_window_is_given_the_last_second_of_the_previous_one(style, monkeypatch):
    calls = []
    sample = style.diffusion.sample

    def spy(model, cond, ctx=None, ctx_len=0, steps=50, generator=None):
        out = sample(model, cond, ctx, ctx_len, steps, generator)
        calls.append((None if ctx is None else ctx[0, :ctx_len].clone(), ctx_len, out[0].clone()))
        return out

    monkeypatch.setattr(style.diffusion, "sample", spy)
    walk = style.apply(planner_line(14), rng=1)                               # 701 samples: windows start at 0, 150, ... 600
    assert len(walk) == 701 and len(calls) == 5
    assert calls[0][0] is None and calls[0][1] == 0                           # the first window starts from nothing
    for (_, _, previous), (ctx, ctx_len, out) in zip(calls, calls[1:]):
        assert ctx_len == CONTEXT
        assert torch.equal(ctx, previous[WINDOW - CONTEXT:])                   # its last second ...
        assert torch.equal(out[:CONTEXT], ctx)                                 # ... is the first second of the next


def test_a_short_path_takes_one_window_and_a_tiny_one_is_refused(style):
    assert len(style.apply(planner_line(1.0), rng=0)) == 51
    with pytest.raises(ValueError, match="too short"):
        style.apply(planner_line(0.05), rng=0)


def test_a_speed_the_model_never_saw_is_clamped_with_a_warning(style):
    with pytest.warns(UserWarning, match="outside the .* m/s the model was trained on"):
        walk = style.apply(planner_line(4, speed=6.0), rng=0)
    assert np.isfinite(walk.positions).all()


def test_no_warning_inside_the_trained_range(style, recwarn):
    style.apply(planner_line(4, speed=1.3), rng=0)
    assert not [w for w in recwarn if "trained on" in str(w.message)]
```


- [ ] **Step 2: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_style.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'backend.motion.style'`.


- [ ] **Step 3: Write the implementation**


Create `backend/motion/style.py`:

```python
"""DiffusionMotionStyle: planned path -> the same path walked by a person."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from backend.motion.dataset import CONTEXT, HOP, WINDOW
from backend.motion.decompose import compose, conditioning, extract_path
from backend.motion.model import Diffusion, torch
from backend.motion.tracks import DT_US
from backend.motion.train import load_checkpoint, pick_device
from backend.trajectories.trajectory import WaypointTrajectory


class DiffusionMotionStyle:
    def __init__(self, model, cfg, stats, device, sample_steps: int = 50):
        self.model, self.cfg, self.stats, self.device = model, cfg, stats, device
        self.diffusion = Diffusion(cfg.diffusion_steps)
        self.sample_steps = sample_steps

    @classmethod
    def load(cls, path, device: Optional[str] = None, sample_steps: int = 50) -> "DiffusionMotionStyle":
        dev = pick_device(device)
        model, cfg, stats, _ = load_checkpoint(path, dev)
        return cls(model, cfg, stats, dev, sample_steps)

    def apply(self, traj: WaypointTrajectory, rng=None) -> WaypointTrajectory:
        """``traj``'s path (its positions) walked with sampled human motion; same start, end and number of seconds.
        The planner's own orientations are replaced. A seed (or Generator) makes the result reproducible."""
        rng = np.random.default_rng(rng)
        n = int(traj.duration_us // DT_US) + 1
        if n < 5:
            raise ValueError(f"the trajectory is too short for motion synthesis ({traj.duration_us} us)")
        times = traj.start_us + np.arange(n, dtype=np.int64) * DT_US
        path = extract_path(traj.poses_at(times)[:, :3])
        cond = conditioning(path)
        lo, hi = self.stats.speed_range
        if cond[:, 0].min() < lo or cond[:, 0].max() > hi:
            warnings.warn(f"path speed {cond[:, 0].min():.2f}-{cond[:, 0].max():.2f} m/s is outside the "
                          f"{lo:.2f}-{hi:.2f} m/s the model was trained on: clamped", stacklevel=2)
            cond = cond.copy()
            cond[:, 0] = np.clip(cond[:, 0], lo, hi)
        cond_n = self.stats.norm_cond(cond).astype(np.float32)

        k = 1 if n <= WINDOW else int(np.ceil((n - WINDOW) / HOP)) + 1
        total = (k - 1) * HOP + WINDOW
        cond_p = np.concatenate([cond_n, np.repeat(cond_n[-1:], total - n, axis=0)]) if total > n else cond_n
        gen = torch.Generator().manual_seed(int(rng.integers(2 ** 62)))
        res = np.zeros((total, 6), np.float32)
        for i in range(k):
            s = i * HOP
            c = torch.from_numpy(cond_p[s:s + WINDOW])[None].to(self.device)
            if i == 0:
                out = self.diffusion.sample(self.model, c, steps=self.sample_steps, generator=gen)
            else:
                ctx = torch.from_numpy(res[s:s + WINDOW].copy())[None].to(self.device)  # first CONTEXT = previous tail
                out = self.diffusion.sample(self.model, c, ctx, CONTEXT, self.sample_steps, gen)
            res[s:s + WINDOW] = out[0].cpu().numpy()
        positions, quats = compose(path, self.stats.denorm_res(res[:n].astype(np.float64)))
        return WaypointTrajectory(times, positions, quats)
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_style.py`
Expected: 6 passed.

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/motion tests/test_motion_style.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/motion/style.py tests/test_motion_style.py
git commit -m "feat(motion): replay a planned path with sampled human motion"
```

---

### Task 8: The `motion` field of street trajectory specs

**Files:**
- Modify: `backend/pipeline.py` (`_build_one`, new `_motion_style`)
- Modify: `tests/test_trajectories.py` (two tests appended)

**Interfaces:**
- Consumes: `registry.resolve_model`, `style.DiffusionMotionStyle.load/apply`.
- Produces: `build_trajectories({"type": "street", ..., "motion": {"model": id} | {"path": file}}, scene, cache)`; the loaded model is kept in `cache` under `("motion", resolved path)`; the same spec and seed always give the same walks (the planner draws first, then the style, from one `Generator`).

- [ ] **Step 1: Write the failing tests**


Append the two tests (the whole diff of `tests/test_trajectories.py` below is just those two tests):


Apply this change to `tests/test_trajectories.py` (a unified diff against the current files: `git apply` takes it as is, or make the same edits by hand):

```diff
--- a/tests/test_trajectories.py
+++ b/tests/test_trajectories.py
@@ -778,3 +778,45 @@
         assert np.allclose(tr.positions[:, 1], 1.2, atol=0.16)
     with pytest.raises(ValueError, match="need a scene"):
         build_trajectories({"type": "street", "duration_s": 1.0})
+
+
+# ---- street walks with learned human motion -------------------------------------------------------------------------
+def test_street_spec_with_motion_replaces_the_glide(motion_checkpoint):
+    from backend.pipeline import build_trajectories
+
+    class SceneLike:
+        def to_trimesh(self):
+            return make_town()
+
+    spec = {"type": "street", "duration_s": 4.0, "count": 2, "seed": 3, "speed": 1.3, "eye_height": 1.6}
+    plain = build_trajectories(spec, SceneLike())
+    cache = {}
+    human = build_trajectories({**spec, "motion": {"path": str(motion_checkpoint)}}, SceneLike(), cache)
+    again = build_trajectories({**spec, "motion": {"path": str(motion_checkpoint)}}, SceneLike(), cache)
+    assert len(human) == 2 and sum(k[0] == "motion" for k in cache if isinstance(k, tuple)) == 1   # loaded once
+    for p, h, h2 in zip(plain, human, again):
+        assert h.start_us == p.start_us and abs(h.end_us - p.end_us) < 20_000
+        assert len(h) > len(p) and np.array_equal(h.positions, h2.positions)       # sampled at 50 Hz, reproducible
+        assert not np.allclose(h.quats[:, 2], p.quats[0, 2])                       # it rolls: the glide did not
+
+
+def test_motion_errors_are_specific(tmp_path, monkeypatch):
+    import sys
+
+    from backend.pipeline import build_trajectories
+
+    class SceneLike:
+        def to_trimesh(self):
+            return make_town()
+
+    spec = {"type": "street", "duration_s": 2.0, "seed": 1}
+    with pytest.raises(FileNotFoundError, match="motion model file not found"):
+        build_trajectories({**spec, "motion": {"path": str(tmp_path / "gone.pt")}}, SceneLike())
+    with pytest.raises(ValueError, match="invalid motion model id"):
+        build_trajectories({**spec, "motion": {"model": "../x"}}, SceneLike())
+    (tmp_path / "m.pt").write_bytes(b"x")
+    for name in ("backend.motion.style", "backend.motion.train", "backend.motion.model"):
+        monkeypatch.delitem(sys.modules, name, raising=False)
+    monkeypatch.setitem(sys.modules, "torch", None)                                   # torch is not installed
+    with pytest.raises(ImportError, match=r"pip install -e \.\[torch\]"):
+        build_trajectories({**spec, "motion": {"path": str(tmp_path / "m.pt")}}, SceneLike())
```


- [ ] **Step 2: Run them to verify they fail**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_trajectories.py -k motion`
Expected: FAIL: the `motion` key is ignored, so the first test's assertions fail (and the error test raises nothing).


- [ ] **Step 3: Write the implementation**


Apply this change to `backend/pipeline.py` (a unified diff against the current files: `git apply` takes it as is, or make the same edits by hand):

```diff
--- a/backend/pipeline.py
+++ b/backend/pipeline.py
@@ -40,7 +40,9 @@
        "min_clearance": float}
       {"type": "street", "duration_s": float, "count": int, "seed": int | None, "speed": float (m/s, 1.4),
        "eye_height": float (m, 1.6), "min_clearance": float (m to obstacles, 1.0),
-       "edge_margin": float (m to the scene edge, 5.0)}  -- eye-height walks, +Y up
+       "edge_margin": float (m to the scene edge, 5.0),
+       "motion": {"model": id} | {"path": file} | None}  -- eye-height walks, +Y up; ``motion`` replaces the planner's
+       glide by sampled human head motion (a trained model, see backend.motion; needs torch)
     Generated types need ``scene`` (their free space is computed from its geometry, once per call). ``cache``: a dict
     kept by the caller to reuse that free space across calls on the same scene (generation is deterministic given the
     seed, so cached and fresh results are identical).
@@ -93,10 +95,28 @@
             gen = cache[key] = StreetTrajectoryGenerator(scene.to_trimesh(), **params)
         rng = np.random.default_rng(spec.get("seed"))
         duration_us = int(round(float(spec["duration_s"]) * 1e6))
-        return gen.generate_many(int(spec.get("count", 1)), duration_us, rng=rng)
+        trajs = gen.generate_many(int(spec.get("count", 1)), duration_us, rng=rng)
+        if spec.get("motion"):
+            style = _motion_style(spec["motion"], cache)
+            trajs = [style.apply(t, rng=rng) for t in trajs]
+        return trajs
     raise ValueError(f"unknown trajectory type: {kind!r}")
 
 
+def _motion_style(motion: Dict[str, Any], cache: Dict[Any, Any]):
+    """The loaded motion model a ``motion`` spec names (kept in ``cache``: loading is the slow part)."""
+    from backend.motion.registry import resolve_model
+
+    path = resolve_model(motion)
+    key = ("motion", str(path.resolve()))
+    style = cache.get(key)
+    if style is None:
+        from backend.motion.style import DiffusionMotionStyle  # imports torch: fails with an install hint without it
+
+        style = cache[key] = DiffusionMotionStyle.load(path)
+    return style
+
+
 def run_to_disk(
     scene: Union[Scene, str, Path, Dict[str, Any]],
     trajectories: Union[Trajectory, Dict[str, Any], Sequence[Any]],
```


- [ ] **Step 4: Run the tests**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_trajectories.py`
Expected: 58 passed (the street / motion tests included).


- [ ] **Step 5: Commit**

```bash
git add backend/pipeline.py tests/test_trajectories.py
git commit -m "feat(motion): street trajectory specs can name a human-motion model"
```

---

### Task 9: Realism metrics

**Files:**
- Create: `backend/motion/evaluate.py`
- Test: `tests/test_motion_evaluate.py`

**Interfaces:**
- Consumes: `dataset.Sequence/assign_splits/build_sequences/recording_key`, `decompose.extract_path/split`, `style.DiffusionMotionStyle` (for `evaluate_model`; any object with `.apply(traj, rng)` and `.cfg` works), `trajectories.WaypointTrajectory`.
- Produces: `evaluate.windows(residual, speed, hop=100)`, `log_psd(w)`, `log_spectral_distance(a, b)`, `step_frequency(w)`, `motion_features(w)`, `classifier_accuracy(a, b, folds=5, seed=0)`, `compare(real: [Sequence], generated: [residual arrays]) -> [bin dicts]`, `evaluate_model(style, tracks, max_sequences=0, seed=0) -> [bin dicts]`, `format_report(report) -> str`. A bin dict has `speed` (lo, hi m/s), `windows`, `spectral_distance`/`*_w1` dicts with `generated`, `plain` (no motion) and `real_other` (noise floor) entries, `step_hz`, `classifier_accuracy`.

**Domain note.** `evaluate_model` replays each held-out real recording along its own smooth path (so conditions match) and lets the model walk it. Reading a report: `generated` should be close to `real_other` and far below `plain`; `classifier_accuracy` near 0.5 means a classifier cannot tell generated from real.

- [ ] **Step 1: Write the failing test**


Create `tests/test_motion_evaluate.py`:

```python
"""motion: the realism metrics tell a right process from a wrong one."""
from __future__ import annotations

import numpy as np
import pytest

from backend.motion.dataset import Sequence
from backend.motion.evaluate import classifier_accuracy, compare, format_report, step_frequency, windows


def sequence(rng, name, freq=3.6, amp=0.02, n=1500):
    t = np.arange(n) / 50.0
    residual = np.zeros((n, 6))
    residual[:, 1] = amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 6)) + 0.002 * rng.standard_normal(n)
    residual[:, 3:] = 0.01 * rng.standard_normal((n, 3))
    cond = np.zeros((n, 3))
    cond[:, 0] = 1.4
    return Sequence("s", name, residual, cond)


@pytest.fixture(scope="module")
def runs():
    rng = np.random.default_rng(0)
    real = [sequence(rng, f"r{i}") for i in range(6)]
    same_process = [sequence(rng, f"g{i}").residual for i in range(6)]
    other_process = [sequence(rng, f"b{i}", freq=2.0, amp=0.05).residual for i in range(6)]
    return real, same_process, other_process


def test_the_right_process_is_close_and_far_better_than_no_motion(runs):
    real, good, _ = runs
    (b,) = compare(real, good)
    assert b["speed"] == (1.2, 1.6) and b["windows"] == 84
    assert b["spectral_distance"]["generated"] < 0.1 and b["spectral_distance"]["plain"] > 3.0
    assert b["spectral_distance"]["generated"] < 3 * b["spectral_distance"]["real_other"] + 0.05
    assert b["step_hz"]["real"] == pytest.approx(3.5, abs=0.25) and b["step_hz"]["generated"] == pytest.approx(3.5, abs=0.25)
    assert b["classifier_accuracy"] < 0.8


def test_the_wrong_process_is_found_out(runs):
    real, _, bad = runs
    (b,) = compare(real, bad)
    assert b["spectral_distance"]["generated"] > 0.3
    assert b["step_hz"]["generated"] == pytest.approx(2.0, abs=0.3)
    assert b["acceleration_w1"]["generated"] > 3 * b["acceleration_w1"]["real_other"]
    assert b["classifier_accuracy"] > 0.9


def test_report_text(runs):
    real, good, _ = runs
    text = format_report(compare(real, good))
    for part in ("speed 1.2-1.6 m/s (84 windows)", "spectral distance", "step frequency", "jerk", "classifier accuracy"):
        assert part in text


def test_windows_and_helpers():
    w, speed = windows(np.zeros((500, 6)), np.full(500, 1.3))
    assert w.shape == (4, 200, 6) and speed == pytest.approx([1.3] * 4)       # starts 0, 100, 200, 300
    assert windows(np.zeros((150, 6)), np.ones(150))[0].shape == (0, 200, 6)
    assert np.isnan(classifier_accuracy(np.zeros((2, 200, 6)), np.zeros((2, 200, 6))))
    bob = np.zeros((1, 200, 6))
    bob[0, :, 1] = np.sin(2 * np.pi * 3.0 * np.arange(200) / 50)
    assert step_frequency(bob) == pytest.approx(3.0, abs=0.4)


def test_inputs_are_checked(runs):
    real, good, _ = runs
    with pytest.raises(ValueError, match="6 real sequences but 5 generated"):
        compare(real, good[:5])
    short = Sequence("s", "x", np.zeros((100, 6)), np.ones((100, 3)))
    with pytest.raises(ValueError, match="long enough"):
        compare([short], [np.zeros((100, 6))])


def test_held_out_recordings_are_replayed_and_compared(motion_checkpoint):
    from motion_helpers import synthetic_walk

    from backend.motion.evaluate import evaluate_model
    from backend.motion.style import DiffusionMotionStyle

    style = DiffusionMotionStyle.load(motion_checkpoint, device="cpu", sample_steps=6)
    tracks = [synthetic_walk(i) for i in range(10, 18)]            # walks 10, 11, 12 and 17 are held out (test_frac 0.5)
    report = evaluate_model(style, tracks, max_sequences=2, seed=0)
    assert report and all(b["spectral_distance"]["generated"] < b["spectral_distance"]["plain"] for b in report)
    with pytest.raises(ValueError, match="no held-out recording"):
        evaluate_model(style, [synthetic_walk(i) for i in range(13, 17)])
```


- [ ] **Step 2: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_evaluate.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'backend.motion.evaluate'`.


- [ ] **Step 3: Write the implementation**


Create `backend/motion/evaluate.py`:

```python
"""Realism metrics: generated vs held-out real residual motion, per speed bin. numpy / scipy only (no torch)."""
from __future__ import annotations

from typing import Dict, List, Sequence as Seq, Tuple

import numpy as np
from scipy import signal
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

from backend.motion.dataset import Sequence, assign_splits, build_sequences, recording_key
from backend.motion.decompose import extract_path, split
from backend.motion.tracks import RATE_HZ, Track
from backend.trajectories.trajectory import WaypointTrajectory

WIN = 200
SPEED_BINS = ((0.2, 0.8), (0.8, 1.2), (1.2, 1.6), (1.6, 3.0))
_NPERSEG = 64
_FLOOR = 1e-12


def windows(residual: np.ndarray, speed: np.ndarray, hop: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """((n, WIN, 6) residual windows, (n,) mean path speed of each)."""
    starts = range(0, len(residual) - WIN + 1, hop)
    if not len(starts):
        return np.zeros((0, WIN, residual.shape[1])), np.zeros(0)
    return (np.stack([residual[s:s + WIN] for s in starts]),
            np.array([speed[s:s + WIN].mean() for s in starts]))


def log_psd(w: np.ndarray) -> np.ndarray:
    """(n, F, 6) log10 power spectral density of each window and channel."""
    _, p = signal.welch(w - w.mean(axis=1, keepdims=True), fs=RATE_HZ, nperseg=_NPERSEG, axis=1)
    return np.log10(p + _FLOOR)


def log_spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference (log10 units) between the mean log spectra of two window sets, frequencies 0.3-15 Hz."""
    f = np.fft.rfftfreq(_NPERSEG, 1.0 / RATE_HZ)
    keep = (f >= 0.3) & (f <= 15.0)
    pa, pb = log_psd(a).mean(axis=0)[keep], log_psd(b).mean(axis=0)[keep]
    return float(np.sqrt(np.mean((pa - pb) ** 2)))


def step_frequency(w: np.ndarray) -> float:
    """Frequency (Hz) of the strongest vertical-offset component between 0.8 and 5 Hz (the bob)."""
    f, p = signal.welch(w[:, :, 1] - w[:, :, 1].mean(axis=1, keepdims=True), fs=RATE_HZ, nperseg=128, axis=1)
    band = (f >= 0.8) & (f <= 5.0)
    return float(f[band][np.argmax(p.mean(axis=0)[band])])


def motion_features(w: np.ndarray) -> Dict[str, np.ndarray]:
    """Pooled samples of angular speed (rad/s), linear acceleration (m/s2) and jerk (m/s3) of the residual."""
    pos, rot = w[:, :, :3], w[:, :, 3:]
    return {
        "angular_speed": np.linalg.norm(np.diff(rot, axis=1), axis=2).ravel() * RATE_HZ,
        "acceleration": np.linalg.norm(np.diff(pos, 2, axis=1), axis=2).ravel() * RATE_HZ ** 2,
        "jerk": np.linalg.norm(np.diff(pos, 3, axis=1), axis=2).ravel() * RATE_HZ ** 3,
    }


def _classifier_features(w: np.ndarray) -> np.ndarray:
    return np.concatenate([log_psd(w).reshape(len(w), -1), np.log(w.std(axis=1) + _FLOOR)], axis=1)


def _logistic_loss(wb: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    z = x @ wb[:-1] + wb[-1]
    return float(np.mean(np.logaddexp(0, z) - y * z) + 1e-2 * wb[:-1] @ wb[:-1])


def classifier_accuracy(a: np.ndarray, b: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Cross-validated accuracy of a logistic regression telling window set ``a`` from ``b`` by their spectra and
    scale. 0.5 = indistinguishable; NaN when there are fewer than ``folds`` windows in a set."""
    if min(len(a), len(b)) < folds:
        return float("nan")
    x = np.concatenate([_classifier_features(a), _classifier_features(b)])
    y = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    order = np.random.default_rng(seed).permutation(len(y))
    correct = 0
    for k in range(folds):
        te = order[k::folds]
        tr = np.setdiff1d(order, te)
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        xt, xe = (x[tr] - mu) / sd, (x[te] - mu) / sd

        wb = minimize(_logistic_loss, np.zeros(x.shape[1] + 1), args=(xt, y[tr]), method="L-BFGS-B",
                      options={"maxiter": 200}).x
        correct += int(np.sum((xe @ wb[:-1] + wb[-1] > 0) == (y[te] > 0.5)))
    return correct / len(y)


def compare(real: Seq[Sequence], generated: Seq[np.ndarray]) -> List[dict]:
    """Per speed bin: metrics of the generated residuals (one array per real sequence, same conditions) against
    the real ones, next to two references: ``plain`` (no motion at all, what ``street`` gives) and ``real_other``
    (even-numbered against odd-numbered real recordings: the noise floor, the best that can be expected)."""
    if len(real) != len(generated):
        raise ValueError(f"{len(real)} real sequences but {len(generated)} generated")
    w_real, w_gen, speed, parity = [], [], [], []
    for i, (seq, gen) in enumerate(zip(real, generated)):
        w, sp = windows(seq.residual, seq.cond[:, 0])
        g, _ = windows(np.asarray(gen), seq.cond[:, 0])
        w_real.append(w)
        w_gen.append(g)
        speed.append(sp)
        parity.append(np.full(len(w), i % 2))
    r_all, g_all = np.concatenate(w_real), np.concatenate(w_gen)
    speed, parity = np.concatenate(speed), np.concatenate(parity)
    if len(r_all) == 0:
        raise ValueError("no real sequence is long enough for a 4 s window")
    report = []
    for lo, hi in SPEED_BINS:
        sel = (speed >= lo) & (speed < hi)
        if sel.sum() < 2:
            continue
        r, g = r_all[sel], g_all[sel]
        even, odd = r_all[sel & (parity == 0)], r_all[sel & (parity == 1)]
        has_ref = len(even) >= 1 and len(odd) >= 1
        fr, fg = motion_features(r), motion_features(g)
        fe, fo = (motion_features(even), motion_features(odd)) if has_ref else (None, None)
        nan = float("nan")
        report.append({
            "speed": (lo, hi), "windows": int(sel.sum()),
            "spectral_distance": {"generated": log_spectral_distance(g, r),
                                  "plain": log_spectral_distance(np.zeros_like(r), r),
                                  "real_other": log_spectral_distance(even, odd) if has_ref else nan},
            "step_hz": {"real": step_frequency(r), "generated": step_frequency(g)},
            **{f"{name}_w1": {"generated": float(wasserstein_distance(fr[name], fg[name])),
                              "real_other": float(wasserstein_distance(fe[name], fo[name])) if has_ref else nan}
               for name in fr},
            "classifier_accuracy": classifier_accuracy(r, g),
        })
    return report


def evaluate_model(style, tracks: Seq[Track], max_sequences: int = 0, seed: int = 0) -> List[dict]:
    """``compare`` on the recordings ``style``'s model never trained on (its own split settings, recomputed from the
    corpus ``tracks``): each is replayed along its own smooth path and walked by the model."""
    cfg = style.cfg
    seqs = build_sequences(tracks)
    splits = assign_splits(seqs, cfg.val_frac, cfg.test_frac, cfg.split_seed, cfg.holdout_sources)
    held_out = [i for i, s in enumerate(seqs) if splits[recording_key(s)] == "test"]
    if not held_out:
        raise ValueError("no held-out recording in this corpus: raise --test-frac when training, or add data")
    if max_sequences:
        held_out = held_out[:max_sequences]
    generated = []
    for i in held_out:
        track = tracks[i]
        path = extract_path(track.positions)
        planned = WaypointTrajectory(track.times_us, path.positions, track.quats)
        walk = style.apply(planned, rng=seed + i)
        generated.append(split(walk.positions, walk.quats)[1])
    return compare([seqs[i] for i in held_out], generated)


def format_report(report: List[dict]) -> str:
    lines = []
    for b in report:
        lo, hi = b["speed"]
        sd, st = b["spectral_distance"], b["step_hz"]
        lines.append(f"speed {lo:.1f}-{hi:.1f} m/s ({b['windows']} windows)")
        lines.append(f"  spectral distance  generated {sd['generated']:.2f}  plain {sd['plain']:.2f}  "
                     f"real-other {sd['real_other']:.2f}")
        lines.append(f"  step frequency     real {st['real']:.2f} Hz  generated {st['generated']:.2f} Hz")
        for name in ("angular_speed", "acceleration", "jerk"):
            w = b[f"{name}_w1"]
            lines.append(f"  {name:<18} W1 generated {w['generated']:.3g}  real-other {w['real_other']:.3g}")
        lines.append(f"  classifier accuracy {b['classifier_accuracy']:.2f} (0.50 = indistinguishable)")
    return "\n".join(lines)
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_evaluate.py`
Expected: 6 passed.

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/motion tests/test_motion_evaluate.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/motion/evaluate.py tests/test_motion_evaluate.py
git commit -m "feat(motion): realism metrics against held-out recordings"
```

---

### Task 10: Event statistics of plain and human walks

**Files:**
- Create: `backend/motion/events_report.py`
- Test: `tests/test_motion_events.py` (needs OpenGL: marked `gl` through the `gl_context` fixture)

**Interfaces:**
- Consumes: `pipeline.build_trajectories`, `simulator.Simulator`, `config.SimConfig`.
- Produces: `events_report.event_statistics(scene, trajectory, config=None) -> {"events_per_s", "events_per_step_p50", "events_per_step_p95", "flow_px_s_mean", "flow_px_s_p95"}`; `compare_event_statistics(scene, motion, street, config=None) -> {"plain": stats, "human": stats}` (`motion` = `{"model": id}` or `{"path": file}`; `street` = extra street spec keys such as `duration_s`, `seed`, `min_clearance`, `edge_margin`); `format_event_report(report) -> str`. Informational only: there is no real event recording to compare with.

- [ ] **Step 1: Write the failing test**


Create `tests/test_motion_events.py`:

```python
"""motion: the event-statistics report on the demo scene (needs OpenGL and torch)."""
from __future__ import annotations

import numpy as np
import pytest

from backend.motion.events_report import compare_event_statistics, event_statistics, format_event_report
from backend.scene import load_scene
from backend.trajectories import WaypointTrajectory

pytestmark = pytest.mark.usefixtures("gl_context")
STREET = {"duration_s": 2.0, "seed": 1, "min_clearance": 0.5, "edge_margin": 0}


@pytest.fixture(scope="module")
def scene(demo_scene):
    return load_scene(demo_scene.scene_spec)


def test_statistics_of_a_known_trajectory(scene, demo_scene):
    stats = event_statistics(scene, WaypointTrajectory.from_json(demo_scene.trajectory_json))
    assert set(stats) == {"events_per_s", "events_per_step_p50", "events_per_step_p95", "flow_px_s_mean", "flow_px_s_p95"}
    assert stats["events_per_s"] > 0 and stats["flow_px_s_mean"] > 0 and stats["flow_px_s_p95"] >= stats["flow_px_s_mean"]


def test_plain_and_human_walks_are_compared_on_the_same_planned_path(scene, motion_checkpoint):
    pytest.importorskip("torch")
    report = compare_event_statistics(scene, {"path": str(motion_checkpoint)}, STREET)
    assert set(report) == {"plain", "human"}
    for stats in report.values():
        assert all(np.isfinite(v) for v in stats.values()) and stats["events_per_s"] > 0
    assert report["plain"] != report["human"]
    text = format_event_report(report)
    assert "events / s" in text and "plain" in text and "human" in text
```


- [ ] **Step 2: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_events.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'backend.motion.events_report'`.


- [ ] **Step 3: Write the implementation**


Create `backend/motion/events_report.py`:

```python
"""Informational comparison of the event streams a plain street walk and a human-motion walk produce on a scene.

Not a pass / fail test: there is no real event recording to compare against. It shows whether the added motion
changes the event rate and the optical flow in the expected direction (more events, more and faster flow jitter).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from backend.config import CameraConfig, SimConfig
from backend.pipeline import build_trajectories
from backend.simulator import Simulator
from backend.trajectories import Trajectory

DEFAULT_CONFIG = SimConfig(camera=CameraConfig(width=128, height=96), dt_us=10_000, gt_every=5, gt=("flow",))


def event_statistics(scene, trajectory: Trajectory, config: Optional[SimConfig] = None) -> Dict[str, float]:
    """Events per second, events per step (median, 95th percentile) and optical-flow speed in px/s (mean, 95th
    percentile over the valid pixels) of one simulated trajectory."""
    config = config or DEFAULT_CONFIG
    counts, speeds = [], []
    with Simulator(scene, config) as sim:
        for step in sim.run(trajectory):
            counts.append(len(step.events_left))
            gt = step.gt
            if gt is not None and gt.flow is not None and gt.flow_dt_us:
                valid = gt.flow_valid.astype(bool)
                if valid.any():
                    speeds.append(np.linalg.norm(gt.flow[valid], axis=1) * 1e6 / gt.flow_dt_us)
    seconds = (trajectory.end_us - trajectory.start_us) / 1e6
    flow = np.concatenate(speeds) if speeds else np.zeros(1)
    return {
        "events_per_s": float(np.sum(counts) / seconds),
        "events_per_step_p50": float(np.percentile(counts, 50)),
        "events_per_step_p95": float(np.percentile(counts, 95)),
        "flow_px_s_mean": float(flow.mean()),
        "flow_px_s_p95": float(np.percentile(flow, 95)),
    }


def compare_event_statistics(scene, motion: Dict[str, Any], street: Dict[str, Any],
                             config: Optional[SimConfig] = None) -> Dict[str, Dict[str, float]]:
    """``{"plain": ..., "human": ...}``: the same planned street walk (same ``street`` spec and seed), without and
    with the ``motion`` model reference (``{"model": id}`` or ``{"path": file}``)."""
    spec = {"type": "street", "count": 1, "seed": 0, **street}
    cache: Dict[Any, Any] = {}
    (plain,) = build_trajectories(spec, scene, cache)
    (human,) = build_trajectories({**spec, "motion": motion}, scene, cache)
    return {"plain": event_statistics(scene, plain, config), "human": event_statistics(scene, human, config)}


def format_event_report(report: Dict[str, Dict[str, float]]) -> str:
    rows = (("events_per_s", "events / s"), ("events_per_step_p50", "events / step (median)"),
            ("events_per_step_p95", "events / step (95th pct)"), ("flow_px_s_mean", "flow speed, px/s (mean)"),
            ("flow_px_s_p95", "flow speed, px/s (95th pct)"))
    lines = [f"{'':<28}{'plain':>12}{'human':>12}"]
    lines += [f"{label:<28}{report['plain'][key]:>12.1f}{report['human'][key]:>12.1f}" for key, label in rows]
    return "\n".join(lines)
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_events.py`
Expected: 2 passed.

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/motion tests/test_motion_events.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/motion/events_report.py tests/test_motion_events.py
git commit -m "feat(motion): compare event rate and flow of plain and human-motion walks"
```

---

### Task 11: Command line

**Files:**
- Modify: `backend/cli.py`
- Test: `tests/test_motion_cli.py`

**Interfaces:**
- Consumes: everything above (imported lazily inside the commands so `dvsim --help` stays light).
- Produces: `dvsim motion-import NAME FORMAT PATH [--out corpus.npz] [--layout L] [--up X Y Z] [--camera opengl|opencv]`; `dvsim motion-train --corpus C (--name ID | --out FILE) [--init CKPT] [--steps --batch --lr --base --mults --diffusion-steps --seed --val-frac --test-frac --holdout-source S --device]`; `dvsim motion-eval --corpus C --model (FILE|ID) [--scene S --events-duration --min-clearance --edge-margin] [--max-sequences --sample-steps --seed --device]`; `dvsim run ... --generate street --motion (FILE.pt|ID)`; `cli.MOTION_FORMATS` (must equal `importers.FORMATS`); the three `motion-*` commands print `error: ...` to stderr and return 1 on a bad file, missing model, missing torch or empty split.

- [ ] **Step 1: Write the failing test**


Create `tests/test_motion_cli.py`:

```python
"""motion: the dvsim motion-import / motion-train / motion-eval commands."""
from __future__ import annotations

import pytest
from motion_helpers import synthetic_walk, write_tum_walk

from backend.cli import MOTION_FORMATS, _motion_ref, main
from backend.motion.corpus import load_corpus, save_corpus
from backend.motion.importers import FORMATS


def test_the_format_list_of_the_cli_matches_the_importers():
    assert MOTION_FORMATS == FORMATS


def test_import_builds_a_corpus_source_by_source(tmp_path, capsys):
    write_tum_walk(tmp_path / "tum" / "groundtruth.txt")
    corpus = tmp_path / "corpus.npz"
    assert main(["motion-import", "tumrgbd", "tum", str(tmp_path / "tum"), "--out", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "tumrgbd: 1 file(s), 1 segment(s)" in out and "heading agreement 1.00" in out
    assert main(["motion-import", "second", "tum", str(tmp_path / "tum"), "--out", str(corpus)]) == 0
    assert main(["motion-import", "tumrgbd", "tum", str(tmp_path / "tum"), "--out", str(corpus)]) == 0   # replaces
    sources = sorted(t.source for t in load_corpus(corpus))
    assert sources == ["second", "tumrgbd"]


def test_a_wrong_frame_warns_and_an_empty_import_fails(tmp_path, capsys):
    write_tum_walk(tmp_path / "tum" / "groundtruth.txt")
    corpus = tmp_path / "corpus.npz"
    assert main(["motion-import", "x", "tum", str(tmp_path / "tum"), "--camera", "opengl", "--out", str(corpus)]) == 0
    assert "frame is probably wrong" in capsys.readouterr().err
    write_tum_walk(tmp_path / "short" / "groundtruth.txt", seconds=2.0)
    assert main(["motion-import", "s", "tum", str(tmp_path / "short"), "--out", str(tmp_path / "none.npz")]) == 1
    assert "nothing to import" in capsys.readouterr().err and not (tmp_path / "none.npz").exists()


def test_motion_ref_tells_files_from_ids():
    assert _motion_ref("walk-v1") == {"model": "walk-v1"}
    assert _motion_ref("some/dir/walk.pt") == {"path": "some/dir/walk.pt"}


def test_training_and_evaluating_a_model(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    monkeypatch.setenv("DVSIM_DATA_DIR", str(tmp_path / "data"))
    corpus = tmp_path / "corpus.npz"
    save_corpus(corpus, [synthetic_walk(i) for i in range(10, 18)])        # walks 10, 11, 12 and 17 are held out
    train = ["motion-train", "--corpus", str(corpus), "--steps", "20", "--batch", "8", "--base", "16", "--mults", "1", "2",
             "--diffusion-steps", "100", "--val-frac", "0", "--test-frac", "0.5", "--device", "cpu"]
    assert main(train + ["--name", "tiny"]) == 0
    assert "4 training, 0 validation, 4 held-out" in capsys.readouterr().out
    assert (tmp_path / "data" / "motion" / "tiny.pt").is_file()
    assert main(["motion-eval", "--corpus", str(corpus), "--model", "tiny", "--sample-steps", "4", "--device", "cpu",
                 "--max-sequences", "2"]) == 0
    out = capsys.readouterr().out
    assert "spectral distance" in out and "classifier accuracy" in out
    with pytest.raises(SystemExit, match="--name ID"):                      # exactly one of --name / --out
        main(train)


def test_motion_needs_the_street_generator(tmp_path, capsys):
    assert main(["run", "--scene", str(tmp_path / "s.obj"), "--generate", "spline", "--motion", "walk", "--out",
                 str(tmp_path / "o")]) == 2
    assert "--motion needs --generate street" in capsys.readouterr().err


def test_bad_inputs_are_reported_not_raised(tmp_path, capsys):
    assert main(["motion-import", "x", "tum", str(tmp_path / "nowhere")]) == 1
    assert "no such file or directory" in capsys.readouterr().err
    save_corpus(tmp_path / "c.npz", [synthetic_walk(0)])
    assert main(["motion-eval", "--corpus", str(tmp_path / "c.npz"), "--model", str(tmp_path / "gone.pt")]) == 1
    assert "motion model file not found" in capsys.readouterr().err
    assert main(["motion-train", "--corpus", str(tmp_path / "c.npz"), "--out", str(tmp_path / "m.pt"), "--test-frac", "0.99",
                 "--val-frac", "0"]) == 1
    assert "no training recording" in capsys.readouterr().err


@pytest.mark.usefixtures("gl_context")
def test_motion_eval_prints_the_event_comparison(demo_scene, motion_checkpoint, tmp_path, capsys):
    corpus = tmp_path / "corpus.npz"
    save_corpus(corpus, [synthetic_walk(i) for i in range(10, 18)])
    assert main(["motion-eval", "--corpus", str(corpus), "--model", str(motion_checkpoint), "--sample-steps", "4",
                 "--device", "cpu", "--max-sequences", "2", "--scene", str(demo_scene.scene_spec),
                 "--events-duration", "2", "--min-clearance", "0.5", "--edge-margin", "0"]) == 0
    out = capsys.readouterr().out
    assert "spectral distance" in out and "events on the scene" in out and "flow speed" in out
```


- [ ] **Step 2: Run it to verify it fails**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_cli.py`
Expected: FAIL: `ImportError: cannot import name 'MOTION_FORMATS' from 'backend.cli'`.


- [ ] **Step 3: Write the implementation**


Apply this change to `backend/cli.py` (a unified diff against the current files: `git apply` takes it as is, or make the same edits by hand):

```diff
--- a/backend/cli.py
+++ b/backend/cli.py
@@ -1,7 +1,8 @@
-"""`dvsim` command line: demo, run, sample-scene, serve, info."""
+"""`dvsim` command line: demo, run, sample-scene, serve, info, motion-import, motion-train, motion-eval."""
 from __future__ import annotations
 
 import argparse
+import functools
 import json
 import os
 import sys
@@ -11,6 +12,10 @@
 from backend.config import GT_FIELDS, SimConfig
 
 
+# backend.motion.importers.FORMATS, spelled out here so that `dvsim --help` does not import the motion package
+MOTION_FORMATS = ("tum", "aria", "headset", "csv")
+
+
 def _add_sim_args(p: argparse.ArgumentParser) -> None:
     g = p.add_argument_group("simulation (override --config)")
     g.add_argument("--config", type=Path, help="JSON file with a (partial) SimConfig")
@@ -77,6 +82,11 @@
         if a.generate == "street":
             del traj["n_key_points"]
             traj.update(speed=a.walk_speed, eye_height=a.eye_height, edge_margin=a.edge_margin)
+            if a.motion:
+                traj["motion"] = _motion_ref(a.motion)
+        elif a.motion:
+            print("error: --motion needs --generate street", file=sys.stderr)
+            return 2
     dirs = run_to_disk(
         _scene_spec(a.scene, a.texture), traj, _config_from_args(a), a.out,
         visualize=a.visualize, compression=a.compression, prefix=a.prefix, show_bar=True,
@@ -86,6 +96,102 @@
     return 0
 
 
+def _reports_errors(func):
+    """Commands that read user files and models: a bad input is an `error:` line and exit code 1, not a traceback."""
+    @functools.wraps(func)
+    def run(a: argparse.Namespace) -> int:
+        try:
+            return func(a)
+        except (ImportError, FileNotFoundError, ValueError) as exc:
+            print(f"error: {exc}", file=sys.stderr)
+            return 1
+    return run
+
+
+def _motion_ref(value: str) -> dict:
+    """A trained model given as a file (``walk.pt``, any path) or as the id of one in $DVSIM_DATA_DIR/motion."""
+    path = Path(value)
+    return {"path": str(path)} if path.suffix == ".pt" or path.exists() else {"model": value}
+
+
+@_reports_errors
+def _cmd_motion_import(a: argparse.Namespace) -> int:
+    from backend.motion.corpus import import_source, load_corpus, save_corpus
+
+    tracks, report = import_source(a.name, a.path, a.format, a.layout, a.up, a.camera)
+    print(f"{a.name}: {report.files} file(s), {report.segments} segment(s), {report.hours:.2f} h, "
+          f"heading agreement {report.agreement:.2f}")
+    for warning in report.warnings:
+        print(f"warning: {warning}", file=sys.stderr)
+    if not tracks:
+        print("error: nothing to import", file=sys.stderr)
+        return 1
+    kept = [t for t in load_corpus(a.out) if t.source != a.name] if a.out.exists() else []
+    save_corpus(a.out, kept + tracks)
+    print(f"{a.out}: {len(kept) + len(tracks)} segment(s) from {len({t.source for t in kept + tracks})} source(s)")
+    return 0
+
+
+def _model_path(name: Optional[str], out: Optional[Path]) -> Path:
+    from backend.motion.registry import SUFFIX, models_dir
+
+    if bool(name) == bool(out):
+        raise SystemExit("error: give either --name ID (stored in $DVSIM_DATA_DIR/motion) or --out FILE")
+    if out:
+        return out
+    folder = models_dir()
+    folder.mkdir(parents=True, exist_ok=True)
+    return folder / f"{name}{SUFFIX}"
+
+
+@_reports_errors
+def _cmd_motion_train(a: argparse.Namespace) -> int:
+    from backend.motion.corpus import load_corpus
+    from backend.motion.dataset import Stats, assign_splits, build_sequences, select
+    from backend.motion.train import TrainConfig, train
+
+    out = _model_path(a.name, a.out)
+    seqs = build_sequences(load_corpus(a.corpus))
+    cfg = TrainConfig(
+        steps=a.steps, batch=a.batch, lr=a.lr, base=a.base, mults=tuple(a.mults), diffusion_steps=a.diffusion_steps,
+        seed=a.seed, val_frac=a.val_frac, test_frac=a.test_frac, split_seed=a.seed,
+        holdout_sources=tuple(a.holdout_source or ()),
+    )
+    splits = assign_splits(seqs, cfg.val_frac, cfg.test_frac, cfg.split_seed, cfg.holdout_sources)
+    train_seqs, val_seqs = select(seqs, splits, "train"), select(seqs, splits, "val")
+    if not train_seqs:
+        print("error: no training recording left after the split", file=sys.stderr)
+        return 1
+    print(f"{len(train_seqs)} training, {len(val_seqs)} validation, {len(select(seqs, splits, 'test'))} held-out "
+          f"segment(s)")
+    train(train_seqs, val_seqs, Stats.fit(train_seqs), cfg, out, device=a.device, init=a.init)
+    print(out)
+    return 0
+
+
+@_reports_errors
+def _cmd_motion_eval(a: argparse.Namespace) -> int:
+    from backend.motion.corpus import load_corpus
+    from backend.motion.evaluate import evaluate_model, format_report
+    from backend.motion.registry import resolve_model
+    from backend.motion.style import DiffusionMotionStyle
+
+    style = DiffusionMotionStyle.load(resolve_model(_motion_ref(a.model)), device=a.device, sample_steps=a.sample_steps)
+    report = evaluate_model(style, load_corpus(a.corpus), a.max_sequences, a.seed)
+    print(format_report(report))
+    if a.scene:
+        from backend.motion.events_report import compare_event_statistics, format_event_report
+        from backend.scene import load_scene
+
+        street = {"duration_s": a.events_duration, "seed": a.seed}
+        for key in ("min_clearance", "edge_margin"):
+            if getattr(a, key) is not None:
+                street[key] = getattr(a, key)
+        print("\nevents on the scene (same planned walk, without and with the model):")
+        print(format_event_report(compare_event_statistics(load_scene(a.scene), _motion_ref(a.model), street)))
+    return 0
+
+
 def _cmd_demo(a: argparse.Namespace) -> int:
     from backend.pipeline import run_to_disk
     from backend.samples import create_demo_scene
@@ -177,6 +283,10 @@
         "--edge-margin", type=float, default=None,
         help="street: keep walks this far from where the scene ends, m (default 5; 0 to walk up to the edge)",
     )
+    p.add_argument(
+        "--motion", metavar="MODEL",
+        help="street: walk like a person: a trained model (file.pt, or the id of one in $DVSIM_DATA_DIR/motion)",
+    )
     p.add_argument("--out", required=True, type=Path)
     p.add_argument("--prefix", default="seq")
     p.add_argument("--visualize", action="store_true", help="also write mp4 previews")
@@ -199,6 +309,47 @@
     p = sub.add_parser("info", help="check the installation (OpenGL context)")
     p.set_defaults(func=_cmd_info)
 
+    p = sub.add_parser("motion-import", help="add a dataset of human trajectories to a corpus file")
+    p.add_argument("name", help="source name (re-importing a name replaces that source)")
+    p.add_argument("format", choices=MOTION_FORMATS)
+    p.add_argument("path", type=Path, help="a trajectory file, or a folder searched recursively")
+    p.add_argument("--out", type=Path, default=Path("corpus.npz"))
+    p.add_argument("--layout", default="tumvi", help="csv: column layout (tumvi | dvsim)")
+    p.add_argument("--up", nargs=3, type=float, metavar=("X", "Y", "Z"), help="up direction of the dataset's world")
+    p.add_argument("--camera", choices=["opengl", "opencv"], help="camera axes (opencv: Y down, looks along +Z)")
+    p.set_defaults(func=_cmd_motion_import)
+
+    p = sub.add_parser("motion-train", help="train a human-motion model on a corpus (needs torch)")
+    p.add_argument("--corpus", required=True, type=Path)
+    p.add_argument("--name", help="model id: stored as $DVSIM_DATA_DIR/motion/<id>.pt (what the API lists)")
+    p.add_argument("--out", type=Path, help="checkpoint file (instead of --name)")
+    p.add_argument("--init", help="continue from this checkpoint (same --base / --mults)")
+    p.add_argument("--steps", type=int, default=20000)
+    p.add_argument("--batch", type=int, default=64)
+    p.add_argument("--lr", type=float, default=2e-4)
+    p.add_argument("--base", type=int, default=64, help="U-Net width (a multiple of 8)")
+    p.add_argument("--mults", nargs="+", type=int, default=[1, 2, 4], help="U-Net width multipliers per level")
+    p.add_argument("--diffusion-steps", type=int, default=1000)
+    p.add_argument("--seed", type=int, default=0, help="training and split seed")
+    p.add_argument("--val-frac", type=float, default=0.1)
+    p.add_argument("--test-frac", type=float, default=0.1)
+    p.add_argument("--holdout-source", action="append", help="keep every recording of this source for evaluation")
+    p.add_argument("--device", help="cuda | mps | cpu (default: the best available)")
+    p.set_defaults(func=_cmd_motion_train)
+
+    p = sub.add_parser("motion-eval", help="compare a model's motion with held-out real motion (needs torch)")
+    p.add_argument("--corpus", required=True, type=Path)
+    p.add_argument("--model", required=True, help="file.pt, or a model id")
+    p.add_argument("--scene", help="also simulate this scene (.obj/.glb/.json) to compare event statistics")
+    p.add_argument("--events-duration", type=float, default=10.0, help="seconds of walk for the event comparison")
+    p.add_argument("--min-clearance", type=float, help="street walk of the event comparison, as in `run`")
+    p.add_argument("--edge-margin", type=float, help="street walk of the event comparison, as in `run`")
+    p.add_argument("--max-sequences", type=int, default=0, help="evaluate at most this many held-out segments")
+    p.add_argument("--sample-steps", type=int, default=50)
+    p.add_argument("--seed", type=int, default=0)
+    p.add_argument("--device")
+    p.set_defaults(func=_cmd_motion_eval)
+
     args = parser.parse_args(argv)
     return args.func(args)
 
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_motion_cli.py`
Expected: 8 passed (the training test takes a few seconds).

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/cli.py tests/test_motion_cli.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/cli.py tests/test_motion_cli.py
git commit -m "feat(motion): motion-import, motion-train, motion-eval and run --motion"
```

---

### Task 12: HTTP API

**Files:**
- Modify: `backend/api/settings.py`, `backend/api/schemas.py`, `backend/api/planner.py`, `backend/api/preview.py`, `backend/api/routes.py`
- Modify: `tests/test_api.py` (tests appended)

**Interfaces:**
- Consumes: `registry.MODEL_ID/resolve_model/list_models`, `pipeline.build_trajectories`.
- Produces: `Settings.motion_dir`; `settings.MAX_MOTION_PREVIEW_S = 300.0`; `schemas.MotionRef(model: str)` and `StreetSpec.motion: Optional[MotionRef]`; `planner.motion_file(model_id, data_dir) -> Path` (NotFound -> 404, no torch -> 501); `planner.plan_job(body, scenes, trajectories, data_dir)` (the worker spec carries `{"path": file}`, the job JSON keeps `{"model": id}`); `TrajectoryPreviewer.generate(body, scenes, data_dir)`; `GET /api/motion-models -> [{"id", "size_bytes"}]`.

**Security note.** The API only accepts model *ids* (`^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`, files of `$DVSIM_DATA_DIR/motion`); `{"path": ...}` is rejected by the schema (`extra = forbid`), so a client cannot make the server open an arbitrary file.

- [ ] **Step 1: Write the failing tests**


Append these tests (the diff of `tests/test_api.py` below is just the new tests):


Apply this change to `tests/test_api.py` (a unified diff against the current files: `git apply` takes it as is, or make the same edits by hand):

```diff
--- a/tests/test_api.py
+++ b/tests/test_api.py
@@ -1077,3 +1077,88 @@
         ws.send_json(_pose(3))
         _, seen = _recv_until(ws, lambda m: isinstance(m, tuple))
         assert not any(isinstance(m, dict) and m.get("type") == "origin" for m in seen)
+
+
+# ---- human motion models (backend.motion) ---------------------------------------------------------------------------
+STREET = {"type": "street", "duration_s": 3.0, "min_clearance": 0.5, "edge_margin": 0}
+
+
+def _install_motion_model(data_dir, checkpoint, name="tiny"):
+    folder = data_dir / "motion"
+    folder.mkdir(parents=True, exist_ok=True)
+    (folder / f"{name}.pt").write_bytes(Path(checkpoint).read_bytes())
+
+
+def test_motion_models_are_listed(client, data_dir, motion_checkpoint):
+    assert client.get("/api/motion-models").json() == []
+    _install_motion_model(data_dir, motion_checkpoint)
+    (data_dir / "motion" / "readme.txt").write_text("not a model")
+    (listed,) = client.get("/api/motion-models").json()
+    assert listed["id"] == "tiny" and listed["size_bytes"] == Path(motion_checkpoint).stat().st_size
+
+
+def test_street_preview_with_a_motion_model(client, data_dir, demo_scene, motion_checkpoint):
+    _install_motion_model(data_dir, motion_checkpoint)
+    scene = _upload(client, demo_scene)["id"]
+    body = {"scene_id": scene, "trajectory": {**STREET, "seed": 5, "motion": {"model": "tiny"}}}
+    r = client.post("/api/trajectories/generate", json=body)
+    assert r.status_code == 200, r.text
+    first = r.json()
+    assert first["trajectory"]["motion"] == {"model": "tiny"} and first["seed"] == 5      # the id, never a server path
+    assert len(first["trajectories"][0]["points"]) == 151                                 # 3 s at 50 Hz
+    again = client.post("/api/trajectories/generate", json=body).json()
+    assert again["trajectories"] == first["trajectories"]                                 # same seed, same walk
+    plain = client.post("/api/trajectories/generate", json={**body, "trajectory": {**STREET, "seed": 5}}).json()
+    assert len(plain["trajectories"][0]["points"]) < 151                                  # the planner's own waypoints
+
+
+def test_motion_model_errors(client, data_dir, demo_scene, motion_checkpoint, monkeypatch):
+    scene = _upload(client, demo_scene)["id"]
+
+    def generate(motion):
+        return client.post("/api/trajectories/generate",
+                           json={"scene_id": scene, "trajectory": {**STREET, "motion": motion}})
+
+    r = generate({"model": "missing"})
+    assert r.status_code == 404 and "'missing' not found" in r.json()["detail"]
+    for bad in ({"model": "../etc/passwd"}, {"model": ""}, {"path": "/etc/passwd"}, {}, {"model": "a", "extra": 1}):
+        assert generate(bad).status_code == 422
+    r = client.post("/api/jobs", json={"scene_id": scene, "trajectory": {**STREET, "motion": {"model": "missing"}}})
+    assert r.status_code == 404 and client.get("/api/jobs").json() == []                  # nothing was queued
+    _install_motion_model(data_dir, motion_checkpoint)
+    monkeypatch.setattr("backend.api.planner._torch_available", lambda: False)
+    for r in (generate({"model": "tiny"}),
+              client.post("/api/jobs", json={"scene_id": scene, "trajectory": {**STREET, "motion": {"model": "tiny"}}})):
+        assert r.status_code == 501 and "pip install -e .[torch]" in r.json()["detail"]
+    assert client.post("/api/trajectories/generate",
+                       json={"scene_id": scene, "trajectory": {"type": "spline", "duration_s": 1, "motion": {"model": "tiny"}}}
+                       ).status_code == 422                                              # only street walks take motion
+
+
+def test_a_corrupt_motion_model_is_a_clean_error(client, data_dir, demo_scene):
+    (data_dir / "motion").mkdir(parents=True)
+    (data_dir / "motion" / "broken.pt").write_bytes(b"not a checkpoint")
+    scene = _upload(client, demo_scene)["id"]
+    r = client.post("/api/trajectories/generate",
+                    json={"scene_id": scene, "trajectory": {**STREET, "motion": {"model": "broken"}}})
+    assert r.status_code == 422 and "cannot read this file" in r.json()["detail"]
+
+
+def test_job_with_a_motion_model(client, data_dir, demo_scene, motion_checkpoint):
+    _install_motion_model(data_dir, motion_checkpoint)
+    scene = _upload(client, demo_scene)["id"]
+    job_id = _job(client, scene, {**STREET, "duration_s": 1.0, "seed": 2, "motion": {"model": "tiny"}})
+    job = _wait(client, job_id)
+    assert job["status"] == "done", job
+    assert job["trajectory"]["motion"] == {"model": "tiny"}                               # the job JSON keeps the id
+    request = json.loads((data_dir / "jobs" / job_id / "request.json").read_text())
+    assert request["trajectory"]["motion"] == {"path": str(data_dir / "motion" / "tiny.pt")}   # the worker gets the file
+    assert job["progress"]["total_steps"] == 101                                          # 1 s at 10 ms
+
+
+def test_a_long_motion_preview_is_refused(client, demo_scene):
+    scene = _upload(client, demo_scene)["id"]
+    for duration, count in ((301.0, 1), (100.0, 4)):
+        r = client.post("/api/trajectories/generate", json={
+            "scene_id": scene, "trajectory": {**STREET, "duration_s": duration, "count": count, "motion": {"model": "any"}}})
+        assert r.status_code == 422 and "limited to 300 s in total" in r.json()["detail"]
```


- [ ] **Step 2: Run them to verify they fail**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_api.py -k motion`
Expected: FAIL: `/api/motion-models` is 404, `motion` is an unknown key (422).


- [ ] **Step 3: Write the implementation**


Apply this change to `backend/api/settings.py`, `backend/api/schemas.py`, `backend/api/planner.py`, `backend/api/preview.py`, `backend/api/routes.py` (a unified diff against the current files: `git apply` takes it as is, or make the same edits by hand):

```diff
--- a/backend/api/settings.py
+++ b/backend/api/settings.py
@@ -12,6 +12,7 @@
 MAX_SEQUENCES = 100               # `count` of a generated trajectory spec
 MAX_DURATION_S = 3600.0           # duration of a generated trajectory
 MAX_STEPS = 2_000_000             # simulation steps of one sequence
+MAX_MOTION_PREVIEW_S = 300.0      # seconds (duration x count) of human-motion walks one preview request may sample
 MAX_RESOLUTION = 8192             # camera width / height
 MAX_JSON_BYTES = 64 * 1024 * 1024  # body of every request except the scene upload
 
@@ -53,6 +54,10 @@
     def jobs_dir(self) -> Path:
         return self.data_dir / "jobs"
 
+    @property
+    def motion_dir(self) -> Path:
+        return self.data_dir / "motion"
+
     @classmethod
     def from_env(cls, data_dir: Optional[Path] = None) -> "Settings":
         frontend = os.environ.get("DVSIM_FRONTEND_DIR", "").strip()
--- a/backend/api/schemas.py
+++ b/backend/api/schemas.py
@@ -7,6 +7,7 @@
 from pydantic import BaseModel, ConfigDict, Field, field_validator
 
 from backend.api.settings import MAX_DURATION_S, MAX_INLINE_POINTS, MAX_SEQUENCES
+from backend.motion.registry import MODEL_ID
 
 POINT_KEYS = ("timestamp", "x", "y", "z", "rx", "ry", "rz", "qx", "qy", "qz", "qw")
 
@@ -60,6 +61,12 @@
     min_clearance: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
 
 
+class MotionRef(_Strict):
+    """A trained human-motion model of ``$DVSIM_DATA_DIR/motion`` (GET /api/motion-models lists them)."""
+
+    model: str = Field(pattern=MODEL_ID.pattern)
+
+
 class StreetSpec(_Strict):
     type: Literal["street"]
     duration_s: float = Field(gt=0, le=MAX_DURATION_S)
@@ -69,6 +76,7 @@
     eye_height: Optional[float] = Field(default=None, gt=0, le=100, allow_inf_nan=False)
     edge_margin: Optional[float] = Field(default=None, ge=0, le=1000, allow_inf_nan=False)
     min_clearance: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
+    motion: Optional[MotionRef] = None
 
 
 TrajectorySpec = Annotated[
--- a/backend/api/planner.py
+++ b/backend/api/planner.py
@@ -2,8 +2,10 @@
 from __future__ import annotations
 
 import dataclasses
+import importlib.util
 import math
 from dataclasses import dataclass
+from pathlib import Path
 from typing import Any, Dict, Tuple
 
 from fastapi import HTTPException
@@ -11,7 +13,9 @@
 from backend.api.schemas import ImportedSpec, InlineSpec, JobCreate
 from backend.api.settings import MAX_RESOLUTION, MAX_STEPS
 from backend.api.stores import SceneStore, TrajectoryStore, clean_points
+from backend.api.util import NotFound
 from backend.config import SimConfig
+from backend.motion.registry import resolve_model
 
 
 @dataclass
@@ -65,8 +69,23 @@
     return config
 
 
+def _torch_available() -> bool:
+    return importlib.util.find_spec("torch") is not None
+
+
+def motion_file(model_id: str, data_dir: Path) -> Path:
+    """The checkpoint of a motion model id: 404 when there is none, 501 when this server cannot run it (no torch)."""
+    try:
+        path = resolve_model({"model": model_id}, data_dir)
+    except FileNotFoundError as exc:
+        raise NotFound(str(exc)) from None
+    if not _torch_available():
+        raise HTTPException(status_code=501, detail="human motion needs PyTorch on the server: pip install -e .[torch]")
+    return path
+
+
 def _trajectory(
-    spec: Any, trajectories: TrajectoryStore
+    spec: Any, trajectories: TrajectoryStore, data_dir: Path
 ) -> Tuple[Dict[str, Any], Dict[str, Any], float, int]:
     """(worker spec, public summary, duration of one sequence in s, number of sequences)."""
     if isinstance(spec, ImportedSpec):
@@ -85,14 +104,17 @@
         summary = {"type": "inline", "num_points": len(points), "origin": spec.origin}
         return worker, summary, (traj.end_us - traj.start_us) / 1e6, 1
     worker = spec.model_dump(exclude_none=True)
-    return worker, dict(worker), spec.duration_s, spec.count
+    summary = dict(worker)
+    if "motion" in worker:  # the worker has no data directory: it gets the file, the job JSON keeps the id
+        worker["motion"] = {"path": str(motion_file(worker["motion"]["model"], data_dir))}
+    return worker, summary, spec.duration_s, spec.count
 
 
-def plan_job(body: JobCreate, scenes: SceneStore, trajectories: TrajectoryStore) -> JobPlan:
+def plan_job(body: JobCreate, scenes: SceneStore, trajectories: TrajectoryStore, data_dir: Path) -> JobPlan:
     scene = scenes.spec(body.scene_id)
     scene_root = str(scenes.path(body.scene_id))
     config = parse_config(body.config)
-    worker, summary, duration_s, count = _trajectory(body.trajectory, trajectories)
+    worker, summary, duration_s, count = _trajectory(body.trajectory, trajectories, data_dir)
     steps = int(duration_s * 1e6 // config.dt_us) + 1
     if steps > MAX_STEPS:
         raise _invalid(f"too many simulation steps per sequence ({steps} > {MAX_STEPS}): raise dt_us or shorten it")
--- a/backend/api/preview.py
+++ b/backend/api/preview.py
@@ -8,12 +8,15 @@
 
 import secrets
 import threading
+from pathlib import Path
 from typing import Any, Dict, List, Optional
 
 import numpy as np
 from fastapi import HTTPException
 
+from backend.api.planner import motion_file
 from backend.api.schemas import TrajectoryGenerate
+from backend.api.settings import MAX_MOTION_PREVIEW_S
 from backend.api.stores import SceneStore
 
 
@@ -24,13 +27,21 @@
         self._scene = None
         self._cache: Dict[Any, Any] = {}
 
-    def generate(self, body: TrajectoryGenerate, scenes: SceneStore) -> Dict[str, Any]:
+    def generate(self, body: TrajectoryGenerate, scenes: SceneStore, data_dir: Path) -> Dict[str, Any]:
         from backend.pipeline import build_trajectories
         from backend.scene import load_scene
 
         spec_json = scenes.spec(body.scene_id)  # 404 before any work
         spec = body.trajectory.model_dump(exclude_none=True)
         spec.setdefault("seed", secrets.randbelow(2 ** 31))
+        if "motion" in spec and spec["duration_s"] * spec.get("count", 1) > MAX_MOTION_PREVIEW_S:
+            raise HTTPException(
+                status_code=422,
+                detail=f"previewing human-motion walks is limited to {MAX_MOTION_PREVIEW_S:g} s in total "
+                       "(duration x count): shorten it, or run a job")
+        built = dict(spec)  # what is generated; ``spec`` is what is returned (a motion model by its id)
+        if "motion" in built:
+            built["motion"] = {"path": str(motion_file(built["motion"]["model"], data_dir))}
         with self._lock:
             if self._scene_id != body.scene_id:
                 self._scene_id, self._scene, self._cache = None, None, {}
@@ -40,7 +51,7 @@
                     raise HTTPException(status_code=422, detail=f"cannot load the scene: {exc}") from None
                 self._scene_id = body.scene_id
             try:
-                trajs = build_trajectories(spec, self._scene, self._cache)
+                trajs = build_trajectories(built, self._scene, self._cache)
             except (ValueError, RuntimeError) as exc:
                 raise HTTPException(status_code=422, detail=f"cannot generate the trajectory: {exc}") from None
         return {
--- a/backend/api/routes.py
+++ b/backend/api/routes.py
@@ -14,6 +14,7 @@
 from backend.api.planner import plan_job
 from backend.api.schemas import JobCreate, TrajectoryCreate, TrajectoryGenerate
 from backend.api.util import NotFound, UnsafePath, resolve_below
+from backend.motion.registry import list_models
 
 router = APIRouter(prefix="/api")
 
@@ -79,9 +80,14 @@
 @router.post("/trajectories/generate", summary="Generate trajectories on a scene without simulating them")
 async def generate_trajectories(body: TrajectoryGenerate, request: Request) -> Dict[str, Any]:
     state = request.app.state
-    return await run_in_threadpool(state.previewer.generate, body, state.scenes)
+    return await run_in_threadpool(state.previewer.generate, body, state.scenes, state.settings.data_dir)
 
 
+@router.get("/motion-models", summary="Trained human-motion models usable as the `motion` of a street trajectory")
+def motion_models(request: Request) -> List[Dict[str, Any]]:
+    return list_models(request.app.state.settings.data_dir)
+
+
 @router.get("/trajectories")
 def list_trajectories(request: Request) -> List[Dict[str, Any]]:
     return request.app.state.trajectories.list()
@@ -106,7 +112,7 @@
 @router.post("/jobs", status_code=202, summary="Queue a simulation job")
 def create_job(body: JobCreate, request: Request) -> Dict[str, str]:
     state = request.app.state
-    plan = plan_job(body, state.scenes, state.trajectories)
+    plan = plan_job(body, state.scenes, state.trajectories, state.settings.data_dir)
     try:
         job_id = state.jobs.submit(plan)
     except QueueFull as exc:
```


- [ ] **Step 4: Run the tests and ruff**


Run: `.claude/scratch/venv39/bin/python -m pytest -q tests/test_api.py`
Expected: 83 passed (the existing API tests included; a real worker process loads the model).

Run: `.claude/scratch/venv39/bin/python -m ruff check backend/api tests/test_api.py`
Expected: All checks passed.


- [ ] **Step 5: Commit**

```bash
git add backend/api tests/test_api.py
git commit -m "feat(motion): motion models in the API (list, street spec field, 404/501/422 guards)"
```

---

### Task 13: Web interface

**Files:**
- Modify: `frontend/src/utils/simRequest.js`, `frontend/src/api/client.js`, `frontend/src/components/trajectoryOptions.js`
- Modify: `frontend/tests/simRequest.test.mjs`, `frontend/tests/api.test.mjs`

**Interfaces:**
- Consumes: `GET /api/motion-models`.
- Produces: `simRequest.NO_MOTION = ''`, `DEFAULT_SIM_SETTINGS.genMotion = ''`, `buildTrajectorySpec('Generated', ...)` adds `motion: { model }` for the street generator only, `motionOptions(models)` (lil-gui options: `{'None (planner glide)': '', <id>: <id>}`), `client.listMotionModels(options) -> [{id, size_bytes}]`, a "Human motion" dropdown in the Trajectory folder shown for Generated + street.

**Note.** lil-gui rebuilds a dropdown to change its options (`controller.options(...)` appends the new controller at the end of the folder); the code moves its DOM node back before "Preview paths". When the server is unreachable the dropdown just offers the planner's glide.

- [ ] **Step 1: Write the failing tests**


Apply this change to `frontend/tests/simRequest.test.mjs`, `frontend/tests/api.test.mjs` (a unified diff against the current files: `git apply` takes it as is, or make the same edits by hand):

```diff
--- a/frontend/tests/simRequest.test.mjs
+++ b/frontend/tests/simRequest.test.mjs
@@ -4,7 +4,7 @@
 import * as THREE from 'three';
 
 import {
-    DEFAULT_SIM_SETTINGS, buildJobRequest, buildSimConfig, buildTrajectorySpec, parseSeed, restoreSimSettings
+    DEFAULT_SIM_SETTINGS, buildJobRequest, buildSimConfig, buildTrajectorySpec, motionOptions, parseSeed, restoreSimSettings
 } from '../src/utils/simRequest.js';
 
 const sim = (overrides = {}) => ({ ...DEFAULT_SIM_SETTINGS, ...overrides });
@@ -92,6 +92,28 @@
     assert.throws(() => buildTrajectorySpec('Generated', { sim: sim({ genKind: 'spline', genKeyPoints: 1 }) }), /Key points/);
 });
 
+test('Generated street walks can name a human-motion model; other generators ignore it', () => {
+    assert.equal(DEFAULT_SIM_SETTINGS.genMotion, '');
+    assert.deepEqual(
+        buildTrajectorySpec('Generated', { sim: sim({ genKind: 'street', genDuration: 20, genSeed: '1', genMotion: 'walk-v1' }) }),
+        { type: 'street', duration_s: 20, count: 1, seed: 1, motion: { model: 'walk-v1' } }
+    );
+    assert.equal('motion' in buildTrajectorySpec('Generated', { sim: sim({ genKind: 'street', genMotion: '' }) }), false);
+    assert.equal('motion' in buildTrajectorySpec('Generated', { sim: sim({ genKind: 'spline', genMotion: 'walk-v1' }) }), false);
+    assert.equal(buildJobRequest('Generated', { sim: sim({ genKind: 'street', genMotion: 'walk-v1' }) }).trajectory.motion.model, 'walk-v1');
+});
+
+test('the Human motion dropdown offers the glide first, then the trained models', () => {
+    assert.deepEqual(motionOptions([]), { 'None (planner glide)': '' });
+    assert.deepEqual(motionOptions([{ id: 'walk-v1', size_bytes: 5 }, { id: 'run' }]), { 'None (planner glide)': '', 'walk-v1': 'walk-v1', run: 'run' });
+    assert.deepEqual(motionOptions(undefined), { 'None (planner glide)': '' });
+});
+
+test('a stored motion model is restored, a wrongly typed one is not', () => {
+    assert.equal(restoreSimSettings(sim(), { genMotion: 'walk-v1' }).genMotion, 'walk-v1');
+    assert.equal(restoreSimSettings(sim(), { genMotion: 3 }).genMotion, '');
+});
+
 test('Live cannot be simulated', () => {
     assert.throws(() => buildTrajectorySpec('Live', { sim: sim() }), /Live/);
 });
--- a/frontend/tests/api.test.mjs
+++ b/frontend/tests/api.test.mjs
@@ -47,6 +47,7 @@
             return json(res, 201, { id, name: state.scenes.get(id).name, files: fields.filter(f => f.filename).map(f => f.filename) });
         }
         if (pathname === '/api/scenes') return json(res, 200, [...state.scenes.values()]);
+        if (pathname === '/api/motion-models') return json(res, 200, [{ id: 'walk-v1', size_bytes: 4096 }]);
 
         if (pathname === '/api/trajectories/generate' && req.method === 'POST') {
             const payload = JSON.parse((await body(req)).toString());
@@ -199,6 +200,14 @@
     assert.equal(await download.text(), 'file:seq_000/left_events.h5');
 });
 
+test('client: trained human-motion models are listed', async (t) => {
+    const fake = await makeFake();
+    t.after(fake.close);
+    setApiBase(fake.url);
+
+    assert.deepEqual(await client.listMotionModels(), [{ id: 'walk-v1', size_bytes: 4096 }]);
+});
+
 test('client: HTTP errors carry the server detail, network errors are flagged', async (t) => {
     const fake = await makeFake();
     setApiBase(fake.url);
```


- [ ] **Step 2: Run them to verify they fail**


Run: `cd frontend && node --test tests/`
Expected: FAIL: `motionOptions` is not exported / `client.listMotionModels is not a function`.


- [ ] **Step 3: Write the implementation**


Apply this change to `frontend/src/utils/simRequest.js`, `frontend/src/api/client.js`, `frontend/src/components/trajectoryOptions.js` (a unified diff against the current files: `git apply` takes it as is, or make the same edits by hand):

```diff
--- a/frontend/src/utils/simRequest.js
+++ b/frontend/src/utils/simRequest.js
@@ -6,6 +6,7 @@
 
 export const GT_FIELDS = ['rgb', 'depth', 'flow', 'pose'];
 export const GENERATED_KINDS = ['rectilinear', 'spline', 'street'];
+export const NO_MOTION = '';
 
 
 export const DEFAULT_SIM_SETTINGS = {
@@ -31,6 +32,7 @@
     genCount: 1,
     genKeyPoints: 6,
     genSeed: '',
+    genMotion: NO_MOTION,   // street walks: the id of a trained human-motion model, '' = the planner's own glide
 
     // imported / live positions are relative to ...
     originMode: 'Object centre'
@@ -132,6 +134,7 @@
 
             if (!GENERATED_KINDS.includes(spec.type)) throw new Error(`Unknown generator "${spec.type}".`);
             if (spec.type === 'spline') spec.n_key_points = requireInteger(sim.genKeyPoints, 'Key points', 2);
+            if (spec.type === 'street' && sim.genMotion) spec.motion = { model: String(sim.genMotion) };
 
             return spec;
         }
@@ -141,6 +144,15 @@
     }
 }
 
+/** Options of the "Human motion" dropdown: the planner's own glide, then one entry per trained model (GET /api/motion-models). */
+export function motionOptions(models) {
+    const options = { 'None (planner glide)': NO_MOTION };
+
+    (Array.isArray(models) ? models : []).forEach(model => { options[model.id] = model.id; });
+
+    return options;
+}
+
 /** { trajectory, config, visualize }: everything of POST /api/jobs except scene_id. */
 export function buildJobRequest(type, context) {
     return {
--- a/frontend/src/api/client.js
+++ b/frontend/src/api/client.js
@@ -72,6 +72,11 @@
     return request('/api/scenes', options);
 }
 
+/** Trained human-motion models: [{ id, size_bytes }]. */
+export function listMotionModels(options) {
+    return request('/api/motion-models', options);
+}
+
 /**
  * Upload an OBJ (+ optional texture image) as a multipart form, or a multi-file scene: `extras` are the other files
  * ({ blob, name }: more OBJ files, the MTL files, the textures). Resolves to { id, name, files }.
--- a/frontend/src/components/trajectoryOptions.js
+++ b/frontend/src/components/trajectoryOptions.js
@@ -3,7 +3,8 @@
 import { armPicking, clearLiveStart, publishLiveOrigin, setLiveHeading } from "./liveStart";
 import { getSceneFiles, saveSimSettings, sceneStore } from "./simulation";
 import { generatePreview } from "../api/trajectoryPreview.js";
-import { GENERATED_KINDS, buildTrajectorySpec } from "../utils/simRequest.js";
+import { listMotionModels } from "../api/client.js";
+import { GENERATED_KINDS, NO_MOTION, buildTrajectorySpec, motionOptions } from "../utils/simRequest.js";
 import { ORIGIN_MODES } from "../utils/trajectoryOrigin.js";
 import { clearGeneratedPaths, setGeneratedPaths, showGeneratedPaths, updateImportedTrajectory } from "../utils/trajectorySetup.js";
 import { sampleConstantSpeed } from "../utils/trajectorySampling.js";
@@ -86,18 +87,42 @@
     const count = folder.add(sim, 'genCount', 1, 100, 1).name('Sequences');
     const keyPoints = folder.add(sim, 'genKeyPoints', 2, 20, 1).name('Key points');
     const seed = folder.add(sim, 'genSeed').name('Seed (empty = random)');
+    // Street walks can be played by a trained human-motion model; the list comes from the server.
+    let motion = folder.add(sim, 'genMotion', motionOptions([])).name('Human motion');
     const preview = folder.add(actions, 'previewPaths').name('Preview paths');
     const pathToPlay = folder.add(actions, 'pathToPlay', 1, 2, 1).name('Path to play').onChange(selectPath);
     const generated = [kind, duration, count, seed, preview];
 
     // Different settings give different paths: the preview no longer shows what would be simulated.
-    [kind, duration, count, keyPoints, seed].forEach(controller => controller.onChange(() => {
+    const onGeneratorSettingChange = () => {
         clearGeneratedPaths();
         resetPlayback();
         updateVisibility();
         if (THREE_VARIABLES.settings.trajectoryType === 'Generated') showGeneratedPaths();
-    }));
+    };
 
+    [kind, duration, count, keyPoints, seed].forEach(controller => controller.onChange(onGeneratorSettingChange));
+    kind.onChange(() => { if (sim.genKind === 'street') refreshMotionModels(); });
+    motion.onChange(onGeneratorSettingChange);
+
+    // lil-gui rebuilds a dropdown to change its options: the new one is put back before "Preview paths".
+    async function refreshMotionModels() {
+        let models;
+
+        try {
+            models = await listMotionModels();
+        } catch (error) {
+            return; // no server (or an old one): the dropdown keeps offering the planner's glide only
+        }
+
+        if (!models.some(model => model.id === sim.genMotion)) sim.genMotion = NO_MOTION;
+
+        motion = motion.options(motionOptions(models));
+        motion.onChange(onGeneratorSettingChange);
+        folder.$children.insertBefore(motion.domElement, preview.domElement);
+        updateVisibility();
+    }
+
     function resetPlayback() {
         THREE_VARIABLES.isPlaying = false;
         THREE_VARIABLES.animationProgress = 0;
@@ -158,10 +183,12 @@
         heading.updateDisplay();
         generated.forEach(controller => controller.show(type === 'Generated'));
         keyPoints.show(type === 'Generated' && sim.genKind === 'spline');
+        motion.show(type === 'Generated' && sim.genKind === 'street');
         pathToPlay.show(type === 'Generated' && THREE_VARIABLES.generatedPaths.length > 1);
     }
 
     updateVisibility();
+    refreshMotionModels();
 
     return updateVisibility;
 }
```


- [ ] **Step 4: Run the tests and the production build**


Run: `cd frontend && node --test tests/`
Expected: 58 passed (54 before, 4 new).

Run: `cd frontend && npx vite build`
Expected: `built in ...ms` (catches syntax errors in the GUI code, which the node tests do not import).


- [ ] **Step 5: Look at it in the browser** (the dropdown code is not covered by the node tests)

Start the server with a model installed and open the interface:

```bash
mkdir -p data/motion && cp <some trained model>.pt data/motion/walk-v1.pt
.claude/scratch/venv39/bin/python -m backend serve
```

Select Trajectory > Generated > Generator `street`: a "Human motion" dropdown appears before "Preview paths" with `None (planner glide)` and `walk-v1`. Pick `walk-v1` and press "Preview paths": the path shows head bob. Switch the generator to `spline`: the dropdown disappears. Stop the server: reload the page, the dropdown offers only the glide and nothing errors in the console.

- [ ] **Step 6: Commit**

```bash
git add frontend/src frontend/tests
git commit -m "feat(frontend): Human motion selector for street walks"
```

---

### Task 14: Documentation, version, full verification

**Files:**
- Modify: `README.md`, `docs/architecture.md`, `CHANGELOG.md`, `pyproject.toml` (version 0.3.0)
- Modify: `docs/superpowers/specs/2026-10-02-human-motion-design.md` (append the implementation notes)

- [ ] **Step 1: Documentation and version**


Apply this change to `CHANGELOG.md`, `README.md`, `docs/architecture.md`, `pyproject.toml` (a unified diff against the current files: `git apply` takes it as is, or make the same edits by hand):

```diff
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -3,6 +3,27 @@
 All notable changes to DVSim. Versions follow [Semantic Versioning](https://semver.org/) (0.x: the interfaces may
 still change between minor versions).
 
+## [0.3.0] - Unreleased
+
+Human-like camera motion: learn how people move from recorded trajectories and replay it on planned walks.
+
+### Added
+- `backend/motion`: importers for TUM RGB-D, TUM-VI (csv), Project Aria / ADT and DVSim headset trajectories, a
+  common 50 Hz track format in DVSim's frame with a heading-agreement check that catches a wrong camera or up-axis
+  convention, and a split of every track into a smooth path and a human-motion residual.
+- A small conditional 1D diffusion model (x0-prediction, DDIM sampling) of that residual, conditioned on the path's
+  speed, yaw rate and curvature; long clips are generated window by window, each given the last second of the
+  previous one. Optional dependency: `pip install -e ".[torch]"`.
+- `motion` field on `street` trajectory specs (API, `pipeline.build_trajectories`) and `dvsim run --motion MODEL`:
+  the planner's path walked with sampled human head motion, reproducible from the seed.
+- `dvsim motion-import`, `motion-train` and `motion-eval` (spectra, step frequency, acceleration / jerk distributions
+  and a real-vs-generated classifier on held-out recordings, next to the plain-walk baseline; `--scene` adds an
+  event-rate and optical-flow comparison).
+- `GET /api/motion-models` and a "Human motion" selector in the web interface.
+
+### Changed
+- Checkpoints hold tensors and plain data only and are loaded with `weights_only=True`.
+
 ## [0.2.0] - 2026-10-02
 
 Rewrite of the first prototype into an installable package with a CLI, a web API and a dataloader.
--- a/README.md
+++ b/README.md
@@ -24,10 +24,12 @@
   objects with their own position, rotation and scale.
 - Trajectories: recorded JSON paths, paths drawn in the web interface, paths recorded with an AR headset, or generated
   inside the scene: `rectilinear`, `spline`, and `street` (eye-height walks on open ground). Generation is fast: a voxel
-  free-space grid keeps the camera out of the geometry.
+  free-space grid keeps the camera out of the geometry. `street` walks can be played with learned human head motion
+  (bob, sway, roll, gaze shifts): see [Human-like camera motion](#human-like-camera-motion).
 
 **Ways to use it**
-- `dvsim` command line: `demo`, `run`, `sample-scene`, `serve`, `info`.
+- `dvsim` command line: `demo`, `run`, `sample-scene`, `serve`, `info`, and `motion-import` / `motion-train` /
+  `motion-eval` for human-like camera motion.
 - Web interface (three.js): live previews of the camera, events, depth and optical flow, trajectory drawing and
   preview, background simulation jobs with progress and downloads.
 - HTTP API (FastAPI) with background jobs, uploads, and WebSocket status.
@@ -112,7 +114,39 @@
 **Trajectory files** are JSON lists of `{"timestamp": ms, "x", "y", "z", "rx", "ry", "rz"}` (a forward vector) or
 `{..., "qx", "qy", "qz", "qw"}` (a full orientation). They can be recorded with the AR-headset page or drawn in the web
 interface. Positions can be offset with `--origin X Y Z`.
+
+## Human-like camera motion
+
+`street` walks glide: constant speed, no roll. A small diffusion model can learn how people really move a head-mounted
+camera and replay it on the planned path. The planner still keeps the camera out of the geometry; the model only adds
+the motion of the person (bob, sway, roll, gaze shifts, speed changes). Needs PyTorch: `pip install -e ".[torch]"`.
+
+```sh
+# 1. collect recordings of people walking (one call per dataset; re-importing a name replaces that source)
+dvsim motion-import tumrgbd tum /data/tum_rgbd --out corpus.npz        # TUM RGB-D groundtruth.txt files
+dvsim motion-import aria aria /data/adt --out corpus.npz               # Aria / ADT closed_loop_trajectory.csv
+dvsim motion-import mine headset recordings/ --out corpus.npz          # trajectories recorded with the AR headset
+
+# 2. train (a GPU or Apple MPS is used when there is one); the model is stored as $DVSIM_DATA_DIR/motion/walk-v1.pt
+dvsim motion-train --corpus corpus.npz --name walk-v1 --holdout-source mine
 
+# 3. check it against the recordings it never saw (--scene also compares event rates and optical flow)
+dvsim motion-eval --corpus corpus.npz --model walk-v1 --scene town.json
+
+# 4. use it
+dvsim run --scene town.json --generate street --motion walk-v1 --duration 30 --out out/
+```
+
+Each import prints a **heading agreement** per source: how well the camera looks where the person walks (1.0 for a
+head-mounted camera on a walking person). A value below 0.5 means the dataset's frame was misread; fix it with
+`--camera opengl|opencv` and `--up X Y Z`. Formats: `tum` (`t[s] x y z qx qy qz qw`), `aria`, `headset` (DVSim JSON) and
+`csv` (`--layout tumvi | dvsim`). In the web interface the model appears as "Human motion" once the street generator
+is selected; through the API it is `"motion": {"model": "walk-v1"}` on a `street` trajectory spec.
+
+Limits: the model is conditioned on speed and turning only, so it does not choose where to go, and it was trained on
+the speeds it saw (other speeds are clamped, with a warning). Training data decides the style: head-mounted glasses and
+handheld cameras move differently.
+
 ## Web interface
 
 ```sh
@@ -257,7 +291,7 @@
 ```
 
 Layout: `backend/scene` (loading), `graphical_computation` (one-pass renderer: RGB, luminance, depth, flow),
-`event_processing`, `trajectories`, `simulator.py`, `pipeline.py`, `saving`, `vizualisation`, `api`, `dataloader`,
+`event_processing`, `trajectories`, `motion` (learned human motion), `simulator.py`, `pipeline.py`, `saving`, `vizualisation`, `api`, `dataloader`,
 `samples`, `cli.py`; `frontend/` (Vite + three.js); `tests/`.
 
 Release notes are in [`CHANGELOG.md`](CHANGELOG.md).
--- a/docs/architecture.md
+++ b/docs/architecture.md
@@ -42,12 +42,13 @@
   pipeline.py          build_trajectories(...), run_to_disk(...) used by the CLI and the API worker
   event_processing/    EventSimulator (vectorised numpy; noise/mismatch options)
   trajectories/        WaypointTrajectory, JSON loader, generators, FreeSpaceGrid
+  motion/              learned human camera motion: importers, path/residual split, diffusion model, training, style (torch only in model/train/style)
   saving/              streaming HDF5 writers, TUM export, SequenceOutput
   vizualisation/       streaming video sinks
   samples/             procedural demo scene + trajectory (`backend.samples.create_demo_scene`)
   api/                 FastAPI app, job manager (spawned worker processes), static frontend; see api/__init__.py
   dataloader/          ClipStream (numpy) + torch IterableDataset; see dataloader/__init__.py for the training-loop example
-  cli.py, __main__.py  `dvsim` command line (demo / run / sample-scene / serve / info)
+  cli.py, __main__.py  `dvsim` command line (demo / run / sample-scene / serve / info / motion-import / motion-train / motion-eval)
   (compute.py removed: its hard-coded __main__ is replaced by the CLI; old version is in refs/backup/pre-dvsim-refactor)
 frontend/              three.js previewer + simulation control panel (frontend/README.md)
 tests/                 pytest, one file per module (243 backend tests + 64 API tests + dataloader tests)
@@ -108,6 +109,27 @@
 class RectilinearTrajectoryGenerator(mesh, ...).generate(duration_us, rng) -> WaypointTrajectory
 class SplineTrajectoryGenerator(mesh, ...).generate(duration_us, n_key_points, rng) -> WaypointTrajectory
 
+# backend/motion   (only model.py, train.py, style.py import torch; `backend.motion.require_torch()` raises with the install hint)
+RATE_HZ = 50; DT_US = 20_000
+class RawTrack(name, times_us, positions, quats, frame: FrameSpec)   # a recording in its dataset's own frame
+class FrameSpec(up=(x, y, z) of the dataset's world, camera="opengl"|"opencv")
+class Track(source, name, times_us, positions, quats)                # one segment at RATE_HZ in DVSim's frame (+Y up, camera -Z)
+importers.read(path, fmt, layout="tumvi", up=None, camera=None) -> [RawTrack]      # fmt: tum | aria | headset | csv
+normalise.normalise(raw, source, max_gap_s=0.2, min_duration_s=4.0) -> [Track]    # frame, gaps, 50 Hz resampling
+decompose.split(positions, quats) -> (PathFrame, residual (N, 6))    # THE definition of path / residual, shared by training and sampling
+decompose.compose(path, residual) -> (positions, quats)              # exact inverse of split
+decompose.conditioning(path) -> (N, 3)                               # speed m/s, yaw rate rad/s, curvature 1/m
+   # path = Gaussian low-pass (-3 dB at 0.5 Hz) of the positions, ends extended by point reflection; heading = its horizontal
+   # tangent (held when slower than 0.1 m/s), no pitch or roll. residual = [offset (x right, y up, z back) in the path frame, rotation vector]
+dataset: Sequence(source, name, residual, cond), assign_splits (hash of the recording, so segments never straddle splits), Stats, WindowSampler
+   # WINDOW = 200 samples (4 s), CONTEXT = 50 (the 1 s shared with the previous window), HOP = 150
+model.MotionUNet / Diffusion   # x0-prediction, cosine schedule, DDIM; the loss ignores the given context
+train.train(...) -> history; train.load_checkpoint(path) -> (model, TrainConfig, Stats, sources)
+   # checkpoint = {"version": "dvsim-motion-1", config/stats/sources as JSON strings, state_dict}: loaded with weights_only=True
+style.DiffusionMotionStyle.load(path).apply(trajectory, rng) -> WaypointTrajectory   # the trajectory's positions walked by a person; 50 Hz
+registry.resolve_model({"model": id} | {"path": file}) -> Path       # ids are files of $DVSIM_DATA_DIR/motion/<id>.pt (no torch)
+evaluate.compare / evaluate_model / events_report.compare_event_statistics            # motion-eval
+
 # backend/saving
 class EventWriter:  path, width, height, compression="zstd", t_offset_us=0; write(events); close()   # /events/{t,x,y,p}
 class GroundTruthWriter:  path, config, compression="zstd", t_offset_us=0; write(gt: GTFrame); close()  # /ground_truth/{timestamp,flow_dt_us,depth,optical_flow,flow_valid,rgb_left,rgb_right,pose}
@@ -159,6 +181,7 @@
 GET    /api/scenes                      -> [{"id","name","files","created_at"}]
 DELETE /api/scenes/{id}
 POST   /api/trajectories  json {"name": str, "points": [{"timestamp":ms,"x","y","z","rx","ry","rz"} ...]}  -> 201 {"id","name","num_points","duration_s"}
+GET    /api/motion-models               -> [{"id","size_bytes"}]  trained human-motion models ($DVSIM_DATA_DIR/motion/<id>.pt)
 GET    /api/trajectories                -> [{"id","name","num_points","duration_s","source":"upload|headset"}]
 GET    /api/trajectories/{id}           -> {..., "points":[...]}
 DELETE /api/trajectories/{id}           -> 204 (409 if a queued/running job references it)
@@ -175,7 +198,9 @@
                       | {"type":"rectilinear","duration_s":float,"count":int,"seed":int|null}
                       | {"type":"spline","duration_s":float,"n_key_points":int,"count":int,"seed":int|null}
                       | {"type":"street","duration_s":float,"count":int,"seed":int|null,"speed":m/s,"eye_height":m,
-                         "min_clearance":m,"edge_margin":m}   (eye-height walks on open ground; scene +Y up, metres),
+                         "min_clearance":m,"edge_margin":m,"motion":{"model":id}|null}   (eye-height walks on open ground; scene +Y up,
+                         metres; `motion` plays the walk with a trained human-motion model: 404 unknown id, 501 when the server has no
+                         PyTorch; the job keeps the id, the worker gets the file),
           "config": <partial SimConfig dict>, "visualize": bool }             -> 202 {"id","status":"queued"}
 GET    /api/jobs                        -> [job]
 GET    /api/jobs/{id}                   -> job = {"id","status":"queued|running|done|failed|cancelled","error":str|null,
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -4,7 +4,7 @@
 
 [project]
 name = "dvsim"
-version = "0.2.0"
+version = "0.3.0"
 description = "DVSim: event-camera (DVS) simulator with ground-truth depth, optical flow and pose"
 readme = "README.md"
 requires-python = ">=3.9"
```


- [ ] **Step 2: Append the implementation notes to the spec**

Append this section at the end of `docs/superpowers/specs/2026-10-02-human-motion-design.md`:

```markdown

## Implementation notes

Differences from the text above found while planning and building (see the plan's "Spec notes"):

1. The residual's position axes are the path frame's own (x right, y up, z back).
2. `motion-import` writes a corpus file read by `motion-train` / `motion-eval`; `--up` and `--camera` override a dataset's frame.
3. Imports report a heading agreement and warn below 0.5 (wrong frame convention).
4. Windows whose mean path speed is below 0.2 m/s are not trained on.
5. The path low-pass extends the signal by point reflection at both ends.
6. `TrainConfig` stores the split settings; `motion-eval` recomputes the held-out recordings from them.
7. The "real vs real" reference compares even- with odd-numbered held-out recordings.
8. API: 404 unknown model id, 501 no torch, 422 for a motion preview over 300 s in total.
9. TUM-VI is read through the generic `csv` importer (layout `tumvi`), not the `tum` text importer.
```

- [ ] **Step 3: Full verification**


Run: `.claude/scratch/venv39/bin/python -m pytest -q -p no:cacheprovider`
Expected: `525 passed` (the 444 of Task 1 plus 81 new tests).

Run: `.claude/scratch/venv39/bin/python -m ruff check backend tests`
Expected: 20 findings, all in files this work did not touch (the same 20 as before).

Run: `cd frontend && node --test tests/ && npx vite build`
Expected: all pass, build succeeds.


- [ ] **Step 4: Run the documented workflow once on real data** (the part no unit test can prove)

The tests use synthetic walks. With real recordings, the quality question is open. On a Mac with MPS:

```bash
P=.claude/scratch/venv39/bin/python
$P -m backend motion-import tumrgbd tum /path/to/tum_rgbd_handheld --out corpus.npz
$P -m backend motion-import mine headset /path/to/your/headset/recordings --out corpus.npz
$P -m backend motion-train --corpus corpus.npz --name walk-v1 --holdout-source mine --steps 20000
$P -m backend motion-eval --corpus corpus.npz --model walk-v1 --scene /path/to/scene.json
```

Check, in this order:
1. Each import prints `heading agreement` close to 1.00 and no warning. Below 0.5, fix `--camera` / `--up` before anything else.
2. In the eval, `spectral distance generated` is far below `plain` and close to `real-other` for every speed bin; `step frequency` of generated matches real; `classifier accuracy` approaches 0.5.
3. In the event comparison, the human walk has more events per second than the plain walk.

**Do not judge quality on synthetic data.** The test walks are perfectly noise-free, so their high-frequency spectrum sits at the numerical floor and any sub-millimetre model error becomes a huge log-ratio. Measured while planning: a 1500-step toy model (1.1M parameters) and a 6000-step default-size model (4.1M parameters, validation loss 8x lower) both give `spectral distance generated` about 2.4 against `plain` 3.9 and `real-other` 0.12, a classifier accuracy of 1.00 and jerk 15-30x the real one. That shows the pipeline runs and that the model beats "no motion"; it says nothing about real recordings, which carry sensor noise at the level of the model's error.

If the real-data numbers do not meet the criterion in item 2, the levers are, in order: more steps; more DDIM steps (`motion-eval --sample-steps 100`); a velocity / smoothness term in `Diffusion.loss`; a larger `--base`. Record the final numbers in `CHANGELOG.md`.

- [ ] **Step 5: Commit and finish**

```bash
git add README.md docs CHANGELOG.md pyproject.toml
git commit -m "docs: human-like camera motion; v0.3.0"
```

Then use superpowers:finishing-a-development-branch.
