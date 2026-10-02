"""Dataloader mode: numpy representations, the framework-agnostic ClipStream and the torch dataset (torch optional)."""
from __future__ import annotations

import numpy as np
import pytest

from backend.dataloader import (
    bin_end_times_us,
    events_to_count_frames,
    events_to_voxel_grid,
    rebase_events,
)
from backend.types import EVENT_DTYPE

H, W = 6, 8


def _ev(rows):
    """Events from (t, x, y, p) rows."""
    return np.array(rows, dtype=EVENT_DTYPE) if rows else np.empty(0, dtype=EVENT_DTYPE)


# ---------------------------------------------------------------- representations (numpy only)

def test_count_frames_bin_assignment_on_edges():
    # window (1000, 5000], 4 bins of 1000 us: edges at 2000, 3000, 4000, 5000; an edge belongs to the earlier bin
    rows = [(1000, 0, 0, 1),                     # t == t0: previous window, ignored
            (1001, 1, 0, 1), (2000, 2, 0, 1),    # bin 0
            (2001, 3, 0, 1), (3000, 4, 0, 1),    # bin 1
            (3001, 5, 0, 1),                     # bin 2
            (4001, 6, 0, 1), (5000, 7, 0, 1),    # bin 3
            (5001, 0, 1, 1)]                     # > t1: next window, ignored
    frames = events_to_count_frames(_ev(rows), 4, 1000, 5000, H, W)
    assert frames.shape == (4, 2, H, W) and frames.dtype == np.float32
    where = {(b, x) for b, x in zip(*np.nonzero(frames[:, 1, 0, :])[:2])}
    assert where == {(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (3, 7)}
    assert frames.sum() == 7                     # every in-window event lands in exactly one bin


def test_count_frames_uneven_bin_width_matches_end_times():
    t0, t1, n = 0, 10_007, 3                     # bin ends floor(k * 10007 / 3) = 3335, 6671, 10007
    ends = bin_end_times_us(n, t0, t1)
    assert ends.tolist() == [3335, 6671, 10007]
    rows = [(e + d, 0, 0, 1) for e in ends for d in (0, 1) if e + d <= t1]
    frames = events_to_count_frames(_ev(rows), n, t0, t1, H, W)
    assert frames[:, 1, 0, 0].tolist() == [1, 2, 2]  # ends[0] in 0; ends[0]+1, ends[1] in 1; ends[1]+1, ends[2] in 2


def test_count_frames_empty_and_single_bin():
    frames = events_to_count_frames(_ev([]), 5, 0, 5000, H, W)
    assert frames.shape == (5, 2, H, W) and not frames.any()
    assert events_to_count_frames(_ev([]), 5, 0, 5000, H, W, split_polarity=False).shape == (5, 1, H, W)
    assert events_to_voxel_grid(_ev([]), 5, 0, 5000, H, W).shape == (5, H, W)
    one = events_to_count_frames(_ev([(1, 0, 0, 0), (5000, 0, 0, 0)]), 1, 0, 5000, H, W)
    assert one.shape == (1, 2, H, W) and one[0, 0, 0, 0] == 2


def test_count_frames_polarity_split_signed_binary():
    rows = [(10, 2, 3, 1), (20, 2, 3, 1), (30, 2, 3, 0),     # pixel (x=2,y=3): 2 ON, 1 OFF
            (40, 5, 1, 0), (50, 5, 1, 0),                    # pixel (5,1): 2 OFF
            (60, 0, 0, 1)]                                   # pixel (0,0): 1 ON
    ev = _ev(rows)
    split = events_to_count_frames(ev, 1, 0, 100, H, W)
    assert split[0, 1, 3, 2] == 2 and split[0, 0, 3, 2] == 1 and split[0, 0, 1, 5] == 2 and split[0, 1, 0, 0] == 1
    assert split.sum() == 6
    signed = events_to_count_frames(ev, 1, 0, 100, H, W, split_polarity=False)
    assert signed.shape == (1, 1, H, W)
    assert signed[0, 0, 3, 2] == 1 and signed[0, 0, 1, 5] == -2 and signed[0, 0, 0, 0] == 1
    spikes = events_to_count_frames(ev, 1, 0, 100, H, W, binary=True)
    assert spikes.max() == 1 and spikes.sum() == 4 and spikes.dtype == np.float32
    sgn = events_to_count_frames(ev, 1, 0, 100, H, W, split_polarity=False, binary=True)
    assert sgn[0, 0, 3, 2] == 1 and sgn[0, 0, 1, 5] == -1 and np.abs(sgn).sum() == 3


def test_count_frames_random_conservation_and_per_bin_counts():
    rng = np.random.default_rng(0)
    n = 2000
    ev = _ev(list(zip(np.sort(rng.integers(1, 10_000, n)), rng.integers(0, W, n), rng.integers(0, H, n),
                      rng.integers(0, 2, n))))
    frames = events_to_count_frames(ev, 7, 0, 10_000, H, W)
    assert frames.sum() == n
    ref = np.zeros((7, 2, H, W))
    for t, x, y, p in ev:
        ref[-(-int(t) * 7 // 10_000) - 1, p, y, x] += 1
    np.testing.assert_array_equal(frames, ref)


def test_voxel_grid_interpolation_weights():
    # T=5 over (0, 4000]: bin k centred on 1000 * k. Event at t=2500 -> t*=2.5: half in bins 2 and 3.
    ev = _ev([(2500, 1, 2, 1), (1000, 3, 3, 0), (3250, 4, 4, 1)])
    vox = events_to_voxel_grid(ev, 5, 0, 4000, H, W)
    assert vox.shape == (5, H, W) and vox.dtype == np.float32
    assert vox[2, 2, 1] == pytest.approx(0.5) and vox[3, 2, 1] == pytest.approx(0.5)
    assert vox[1, 3, 3] == pytest.approx(-1.0) and vox[0, 3, 3] == 0 and vox[2, 3, 3] == 0   # exactly on a centre
    assert vox[3, 4, 4] == pytest.approx(0.75) and vox[4, 4, 4] == pytest.approx(0.25)
    assert vox.sum() == pytest.approx(1.0)                                                # 2 ON - 1 OFF


def test_voxel_grid_weights_sum_to_signed_count_and_edges():
    rng = np.random.default_rng(1)
    n = 3000
    t = np.sort(rng.integers(1, 20_001, n))
    ev = _ev(list(zip(t, rng.integers(0, W, n), rng.integers(0, H, n), rng.integers(0, 2, n))))
    vox = events_to_voxel_grid(ev, 6, 0, 20_000, H, W)
    signed = int((ev["p"] == 1).sum() - (ev["p"] == 0).sum())
    assert vox.sum() == pytest.approx(signed, abs=1e-2)
    # per pixel the bins add up to the signed count of that pixel
    ref = np.zeros((H, W))
    np.add.at(ref, (ev["y"], ev["x"]), np.where(ev["p"] == 1, 1.0, -1.0))
    np.testing.assert_allclose(vox.sum(axis=0), ref, atol=1e-4)
    last = events_to_voxel_grid(_ev([(20_000, 0, 0, 1)]), 6, 0, 20_000, H, W)   # t == t1 -> all weight on the last bin
    assert last[5, 0, 0] == 1 and last.sum() == 1
    single = events_to_voxel_grid(_ev([(7, 0, 0, 1), (9, 0, 0, 0)]), 1, 0, 10, H, W)
    assert single.shape == (1, H, W) and single.sum() == 0


def test_rebase_and_input_validation():
    ev = _ev([(5000, 1, 1, 1), (6000, 2, 2, 0)])
    out = rebase_events(ev, 4000)
    assert out.dtype == EVENT_DTYPE and out["t"].tolist() == [1000, 2000] and ev["t"].tolist() == [5000, 6000]
    with pytest.raises(ValueError):
        events_to_count_frames(_ev([(1, W, 0, 1)]), 1, 0, 10, H, W)            # x out of the sensor
    with pytest.raises(ValueError):
        events_to_count_frames(_ev([]), 1, 10, 10, H, W)                       # empty window
    with pytest.raises(ValueError):
        events_to_voxel_grid(_ev([]), 0, 0, 10, H, W)


# ---------------------------------------------------------------- ClipStream (needs a GL context)

import gc  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

from backend.config import SimConfig  # noqa: E402
from backend.dataloader import ClipStream  # noqa: E402
from backend.scene import load_scene  # noqa: E402
from backend.simulator import Simulator  # noqa: E402
from backend.trajectories import WaypointTrajectory  # noqa: E402

gl = pytest.mark.usefixtures("gl_context")
DT = 10_000
STEPS = 10
SPAN = STEPS * DT


def _cfg(**kw):
    d = {"camera": {"width": 32, "height": 24}, "dt_us": DT}
    d.update(kw)
    return SimConfig.from_dict(d)


@pytest.fixture(scope="module")
def scene(demo_scene):
    return load_scene(demo_scene.scene_spec)


@pytest.fixture(scope="module")
def scene_b(demo_scene):
    return load_scene({"objects": [{"path": str(demo_scene.room_obj), "texture": str(demo_scene.room_texture)}]})


@pytest.fixture(scope="module")
def records(demo_scene):
    return json.loads(demo_scene.trajectory_json.read_text())


def _traj(records, seconds):
    return WaypointTrajectory.from_json([r for r in records if r["timestamp"] <= seconds * 1000.0 + 1e-6])


def _take(stream, n):
    with stream:
        return list(itertools.islice(stream, n))


def _equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return np.array_equal(a, b)


@gl
@pytest.mark.parametrize("kw,shape", [
    ({}, (STEPS, 2, 24, 32)),
    ({"binary": True}, (STEPS, 2, 24, 32)),
    ({"split_polarity": False}, (STEPS, 1, 24, 32)),
    ({"num_bins": 4}, (4, 2, 24, 32)),
    ({"representation": "voxel", "num_bins": 5}, (5, 24, 32)),
])
def test_clip_shapes_dtypes_and_representations(scene, records, kw, shape):
    stream = ClipStream(scene, _cfg(), _traj(records, 0.3), clip_steps=STEPS, **kw)
    clips = _take(stream, 2)
    assert len(clips) == 2
    for clip in clips:
        assert clip["events"].shape == shape and clip["events"].dtype == np.float32
        assert clip["t_us"].shape == (shape[0],) and clip["t_us"][-1] == SPAN and clip["t_us"].dtype == np.int64
        assert set(clip["gt"]) == {"depth", "flow", "flow_valid"}
        assert "events_right" not in clip
    assert clips[0]["events"].any()
    if kw.get("binary"):
        assert set(np.unique(clips[0]["events"])) <= {0.0, 1.0}


@gl
def test_raw_events_and_gt_layout_match_a_plain_simulator_run(scene, records):
    traj = _traj(records, 0.55)                                   # 56 steps: 5 clips + 5 leftover steps (dropped)
    gt = ("rgb", "depth", "flow", "pose")
    stream = ClipStream(scene, _cfg(), traj, clip_steps=STEPS, representation="raw", gt=gt, num_sequences=1)
    clips = _take(stream, 100)
    assert len(clips) == 5                                        # the trailing partial clip is dropped
    with Simulator(scene, stream.config) as sim:
        steps = list(sim.run(traj))
    assert len(steps) == 56 and len(steps[0].events_left) == 0
    for j, clip in enumerate(clips):
        window = steps[j * STEPS + 1:(j + 1) * STEPS + 1]         # clip j = steps jS+1 .. (j+1)S
        ev = clip["events"]
        assert ev.dtype == EVENT_DTYPE and (np.diff(ev["t"]) >= 0).all()
        assert (ev["t"] > 0).all() and (ev["t"] <= SPAN).all()
        expected = np.concatenate([s.events_left for s in window])
        np.testing.assert_array_equal(ev["t"] + clip["meta"]["start_us"], expected["t"])   # contiguous, exact
        for f in "xyp":
            np.testing.assert_array_equal(ev[f], expected[f])
        assert clip["gt_t_us"].tolist() == [DT * (i + 1) for i in range(STEPS)]
        g = clip["gt"]
        assert g["depth"].shape == (STEPS, 1, 24, 32) and g["flow"].shape == (STEPS, 2, 24, 32)
        assert g["flow_valid"].shape == (STEPS, 1, 24, 32) and g["flow_valid"].dtype == np.uint8
        assert g["rgb"].shape == (STEPS, 3, 24, 32) and g["rgb"].dtype == np.uint8
        assert g["pose"].shape == (STEPS, 7) and g["pose"].dtype == np.float64
        for i, s in enumerate(window):
            np.testing.assert_array_equal(g["depth"][i, 0], s.gt.depth)
            np.testing.assert_array_equal(g["flow"][i, 0], s.gt.flow[..., 0])
            np.testing.assert_array_equal(g["flow"][i, 1], s.gt.flow[..., 1])
            np.testing.assert_array_equal(g["flow_valid"][i, 0], s.gt.flow_valid)
            np.testing.assert_array_equal(g["rgb"][i], np.moveaxis(s.gt.rgb_left, -1, 0))
            np.testing.assert_array_equal(g["pose"][i], s.gt.pose)
    first = np.concatenate([s.events_left for s in steps[1:]])
    assert len(first) > 0 and sum(len(c["events"]) for c in clips) == sum(len(s.events_left) for s in steps[1:51])


@gl
def test_count_frames_agree_with_raw_events(scene, records):
    traj = _traj(records, 0.25)
    raw = _take(ClipStream(scene, _cfg(), traj, representation="raw", num_bins=4), 1)[0]
    frames = _take(ClipStream(scene, _cfg(), traj, num_bins=4), 1)[0]
    ref = events_to_count_frames(raw["events"], 4, 0, SPAN, 24, 32)
    np.testing.assert_array_equal(frames["events"], ref)
    assert frames["events"].sum() == len(raw["events"])
    vox = _take(ClipStream(scene, _cfg(), traj, representation="voxel", num_bins=4), 1)[0]
    np.testing.assert_allclose(vox["events"], events_to_voxel_grid(raw["events"], 4, 0, SPAN, 24, 32))


@gl
def test_gt_subset_and_gt_every(scene, records):
    traj = _traj(records, 0.25)
    none = _take(ClipStream(scene, _cfg(), traj, gt=()), 1)[0]
    assert none["gt"] == {} and none["gt_t_us"].shape == (0,)
    pose = _take(ClipStream(scene, _cfg(), traj, gt=("pose",)), 1)[0]
    assert set(pose["gt"]) == {"pose"}
    clips = _take(ClipStream(scene, _cfg(gt_every=4), traj), 3)
    assert [c["gt"]["depth"].shape[0] for c in clips] == [2, 3, 2]          # steps 4,8 | 12,16,20 | 24,28
    assert clips[0]["gt_t_us"].tolist() == [4 * DT, 8 * DT] and clips[1]["gt_t_us"].tolist() == [2 * DT, 6 * DT, 10 * DT]
    sparse = _take(ClipStream(scene, _cfg(gt_every=15), traj), 2)
    assert [c["gt"]["flow"].shape for c in sparse] == [(0, 2, 24, 32), (1, 2, 24, 32)]
    assert sparse[0]["gt"]["depth"].dtype == np.float32 and sparse[0]["gt"]["flow_valid"].shape == (0, 1, 24, 32)


@gl
def test_stereo_events_right(scene, records):
    traj = _traj(records, 0.25)
    cfg = _cfg(camera={"width": 32, "height": 24, "stereo": True}, gt=())
    clip = _take(ClipStream(scene, cfg, traj, representation="raw", gt=("rgb",)), 1)[0]
    with Simulator(scene, cfg) as sim:
        steps = list(itertools.islice(sim.run(traj), STEPS + 1))
    expected = np.concatenate([s.events_right for s in steps[1:]])
    np.testing.assert_array_equal(clip["events_right"]["t"], expected["t"])
    np.testing.assert_array_equal(clip["events_right"]["x"], expected["x"])
    assert len(expected) > 0 and not np.array_equal(clip["events"]["x"], clip["events_right"]["x"])
    assert clip["gt"]["rgb"].shape == (STEPS, 3, 24, 32) and clip["gt"]["rgb_right"].shape == (STEPS, 3, 24, 32)
    counts = _take(ClipStream(scene, cfg, traj), 1)[0]
    assert counts["events_right"].shape == counts["events"].shape == (STEPS, 2, 24, 32)


def _generated(scene, **kw):
    kw.setdefault("trajectory_duration_s", 0.35)               # 36 steps: 3 clips of 10 steps per sequence
    return ClipStream(scene, _cfg(), **kw)


@gl
@pytest.mark.parametrize("generator", ["spline", "rectilinear"])
def test_generated_trajectories_and_is_first_clip(scene, generator):
    clips = _take(_generated(scene, generator=generator, num_sequences=2), 100)
    assert len(clips) == 6                                     # 2 sequences x 3 clips, partial tail dropped
    meta = [c["meta"] for c in clips]
    assert [m["sequence_id"] for m in meta] == [0, 0, 0, 1, 1, 1]
    assert [m["clip_index"] for m in meta] == [0, 1, 2, 0, 1, 2]
    assert [m["is_first_clip"] for m in meta] == [True, False, False, True, False, False]
    assert all(type(m["is_first_clip"]) is bool and type(m["start_us"]) is int for m in meta)
    for a, b in zip(meta, meta[1:]):
        if b["clip_index"]:
            assert b["start_us"] == a["start_us"] + SPAN       # clips of a sequence are back to back
    assert sum(float(c["events"].sum()) for c in clips) > 0
    assert all(c["events"].shape == (STEPS, 2, 24, 32) for c in clips)


@gl
def test_street_generator_walks_the_floor(scene):
    opts = {"eye_height": 1.2, "min_clearance": 0.5, "edge_margin": 0}
    stream = _generated(scene, generator="street", street_options=opts, num_sequences=2, gt=("pose",))
    clips = _take(stream, 100)
    assert [c["meta"]["sequence_id"] for c in clips] == [0, 0, 0, 1, 1, 1]
    poses = np.concatenate([c["gt"]["pose"] for c in clips])
    height = poses[:, 1]                                       # x y z qx qy qz qw; the floor is y = 0
    assert (height >= 1.2 - 1e-3).all() and (height <= 1.2 + 0.3).all()        # may step onto low (< 0.3 m) objects
    assert sum(float(c["events"].sum()) for c in clips) > 0
    with pytest.raises(TypeError):
        _take(_generated(scene, generator="street", street_options={"bogus": 1}, num_sequences=1), 1)


@gl
def test_determinism_and_seed_sensitivity(scene):
    cfg = _cfg(events={"sigma_c": 0.3})
    kw = dict(trajectory_duration_s=0.25, num_sequences=2)
    a = _take(ClipStream(scene, cfg, seed=5, **kw), 100)
    b = _take(ClipStream(scene, cfg, seed=5, **kw), 100)
    c = _take(ClipStream(scene, cfg, seed=6, **kw), 100)
    assert len(a) == len(b) == len(c) == 4
    assert all(_equal(x, y) for x, y in zip(a, b))                      # same seed: identical clips
    assert not any(np.array_equal(x["events"], y["events"]) for x, y in zip(a, c))
    assert not np.array_equal(a[0]["gt"]["depth"], c[0]["gt"]["depth"])   # different trajectory as well


@gl
def test_event_model_seed_differs_between_sequences_but_is_reproducible(scene, records):
    traj = _traj(records, 0.15)
    kw = dict(num_sequences=2, seed=3)
    noisy = _take(ClipStream(scene, _cfg(events={"sigma_c": 0.3}), [traj], **kw), 100)
    again = _take(ClipStream(scene, _cfg(events={"sigma_c": 0.3}), [traj], **kw), 100)
    assert [c["meta"]["sequence_id"] for c in noisy] == [0, 1]           # same trajectory, two sequences
    assert not np.array_equal(noisy[0]["events"], noisy[1]["events"])   # different threshold mismatch draws
    assert _equal(noisy[0], again[0]) and _equal(noisy[1], again[1])
    ideal = _take(ClipStream(scene, _cfg(), [traj], **kw), 100)
    assert np.array_equal(ideal[0]["events"], ideal[1]["events"])       # control: no mismatch, no seed dependence


@gl
def test_multi_scene_sampling_and_lru(scene, scene_b):
    stream = _generated([scene, scene_b], trajectory_duration_s=0.15, num_sequences=12, max_cached_scenes=1)
    with stream:
        clips, alive = [], []
        for clip in stream:
            clips.append(clip)
            alive.append(len(stream._slots))
    seen = [c["meta"]["scene_index"] for c in clips]
    assert set(seen) == {0, 1}                                           # both scenes are sampled ...
    assert max(alive) == 1                                               # ... and only one simulator stays alive
    assert len(seen) == 12
    again = _take(_generated([scene, scene_b], trajectory_duration_s=0.15, num_sequences=12, max_cached_scenes=2), 100)
    assert all(_equal(a, b) for a, b in zip(clips, again))               # evicting / reloading a scene changes nothing
    stream2 = _generated([scene, scene_b], trajectory_duration_s=0.15, num_sequences=12, max_cached_scenes=2)
    with stream2:
        list(stream2)
        assert len(stream2._slots) == 2 and all(s.sim is not None and s.grid is not None for s in stream2._slots.values())


@gl
def test_scene_specs_are_loaded_lazily_from_paths_and_dicts(demo_scene):
    specs = [str(demo_scene.scene_spec), {"objects": [{"path": str(demo_scene.room_obj),
                                                       "texture": str(demo_scene.room_texture)}]}]
    stream = ClipStream(specs, _cfg(), trajectory_duration_s=0.15, num_sequences=4, seed=1)
    assert not stream._slots                                             # nothing is loaded before iteration
    clips = _take(stream, 100)
    assert len(clips) == 4 and all(c["events"].shape == (STEPS, 2, 24, 32) for c in clips)


@gl
def test_given_trajectories_are_cycled(scene, records, demo_scene):
    a = _traj(records, 0.15)
    b = WaypointTrajectory.from_json([r for r in records if 100.0 <= r["timestamp"] <= 200.0])
    clips = _take(ClipStream(scene, _cfg(), [a, b], num_sequences=3, seed=0, gt=("pose", "depth")), 100)
    assert [c["meta"]["sequence_id"] for c in clips] == [0, 1, 2]
    assert _equal(clips[0], {**clips[2], "meta": clips[0]["meta"]})      # third sequence replays the first trajectory
    assert not np.array_equal(clips[0]["gt"]["pose"], clips[1]["gt"]["pose"])
    spec = {"type": "file", "path": str(demo_scene.trajectory_json), "origin": [0, 0, 0]}
    from_spec = _take(ClipStream(scene, _cfg(), spec, num_sequences=1, gt=("pose",)), 1)[0]
    np.testing.assert_allclose(from_spec["gt"]["pose"][0, :3], [-2.1, 1.5, -0.75], atol=0.05)
    gen_spec = {"type": "spline", "duration_s": 0.15, "count": 2, "n_key_points": 4, "seed": None}
    g1 = _take(ClipStream(scene, _cfg(), gen_spec, num_sequences=2, seed=9, gt=("pose",)), 100)
    g2 = _take(ClipStream(scene, _cfg(), gen_spec, num_sequences=2, seed=9, gt=("pose",)), 100)
    assert len(g1) == 2 and all(_equal(x, y) for x, y in zip(g1, g2))    # spec seeds derive from the stream seed
    assert not np.array_equal(g1[0]["gt"]["pose"], g1[1]["gt"]["pose"])


def _alive(ctx):
    try:
        ctx.buffer(b"abcd").release()
    except AttributeError:                                               # a released moderngl context is invalid
        return False
    return True


@gl
def test_close_releases_simulators_and_context(scene):
    stream = _generated(scene)
    next(stream)
    sim = stream._slots[0].sim
    ctx = stream._ctx
    assert not sim.renderer._released and _alive(ctx)
    stream.close()
    assert sim.renderer._released and not _alive(ctx) and not stream._slots
    with pytest.raises(StopIteration):
        next(stream)
    stream.close()                                                       # idempotent

    from backend.graphical_computation.utils.gl_context import create_gl_context

    own = create_gl_context()                                            # a caller-owned context is left alone
    try:
        with _generated(scene, context=own) as shared:
            next(shared)
            sim2 = shared._slots[0].sim
        assert sim2.renderer._released and _alive(own)
    finally:
        own.release()

    dropped = _generated(scene)
    next(dropped)
    sim3 = dropped._slots[0].sim
    del dropped
    gc.collect()
    assert sim3.renderer._released                                       # garbage collection releases the GL objects


def test_constructor_validation(demo_scene):
    spec = str(demo_scene.scene_spec)
    with pytest.raises(ValueError, match="representation"):
        ClipStream(spec, representation="dense")
    with pytest.raises(ValueError, match="ground-truth"):
        ClipStream(spec, gt=("normals",))
    with pytest.raises(ValueError, match="generator"):
        ClipStream(spec, generator="brownian")
    with pytest.raises(ValueError, match="shorter than one clip"):
        ClipStream(spec, _cfg(), trajectory_duration_s=0.05)
    with pytest.raises(ValueError):
        ClipStream([])


@gl
def test_given_trajectory_shorter_than_a_clip_raises(scene, records):
    with pytest.raises(ValueError, match="shorter than one clip"):
        _take(ClipStream(scene, _cfg(), _traj(records, 0.05)), 1)


# ---------------------------------------------------------------- torch dataset (optional dependency)

import pickle  # noqa: E402
import sys  # noqa: E402


def test_torch_names_give_a_clear_error_without_torch(monkeypatch):
    import backend.dataloader as pkg

    monkeypatch.setitem(sys.modules, "torch", None)                      # `import torch` now raises ImportError
    monkeypatch.delitem(sys.modules, "backend.dataloader.dataset", raising=False)
    monkeypatch.delattr(pkg, "dataset", raising=False)
    for name in ("DVSimIterableDataset", "make_dataloader", "collate_raw"):
        with pytest.raises(ImportError, match=r"dvsim\[torch\]"):
            getattr(pkg, name)
    with pytest.raises(AttributeError):
        pkg.__getattr__("not_a_name")
    assert pkg.ClipStream is ClipStream                                  # the numpy side never needs torch


def _dataset(scene, records=None, **kw):
    from backend.dataloader import DVSimIterableDataset

    traj = [_traj(records, 0.55)] if records is not None else None
    kw.setdefault("trajectory_duration_s", 0.25)
    if traj is not None:
        kw.pop("trajectory_duration_s")
    return DVSimIterableDataset(scene, _cfg(), traj, **kw)


@gl
def test_dataset_yields_tensors(scene, records):
    torch = pytest.importorskip("torch")
    ds = _dataset(scene, records, samples_per_epoch=3, gt=("depth", "flow", "pose"))
    with _closing(ds):
        clips = list(ds)
    assert len(ds) == 3 and len(clips) == 3
    c = clips[0]
    assert c["events"].dtype == torch.float32 and c["events"].shape == (STEPS, 2, 24, 32)
    assert c["t_us"].dtype == torch.int64 and c["gt_t_us"].shape == (STEPS,)
    assert c["gt"]["depth"].shape == (STEPS, 1, 24, 32) and c["gt"]["flow"].shape == (STEPS, 2, 24, 32)
    assert c["gt"]["flow_valid"].dtype == torch.uint8 and c["gt"]["pose"].dtype == torch.float64
    assert c["meta"] == {"scene_index": 0, "sequence_id": 0, "clip_index": 0, "start_us": 0, "is_first_clip": True}
    assert [x["meta"]["clip_index"] for x in clips] == [0, 1, 2]
    reference = _take(ClipStream(scene, _cfg(), _traj(records, 0.55), gt=("depth", "flow", "pose")), 3)
    for got, want in zip(clips, reference):
        np.testing.assert_array_equal(got["events"].numpy(), want["events"])
        np.testing.assert_array_equal(got["gt"]["flow"].numpy(), want["gt"]["flow"])


class _closing:
    def __init__(self, ds):
        self.ds = ds

    def __enter__(self):
        return self.ds

    def __exit__(self, *exc):
        self.ds.close()


@gl
def test_dataloader_in_process_batches(scene, records):
    torch = pytest.importorskip("torch")
    from backend.dataloader import make_dataloader

    ds = _dataset(scene, records, samples_per_epoch=5, gt=("depth", "flow"))
    loader = make_dataloader(ds, batch_size=2)
    assert len(loader) == 3
    with _closing(ds):
        batches = list(loader)
    assert [b["events"].shape[0] for b in batches] == [2, 2, 1]
    b = batches[0]
    assert b["events"].shape == (2, STEPS, 2, 24, 32) and b["gt"]["depth"].shape == (2, STEPS, 1, 24, 32)
    assert b["t_us"].shape == (2, STEPS) and b["gt_t_us"].shape == (2, STEPS)
    assert b["meta"]["is_first_clip"].dtype == torch.bool and b["meta"]["is_first_clip"].tolist() == [True, False]
    assert b["meta"]["clip_index"].tolist() == [0, 1] and b["meta"]["start_us"].tolist() == [0, SPAN]


@gl
def test_raw_representation_is_collated_ragged(scene, records):
    torch = pytest.importorskip("torch")
    from backend.dataloader import make_dataloader

    ds = _dataset(scene, records, samples_per_epoch=2, representation="raw", gt=())
    with _closing(ds):
        clips = list(ds)
    ev = clips[0]["events"]
    assert set(ev) == {"t", "x", "y", "p"} and ev["t"].dtype == torch.int64 and ev["x"].dtype == torch.int32
    assert ev["p"].dtype == torch.uint8 and len(ev["t"]) > 0
    with _closing(ds):
        (batch,) = list(make_dataloader(ds, batch_size=2))
    merged = batch["events"]
    n0, n1 = len(clips[0]["events"]["t"]), len(clips[1]["events"]["t"])
    assert len(merged["t"]) == n0 + n1 == len(merged["batch"])
    assert merged["batch"].tolist() == [0] * n0 + [1] * n1
    assert batch["t_us"].shape == (2, STEPS)


def test_dataset_len_and_validation(demo_scene):
    pytest.importorskip("torch")
    from backend.dataloader import DVSimIterableDataset

    spec = str(demo_scene.scene_spec)
    assert len(DVSimIterableDataset(spec, _cfg(), samples_per_epoch=7)) == 7
    with pytest.raises(TypeError):
        len(DVSimIterableDataset(spec, _cfg()))
    with pytest.raises(ValueError, match="representation"):
        DVSimIterableDataset(spec, _cfg(), representation="dense")     # bad arguments fail at construction
    with pytest.raises(ValueError, match="context"):
        DVSimIterableDataset(spec, _cfg(), context=object())


@gl
def test_epochs_seeds_and_pickling(scene):
    torch = pytest.importorskip("torch")
    kw = dict(samples_per_epoch=2, seed=1)
    ds = _dataset(scene, **kw)
    with _closing(ds):
        e0, e1 = list(ds), list(ds)                                    # every iter() is a new epoch
        ds.set_epoch(0)
        again = list(ds)
    same = lambda a, b: all(torch.equal(x["events"], y["events"]) for x, y in zip(a, b))
    assert not same(e0, e1) and same(e0, again)
    fresh = _dataset(scene, **kw)
    with _closing(fresh):
        assert same(e0, list(fresh))                                   # a rerun with the same seed reproduces
    other = _dataset(scene, samples_per_epoch=2, seed=2)
    with _closing(other):
        assert not same(e0, list(other))
    ds2 = _dataset(scene, **kw)
    with _closing(ds2):
        list(ds2)                                                      # a live stream exists ...
        clone = pickle.loads(pickle.dumps(ds2))                        # ... but is not pickled
        assert clone._stream is None and clone.epoch == 1 and len(clone) == 2
        clone.set_epoch(0)
        with _closing(clone):
            assert same(e0, list(clone))


@gl
def test_seed_none_follows_torch_seed(scene):
    torch = pytest.importorskip("torch")
    ds = _dataset(scene, samples_per_epoch=1, seed=None)
    with _closing(ds):
        torch.manual_seed(11)
        a = next(iter(ds))
        ds.set_epoch(0)
        torch.manual_seed(11)
        b = next(iter(ds))
        ds.set_epoch(0)
        torch.manual_seed(12)
        c = next(iter(ds))
    assert torch.equal(a["events"], b["events"]) and not torch.equal(a["events"], c["events"])


@pytest.mark.slow
def test_spawned_workers_stream_distinct_reproducible_clips(demo_scene, gl_context):
    torch = pytest.importorskip("torch")
    from backend.dataloader import DVSimIterableDataset, make_dataloader

    def make():
        ds = DVSimIterableDataset(str(demo_scene.scene_spec), _cfg(), samples_per_epoch=8, seed=4,
                                  trajectory_duration_s=0.25, gt=("depth",))
        return make_dataloader(ds, batch_size=2, num_workers=2)

    loader = make()
    assert loader.persistent_workers and loader.multiprocessing_context.get_start_method() == "spawn"
    assert len(loader) == 4
    first, second = list(loader), list(loader)                         # two epochs on the same persistent workers
    assert len(first) == 4 and all(b["events"].shape == (2, STEPS, 2, 24, 32) for b in first)
    assert first[0]["gt"]["depth"].shape == (2, STEPS, 1, 24, 32)
    assert not torch.equal(first[0]["events"], first[1]["events"])     # batch 0 came from worker 0, batch 1 from worker 1
    assert not torch.equal(first[0]["events"], second[0]["events"])    # persistent workers move on to a new epoch
    del loader
    rerun = list(make())
    assert all(torch.equal(a["events"], b["events"]) for a, b in zip(first, rerun))
    assert all(torch.equal(a["gt"]["depth"], b["gt"]["depth"]) for a, b in zip(first, rerun))
