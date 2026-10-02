"""Output writers: events / ground-truth HDF5, TUM export, SequenceOutput and the streaming video visualiser."""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401  (registers the zstd filter for reading)
import numpy as np
import pytest

from backend.config import GT_FIELDS, CameraConfig, SimConfig
from backend.pose import Pose
from backend.saving import EventWriter, GroundTruthWriter, SequenceOutput, write_tum
from backend.types import EVENT_DTYPE, GTFrame, StepOutput
from backend.vizualisation import Visualizer
from backend.vizualisation.visualizer import _COLORWHEEL, depth_frame, events_frame, flow_frame, flow_to_rgb

REPO = Path(__file__).resolve().parent.parent
COMPRESSIONS = ("zstd", "gzip", "none")
ZSTD_ID, DEFLATE_ID = 32015, 1


def make_events(rng, n, w, h, t0=0):
    ev = np.empty(n, EVENT_DTYPE)
    ev["t"] = t0 + np.cumsum(rng.integers(0, 4, n))
    ev["x"] = rng.integers(0, w, n)
    ev["y"] = rng.integers(0, h, n)
    ev["p"] = rng.integers(0, 2, n)
    return ev


def filter_ids(ds):
    plist = ds.id.get_create_plist()
    return [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]


def cfg_for(fields, stereo, w=16, h=12, **kw):
    return SimConfig(camera=CameraConfig(width=w, height=h, stereo=stereo), gt=tuple(fields), **kw)


def make_gt(rng, cfg, t_us, pose=True):
    cam = cfg.camera
    hw = (cam.height, cam.width)
    return GTFrame(
        t_us=t_us,
        pose=rng.normal(size=7) if pose else None,
        rgb_left=rng.integers(0, 256, hw + (3,), dtype=np.uint8),
        rgb_right=rng.integers(0, 256, hw + (3,), dtype=np.uint8) if cam.stereo else None,
        depth=rng.random(hw, dtype=np.float32) * 5,
        flow=rng.normal(size=hw + (2,)).astype(np.float32),
        flow_valid=rng.integers(0, 2, hw, dtype=np.uint8),
        flow_dt_us=cfg.dt_us * cfg.gt_every,
    )


def expected_datasets(fields, stereo):
    names = ["timestamp"]
    if "rgb" in fields:
        names += ["rgb_left"] + (["rgb_right"] if stereo else [])
    if "depth" in fields:
        names.append("depth")
    if "flow" in fields:
        names += ["optical_flow", "flow_valid", "flow_dt_us"]
    if "pose" in fields:
        names.append("pose")
    return names


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    info = dict(count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return frames, info


# ----------------------------------------------------------------------------------------------- events


def test_events_round_trip_multi_chunk(tmp_path):
    rng = np.random.default_rng(0)
    w, h = 346, 260
    events = make_events(rng, 1_300_000, w, h, t0=1000)
    path = tmp_path / "e.h5"
    with EventWriter(path, w, h) as writer:
        pos = 0
        while pos < len(events):
            n = int(rng.integers(1, 5000))
            writer.write(events[pos:pos + n])
            pos += n
        assert writer.num_events == len(events)
    with h5py.File(path, "r") as f:
        g = f["events"]
        assert (g.attrs["width"], g.attrs["height"], g.attrs["time_unit"]) == (w, h, "us")
        assert g.attrs["num_events"] == len(events)
        for name, dtype in (("t", np.uint32), ("x", np.uint16), ("y", np.uint16), ("p", np.uint8)):
            assert g[name].dtype == dtype
            assert g[name].chunks == (1 << 18,)
            assert ZSTD_ID in filter_ids(g[name])
            assert np.array_equal(g[name][:], events[name].astype(dtype))
        t = g["t"][:]
        assert np.all(t[1:] >= t[:-1])


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_events_compression_modes(tmp_path, compression):
    rng = np.random.default_rng(1)
    events = make_events(rng, 50_000, 64, 48)
    path = tmp_path / "e.h5"
    with EventWriter(path, 64, 48, compression=compression) as writer:
        writer.write(events)
    with h5py.File(path, "r") as f:
        ids = filter_ids(f["events/x"])
        assert {"zstd": ZSTD_ID in ids, "gzip": DEFLATE_ID in ids, "none": ids == []}[compression]
        assert np.array_equal(f["events/t"][:], events["t"])
        assert np.array_equal(f["events/p"][:], events["p"])


def test_events_unknown_compression(tmp_path):
    with pytest.raises(ValueError, match="compression"):
        EventWriter(tmp_path / "e.h5", 8, 8, compression="lz4")


def test_events_zero_events_file_is_valid(tmp_path):
    path = tmp_path / "e.h5"
    with EventWriter(path, 64, 48) as writer:
        writer.write(np.empty(0, EVENT_DTYPE))
        writer.write(np.empty(0, EVENT_DTYPE))
    with h5py.File(path, "r") as f:
        for name in "txyp":
            assert f["events"][name].shape == (0,)
        assert f["events"].attrs["num_events"] == 0
        assert f["events"].attrs["width"] == 64


def test_events_uint32_overflow_is_an_error(tmp_path):
    path = tmp_path / "e.h5"
    ok = make_events(np.random.default_rng(2), 100, 64, 48)
    bad = ok[:5].copy()
    bad["t"] = np.array([10, 20, 2 ** 32 - 1, 2 ** 32, 2 ** 32 + 5])
    with EventWriter(path, 64, 48) as writer:
        writer.write(ok)
        with pytest.raises(ValueError, match="uint32"):
            writer.write(bad)
        assert writer.num_events == 100
        edge = ok[:1].copy()
        edge["t"] = 2 ** 32 - 1
        writer.write(edge)  # the last representable microsecond is accepted
    with h5py.File(path, "r") as f:
        assert f["events/t"].shape == (101,)
        assert f["events/t"][-1] == 2 ** 32 - 1


def test_events_t_offset_is_recorded_and_absolute_times_are_a_writer_input_error(tmp_path):
    rng = np.random.default_rng(10)
    ev = make_events(rng, 20, 64, 48, t0=5)
    offset = 5_000_000_000                      # an absolute clock value beyond uint32
    with EventWriter(tmp_path / "e.h5", 64, 48, t_offset_us=offset) as writer:
        writer.write(ev)                        # relative times are stored unchanged
        absolute = ev.copy()
        absolute["t"] += offset
        with pytest.raises(ValueError, match="t_offset_us") as err:
            writer.write(absolute)              # absolute times do not fit and the message says how to fix it
        assert "sequence start" in str(err.value)
    with h5py.File(tmp_path / "e.h5", "r") as f:
        attr = f["events"].attrs["t_offset_us"]
        assert attr.dtype == np.int64 and int(attr) == offset
        assert np.array_equal(f["events/t"][:], ev["t"])
    with EventWriter(tmp_path / "d.h5", 64, 48):
        pass
    with h5py.File(tmp_path / "d.h5", "r") as f:
        assert int(f["events"].attrs["t_offset_us"]) == 0


def test_events_negative_time_sorting_and_bounds_errors(tmp_path):
    ev = make_events(np.random.default_rng(3), 10, 64, 48, t0=100)
    with EventWriter(tmp_path / "e.h5", 64, 48) as writer:
        neg = ev.copy()
        neg["t"] -= 1000
        with pytest.raises(ValueError, match="uint32"):
            writer.write(neg)
        with pytest.raises(ValueError, match="sorted"):
            writer.write(ev[::-1].copy())
        writer.write(ev)
        back = ev[-1:].copy()
        back["t"] -= 1
        with pytest.raises(ValueError, match="back in time"):
            writer.write(back)
        outside = ev[:1].copy()
        outside["t"] = 10 ** 6
        outside["x"] = 64
        with pytest.raises(ValueError, match="sensor"):
            writer.write(outside)
        assert writer.num_events == 10


def test_events_write_after_close(tmp_path):
    writer = EventWriter(tmp_path / "e.h5", 8, 8)
    writer.close()
    writer.close()  # idempotent
    with pytest.raises(ValueError, match="closed"):
        writer.write(make_events(np.random.default_rng(0), 3, 8, 8))


# ------------------------------------------------------------------------------------------ ground truth


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("fields", [c for r in range(len(GT_FIELDS) + 1) for c in itertools.combinations(GT_FIELDS, r)])
def test_gt_round_trip_field_subsets(tmp_path, fields, stereo):
    rng = np.random.default_rng(4)
    cfg = cfg_for(fields, stereo, dt_us=5000, gt_every=2)
    frames = [make_gt(rng, cfg, 10_000 * (i + 1)) for i in range(6)]
    path = tmp_path / "gt.h5"
    with GroundTruthWriter(path, cfg) as writer:
        for gt in frames:
            writer.write(gt)
        assert writer.num_frames == 6
    with h5py.File(path, "r") as f:
        g = f["ground_truth"]
        assert sorted(g.keys()) == sorted(expected_datasets(fields, stereo))
        assert g.attrs["num_frames"] == 6 and g.attrs["pose_order"] == "x y z qx qy qz qw"
        assert np.array_equal(g["timestamp"][:], [gt.t_us for gt in frames])
        assert g["timestamp"].dtype == np.uint32
        stacks = {
            "rgb_left": ("rgb_left", np.uint8), "rgb_right": ("rgb_right", np.uint8), "depth": ("depth", np.float32),
            "optical_flow": ("flow", np.float32), "flow_valid": ("flow_valid", np.uint8), "pose": ("pose", np.float64),
        }
        for name, (attr, dtype) in stacks.items():
            if name in g:
                assert g[name].dtype == dtype
                assert np.array_equal(g[name][:], np.stack([getattr(gt, attr) for gt in frames]))
        if "flow" in fields:
            assert g["flow_dt_us"].dtype == np.uint32
            assert np.all(g["flow_dt_us"][:] == 10_000)
        if "depth" in fields:
            assert g["depth"].chunks == (1, 12, 16)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_gt_compression_modes(tmp_path, compression, stereo):
    rng = np.random.default_rng(5)
    cfg = cfg_for(GT_FIELDS, stereo)
    frames = [make_gt(rng, cfg, 1000 * i) for i in range(4)]
    path = tmp_path / "gt.h5"
    with GroundTruthWriter(path, cfg, compression=compression) as writer:
        for gt in frames:
            writer.write(gt)
    with h5py.File(path, "r") as f:
        g = f["ground_truth"]
        ids = filter_ids(g["optical_flow"])
        assert {"zstd": ZSTD_ID in ids, "gzip": DEFLATE_ID in ids, "none": ids == []}[compression]
        assert np.array_equal(g["optical_flow"][:], np.stack([gt.flow for gt in frames]))
        assert np.array_equal(g["rgb_left"][:], np.stack([gt.rgb_left for gt in frames]))
        assert ("rgb_right" in g) == stereo


def test_gt_t_offset_is_recorded(tmp_path):
    rng = np.random.default_rng(11)
    cfg = cfg_for(("depth",), False)
    with GroundTruthWriter(tmp_path / "gt.h5", cfg, t_offset_us=7_200_000_000) as writer:
        writer.write(make_gt(rng, cfg, 1000))
    with h5py.File(tmp_path / "gt.h5", "r") as f:
        attr = f["ground_truth"].attrs["t_offset_us"]
        assert attr.dtype == np.int64 and int(attr) == 7_200_000_000
        assert f["ground_truth/timestamp"][0] == 1000
        assert "relative" in f["ground_truth"].attrs["convention_time"]


def test_gt_zero_frames_file_is_valid(tmp_path):
    cfg = cfg_for(GT_FIELDS, True)
    path = tmp_path / "gt.h5"
    GroundTruthWriter(path, cfg).close()
    with h5py.File(path, "r") as f:
        g = f["ground_truth"]
        assert g.attrs["num_frames"] == 0
        assert g["depth"].shape == (0, 12, 16) and g["optical_flow"].shape == (0, 12, 16, 2)
        assert g["rgb_right"].shape == (0, 12, 16, 3) and g["pose"].shape == (0, 7)


def test_gt_missing_field_is_an_error_and_leaves_file_consistent(tmp_path):
    rng = np.random.default_rng(6)
    cfg = cfg_for(("depth", "flow", "rgb"), True)
    path = tmp_path / "gt.h5"
    with GroundTruthWriter(path, cfg) as writer:
        good = make_gt(rng, cfg, 1000)
        for attr in ("depth", "flow", "flow_valid", "flow_dt_us", "rgb_left", "rgb_right"):
            frame = make_gt(rng, cfg, 2000)
            setattr(frame, attr, None)
            with pytest.raises(ValueError, match=attr):
                writer.write(frame)
        assert writer.num_frames == 0
        writer.write(good)
    with h5py.File(path, "r") as f:
        assert {k: v.shape[0] for k, v in f["ground_truth"].items()} == {k: 1 for k in f["ground_truth"]}


def test_gt_bad_inputs(tmp_path):
    rng = np.random.default_rng(7)
    cfg = cfg_for(("rgb", "depth"), False)
    with GroundTruthWriter(tmp_path / "gt.h5", cfg) as writer:
        frame = make_gt(rng, cfg, 1000)
        frame.rgb_left = frame.rgb_left.astype(np.float32) / 255
        with pytest.raises(TypeError, match="uint8"):
            writer.write(frame)
        frame = make_gt(rng, cfg, 1000)
        frame.depth = frame.depth[:-1]
        with pytest.raises(ValueError, match="shape"):
            writer.write(frame)
        frame = make_gt(rng, cfg, 2 ** 32)
        with pytest.raises(ValueError, match="uint32"):
            writer.write(frame)
        assert writer.num_frames == 0


def test_gt_unlisted_fields_are_ignored(tmp_path):
    rng = np.random.default_rng(8)
    cfg = cfg_for(("depth",), False)
    with GroundTruthWriter(tmp_path / "gt.h5", cfg) as writer:
        writer.write(make_gt(rng, cfg, 1000))  # frame also carries rgb / flow / pose
    with h5py.File(tmp_path / "gt.h5", "r") as f:
        assert sorted(f["ground_truth"].keys()) == ["depth", "timestamp"]


# --------------------------------------------------------------------------------------------------- TUM


def test_tum_round_trip(tmp_path):
    rng = np.random.default_rng(9)
    n = 200
    t_us = np.sort(rng.integers(0, 3_000_000_000, n))
    poses = np.concatenate([rng.normal(size=(n, 3)) * 10, rng.normal(size=(n, 4))], axis=1)
    path = tmp_path / "traj.txt"
    write_tum(path, t_us, poses)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("# ") and len([ln for ln in lines if not ln.startswith("#")]) == n
    data = np.loadtxt(path, comments="#")
    assert data.shape == (n, 8)
    assert np.max(np.abs(data[:, 0] - t_us / 1e6)) < 1e-6
    assert np.max(np.abs(data[:, 1:] - poses)) < 1e-6
    first = lines[[i for i, ln in enumerate(lines) if not ln.startswith("#")][0]].split()[0]
    assert first == f"{t_us[0] // 1_000_000}.{t_us[0] % 1_000_000:06d}"


def test_tum_offset_comment(tmp_path):
    poses = np.zeros((2, 7))
    poses[:, 6] = 1.0
    write_tum(tmp_path / "a.txt", [0, 10_000], poses, t_offset_us=4_300_000_000)
    lines = (tmp_path / "a.txt").read_text().splitlines()
    assert "# t_offset_us = 4300000000; timestamps below are relative to it" in lines
    assert [ln.startswith("#") for ln in lines] == [True, True, True, False, False]
    assert np.allclose(np.loadtxt(tmp_path / "a.txt", comments="#")[:, 0], [0.0, 0.01])
    write_tum(tmp_path / "b.txt", [0], poses[:1])
    assert "t_offset_us" not in (tmp_path / "b.txt").read_text()


def test_tum_accepts_pose_objects_and_empty(tmp_path):
    poses = [Pose.from_look([i, 0.5, -i], [0, 0, -1]) for i in range(3)]
    write_tum(tmp_path / "a.txt", [0, 1_500_000, 2_000_001], poses)
    data = np.loadtxt(tmp_path / "a.txt", comments="#")
    assert np.allclose(data[:, 0], [0.0, 1.5, 2.000001], atol=1e-9)
    assert np.allclose(data[:, 1:], np.stack([p.as_array() for p in poses]), atol=1e-8)
    write_tum(tmp_path / "b.txt", [], np.empty((0, 7)))
    assert all(ln.startswith("#") for ln in (tmp_path / "b.txt").read_text().splitlines())
    with pytest.raises(ValueError):
        write_tum(tmp_path / "c.txt", [0, 1], np.zeros((3, 7)))


# ------------------------------------------------------------------------------------ visualiser pieces


def test_events_frame_colours():
    acc = np.zeros((4, 5), np.int64)
    acc[1, 1] = 3
    acc[2, 3] = -1
    acc[0, 0] = 1
    out = events_frame(acc)
    assert out.dtype == np.uint8 and out.shape == (4, 5, 3)
    assert tuple(out[1, 1]) == (255, 0, 0)                # strongest ON: pure red
    assert tuple(out[3, 4]) == (255, 255, 255)            # white background
    assert out[2, 3, 2] == 255 and out[2, 3, 0] == out[2, 3, 1] < 200   # OFF: blue
    assert out[0, 0, 0] == 255 and 0 < out[0, 0, 1] < 255  # weaker ON: lighter red
    assert np.all(events_frame(np.zeros((3, 3), np.int64)) == 255)


def _torch_reference_flow_to_rgb(flow_hw2, scale):
    """The user's reference implementation (Middlebury / Baker et al. 2007), verbatim, on a torch tensor."""
    import math

    torch = pytest.importorskip("torch")
    flow_uv = torch.from_numpy(np.ascontiguousarray(flow_hw2.transpose(2, 0, 1))).double()
    wheel = torch.from_numpy(_COLORWHEEL)
    ncols = wheel.shape[0]
    u, v = flow_uv[0] / scale, flow_uv[1] / scale
    rad = (u ** 2 + v ** 2).sqrt()
    angle = torch.atan2(-v, -u) / math.pi
    fk = (angle + 1) / 2 * (ncols - 1)
    k0 = fk.floor().long()
    k1 = (k0 + 1) % ncols
    f = (fk - k0).unsqueeze(-1)
    col = (1 - f) * (wheel[k0] / 255.0) + f * (wheel[k1] / 255.0)
    in_range = (rad <= 1.0).unsqueeze(-1)
    rad = rad.clamp(max=1.0).unsqueeze(-1)
    col = torch.where(in_range, 1 - rad * (1 - col), col * 0.75)
    return (col * 255).round().clamp(0, 255).byte().numpy()


def test_colorwheel_is_the_middlebury_wheel():
    assert _COLORWHEEL.shape == (55, 3)
    assert tuple(_COLORWHEEL[0]) == (255, 0, 0)          # RY starts at red
    assert tuple(_COLORWHEEL[15]) == (255, 255, 0)       # YG starts at yellow
    assert tuple(_COLORWHEEL[21]) == (0, 255, 0)         # GC starts at green
    assert tuple(_COLORWHEEL[25]) == (0, 255, 255)       # CB starts at cyan
    assert tuple(_COLORWHEEL[36]) == (0, 0, 255)         # BM starts at blue
    assert tuple(_COLORWHEEL[49]) == (255, 0, 255)       # MR starts at magenta
    assert tuple(_COLORWHEEL[16]) == (255 - 42, 255, 0)  # floor(255 * 1 / 6) = 42


def test_flow_to_rgb_matches_the_torch_reference_exactly():
    rng = np.random.default_rng(0)
    flow = rng.normal(0.0, 6.0, (40, 50, 2)).astype(np.float32)   # plenty of pixels on both sides of the scale
    flow[0, :4] = [[0, 0], [5, 0], [0, 5], [-5, 0]]
    for scale in (4.0, 8.0):
        assert np.array_equal(flow_to_rgb(flow, scale), _torch_reference_flow_to_rgb(flow, scale))


def test_flow_to_rgb_colorimetry():
    flow = np.array([[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [40.0, 0.0]]], np.float32)   # 0, half, full, 10x scale
    out = flow_to_rgb(flow, 4.0).astype(int)
    full = out[0, 2]
    assert tuple(out[0, 0]) == (255, 255, 255)                           # zero flow: white
    assert np.all(np.abs(out[0, 1] - (255 - 0.5 * (255 - full))) <= 1)   # half scale: half saturated
    assert np.all(np.abs(out[0, 3] - 0.75 * full) <= 1)                  # beyond scale: dimmed full hue
    assert full[0] == 255 and full[1] == 0                               # flow to the right is red(-magenta)


def test_flow_frame_invalid_pixels_are_black_and_distinct_from_zero_flow():
    flow = np.zeros((4, 4, 2), np.float32)
    valid = np.ones((4, 4), np.uint8)
    valid[:, 0] = 0
    flow[0, 1] = np.nan
    out = flow_frame(flow, valid, max_px=4.0)
    assert np.all(out[:, 0] == 0) and np.all(out[0, 1] == 0)   # invalid and NaN: black
    assert np.all(out[1:, 1:] == 255)                           # valid zero flow: white
    assert np.all(flow_frame(flow, None, 4.0)[1:, 1:] == 255)


def test_depth_frame_fixed_range():
    depth = np.array([[0.0, 1.0, 2.0, 3.0, 9.0, np.nan]], np.float32)
    g = depth_frame(depth, 1.0, 3.0)[0, :, 0].astype(int)
    assert g[0] == 0 and g[5] == 0                 # invalid: black
    assert g[1] == 255 and g[3] == 32 and g[4] == 32   # near bright, far / out of range dark but not black
    assert g[1] > g[2] > g[3]


# ------------------------------------------------------------------------------ sequence end to end


def fake_steps(cfg, n_steps=30):
    """ON block (8x8 at (8,16)) in window 0 (steps 1-3), OFF block (8x8 at (40,24)) in window 1 (steps 4-6); the
    right camera sees the opposite polarities. Constant depth / flow / RGB on ground-truth steps."""
    cam = cfg.camera
    hw = (cam.height, cam.width)
    ys, xs = np.mgrid[0:8, 0:8]
    angle = np.radians(120.0)
    flow_max = 0.1 * max(cam.width, cam.height)
    flow = np.broadcast_to(np.array([np.cos(angle), np.sin(angle)], np.float32) * flow_max, hw + (2,)).copy()
    depth = np.full(hw, 2.0, np.float32)
    depth[:, : cam.width // 2] = 0.0            # left half: no surface
    for k in range(n_steps):
        t = k * cfg.dt_us

        def block(x0, y0, polarity):
            ev = np.empty(64, EVENT_DTYPE)
            ev["t"], ev["x"], ev["y"], ev["p"] = t, xs.ravel() + x0, ys.ravel() + y0, polarity
            return ev

        empty = np.empty(0, EVENT_DTYPE)
        left = block(8, 16, 1) if 1 <= k <= 3 else block(40, 24, 0) if 4 <= k <= 6 else empty
        right = block(8, 16, 0) if 1 <= k <= 3 else block(40, 24, 1) if 4 <= k <= 6 else empty
        gt = None
        if k % cfg.gt_every == 0:
            gt = GTFrame(
                t_us=t, pose=np.array([0.01 * k, 0, 0, 0, 0, 0, 1.0]),
                rgb_left=np.full(hw + (3,), (200, 40, 40), np.uint8), rgb_right=np.full(hw + (3,), (40, 40, 200), np.uint8),
                depth=depth, flow=flow, flow_valid=np.ones(hw, np.uint8), flow_dt_us=cfg.dt_us * cfg.gt_every)
        yield StepOutput(step=k, t_us=t, events_left=left, events_right=right if cam.stereo else None, gt=gt)


VIZ_NAMES = ("left_events", "right_events", "left_rgb", "right_rgb", "depth", "optical_flow")


@pytest.fixture(scope="module")
def stereo_cfg():
    return SimConfig(camera=CameraConfig(width=64, height=48, stereo=True), dt_us=10_000, gt_every=3)


@pytest.fixture(scope="module")
def sequence(tmp_path_factory, stereo_cfg):
    seq = tmp_path_factory.mktemp("seq") / "seq_000"
    with SequenceOutput(seq, stereo_cfg, visualize=True, meta={"sequence": 0, "start_us": np.int64(0)}) as out:
        for step in fake_steps(stereo_cfg):
            out.write_step(step)
    return seq


def test_sequence_file_set_and_meta(sequence, stereo_cfg):
    expected = {"left_events.h5", "right_events.h5", "ground_truth.h5", "trajectory_tum.txt", "meta.json"}
    expected |= {f"viz_{n}.mp4" for n in VIZ_NAMES}
    assert {p.name for p in sequence.iterdir()} == expected
    meta = json.loads((sequence / "meta.json").read_text())
    assert meta["complete"] is True
    assert meta["n_events_left"] == meta["n_events_right"] == 6 * 64
    assert meta["n_gt_frames"] == 10 and meta["n_steps"] == 30
    assert meta["start_us"] == 0 and meta["end_us"] == 290_000 and meta["duration_us"] == 290_000
    assert SimConfig.from_dict(meta["config"]) == stereo_cfg
    assert meta["meta"] == {"sequence": 0, "start_us": 0}
    assert {"pose", "depth", "optical_flow", "rgb", "events", "time"} <= set(meta["conventions"])
    assert {"python", "numpy", "h5py"} <= set(meta["versions"])
    assert sorted(meta["files"]) == sorted(expected - {"meta.json"})


def test_sequence_hdf5_contents(sequence, stereo_cfg):
    steps = list(fake_steps(stereo_cfg))
    for side, attr in (("left", "events_left"), ("right", "events_right")):
        expected = np.concatenate([getattr(s, attr) for s in steps])
        with h5py.File(sequence / f"{side}_events.h5", "r") as f:
            for name in "txyp":
                assert np.array_equal(f["events"][name][:], expected[name])
            assert (f["events"].attrs["width"], f["events"].attrs["height"]) == (64, 48)
    gt_steps = [s.gt for s in steps if s.gt is not None]
    with h5py.File(sequence / "ground_truth.h5", "r") as f:
        g = f["ground_truth"]
        assert np.array_equal(g["timestamp"][:], np.arange(10) * 30_000)
        assert np.array_equal(g["pose"][:], np.stack([gt.pose for gt in gt_steps]))
        assert np.array_equal(g["rgb_left"][3], gt_steps[3].rgb_left)
        assert np.all(g["flow_dt_us"][:] == 30_000)


def test_sequence_tum(sequence):
    data = np.loadtxt(sequence / "trajectory_tum.txt", comments="#")
    assert data.shape == (10, 8)
    assert np.allclose(data[:, 0], np.arange(10) * 0.03, atol=1e-6)
    assert np.allclose(data[:, 1], np.arange(10) * 0.03, atol=1e-6)   # x = 0.01 * step
    assert np.allclose(data[:, 7], 1.0)


def test_sequence_videos_frame_counts_and_size(sequence):
    expected = {"left_events": 10, "right_events": 10, "left_rgb": 10, "right_rgb": 10, "depth": 10,
                "optical_flow": 10}
    for name, count in expected.items():
        frames, info = read_video(sequence / f"viz_{name}.mp4")
        assert (info["width"], info["height"]) == (64, 48), name
        assert info["count"] == len(frames) == count, name
        assert abs(info["fps"] - 1e6 / 30_000) < 0.1, name
        assert frames[0].shape == (48, 64, 3)


def test_sequence_event_video_colours(sequence):
    left, _ = read_video(sequence / "viz_left_events.mp4")
    right, _ = read_video(sequence / "viz_right_events.mp4")
    on_block, off_block = (slice(16 + 2, 24 - 2), slice(8 + 2, 16 - 2)), (slice(24 + 2, 32 - 2), slice(40 + 2, 48 - 2))
    r, g, b = (left[0][on_block].reshape(-1, 3).mean(axis=0))
    assert r > 200 and g < 90 and b < 90                    # window 0: ON events are red
    r, g, b = (left[1][off_block].reshape(-1, 3).mean(axis=0))
    assert b > 200 and r < 90 and g < 90                    # window 1: OFF events are blue
    assert left[0][off_block].reshape(-1, 3).mean() > 230    # nothing there in window 0: white
    assert left[5][on_block].reshape(-1, 3).mean() > 230
    r, g, b = (right[0][on_block].reshape(-1, 3).mean(axis=0))
    assert b > 200 and r < 90                               # the right camera has the opposite polarity
    r, g, b = (right[1][off_block].reshape(-1, 3).mean(axis=0))
    assert r > 200 and g < 90 and b < 90


def test_sequence_rgb_videos_keep_channel_order(sequence):
    left, _ = read_video(sequence / "viz_left_rgb.mp4")
    right, _ = read_video(sequence / "viz_right_rgb.mp4")
    r, g, b = left[0][10:38, 10:54].reshape(-1, 3).mean(axis=0)
    assert r > 170 and g < 80 and b < 80                     # (200, 40, 40) is red, not blue
    r, g, b = right[0][10:38, 10:54].reshape(-1, 3).mean(axis=0)
    assert b > 170 and r < 80 and g < 80


def test_sequence_flow_and_depth_videos(sequence):
    flow, _ = read_video(sequence / "viz_optical_flow.mp4")
    depth, _ = read_video(sequence / "viz_depth.mp4")
    # constant flow (angle 120 deg, |flow| = flow_max): the Middlebury colour of that flow on the whole frame
    angle = np.radians(120.0)
    expected = flow_to_rgb(np.array([[[np.cos(angle), np.sin(angle)]]]) * 6.4, 6.4)[0, 0].astype(float)
    got = flow[4][8:40, 8:56].reshape(-1, 3).mean(axis=0)
    assert np.all(np.abs(got - expected) < 12), (got, expected)       # mp4 compression tolerance
    # depth: left half invalid = black; right half = the fixed range's minimum = brightest
    assert depth[2][:, 4:28].mean() < 12
    assert depth[2][:, 36:60].mean() > 230


def simple_steps(cfg, n_steps, rng=None):
    """Random events at each step time plus a full GT frame every `gt_every` steps (white RGB); any sensor size."""
    rng = rng or np.random.default_rng(12)
    cam = cfg.camera
    for k in range(n_steps):
        t = k * cfg.dt_us
        left, right = (make_events(rng, 30, cam.width, cam.height) for _ in range(2))
        left["t"] = right["t"] = t
        gt = None
        if k % cfg.gt_every == 0:
            gt = make_gt(rng, cfg, t)
            gt.rgb_left[:] = 255
            if cam.stereo:
                gt.rgb_right[:] = 255
            gt.flow_valid[:] = 1
        yield StepOutput(step=k, t_us=t, events_left=left, events_right=right if cam.stereo else None, gt=gt)


def test_sequence_records_t_offset_everywhere(tmp_path):
    offset = 6_000_000_000                       # e.g. a headset performance.now() far beyond uint32
    cfg = SimConfig(camera=CameraConfig(width=64, height=48, stereo=True), gt_every=2)
    with SequenceOutput(tmp_path / "s", cfg, t_offset_us=offset) as out:
        for step in simple_steps(cfg, 6):
            out.write_step(step)
    seq = tmp_path / "s"
    meta = json.loads((seq / "meta.json").read_text())
    assert meta["t_offset_us"] == offset and meta["start_us"] == 0 and meta["end_us"] == 50_000
    assert "RELATIVE" in meta["conventions"]["time"] and "t_offset_us" in meta["conventions"]["time"]
    for name, group in (("left_events.h5", "events"), ("right_events.h5", "events"), ("ground_truth.h5", "ground_truth")):
        with h5py.File(seq / name, "r") as f:
            assert int(f[group].attrs["t_offset_us"]) == offset
    with h5py.File(seq / "ground_truth.h5", "r") as f:
        assert np.array_equal(f["ground_truth/timestamp"][:], [0, 20_000, 40_000])
    lines = (seq / "trajectory_tum.txt").read_text().splitlines()
    assert f"# t_offset_us = {offset}; timestamps below are relative to it" in lines
    assert np.allclose(np.loadtxt(seq / "trajectory_tum.txt", comments="#")[:, 0], [0.0, 0.02, 0.04])
    default = tmp_path / "d"
    with SequenceOutput(default, cfg) as out:
        pass
    assert json.loads((default / "meta.json").read_text())["t_offset_us"] == 0


def test_sequence_meta_files_lists_everything_in_the_directory_at_close(tmp_path):
    cfg = SimConfig(camera=CameraConfig(width=64, height=48), gt=("pose",), gt_every=2)
    with SequenceOutput(tmp_path / "s", cfg, meta={"sequence": 3}) as out:
        for step in simple_steps(cfg, 4):
            out.write_step(step)
        (tmp_path / "s" / "trajectory.json").write_text("[]")     # the pipeline adds files before closing
        (tmp_path / "s" / "notes.txt").write_text("x")
    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    assert meta["files"] == sorted(p.name for p in (tmp_path / "s").iterdir() if p.name != "meta.json")
    assert {"trajectory.json", "notes.txt", "trajectory_tum.txt", "ground_truth.h5", "left_events.h5"} <= set(meta["files"])
    assert meta["meta"] == {"sequence": 3}


def test_sequence_without_gt_or_viz(tmp_path):
    cfg = SimConfig(camera=CameraConfig(width=64, height=48), gt=(), gt_every=3)
    with SequenceOutput(tmp_path / "s", cfg) as out:
        for step in fake_steps(cfg, 8):
            out.write_step(step)
    assert sorted(p.name for p in (tmp_path / "s").iterdir()) == ["left_events.h5", "meta.json"]
    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    assert meta["n_gt_frames"] is None and meta["n_events_right"] is None and meta["n_events_left"] == 6 * 64


def test_sequence_gt_without_pose_has_no_tum(tmp_path):
    cfg = SimConfig(camera=CameraConfig(width=64, height=48), gt=("depth",), gt_every=3)
    with SequenceOutput(tmp_path / "s", cfg) as out:
        for step in fake_steps(cfg, 8):
            out.write_step(step)
    names = {p.name for p in (tmp_path / "s").iterdir()}
    assert names == {"left_events.h5", "ground_truth.h5", "meta.json"}


def test_sequence_error_closes_everything(tmp_path):
    cfg = SimConfig(camera=CameraConfig(width=64, height=48, stereo=True), gt_every=3)
    steps = list(fake_steps(cfg, 8))
    with pytest.raises(ValueError, match="events_right"):
        with SequenceOutput(tmp_path / "s", cfg, visualize=True) as out:
            for step in steps[:4]:
                out.write_step(step)
            bad = steps[4]
            bad.events_right = None
            out.write_step(bad)
    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    assert meta["complete"] is False
    for name in ("left_events.h5", "right_events.h5", "ground_truth.h5"):
        with h5py.File(tmp_path / "s" / name, "r") as f:   # not left open / corrupt
            assert len(f) == 1
    assert read_video(tmp_path / "s" / "viz_left_rgb.mp4")[1]["count"] == 2


def test_sequence_write_after_close(tmp_path):
    cfg = SimConfig(camera=CameraConfig(width=64, height=48), gt=())
    out = SequenceOutput(tmp_path / "s", cfg)
    out.close()
    out.close()
    with pytest.raises(ValueError, match="closed"):
        out.write_step(next(iter(fake_steps(cfg, 1))))


def test_visualizer_time_accumulation_and_depth_range(tmp_path, stereo_cfg):
    with Visualizer(tmp_path, stereo_cfg, time_accumulation_us=60_000, depth_range=(1.0, 3.0), prefix="v_") as viz:
        assert viz.time_accumulation_us == 60_000
        for step in fake_steps(stereo_cfg):
            viz.add(step)
        assert {p.name for p in viz.files} == {f"v_{n}.mp4" for n in VIZ_NAMES}
    frames, info = read_video(tmp_path / "v_left_events.mp4")
    assert info["count"] == 5 and abs(info["fps"] - 1e6 / 60_000) < 0.1  # 4 full windows + the partial last one
    r, g, b = frames[0][18:22, 10:14].reshape(-1, 3).mean(axis=0)   # steps 1-6 share window 0: both blocks are in it
    assert r > 200 and g < 90 and b < 90
    r, g, b = frames[0][26:30, 42:46].reshape(-1, 3).mean(axis=0)
    assert b > 200 and r < 90 and g < 90
    depth, _ = read_video(tmp_path / "v_depth.mp4")
    assert 100 < depth[0][:, 36:60].mean() < 170                                # depth 2.0 in the range (1, 3): mid gray


def test_visualizer_depth_range_defaults_to_first_frame_and_stays_fixed(tmp_path):
    cfg = SimConfig(camera=CameraConfig(width=64, height=48), gt=("depth",), dt_us=10_000, gt_every=1)
    d0 = np.full((48, 64), 1.0, np.float32)
    d0[:, 32:] = 3.0
    d1 = np.full((48, 64), 3.0, np.float32)
    d1[:, :8] = 0.0
    with Visualizer(tmp_path, cfg) as viz:
        for k, d in enumerate((d0, d1)):
            viz.add(StepOutput(step=k, t_us=k * 10_000, events_left=np.empty(0, EVENT_DTYPE), gt=GTFrame(t_us=k * 10_000, depth=d)))
        assert viz.depth_range == (1.0, 3.0)
    frames, _ = read_video(tmp_path / "viz_depth.mp4")
    assert frames[0][:, 4:28].mean() > 230 and 20 < frames[0][:, 36:60].mean() < 60
    assert 20 < frames[1][:, 20:60].mean() < 60        # with a per-frame range this would have been rescaled
    assert frames[1][:, 0:6].mean() < 12               # invalid stays black


@pytest.mark.parametrize("width, height", [(33, 17), (64, 17), (33, 48)])
def test_odd_frame_sizes_are_padded_to_even_video_size(tmp_path, width, height):
    """mp4v silently crops odd sizes (33x17 -> 32x16); frames are padded with black instead."""
    cfg = SimConfig(camera=CameraConfig(width=width, height=height, stereo=True), gt_every=2)
    pw, ph = width + width % 2, height + height % 2
    with SequenceOutput(tmp_path / "s", cfg, visualize=True) as out:
        for step in simple_steps(cfg, 8):
            out.write_step(step)
    videos = sorted((tmp_path / "s").glob("viz_*.mp4"))
    assert {v.stem[4:] for v in videos} == set(VIZ_NAMES)
    for video in videos:
        frames, info = read_video(video)
        assert (info["width"], info["height"]) == (pw, ph), video.name
        assert frames[0].shape == (ph, pw, 3)
        assert info["count"] == len(frames) == 4, video.name
    rgb, _ = read_video(tmp_path / "s" / "viz_left_rgb.mp4")       # white frames: content bright, padding black
    assert rgb[0][:height, :width].mean() > 200
    if width % 2:
        assert rgb[0][:, width].mean() < 100
    if height % 2:
        assert rgb[0][height, :].mean() < 100


def test_visualizer_rejects_wrong_frame_size(tmp_path):
    cfg = SimConfig(camera=CameraConfig(width=64, height=48), gt=("rgb",), gt_every=1)
    frame = GTFrame(t_us=0, rgb_left=np.zeros((40, 64, 3), np.uint8))
    with Visualizer(tmp_path, cfg) as viz:
        with pytest.raises(ValueError, match="expected uint8 frame"):
            viz.add(StepOutput(step=0, t_us=0, events_left=np.empty(0, EVENT_DTYPE), gt=frame))


# ------------------------------------------------------------------------------------------ memory


def _peak_rss_after_writing(path, n_frames):
    """Peak RSS (MB) of a fresh process that streams `n_frames` flow frames of 260x346 into `path`."""
    script = textwrap.dedent(f"""
        import resource, sys
        import numpy as np
        from backend.config import CameraConfig, SimConfig
        from backend.saving import GroundTruthWriter
        from backend.types import GTFrame

        H, W = 260, 346
        cfg = SimConfig(camera=CameraConfig(width=W, height=H), gt=("flow",))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        base = np.stack([np.sin(xx / 40), np.cos(yy / 30)], -1).astype(np.float32)
        valid = np.ones((H, W), np.uint8)
        writer = GroundTruthWriter({str(path)!r}, cfg)
        for i in range({n_frames}):
            writer.write(GTFrame(t_us=i * 1000, flow=base * (1 + i * 0.001), flow_valid=valid, flow_dt_us=1000))
        writer.close()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(rss / 1e6 if sys.platform == "darwin" else rss / 1e3)
    """)
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=300,
                          env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert proc.returncode == 0, proc.stderr
    return float(proc.stdout.split()[-1])


def test_gt_writer_memory_is_streaming(tmp_path):
    """Peak RSS must not scale with sequence length: 800 extra frames are ~576 MB of raw flow, so a writer that
    accumulated them would grow by hundreds of MB. (Allocator and HDF5 filter buffers make the absolute level
    library-dependent and a few tens of MB noisy, hence the comparison between two lengths, best of two runs each.)"""
    small = min(_peak_rss_after_writing(tmp_path / "small.h5", 200) for _ in range(2))
    large = min(_peak_rss_after_writing(tmp_path / "large.h5", 1000) for _ in range(2))
    assert large - small < 64.0, f"peak RSS grew from {small:.0f} MB (200 frames) to {large:.0f} MB (1000 frames)"
    with h5py.File(tmp_path / "large.h5", "r") as f:
        assert f["ground_truth/optical_flow"].shape == (1000, 260, 346, 2)
        assert f["ground_truth/timestamp"][-1] == 999_000


def test_flow_video_scale_defaults_to_first_frame_p99_and_stays_fixed():
    from backend.vizualisation.visualizer import _flow_scale

    flow = np.zeros((10, 10, 2), np.float32)
    flow[..., 0] = np.arange(100, dtype=np.float32).reshape(10, 10)
    valid = np.ones((10, 10), np.uint8)
    assert _flow_scale(flow, valid) == pytest.approx(np.percentile(np.arange(100), 99))
    valid[:] = 0
    assert _flow_scale(flow, valid) == 1.0                          # nothing valid: neutral scale
    assert _flow_scale(np.zeros((4, 4, 2), np.float32), None) == 1.0  # static camera: never divide by ~0
