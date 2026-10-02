"""Integration tests: Simulator, pipeline.run_to_disk / build_trajectories and the CLI on the demo scene."""
from __future__ import annotations

import json

import h5py
import hdf5plugin  # noqa: F401  (registers zstd)
import numpy as np
import pytest

from backend.cli import main as cli_main
from backend.config import SimConfig
from backend.pipeline import Cancelled, build_trajectories, run_to_disk
from backend.scene import load_scene
from backend.simulator import Simulator
from backend.trajectories import WaypointTrajectory

pytestmark = pytest.mark.usefixtures("gl_context")

C = 0.2
LOG_EPS = 1e-3


def _config(**kw):
    d = {"camera": {"width": 64, "height": 48}, "dt_us": 20_000, "events": {"c_pos": C, "c_neg": C}}
    d.update(kw)
    return SimConfig.from_dict(d)


@pytest.fixture(scope="module")
def scene(demo_scene):
    return load_scene(demo_scene.scene_spec)


@pytest.fixture(scope="module")
def trajectory(demo_scene):
    return WaypointTrajectory.from_json(demo_scene.trajectory_json)


def _luminance(rgb):
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def test_step_count_timing_and_event_windows(scene, trajectory):
    cfg = _config(gt=())
    with Simulator(scene, cfg) as sim:
        steps = list(sim.run(trajectory))
    assert len(steps) == sim.total_steps(trajectory) == 201
    assert [s.t_us for s in steps[:3]] == [0, 20_000, 40_000]
    assert steps[-1].t_us == trajectory.end_us
    assert len(steps[0].events_left) == 0                       # first frame only initialises the reference
    assert all(s.gt is None for s in steps)                     # gt=() -> no ground truth at all
    for prev, cur in zip(steps, steps[1:]):
        ev = cur.events_left
        assert ((ev["t"] > prev.t_us) & (ev["t"] <= cur.t_us)).all()
        assert (np.diff(ev["t"]) >= 0).all()
    assert sum(len(s.events_left) for s in steps) > 1000


def test_ground_truth_cadence_and_content(scene, trajectory):
    cfg = _config(gt_every=5, camera={"width": 64, "height": 48, "stereo": True})
    with Simulator(scene, cfg) as sim:
        steps = list(sim.run(trajectory))
    gt_steps = [s for s in steps if s.gt is not None]
    assert [s.step for s in gt_steps] == list(range(0, 201, 5))
    g = gt_steps[3].gt
    assert g.rgb_left.shape == (48, 64, 3) and g.rgb_left.dtype == np.uint8
    assert g.rgb_right.shape == (48, 64, 3) and (g.rgb_left != g.rgb_right).any()      # real stereo pair
    assert g.depth.shape == (48, 64) and g.depth.dtype == np.float32 and (g.depth > 0).mean() > 0.99
    assert g.flow.shape == (48, 64, 2) and np.isfinite(g.flow).all()
    assert g.flow_dt_us == 5 * 20_000 and g.flow_valid.mean() > 0.99
    assert g.pose.shape == (7,) and g.pose[:3] == pytest.approx(trajectory.pose_at(g.t_us).position)
    assert all(s.events_right is not None for s in steps)


def test_last_ground_truth_frame_has_no_valid_flow(scene, trajectory):
    with Simulator(scene, _config()) as sim:
        steps = list(sim.run(trajectory))
    last = steps[-1].gt
    assert last.flow_dt_us == 0 and last.flow_valid.sum() == 0 and not last.flow.any()
    assert steps[-2].gt.flow_dt_us == 20_000 and steps[-2].gt.flow_valid.any()


def test_gt_fields_are_only_computed_when_requested(scene, trajectory):
    with Simulator(scene, _config(gt=["depth"])) as sim:
        g = next(s.gt for s in sim.run(trajectory))
    assert g.depth is not None
    assert g.rgb_left is None and g.flow is None and g.pose is None


def test_deterministic(scene, trajectory):
    cfg = _config(events={"c_pos": C, "c_neg": C, "sigma_c": 0.05, "shot_noise_hz": 1.0, "seed": 3}, gt=())

    def events():
        with Simulator(scene, cfg) as sim:
            return np.concatenate([s.events_left for s in sim.run(trajectory)])

    a, b = events(), events()
    assert len(a) == len(b) > 0 and (a == b).all()


def test_events_agree_with_rendered_frames(scene, trajectory):
    """Net ON-OFF events per pixel == change of log-luminance between first and last RGB frame, divided by C (+-1)."""
    cfg = _config(gt=["rgb"], gt_every=200)                     # GT frames at step 0 and step 200 only
    with Simulator(scene, cfg) as sim:
        steps = list(sim.run(trajectory))
    first, last = steps[0].gt.rgb_left, steps[-1].gt.rgb_left
    net = np.zeros(first.shape[:2], np.int64)
    for s in steps:
        ev = s.events_left
        np.add.at(net, (ev["y"], ev["x"]), 2 * ev["p"].astype(np.int64) - 1)
    expected = (np.log(_luminance(last) + LOG_EPS) - np.log(_luminance(first) + LOG_EPS)) / C
    mid = (_luminance(first) > 0.02) & (_luminance(last) > 0.02)  # 8-bit sRGB output is coarse near black
    assert np.mean(np.abs(net - expected)[mid] <= 1.0) > 0.97


# ------------------------------------------------------------------------------------------------ pipeline

def test_run_to_disk_writes_a_complete_sequence(demo_scene, tmp_path):
    calls = []
    cfg = _config(camera={"width": 48, "height": 32})
    dirs = run_to_disk(
        demo_scene.scene_spec, {"type": "file", "path": str(demo_scene.trajectory_json)}, cfg, tmp_path,
        prefix="t", progress=lambda *a: calls.append(a),
    )
    assert [d.name for d in dirs] == ["t_000"]
    names = {p.name for p in dirs[0].iterdir()}
    assert {"left_events.h5", "ground_truth.h5", "meta.json", "trajectory_tum.txt", "trajectory.json"} <= names
    meta = json.loads((dirs[0] / "meta.json").read_text())
    assert meta["complete"] is True and meta["n_gt_frames"] == 201 and meta["n_events_left"] > 0
    assert calls[0] == (0, 1, 1, 201) and calls[-1] == (0, 1, 201, 201)
    with h5py.File(dirs[0] / "ground_truth.h5") as f:
        assert set(f["ground_truth"]) >= {"depth", "optical_flow", "flow_valid", "rgb_left", "pose", "timestamp"}


def test_cancel_closes_files_and_marks_incomplete(demo_scene, tmp_path):
    n = []

    def cancel():
        n.append(1)
        return len(n) > 10

    with pytest.raises(Cancelled):
        run_to_disk(
            demo_scene.scene_spec, {"type": "file", "path": str(demo_scene.trajectory_json)},
            _config(camera={"width": 48, "height": 32}), tmp_path, prefix="c", cancel=cancel,
        )
    meta = json.loads((tmp_path / "c_000" / "meta.json").read_text())
    assert meta["complete"] is False
    with h5py.File(tmp_path / "c_000" / "left_events.h5") as f:      # readable, i.e. properly closed
        assert "events" in f


def test_build_trajectories_specs(scene, demo_scene):
    records = json.loads(demo_scene.trajectory_json.read_text())
    inline = build_trajectories({"type": "inline", "points": records, "origin": [1.0, 0.0, 0.0]})
    from_file = build_trajectories({"type": "file", "path": str(demo_scene.trajectory_json)})
    assert inline[0].pose_at(0).position[0] == pytest.approx(from_file[0].pose_at(0).position[0] + 1.0)
    generated = build_trajectories([
        {"type": "rectilinear", "duration_s": 2.0, "count": 2, "seed": 1},
        {"type": "spline", "duration_s": 1.5, "n_key_points": 6, "count": 2, "seed": 1},
    ], scene)
    assert len(generated) == 4 and generated[0].duration_us == 2_000_000 and generated[2].duration_us == 1_500_000
    again = build_trajectories({"type": "spline", "duration_s": 1.5, "n_key_points": 6, "count": 2, "seed": 1}, scene)
    assert np.allclose(generated[2].pose_at(700_000).as_array(), again[0].pose_at(700_000).as_array())
    with pytest.raises(ValueError):
        build_trajectories({"type": "warp"})
    with pytest.raises(ValueError):
        build_trajectories({"type": "spline", "duration_s": 1.0})           # generated types need a scene


# ------------------------------------------------------------------------------------------------------- CLI

def test_cli_run_with_generated_trajectories(demo_scene, tmp_path):
    rc = cli_main([
        "run", "--scene", str(demo_scene.scene_spec), "--generate", "spline", "--count", "2", "--duration", "0.4",
        "--out", str(tmp_path), "--width", "40", "--height", "30", "--gt", "depth,pose", "--dt-us", "20000",
        "--c-pos", "0.3", "--seed", "5",
    ])
    assert rc == 0
    for i in range(2):
        meta = json.loads((tmp_path / f"seq_{i:03d}" / "meta.json").read_text())
        assert meta["config"]["camera"]["width"] == 40 and meta["config"]["events"]["c_pos"] == 0.3
        assert meta["config"]["gt"] == ["depth", "pose"]


def test_cli_run_with_street_walks(demo_scene, tmp_path):
    rc = cli_main([
        "run", "--scene", str(demo_scene.scene_spec), "--generate", "street", "--duration", "0.4", "--seed", "2",
        "--walk-speed", "0.5", "--eye-height", "1.2", "--edge-margin", "0", "--out", str(tmp_path),
        "--width", "40", "--height", "30", "--gt", "pose", "--dt-us", "20000",
    ])
    assert rc == 0
    records = json.loads((tmp_path / "seq_000" / "trajectory.json").read_text())
    p = np.array([[r["x"], r["y"], r["z"]] for r in records])
    assert np.allclose(p[:, 1], 1.2, atol=1e-3)                              # the demo floor is y = 0
    assert np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1).sum() == pytest.approx(0.2, rel=0.01)


def test_cli_rejects_ambiguous_trajectory_arguments(demo_scene, tmp_path, capsys):
    assert cli_main(["run", "--scene", str(demo_scene.scene_spec), "--out", str(tmp_path)]) == 2
    assert "--trajectory" in capsys.readouterr().err


def test_cli_info_reports_opengl(capsys):
    assert cli_main(["info"]) == 0
    assert "OpenGL" in capsys.readouterr().out


# ---------------------------------------------------------------------------- stereo, per-eye config, time base

def _quad_scene(tmp_path, plane):
    """Textured square (white with a red square in the middle) 2 units in front of the origin.

    plane "-z": in the plane z=-2 (seen by a camera looking along -Z); plane "+x": in the plane x=+2."""
    from PIL import Image

    tex = np.full((64, 64, 3), 255, np.uint8)
    tex[24:40, 24:40] = (255, 0, 0)
    Image.fromarray(tex).save(tmp_path / "quad.png")
    corners = {
        "-z": [(-2, -2, -2), (2, -2, -2), (2, 2, -2), (-2, 2, -2)],
        "+x": [(2, -2, -2), (2, -2, 2), (2, 2, 2), (2, 2, -2)],
    }[plane]
    lines = [f"v {x} {y} {z}" for x, y, z in corners] + ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    lines += ["f 1/1 2/2 3/3", "f 1/1 3/3 4/4"]
    (tmp_path / "quad.obj").write_text("\n".join(lines) + "\n")
    return {"objects": [{"path": str(tmp_path / "quad.obj"), "texture": str(tmp_path / "quad.png")}]}


def _red_centroid_x(rgb):
    mask = (rgb[..., 0] > 200) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
    assert mask.sum() > 20
    return np.nonzero(mask)[1].mean()


@pytest.mark.parametrize("plane, front", [("-z", (0, 0, -1)), ("+x", (1, 0, 0))])
def test_stereo_baseline_is_along_the_camera_right_axis(tmp_path, plane, front):
    """Right eye = left eye moved along camera +X: a point 2 units ahead shows a disparity f*B/Z whatever the
    viewing direction (the old code offset along world Z and only worked when looking along +-X)."""
    width, height, fov, baseline, depth = 128, 96, 60.0, 0.2, 2.0
    cfg = SimConfig.from_dict({
        "camera": {"width": width, "height": height, "fov_y_deg": fov, "stereo": True, "stereo_baseline": baseline},
        "gt": ["rgb"], "dt_us": 20_000,
    })
    traj = WaypointTrajectory.from_waypoints([0, 40_000], [[0, 0, 0], [0, 0, 0]], fronts=[front, front])
    with Simulator(load_scene(_quad_scene(tmp_path, plane)), cfg) as sim:
        gt = next(sim.run(traj)).gt
    focal = (height / 2) / np.tan(np.radians(fov / 2))
    disparity = _red_centroid_x(gt.rgb_left) - _red_centroid_x(gt.rgb_right)
    assert disparity == pytest.approx(focal * baseline / depth, abs=0.7)


def test_right_eye_uses_its_own_event_config(scene, trajectory):
    cfg = _config(camera={"width": 64, "height": 48, "stereo": True}, gt=(),
                  events_right={"c_pos": 0.6, "c_neg": 0.6})
    with Simulator(scene, cfg) as sim:
        steps = list(sim.run(trajectory))
    left = sum(len(s.events_left) for s in steps)
    right = sum(len(s.events_right) for s in steps)
    assert left > 1000 and right < 0.6 * left           # 3x the threshold -> roughly a third of the events


def _shifted(records, offset_ms):
    return [dict(r, timestamp=r["timestamp"] + offset_ms) for r in records]


def test_output_time_is_relative_to_the_trajectory_start(scene, demo_scene):
    """Headset recordings use performance.now(): hours-long offsets must not leak into the outputs."""
    records = json.loads(demo_scene.trajectory_json.read_text())[:11]
    offset_ms = 4.3e6                                   # 4.3e9 us: beyond uint32
    late = WaypointTrajectory.from_json(_shifted(records, offset_ms))
    early = WaypointTrajectory.from_json(records)
    assert late.start_us == pytest.approx(4.3e9)
    with Simulator(scene, _config(gt=["pose"])) as sim:
        a, b = list(sim.run(late)), list(sim.run(early))
    assert [s.t_us for s in a] == [s.t_us for s in b] == [0, 20_000, 40_000, 60_000, 80_000, 100_000]
    assert all((x.events_left == y.events_left).all() for x, y in zip(a, b))       # same path, same events
    assert a[0].gt.pose == pytest.approx(b[0].gt.pose)


def test_run_to_disk_records_the_time_offset(demo_scene, tmp_path):
    records = json.loads(demo_scene.trajectory_json.read_text())[:11]
    late = {"type": "inline", "points": _shifted(records, 4.3e6)}
    dirs = run_to_disk(demo_scene.scene_spec, late, _config(camera={"width": 32, "height": 24}), tmp_path)
    meta = json.loads((dirs[0] / "meta.json").read_text())
    assert meta["t_offset_us"] == 4_300_000_000 and meta["duration_us"] == 100_000
    assert meta["meta"]["trajectory_start_us"] == 4_300_000_000
    with h5py.File(dirs[0] / "ground_truth.h5") as f:
        assert f["ground_truth/timestamp"][0] == 0 and f["ground_truth/timestamp"][-1] == 100_000


def test_too_long_trajectory_is_rejected_before_rendering(scene):
    long_traj = WaypointTrajectory.from_waypoints([0, int(5e9)], [[0, 0, 0], [1, 0, 0]])   # 83 min
    with Simulator(scene, _config(gt=())) as sim:
        with pytest.raises(ValueError, match="uint32"):
            next(sim.run(long_traj))


def test_event_seed_overrides_the_noise_seed_per_run(scene, trajectory):
    cfg = _config(events={"c_pos": C, "c_neg": C, "shot_noise_hz": 5.0, "seed": 0}, gt=())

    def events(**kw):
        with Simulator(scene, cfg) as sim:
            return np.concatenate([s.events_left for s in sim.run(trajectory, **kw)])

    default, same, other = events(), events(), events(event_seed=7)
    assert (default == same).all()
    assert (events(event_seed=7) == other).all()                       # same override seed -> same events
    assert len(other) != len(default) or (other != default).any()     # another seed -> other noise


def test_scene_seen_from_far_away_is_not_clipped(tmp_path):
    """The clip planes follow the camera path: a 4x4 quad (diagonal 5.7) seen from 82 units, i.e. 14 diagonals
    away, was silently invisible when far was fixed at 10 diagonals."""
    cfg = SimConfig.from_dict({"camera": {"width": 128, "height": 96}, "gt": ["depth"], "dt_us": 20_000})
    traj = WaypointTrajectory.from_waypoints([0, 40_000], [[0, 0, 80], [0, 0, 80]], fronts=[(0, 0, -1)] * 2)
    with Simulator(load_scene(_quad_scene(tmp_path, "-z")), cfg) as sim:
        depth = next(sim.run(traj)).gt.depth
    # the quad is small and far (~4x4 px at this distance/FOV): what matters is that it is visible at all
    # (it was 0 px when far was fixed at 10 diagonals) and at the right depth
    assert (depth > 0).sum() > 0 and depth[depth > 0].mean() == pytest.approx(82.0, abs=0.5)


def test_simulator_refuses_threads_it_was_not_created_in(scene, trajectory):
    import threading

    errors = []

    def use():
        try:
            next(sim.run(trajectory))
        except RuntimeError as exc:
            errors.append(exc)

    with Simulator(scene, _config(gt=())) as sim:
        thread = threading.Thread(target=use)
        thread.start()
        thread.join()
    assert len(errors) == 1 and "thread" in str(errors[0]).lower()
