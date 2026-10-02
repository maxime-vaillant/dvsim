"""Trajectories: waypoint interpolation, legacy JSON, free-space grid and the random generators (analytic checks)."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
import trimesh
import trimesh.transformations as tf

from backend.pose import Pose, slerp
from backend.trajectories import (
    FreeSpaceGrid,
    ImportedTrajectory,
    RectilinearTrajectoryGenerator,
    SplineTrajectoryGenerator,
    StreetTrajectoryGenerator,
    Trajectory,
    WaypointTrajectory,
)
from backend.trajectories.trajectory import quats_to_fronts, slerp_many

ROOM_HALF = np.array([5.0, 2.0, 4.0])            # room x in [-5,5], y in [-2,2], z in [-4,4]
PILLAR_C, PILLAR_HALF = np.array([1.0, 0.0, 1.0]), np.array([0.5, 1.5, 0.5])
MIN_CLEARANCE = 0.3


def angle_between(q0, q1) -> float:
    """Rotation angle (rad, in [0, pi]) between two unit quaternions."""
    s = 1.0 if q0 @ q1 >= 0 else -1.0
    return 4.0 * np.arctan2(np.linalg.norm(q1 - s * q0), np.linalg.norm(q1 + s * q0))


def axis_angle(axis, angle) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return np.concatenate([a * np.sin(angle / 2), [np.cos(angle / 2)]])


def same_rotation(q0, q1, tol=1e-9) -> bool:
    return abs(abs(float(np.dot(q0, q1))) - 1.0) < tol


def box_sd(p, half, center=(0, 0, 0)):
    """Signed distance to an axis-aligned box (negative inside)."""
    q = np.abs(p - np.asarray(center)) - half
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def room_clearance(p, pillar=True):
    """Analytic clearance inside the room (walls, and the pillar if present)."""
    p = np.atleast_2d(p)
    c = -box_sd(p, ROOM_HALF)
    return np.minimum(c, box_sd(p, PILLAR_HALF, PILLAR_C)) if pillar else c


def make_room() -> trimesh.Trimesh:
    room = trimesh.creation.box(extents=2 * ROOM_HALF)
    room.invert()
    return room


def make_pillar_room() -> trimesh.Trimesh:
    pillar = trimesh.creation.box(extents=2 * PILLAR_HALF)
    pillar.apply_translation(PILLAR_C)
    return trimesh.util.concatenate([make_room(), pillar])


@pytest.fixture(scope="module")
def room_grid():
    return FreeSpaceGrid(make_room())


@pytest.fixture(scope="module")
def pillar_grid():
    return FreeSpaceGrid(make_pillar_room())


# ---------------------------------------------------------------- WaypointTrajectory
def test_exact_at_waypoints_and_midpoints():
    t = np.array([0, 1_000_000, 3_000_000])
    p = np.array([[0.0, 0, 0], [2, 0, 0], [2, 4, 0]])
    q = np.stack([axis_angle([0, 1, 0], a) for a in (0.0, 0.8, 1.6)])
    tr = WaypointTrajectory(t, p, q)
    assert p.flags.writeable and q.flags.writeable and not tr.positions.flags.writeable   # caller's arrays untouched
    for i in range(3):
        pose = tr.pose_at(t[i])
        assert np.array_equal(pose.position, p[i])
        assert same_rotation(pose.quat, q[i])
    mid = tr.pose_at(500_000)
    assert np.allclose(mid.position, [1, 0, 0], atol=1e-12)
    assert same_rotation(mid.quat, axis_angle([0, 1, 0], 0.4))
    mid = tr.pose_at(2_000_000)  # unequal spacing: halfway through the second interval
    assert np.allclose(mid.position, [2, 2, 0], atol=1e-12)
    assert same_rotation(mid.quat, axis_angle([0, 1, 0], 1.2))
    assert (tr.start_us, tr.end_us, tr.duration_us, len(tr)) == (0, 3_000_000, 3_000_000, 3)
    assert isinstance(tr, Trajectory)


def test_slerp_constant_angular_rate_closed_form():
    axis, total = np.array([1.0, 2.0, -0.5]), 2.4  # rad, well below pi
    q0 = axis_angle([0.3, 0.1, 0.9], 0.7)
    q1 = tf_quat_compose(axis_angle(axis, total), q0)
    tr = WaypointTrajectory([10_000_000, 12_000_000], [[0, 0, 0], [1, 0, 0]], [q0, q1])
    ts = np.linspace(10_000_000, 12_000_000, 41)
    poses = tr.poses_at(ts)
    for t, pose in zip(ts, poses):
        alpha = (t - 10_000_000) / 2_000_000
        assert abs(angle_between(q0, pose[3:]) - alpha * total) < 1e-9
        assert same_rotation(pose[3:], tf_quat_compose(axis_angle(axis, alpha * total), q0), tol=1e-12)
    steps = [angle_between(a[3:], b[3:]) for a, b in zip(poses[:-1], poses[1:])]
    assert np.ptp(steps) < 1e-9  # constant rate


def tf_quat_compose(a, b):
    """Hamilton product a * b for (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def test_slerp_many_matches_pose_slerp_and_opposite_signs():
    rng = np.random.default_rng(0)
    q0 = rng.normal(size=(50, 4)); q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    q1 = rng.normal(size=(50, 4)); q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    a = rng.uniform(size=50)
    ref = np.stack([slerp(x, y, float(t)) for x, y, t in zip(q0, q1, a)])
    assert np.allclose(slerp_many(q0, q1, a), ref, atol=1e-12)
    assert np.allclose(slerp_many(q0, -q1, a), ref, atol=1e-12)   # -q is the same rotation: same short path
    assert np.allclose(slerp_many(q0, q1, np.zeros(50)), q0, atol=1e-15)
    assert np.allclose(slerp_many(q0, q1, np.ones(50)), np.where((q0 * q1).sum(1, keepdims=True) < 0, -q1, q1), atol=1e-15)


def test_pose_at_is_pure_and_clamped():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.integers(1000, 50_000, size=20)) + 7_000_000
    tr = WaypointTrajectory.from_waypoints(t, rng.normal(size=(20, 3)), fronts=rng.normal(size=(20, 3)))
    queries = rng.uniform(t[0], t[-1], size=64)
    fwd = [tr.pose_at(q).as_array() for q in queries]
    for i in rng.permutation(64):  # any order, any repetition
        assert np.array_equal(tr.pose_at(queries[i]).as_array(), fwd[i])
    assert np.array_equal(tr.pose_at(queries[3]).as_array(), fwd[3])
    assert np.array_equal(tr.pose_at(t[0] - 5e6).as_array(), tr.pose_at(t[0]).as_array())
    assert np.array_equal(tr.pose_at(t[-1] + 5e6).as_array(), tr.pose_at(t[-1]).as_array())
    assert np.allclose(tr.pose_at(t[0]).position, tr.positions[0]) and np.allclose(tr.pose_at(t[-1]).position, tr.positions[-1])
    assert np.allclose(tr.poses_at(queries), np.stack(fwd), atol=1e-14, rtol=0)   # bulk and scalar paths agree
    assert np.allclose(tr.poses_at([t[0] - 1e9, t[-1] + 1e9]), np.stack([tr.pose_at(t[0]).as_array(), tr.pose_at(t[-1]).as_array()]), atol=1e-14, rtol=0)
    with pytest.raises(ValueError):
        tr.pose_at(float("nan"))


ZIGZAG_T = np.array([1_000_000, 1_200_000, 2_700_000, 2_700_500, 5_700_500])   # segments of 0.2 s, 1.5 s, 0.5 ms, 3 s
ZIGZAG_P = np.array([[0.0, 0, 0], [4, 0, 0], [1, 3, 0], [1, 3.5, 0], [-6, 1, 2]])   # out, back, tiny hop, far away
ZIGZAG_YAW = np.array([0.0, 1.0, -0.5, 2.0, 0.3])


def zigzag() -> WaypointTrajectory:
    return WaypointTrajectory(ZIGZAG_T, ZIGZAG_P, np.stack([axis_angle([0, 1, 0], a) for a in ZIGZAG_YAW]))


def zigzag_reference(t):
    """Independent reference: np.interp holds the end values outside the range; slerp about a fixed axis is linear in angle."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    pos = np.stack([np.interp(t, ZIGZAG_T, ZIGZAG_P[:, k]) for k in range(3)], axis=1)
    return pos, np.interp(t, ZIGZAG_T, ZIGZAG_YAW)


def assert_matches_zigzag_reference(tr, times, tol=1e-9):
    times = np.asarray(times, dtype=float)
    ref_pos, ref_yaw = zigzag_reference(times)
    bulk = tr.poses_at(times)
    assert bulk.shape == (len(times), 7)
    for t, row, pos, yaw in zip(times, bulk, ref_pos, ref_yaw):
        assert np.allclose(row[:3], pos, atol=tol, rtol=0), (t, row[:3], pos)   # an unclamped alpha extrapolates far off
        assert same_rotation(row[3:], axis_angle([0, 1, 0], yaw), tol=1e-9), t
        pose = tr.pose_at(t)
        assert np.allclose(pose.position, pos, atol=tol, rtol=0) and same_rotation(pose.quat, axis_angle([0, 1, 0], yaw), tol=1e-9), t


def test_time_is_clamped_to_the_first_and_last_waypoint():
    tr = zigzag()
    first, last = tr.pose_at(tr.start_us), tr.pose_at(tr.end_us)
    assert np.array_equal(first.position, ZIGZAG_P[0]) and np.array_equal(last.position, ZIGZAG_P[-1])
    before = [tr.start_us - 1, tr.start_us - 0.5, tr.start_us - 1e6, tr.start_us - 1e12, -1e18, 0]
    after = [tr.end_us + 1, tr.end_us + 0.5, tr.end_us + 1e6, tr.end_us + 1e12, 1e18, 1e300]
    for t in before:
        pose = tr.pose_at(t)
        assert np.array_equal(pose.position, ZIGZAG_P[0]), t                # exactly the first waypoint, not extrapolated
        assert np.array_equal(pose.as_array(), first.as_array()), t
    for t in after:
        pose = tr.pose_at(t)
        assert np.array_equal(pose.position, ZIGZAG_P[-1]), t
        assert np.array_equal(pose.as_array(), last.as_array()), t
    bulk = tr.poses_at(before + after)
    assert np.all(bulk[: len(before)] == first.as_array()) and np.all(bulk[len(before):] == last.as_array())
    assert same_rotation(first.quat, axis_angle([0, 1, 0], ZIGZAG_YAW[0])) and same_rotation(last.quat, axis_angle([0, 1, 0], ZIGZAG_YAW[-1]))
    for t in (-np.inf, np.inf):
        with pytest.raises(ValueError, match="finite"):
            tr.poses_at([t])


def test_alpha_never_extrapolates_at_and_around_waypoint_times():
    tr = zigzag()
    exact = np.concatenate([ZIGZAG_T, ZIGZAG_T.astype(float)])
    assert_matches_zigzag_reference(tr, exact)
    for i, t in enumerate(ZIGZAG_T):     # exactly at a waypoint time: that waypoint, bit for bit
        assert np.array_equal(tr.pose_at(t).position, ZIGZAG_P[i]), i
        assert np.array_equal(tr.poses_at([t])[0, :3], ZIGZAG_P[i]), i
    eps = np.array([-1.0, -0.5, -1e-6, 1e-6, 0.5, 1.0])              # just inside / outside every segment boundary
    near = (ZIGZAG_T[:, None] + eps[None, :]).reshape(-1)
    assert_matches_zigzag_reference(tr, near)
    rng = np.random.default_rng(6)
    assert_matches_zigzag_reference(tr, rng.uniform(ZIGZAG_T[0] - 2e6, ZIGZAG_T[-1] + 2e6, size=400))
    # inside the 0.5 ms hop, a query 1 us past its end must already belong to the next (3 s) segment
    hop_end = ZIGZAG_T[3]
    p = tr.pose_at(hop_end + 1).position
    assert np.allclose(p, ZIGZAG_P[3] + (ZIGZAG_P[4] - ZIGZAG_P[3]) / 3_000_000, atol=1e-12)
    p = tr.pose_at(hop_end - 1).position
    assert np.allclose(p, ZIGZAG_P[3] + (ZIGZAG_P[2] - ZIGZAG_P[3]) / 500, atol=1e-12)
    # every query stays in the bounding box of its two neighbouring waypoints (convex combination, alpha in [0, 1])
    q = rng.uniform(ZIGZAG_T[0] - 1e6, ZIGZAG_T[-1] + 1e6, size=500)
    pos = tr.poses_at(q)[:, :3]
    assert np.all(pos >= ZIGZAG_P.min(axis=0) - 1e-12) and np.all(pos <= ZIGZAG_P.max(axis=0) + 1e-12)


def test_two_waypoint_trajectory():
    tr = WaypointTrajectory.from_waypoints([5_000_000, 6_000_000], [[0.0, 1, 2], [4, 1, -2]], fronts=[[0, 0, -1], [1, 0, 0]])
    assert (len(tr), tr.start_us, tr.end_us, tr.duration_us) == (2, 5_000_000, 6_000_000, 1_000_000)
    assert np.allclose(tr.pose_at(5_250_000).position, [1.0, 1, 1]) and np.allclose(tr.pose_at(5_500_000).position, [2.0, 1, 0])
    mid = tr.pose_at(5_500_000)
    assert same_rotation(mid.quat, slerp(tr.quats[0], tr.quats[1], 0.5), tol=1e-12)
    assert np.allclose(tr.pose_at(5_500_000).forward, [np.sqrt(0.5), 0, -np.sqrt(0.5)], atol=1e-12)
    out = tr.poses_at([4_999_999, 5_000_000, 5_500_000, 6_000_000, 6_000_001])
    assert np.array_equal(out[[0, 1], :3], [[0, 1, 2], [0, 1, 2]]) and np.array_equal(out[[3, 4], :3], [[4, 1, -2], [4, 1, -2]])
    assert np.allclose(out[2, :3], [2, 1, 0])
    assert np.array_equal(tr.pose_at(-1e9).position, [0, 1, 2]) and np.array_equal(tr.pose_at(1e9).position, [4, 1, -2])
    ms = WaypointTrajectory.from_json([rec(1000.5), rec(1010.5, x=1.0)])                 # two legacy records
    assert (len(ms), ms.start_us, ms.end_us) == (2, 1_000_500, 1_010_500) and np.allclose(ms.pose_at(1_005_500).position, [0.5, 0, 0])


def test_query_times_may_be_ints_floats_numpy_scalars_or_arrays():
    tr = zigzag()
    t = 1_800_000
    ref = tr.pose_at(t).as_array()
    for q in (t, float(t), np.int64(t), np.float64(t), np.int32(t), np.array(t), np.array(float(t)), np.array([t])):
        assert np.array_equal(tr.pose_at(q).as_array(), ref), type(q)
    assert np.array_equal(tr.poses_at(t), tr.poses_at([t])) and tr.poses_at(t).shape == (1, 7)
    assert np.array_equal(tr.poses_at(np.array([t, t + 1], dtype=np.int64)), tr.poses_at([float(t), t + 1.0]))
    assert_matches_zigzag_reference(tr, [1_000_000.25, 1_100_000.5, 1_800_000.75, 2_700_250.5, 5_000_000.125, 5_700_500.0])
    a, b = tr.pose_at(1_100_000.0).position, tr.pose_at(1_100_000.5).position    # sub-microsecond queries interpolate
    assert 0 < np.linalg.norm(b - a) < 1e-4
    assert np.allclose(tr.pose_at(1_100_000.5).position, zigzag_reference(1_100_000.5)[0][0], atol=1e-12)


LEGACY = [
    {"timestamp": 5_123_456.789, "x": 1.0, "y": 2.0, "z": 3.0, "rx": 0.0, "ry": 0.0, "rz": -1.0},
    {"timestamp": 5_123_466.789, "x": 1.5, "y": 2.0, "z": 3.0, "rx": 3.0, "ry": 0.0, "rz": -4.0},
    {"timestamp": 5_123_496.789, "x": 2.5, "y": 2.5, "z": 2.0, "rx": 1.0, "ry": 1.0, "rz": 0.0},
]


def test_legacy_json_ms_forward_vectors_and_offset_start():
    tr = WaypointTrajectory.from_json(LEGACY)
    assert tr.start_us == 5_123_456_789 and tr.end_us == 5_123_496_789 and tr.duration_us == 40_000
    for rec, t in zip(LEGACY, tr.times_us):
        pose = tr.pose_at(t)
        assert np.allclose(pose.position, [rec["x"], rec["y"], rec["z"]])
        f = np.array([rec["rx"], rec["ry"], rec["rz"]])
        assert np.allclose(pose.forward, f / np.linalg.norm(f), atol=1e-12)
        assert abs(pose.up @ pose.forward) < 1e-12 and pose.right[1] == pytest.approx(0.0, abs=1e-12)  # no roll
    assert np.allclose(tr.pose_at(5_123_461_789).position, [1.25, 2.0, 3.0])  # halfway through the first interval


def test_records_round_trip(tmp_path):
    tr = WaypointTrajectory.from_json(LEGACY)
    rec = tr.to_records()
    assert rec[0]["timestamp"] == pytest.approx(5_123_456.789) and set(rec[0]) == {
        "timestamp", "x", "y", "z", "rx", "ry", "rz", "qx", "qy", "qz", "qw"}
    back = WaypointTrajectory.from_json(rec)
    assert np.array_equal(back.times_us, tr.times_us) and np.allclose(back.positions, tr.positions)
    assert all(same_rotation(a, b, 1e-12) for a, b in zip(back.quats, tr.quats))
    only_forward = [{k: r[k] for k in ("timestamp", "x", "y", "z", "rx", "ry", "rz")} for r in rec]  # legacy readers
    fwd = WaypointTrajectory.from_json(only_forward)
    assert all(same_rotation(a, b, 1e-9) for a, b in zip(fwd.quats, tr.quats))
    path = tmp_path / "traj.json"
    tr.save_json(path)
    assert isinstance(json.loads(path.read_text()), list)
    for loaded in (WaypointTrajectory.from_json(path), WaypointTrajectory.from_json(str(path)), ImportedTrajectory(path),
                   ImportedTrajectory([str(path)]), WaypointTrajectory.from_json({"points": rec})):
        assert np.array_equal(loaded.times_us, tr.times_us) and np.allclose(loaded.positions, tr.positions)
    with pytest.raises(ValueError, match="one path"):
        ImportedTrajectory([str(path), str(path)])


def test_quaternion_json():
    q = [axis_angle([0, 0, 1], 0.5), axis_angle([1, 1, 0], 1.0)]
    recs = [{"timestamp": 0, "x": 0, "y": 0, "z": 0, "qx": q[0][0], "qy": q[0][1], "qz": q[0][2], "qw": q[0][3]},
            {"timestamp": 1000, "x": 1, "y": 0, "z": 0, "qx": q[1][0], "qy": q[1][1], "qz": q[1][2], "qw": q[1][3],
             "rx": 5.0, "ry": 0.0, "rz": 0.0}]  # quaternion wins over a forward vector
    tr = WaypointTrajectory.from_json(recs)
    assert tr.end_us == 1_000_000
    assert all(same_rotation(a, b) for a, b in zip(tr.quats, q))
    assert same_rotation(tr.pose_at(500_000).quat, slerp(q[0], q[1], 0.5))


def test_origin_offset():
    base = WaypointTrajectory.from_json(LEGACY)
    moved = WaypointTrajectory.from_json(LEGACY, origin=(10.0, -1.0, 0.5))
    assert np.allclose(moved.positions - base.positions, [10.0, -1.0, 0.5])
    assert np.array_equal(moved.quats, base.quats) and np.array_equal(moved.times_us, base.times_us)
    assert np.allclose(moved.to_records()[0]["x"], LEGACY[0]["x"] + 10.0)
    with pytest.raises(ValueError, match="origin"):
        WaypointTrajectory.from_json(LEGACY, origin=(1, 2))


def rec(t, **kw):
    r = {"timestamp": t, "x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": -1.0}
    r.update(kw)
    return r


@pytest.mark.parametrize("records, match", [
    ([rec(0)], "at least 2"),
    ([], "at least 2"),
    ([rec(0), rec(10), rec(5)], "backwards"),
    ([rec(0), rec(10), rec(10)], "duplicate"),
    ([rec(0), rec(0.0004)], "duplicate"),   # equal after rounding to microseconds
    ([rec(0), rec(10, x=float("nan"))], "non-finite"),
    ([rec(0), rec(float("inf"))], "non-finite"),
    ([rec(0), rec(10, rx=0.0, ry=0.0, rz=0.0)], "zero length"),
    ([rec(0), {k: v for k, v in rec(10).items() if k != "y"}], "missing y"),
    ([rec(0), {k: v for k, v in rec(10).items() if k not in ("rx", "ry", "rz")}], "rx,ry,rz"),
    ([rec(0), rec(10, x="abc")], "non-numeric"),
    ([rec(0), "not a record"], "expected an object"),
])
def test_invalid_records_raise_clear_errors(records, match):
    with pytest.raises(ValueError, match=match):
        WaypointTrajectory.from_json(records)


def test_duplicate_timestamps_can_be_dropped():
    tr = WaypointTrajectory.from_json([rec(0, x=1.0), rec(10, x=2.0), rec(10, x=3.0), rec(20, x=4.0)], on_duplicate="drop")
    assert np.array_equal(tr.times_us, [0, 10_000, 20_000]) and np.allclose(tr.positions[:, 0], [1, 2, 4])
    with pytest.raises(ValueError, match="on_duplicate"):
        WaypointTrajectory.from_json(LEGACY, on_duplicate="ignore")


def test_invalid_waypoints_raise():
    p, f = np.zeros((3, 3)), np.tile([0.0, 0, -1], (3, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        WaypointTrajectory.from_waypoints([0, 5, 5], p, fronts=f)
    with pytest.raises(ValueError, match="at least 2"):
        WaypointTrajectory.from_waypoints([0], p[:1], fronts=f[:1])
    with pytest.raises(ValueError, match="either quats or fronts"):
        WaypointTrajectory.from_waypoints([0, 1, 2], p, quats=np.tile([0.0, 0, 0, 1], (3, 1)), fronts=f)
    with pytest.raises(ValueError, match="zero"):
        WaypointTrajectory.from_waypoints([0, 1, 2], p, fronts=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        WaypointTrajectory.from_waypoints([0, 1, 2], np.full((3, 3), np.nan), fronts=f)
    with pytest.raises(ValueError, match="expected positions"):
        WaypointTrajectory.from_waypoints([0, 1, 2], p[:2], fronts=f)
    with pytest.raises(ValueError, match="zero quaternion"):
        WaypointTrajectory([0, 1], p[:2], np.zeros((2, 4)))


def test_default_orientation_looks_down_minus_z():
    tr = WaypointTrajectory.from_waypoints([0, 1000], [[0, 0, 0], [1, 1, 1]])
    assert np.allclose(tr.pose_at(500).forward, [0, 0, -1])


def test_looking_straight_up_and_down():
    tr = WaypointTrajectory.from_waypoints([0, 1_000_000, 2_000_000], np.zeros((3, 3)), fronts=[[0, 1, 0], [0, -1, 0], [0, 1, 0]])
    assert np.allclose(tr.pose_at(0).forward, [0, 1, 0], atol=1e-12)
    assert np.allclose(tr.pose_at(1_000_000).forward, [0, -1, 0], atol=1e-12)
    for t in np.linspace(0, 2_000_000, 21):
        pose = tr.pose_at(t)
        assert np.all(np.isfinite(pose.as_array())) and abs(np.linalg.norm(pose.quat) - 1) < 1e-12
        assert np.linalg.det(pose.rotation()) == pytest.approx(1.0)
    mid = tr.pose_at(500_000)  # half way between up and down: horizontal
    assert abs(mid.forward[1]) < 1e-9
    assert np.allclose(Pose.from_look(np.zeros(3), [0, 1, 0]).forward, [0, 1, 0])


def test_headset_scale_bulk_load_is_fast():
    n = 6000  # 60 s at 100 Hz
    rng = np.random.default_rng(3)
    recs = [rec(9e5 + 10 * i, x=float(i) * 1e-3, y=1.5, z=0.0, rx=float(rng.normal()), ry=0.1, rz=float(rng.normal())) for i in range(n)]
    t0 = time.perf_counter()
    tr = WaypointTrajectory.from_json(recs)
    ts = np.arange(tr.start_us, tr.end_us, 10_000)
    tr.poses_at(ts)
    assert time.perf_counter() - t0 < 5.0
    assert len(tr) == n


# ---------------------------------------------------------------- FreeSpaceGrid
def test_grid_box_room(room_grid):
    g = room_grid
    assert g.enclosed and g.pitch == pytest.approx(MIN_CLEARANCE / 3)
    assert np.allclose(g.bounds, [-ROOM_HALF, ROOM_HALF])
    inside = np.array([[0, 0, 0], [4.5, 0, 0], [-4.9, 1.9, 3.9], [4.75, 0, 0]])
    outside = np.array([[5.5, 0, 0], [0, 3.0, 0], [0, 0, -4.5], [50, 50, 50]])
    assert g.is_free(inside[:1], 1.5)[0] and g.is_free(inside[:2], 0.3).all()
    assert not g.is_free(inside[2:3], 0.3)[0] and not g.is_free(inside[3:], 0.3)[0]  # 0.25 from the wall
    assert not g.is_free(outside, 0.0).any()
    assert np.all(g.clearance_at(outside) < 0)
    assert not g.is_free([[np.nan, 0, 0]], 0.0)[0]
    rng = np.random.default_rng(0)
    p = rng.uniform(-ROOM_HALF + 0.05, ROOM_HALF - 0.05, size=(5000, 3))
    err = np.abs(g.clearance_at(p) - room_clearance(p, pillar=False))
    assert err.max() <= g.pitch


def test_grid_room_with_pillar(pillar_grid):
    g = pillar_grid
    assert g.enclosed
    assert g.is_free([[0, 0, 0], [-3, 0, -2], [3, 0, 3]], 0.3).all()
    assert not g.is_free([[1.0, 0.0, 1.0], [0.9, 0.0, 1.1], [1.0, 1.0, 1.0]], 0.0).any()      # inside the pillar
    assert not g.is_free([[0.35, 0, 1.0], [1.0, 0, 1.75], [1.0, 1.65, 1.0]], 0.3).any()       # closer than 0.3 to it
    assert g.is_free([[0.1, 0, 1.0]], 0.3)[0]                                                 # 0.4 from it
    rng = np.random.default_rng(1)
    p = rng.uniform(-ROOM_HALF + 0.05, ROOM_HALF - 0.05, size=(8000, 3))
    truth = room_clearance(p)
    err = np.abs(g.clearance_at(p) - truth)
    assert err.max() <= 1.3 * g.pitch and np.percentile(err, 99) <= g.pitch
    flipped = (g.clearance_at(p) > 0) != (truth > 0)
    assert np.all(np.abs(truth[flipped]) < g.pitch)     # free / solid only disagree within a voxel of a surface
    # a solid resting on the floor (faces coincide with the room's) is solid too
    table = trimesh.creation.box(extents=(1, 1, 1))
    table.apply_translation((-2.5, -1.5, -2.0))
    gt = FreeSpaceGrid(trimesh.util.concatenate([make_room(), table]))
    assert gt.enclosed and not gt.is_free([[-2.5, -1.5, -2.0]], 0.0)[0] and gt.is_free([[-2.5, 0.0, -2.0]], 0.3)[0]


def test_grid_segments(pillar_grid):
    g = pillar_grid
    a = np.array([[-3.0, 0, 1.0], [-3.0, 0, -2.0], [-3.0, 0, 1.0], [-4.9, 0, 0], [-3.0, 0, 1.0]])
    b = np.array([[4.0, 0, 1.0], [4.0, 0, -2.0], [4.0, 0, 3.0], [4.9, 0, 0], [-3.0, 0, 1.0]])
    assert g.segments_free(a, b, 0.3).tolist() == [False, True, True, False, True]   # through pillar / free / free / wall clearance / point
    assert g.segments_free(a, b, 0.3, n_samples=200).tolist() == [False, True, True, False, True]
    assert g.segments_free(a[:0], b[:0], 0.3).shape == (0,)
    # a segment that grazes past the pillar at 0.2 is rejected for 0.3 but accepted for 0.05
    assert not g.segments_free([[-3.0, 0, 0.3]], [[4.0, 0, 0.3]], 0.3)[0]
    assert g.segments_free([[-3.0, 0, 0.3]], [[4.0, 0, 0.3]], 0.05)[0]


def test_sample_free(pillar_grid):
    g = pillar_grid
    p = g.sample_free(2000, 0.3, np.random.default_rng(0))
    assert p.shape == (2000, 3) and g.is_free(p, 0.3).all()
    assert (room_clearance(p) >= 0.3 - g.pitch).all()
    assert np.array_equal(g.sample_free(50, 0.3, 7), g.sample_free(50, 0.3, 7))
    assert p[:, 0].min() < -4 and p[:, 0].max() > 4          # covers the room
    with pytest.raises(RuntimeError, match="no free space"):
        g.sample_free(10, 5.0, 0)


def test_grid_rotated_room_is_not_leaky():
    rot = tf.rotation_matrix(0.6, [1.0, 1.0, 0.3])
    room = make_room()
    room.apply_transform(rot)
    g = FreeSpaceGrid(room)
    assert g.enclosed
    p = np.random.default_rng(2).uniform(-3.5, 3.5, size=(4000, 3))
    local = (p - rot[:3, 3]) @ rot[:3, :3]                    # R^T p
    truth = -box_sd(local, ROOM_HALF)
    ok = truth > 0.4
    assert g.is_free(p[ok], 0.0).all()
    assert np.abs(g.clearance_at(p) - truth)[truth > 0].max() <= 1.2 * g.pitch


def test_grid_sphere_is_free_inside():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=3.0)
    g = FreeSpaceGrid(sphere, pitch=0.1)
    p = np.random.default_rng(3).uniform(-3.2, 3.2, size=(4000, 3))
    r = np.linalg.norm(p, axis=1)
    err = np.abs(g.clearance_at(p) - (3.0 - r))[r < 3.0]
    assert g.enclosed and err.max() <= g.pitch
    assert not g.is_free(p[r > 3.05], 0.0).any()


def test_grid_accepts_scene_like_objects():
    class Scene:
        def to_trimesh(self):
            return make_room()

    g = FreeSpaceGrid(Scene(), min_clearance=0.6)
    assert g.enclosed and g.pitch == pytest.approx(0.2) and g.is_free([[0, 0, 0]], 0.6)[0]
    with pytest.raises(TypeError):
        FreeSpaceGrid("not a mesh")


def test_grid_pitch_is_capped():
    g = FreeSpaceGrid(make_room(), min_clearance=0.3, max_voxels=40_000)
    assert np.prod(g.shape) <= 40_000 and g.pitch > 0.1
    with pytest.raises(ValueError):
        FreeSpaceGrid(make_room(), pitch=0.001)


def open_box() -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=2 * ROOM_HALF)
    return trimesh.Trimesh(box.vertices, box.faces[box.face_normals[:, 1] < 0.5])   # no lid


def test_leaky_mesh_warns_and_falls_back():
    with pytest.warns(UserWarning, match="does not enclose"):
        g = FreeSpaceGrid(open_box())
    assert not g.enclosed
    assert g.is_free([[0, 0, 0], [0, 1.9, 0]], 0.3).all()            # every non-surface voxel inside the bounds
    assert not g.is_free([[4.95, 0, 0], [0, -1.95, 0]], 0.3).any()   # clearance still measured from the surface
    assert not g.is_free([[0, 0, 4.5], [0, 2.5, 0]], 0.0).any()      # outside the bounds
    assert g.sample_free(20, 0.3, 0).shape == (20, 3)
    tr = SplineTrajectoryGenerator(g).generate(5_000_000, n_key_points=6, rng=0)
    assert g.is_free(tr.positions, 0.3).all()
    tr = RectilinearTrajectoryGenerator(g).generate(5_000_000, rng=0)
    assert g.is_free(tr.positions, 0.3).all()


# ---------------------------------------------------------------- generators
@pytest.fixture(scope="module")
def spline_gen(pillar_grid):
    return SplineTrajectoryGenerator(pillar_grid, min_clearance=MIN_CLEARANCE)


@pytest.fixture(scope="module")
def rect_gen(pillar_grid):
    return RectilinearTrajectoryGenerator(pillar_grid, min_clearance=MIN_CLEARANCE)


def assert_all_free(tr, grid):
    assert grid.is_free(tr.positions, MIN_CLEARANCE).all()
    assert (room_clearance(tr.positions) >= MIN_CLEARANCE - grid.pitch).all()   # analytic, one voxel of slack
    mid = 0.5 * (tr.positions[:-1] + tr.positions[1:])                          # what lerp actually visits
    assert grid.is_free(mid, MIN_CLEARANCE).all()


def test_rectilinear_is_horizontal_free_and_constant_heading(rect_gen, pillar_grid):
    for seed in range(15):
        tr = rect_gen.generate(4_000_000, rng=seed)
        assert (tr.start_us, tr.end_us, len(tr)) == (0, 4_000_000, 2)
        assert_all_free(tr, pillar_grid)
        assert tr.positions[0, 1] == tr.positions[1, 1]                                     # constant y
        d = tr.positions[1] - tr.positions[0]
        assert np.linalg.norm(d) > 1.0
        for t in (0, 1_000_000, 4_000_000):
            fwd = tr.pose_at(t).forward
            assert np.allclose(fwd, d / np.linalg.norm(d), atol=1e-12) and abs(fwd[1]) < 1e-12
        assert pillar_grid.segments_free(tr.positions[:1], tr.positions[1:], MIN_CLEARANCE, n_samples=400)[0]


def test_rectilinear_stops_at_valid_distance_and_options(pillar_grid):
    long = RectilinearTrajectoryGenerator(pillar_grid, valid_distance=3.0).generate(1_000_000, rng=0)
    assert np.linalg.norm(long.positions[1] - long.positions[0]) > 3.0
    start = np.array([-3.0, 0.2, -3.0])
    tr = RectilinearTrajectoryGenerator(pillar_grid).generate(1_000_000, rng=0, initial_point=start)
    assert np.array_equal(tr.positions[0], start) and tr.positions[1, 1] == 0.2
    tr = RectilinearTrajectoryGenerator(pillar_grid, fixed_height=0.5).generate(1_000_000, rng=1)
    assert np.allclose(tr.positions[:, 1], 0.5)
    with pytest.raises(ValueError, match="not in free space"):
        RectilinearTrajectoryGenerator(pillar_grid).generate(1_000_000, initial_point=[1.0, 0.0, 1.0])   # in the pillar
    with pytest.raises(ValueError, match="duration"):
        RectilinearTrajectoryGenerator(pillar_grid).generate(0)
    tiny = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    tiny.invert()
    with pytest.raises(RuntimeError, match="no free space"):   # clearance 0.3 does not fit into a 0.5 m box
        RectilinearTrajectoryGenerator(tiny).generate(1_000_000, rng=0)


def test_spline_waypoints_free_smooth_and_constant_speed(spline_gen, pillar_grid):
    for seed in range(15):
        tr = spline_gen.generate(10_000_000, n_key_points=10, rng=seed)
        assert (tr.start_us, tr.end_us, len(tr)) == (0, 10_000_000, 200)
        assert_all_free(tr, pillar_grid)
        steps = np.linalg.norm(np.diff(tr.positions, axis=0), axis=1)
        assert steps.max() < 0.08 and steps.min() > 0.01                      # fine spacing, no jumps
        ang = [angle_between(a, b) for a, b in zip(tr.quats[:-1], tr.quats[1:])]
        assert np.degrees(max(ang)) < 15.0                                     # bounded angular step (worst case: U-turn)
        fwd = np.stack([tr.pose_at(t).forward for t in np.linspace(0, 10_000_000, 51)])
        assert np.abs(fwd[:, 1]).max() < 1e-9                                  # horizontal heading, no roll
        up = np.stack([tr.pose_at(t).up for t in np.linspace(0, 10_000_000, 51)])
        assert (up[:, 1] > 0.99).all()
        assert steps.max() / steps.min() < 3.0                                 # constant arc-length speed (chords shrink in U-turns)


def test_spline_max_turn_limits_reversals(pillar_grid):
    gen = SplineTrajectoryGenerator(pillar_grid, max_turn_deg=45.0)
    for seed in range(5):
        tr = gen.generate(10_000_000, n_key_points=8, rng=seed)
        assert_all_free(tr, pillar_grid)
        v = np.diff(tr.positions, axis=0)
        steps = np.linalg.norm(v, axis=1)
        assert steps.max() / steps.min() < 1.3                              # gentle curves: chords ~ arcs
        cosang = np.sum(v[:-1] * v[1:], axis=1) / (steps[:-1] * steps[1:])
        assert np.degrees(np.arccos(np.clip(cosang, -1, 1))).max() < 10.0   # direction change per waypoint


def test_spline_100hz_playback_is_smooth(spline_gen):
    tr = spline_gen.generate(10_000_000, n_key_points=10, rng=4)
    poses = tr.poses_at(np.arange(0, 10_000_001, 10_000))
    assert poses.shape == (1001, 7) and np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).max() < 0.03
    ang = [angle_between(a[3:], b[3:]) for a, b in zip(poses[:-1], poses[1:])]
    assert np.degrees(max(ang)) < 6.0


def test_spline_options(spline_gen, pillar_grid):
    start = np.array([-3.0, 0.0, -3.0])
    tr = spline_gen.generate(5_000_000, n_key_points=5, rng=0, initial_point=start, num_samples_per_point=30)
    assert len(tr) == 150 and np.allclose(tr.positions[0], start, atol=1e-9)
    two = spline_gen.generate(1_000_000, n_key_points=2, rng=0)     # a straight step
    assert len(two) == 40 and np.allclose(np.linalg.norm(two.positions[-1] - two.positions[0]), 1.0, atol=1e-6)
    long_step = SplineTrajectoryGenerator(pillar_grid, step=2.0).generate(5_000_000, n_key_points=4, rng=0)
    assert_all_free(long_step, pillar_grid)
    for bad in (1, 0, 2.5, "10", True):
        with pytest.raises(ValueError, match="n_key_points"):
            spline_gen.generate(1_000_000, n_key_points=bad)
    with pytest.raises(ValueError, match="too short"):
        spline_gen.generate(100, n_key_points=10)
    with pytest.raises(ValueError, match="not in free space"):
        spline_gen.generate(1_000_000, initial_point=[9.0, 0.0, 0.0], rng=0)


@pytest.mark.parametrize("make", [
    lambda g: RectilinearTrajectoryGenerator(g),
    lambda g: SplineTrajectoryGenerator(g),
])
def test_seeded_determinism(make, pillar_grid):
    gen = make(pillar_grid)
    a, b, c = gen.generate(3_000_000, rng=11), gen.generate(3_000_000, rng=11), gen.generate(3_000_000, rng=12)
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.quats, b.quats) and np.array_equal(a.times_us, b.times_us)
    assert a.positions.shape != c.positions.shape or not np.allclose(a.positions, c.positions)
    many1, many2 = gen.generate_many(4, 3_000_000, rng=5), gen.generate_many(4, 3_000_000, rng=5)
    assert len(many1) == 4
    assert all(np.array_equal(x.positions, y.positions) for x, y in zip(many1, many2))
    assert not np.allclose(many1[0].positions[0], many1[1].positions[0])   # successive draws differ
    gen_rng = np.random.default_rng(5)
    assert np.array_equal(gen.generate(3_000_000, rng=gen_rng).positions, many1[0].positions)


def test_spline_generate_many_positional_key_points(spline_gen, pillar_grid):
    many = spline_gen.generate_many(3, 2_000_000, 6, rng=0)     # (n, duration_us, n_key_points)
    assert [len(t) for t in many] == [120, 120, 120]
    assert all(pillar_grid.is_free(t.positions, MIN_CLEARANCE).all() for t in many)


def test_generators_accept_a_mesh_directly():
    gen = SplineTrajectoryGenerator(make_pillar_room(), min_clearance=0.4)
    assert isinstance(gen.grid, FreeSpaceGrid) and gen.grid.pitch == pytest.approx(0.4 / 3)
    tr = gen.generate(2_000_000, n_key_points=4, rng=0)
    assert gen.grid.is_free(tr.positions, 0.4).all()


def test_timing_330k_face_mesh():
    mesh = trimesh.creation.icosphere(subdivisions=7, radius=5.0)
    assert len(mesh.faces) == 327_680
    t0 = time.perf_counter()
    grid = FreeSpaceGrid(mesh)
    t1 = time.perf_counter()
    splines = SplineTrajectoryGenerator(grid).generate_many(5, 10_000_000, n_key_points=20, rng=0)
    t2 = time.perf_counter()
    lines = RectilinearTrajectoryGenerator(grid).generate_many(5, 5_000_000, rng=0)
    t3 = time.perf_counter()
    print(f"\n[timing] {len(mesh.faces)} faces: grid {grid.shape} pitch {grid.pitch:.3f}: build {t1 - t0:.2f} s, "
          f"5 splines {t2 - t1:.2f} s, 5 rectilinear {t3 - t2:.2f} s, total {t3 - t0:.2f} s")
    for tr in splines + lines:
        assert grid.is_free(tr.positions, MIN_CLEARANCE).all()
        assert (np.linalg.norm(tr.positions, axis=1) <= 5.0 - MIN_CLEARANCE + grid.pitch).all()
    assert t3 - t0 < 30.0   # the old one-point-at-a-time approach needed minutes


# ---- street walks ---------------------------------------------------------------------------------------------------
TOWN_HALF = 30.0                                  # ground plane x, z in [-30, 30], y = 0
BUILDINGS = [((8.0, 0.0), (3.0, 4.0), 8.0),       # (centre x z, half extents x z, height)
             ((-10.0, 10.0), (2.0, 2.0), 5.0)]
KERB = ((0.0, -12.0), (5.0, 1.0), 0.15)           # a low slab: still walkable


def make_town() -> trimesh.Trimesh:
    ground = trimesh.creation.box(extents=[2 * TOWN_HALF, 0.1, 2 * TOWN_HALF])
    ground.apply_translation([0, -0.05, 0])
    parts = [ground]
    for (cx, cz), (hx, hz), h in BUILDINGS + [KERB]:
        b = trimesh.creation.box(extents=[2 * hx, h, 2 * hz])
        b.apply_translation([cx, h / 2, cz])
        parts.append(b)
    return trimesh.util.concatenate(parts)


def building_distance(xz) -> np.ndarray:
    """Horizontal distance from (M, 2) points to the nearest building footprint (negative inside)."""
    d = [box_sd(np.c_[xz[:, 0], np.zeros(len(xz)), xz[:, 1]], np.array([hx, 1.0, hz]), (cx, 0.0, cz))
         for (cx, cz), (hx, hz), _ in BUILDINGS]
    return np.min(d, axis=0)


@pytest.fixture(scope="module")
def street_gen():
    return StreetTrajectoryGenerator(make_town())


def test_street_map(street_gen):
    g = street_gen
    walkable = lambda x, z: bool(g.walkable[tuple(((np.array([x, z]) - g.origin) // g.cell).astype(int))])  # noqa: E731
    assert not walkable(8.0, 0.0), "a roof is not ground"
    assert not walkable(8.0, 4.5) and walkable(8.0, 5.5), "1 m clearance around the building"
    assert walkable(-20.0, -20.0) and walkable(0.0, -12.0), "open ground and the kerb are walkable"
    assert not walkable(27.0, 0.0), "5 m away from the scene edge"
    assert g.ground_at([[0.0, -12.0]])[0] == pytest.approx(0.15, abs=0.02)
    assert g.ground_at([[-20.0, -20.0]])[0] == pytest.approx(0.0, abs=0.02)
    expected = 50.0 ** 2 - (2 * 4.0) * (2 * 5.0) - 2 * (2 * 3.0) ** 2  # inner square minus buildings + clearance
    assert g.walkable_area == pytest.approx(expected, rel=0.06)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_street_walks(street_gen, seed):
    g, duration_us = street_gen, 30_000_000
    tr = g.generate(duration_us, rng=seed)
    p, xz = tr.positions, tr.positions[:, [0, 2]]
    assert tr.times_us[0] == 0 and tr.times_us[-1] == duration_us
    steps = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    assert steps.sum() == pytest.approx(g.speed * duration_us / 1e6, rel=0.01)
    assert steps.max() == pytest.approx(steps.mean(), rel=0.01), "constant speed"
    assert (building_distance(xz) >= 0.5 * g.min_clearance).all()
    assert (np.abs(xz) <= TOWN_HALF - 4.0).all()
    assert np.allclose(p[:, 1] - g.ground_at(xz), g.eye_height, atol=0.16)  # the kerb is smoothed over
    fronts = quats_to_fronts(tr.quats)
    assert np.degrees(np.arcsin(fronts[:, 1])) == pytest.approx(np.full(len(p), g.pitch_deg), abs=1e-6)
    heading = fronts[:-1, [0, 2]] / np.linalg.norm(fronts[:-1, [0, 2]], axis=1, keepdims=True)
    motion = np.diff(xz, axis=0) / steps[:, None]
    assert np.mean(np.sum(heading * motion, axis=1) > 0.8) > 0.95, "looks where it walks"


def test_street_seeds_and_initial_point(street_gen):
    a, b = street_gen.generate(5_000_000, rng=7), street_gen.generate(5_000_000, rng=7)
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.quats, b.quats)
    assert not np.allclose(a.positions, street_gen.generate(5_000_000, rng=8).positions)
    start = street_gen.generate(5_000_000, rng=0, initial_point=[-20.0, 9.0, -15.0]).positions[0]
    assert np.hypot(start[0] + 20.0, start[2] + 15.0) < 0.5
    near_roof = street_gen.generate(5_000_000, rng=0, initial_point=[8.0, 0.0, 0.0]).positions[0]
    assert building_distance(near_roof[None, [0, 2]])[0] >= street_gen.min_clearance - street_gen.cell
    many = street_gen.generate_many(3, 2_000_000, rng=0)
    assert len(many) == 3 and not np.allclose(many[0].positions[0], many[1].positions[0])


def test_street_small_and_invalid_scenes():
    room = trimesh.creation.box(extents=[6.0, 0.1, 6.0])     # smaller than the edge margin: ignored with a warning
    with pytest.warns(UserWarning, match="edge_margin"):
        g = StreetTrajectoryGenerator(room)
    assert (np.abs(g.generate(3_000_000, rng=0).positions[:, [0, 2]]) <= 2.0 + g.cell).all()
    with pytest.raises(ValueError, match="no walkable ground"):
        StreetTrajectoryGenerator(trimesh.creation.box(extents=[1.2, 0.1, 1.2]))
    indoor = StreetTrajectoryGenerator(make_room(), edge_margin=0)  # a closed box is a room: walk its floor
    p = indoor.generate(5_000_000, rng=0).positions
    assert np.allclose(p[:, 1], -ROOM_HALF[1] + indoor.eye_height, atol=1e-6)
    assert (np.abs(p[:, [0, 2]]) <= ROOM_HALF[[0, 2]] - 0.5 * indoor.min_clearance).all()
    for bad in ({"speed": 0}, {"eye_height": -1}, {"min_clearance": float("nan")}, {"pitch_deg": 95},
                {"edge_margin": -1}):
        with pytest.raises(ValueError):
            StreetTrajectoryGenerator(room, **bad)
    with pytest.raises(ValueError, match="duration_us"):
        g.generate(0)


def test_street_from_pipeline_spec():
    from backend.pipeline import build_trajectories

    class SceneLike:
        def to_trimesh(self):
            return make_town()

    trajs = build_trajectories({"type": "street", "duration_s": 4.0, "count": 2, "seed": 3, "speed": 2.0,
                                "eye_height": 1.2}, SceneLike())
    assert len(trajs) == 2
    for tr in trajs:
        assert tr.end_us - tr.start_us == 4_000_000
        assert np.linalg.norm(np.diff(tr.positions[:, [0, 2]], axis=0), axis=1).sum() == pytest.approx(8.0, rel=0.01)
        assert np.allclose(tr.positions[:, 1], 1.2, atol=0.16)
    with pytest.raises(ValueError, match="need a scene"):
        build_trajectories({"type": "street", "duration_s": 1.0})


# ---- street walks with learned human motion -------------------------------------------------------------------------
def test_street_spec_with_motion_replaces_the_glide(motion_checkpoint):
    from backend.pipeline import build_trajectories

    class SceneLike:
        def to_trimesh(self):
            return make_town()

    spec = {"type": "street", "duration_s": 4.0, "count": 2, "seed": 3, "speed": 1.3, "eye_height": 1.6}
    plain = build_trajectories(spec, SceneLike())
    cache = {}
    human = build_trajectories({**spec, "motion": {"path": str(motion_checkpoint)}}, SceneLike(), cache)
    again = build_trajectories({**spec, "motion": {"path": str(motion_checkpoint)}}, SceneLike(), cache)
    assert len(human) == 2 and sum(k[0] == "motion" for k in cache if isinstance(k, tuple)) == 1   # loaded once
    for p, h, h2 in zip(plain, human, again):
        assert h.start_us == p.start_us and abs(h.end_us - p.end_us) < 20_000
        assert len(h) > len(p) and np.array_equal(h.positions, h2.positions)       # sampled at 50 Hz, reproducible
        assert not np.allclose(h.quats[:, 2], p.quats[0, 2])                       # it rolls: the glide did not


def test_motion_errors_are_specific(tmp_path, monkeypatch):
    import sys

    from backend.pipeline import build_trajectories

    class SceneLike:
        def to_trimesh(self):
            return make_town()

    spec = {"type": "street", "duration_s": 2.0, "seed": 1}
    with pytest.raises(FileNotFoundError, match="motion model file not found"):
        build_trajectories({**spec, "motion": {"path": str(tmp_path / "gone.pt")}}, SceneLike())
    with pytest.raises(ValueError, match="invalid motion model id"):
        build_trajectories({**spec, "motion": {"model": "../x"}}, SceneLike())
    (tmp_path / "m.pt").write_bytes(b"x")
    for name in ("backend.motion.style", "backend.motion.train", "backend.motion.model"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)                                   # torch is not installed
    with pytest.raises(ImportError, match=r"pip install -e \.\[torch\]"):
        build_trajectories({**spec, "motion": {"path": str(tmp_path / "m.pt")}}, SceneLike())
