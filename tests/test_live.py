"""Live rendering: poses arriving one at a time -> exact RGB / events / depth / flow views."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.config import SimConfig
from backend.live import FLOOR_ORIGIN_HEIGHT, VIEWS, LiveRenderer
from backend.scene import load_scene

gl = pytest.mark.usefixtures("gl_context")
W, H = 48, 36


def _cfg(**kw):
    d = {"camera": {"width": W, "height": H}, "dt_us": 10_000}
    d.update(kw)
    return SimConfig.from_dict(d)


@pytest.fixture(scope="module")
def scene(demo_scene):
    return load_scene(demo_scene.scene_spec)


@pytest.fixture
def live(scene, gl_context):
    with LiveRenderer(scene, _cfg()) as renderer:
        yield renderer


FORWARD = (0.0, 0.0, -1.0)


def test_origin_modes(scene, gl_context):
    lo, hi = scene.bounds
    centre = (lo + hi) / 2
    with LiveRenderer(scene, _cfg(), origin="centre") as a:
        assert np.allclose(a.origin, centre)
        assert np.allclose(a.world_pose((1, 2, 3), FORWARD).position, centre + [1, 2, 3])
    with LiveRenderer(scene, _cfg(), origin="floor") as b:
        assert np.allclose(b.origin, [centre[0], lo[1] + FLOOR_ORIGIN_HEIGHT, centre[2]])
    with pytest.raises(ValueError, match="origin"):
        LiveRenderer(scene, _cfg(), origin="moon")
    with pytest.raises(ValueError):
        LiveRenderer(scene, _cfg(), jpeg_quality=0)


@gl
@pytest.mark.parametrize("view", VIEWS)
def test_every_view_is_an_rgb_image(live, view):
    for k in range(3):
        image = live.render((0.1 * k, 0.0, 0.0), FORWARD, 33_000 * k, view)
    scale = 2 if view == "grid" else 1
    assert image.shape == (H * scale, W * scale, 3) and image.dtype == np.uint8
    decoded = cv2.imdecode(np.frombuffer(live.encode(image), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == image.shape                            # a real JPEG of the same size
    with pytest.raises(ValueError, match="view"):
        live.render((0, 0, 0), FORWARD, 200_000, "thermal")


@gl
def test_rgb_matches_the_simulator_frame(scene, gl_context):
    """The live RGB of a pose is the simulator's RGB of that pose (same renderer, same origin convention)."""
    from backend.simulator import Simulator
    from backend.trajectories import WaypointTrajectory

    with LiveRenderer(scene, _cfg(), origin="centre") as live:
        pos = np.array([0.3, 0.2, 0.1])
        image = live.render(pos, FORWARD, 0, "rgb")
        world = live.origin + pos
    traj = WaypointTrajectory.from_waypoints([0, 10_000], [world, world], fronts=[FORWARD, FORWARD])
    cfg = _cfg(gt=("rgb",))
    with Simulator(scene, cfg) as sim:
        frame = next(iter(sim.run(traj))).gt.rgb_left
    assert np.array_equal(image, frame)


@gl
def test_events_follow_the_motion_and_flow_follows_the_speed(live):
    still = [live.render((0, 0, 0), FORWARD, 33_000 * k, "events") for k in range(3)]
    assert all((frame == 255).all() for frame in still), "a static camera makes no events"

    moving = [live.render((0.0, 0.0, -0.05 * k), FORWARD, 100_000 + 33_000 * k, "events") for k in range(1, 6)]
    assert any((frame != 255).any() for frame in moving), "a moving camera does"

    live.render((0, 0, 0), FORWARD, 400_000, "flow")                  # a fresh start, then steady motion
    slow = live.render((0, 0, -0.01), FORWARD, 433_000, "flow")
    fast = live.render((0, 0, -0.07), FORWARD, 466_000, "flow")       # six times further in the same time
    assert (slow != 255).any() or (fast != 255).any()
    # the faster motion saturates more of the colour scale than the slower one (white = no motion)
    assert (fast < 255).sum() >= (slow < 255).sum()


@gl
def test_flow_is_expressed_per_simulation_step(scene, gl_context):
    """Constant velocity: the live flow equals the simulator's (px per dt_us), whatever the frame interval is."""
    from backend.simulator import Simulator
    from backend.trajectories import WaypointTrajectory

    speed = 2.0                                                        # m/s along -Z
    with LiveRenderer(scene, _cfg(gt=("flow",)), origin="centre") as live:
        centre = live.origin.copy()
        live.render((0, 0, 0), FORWARD, 0, "flow")
        # 20 ms frames: 40 mm per frame; the flow is expressed per 10 ms (dt_us), like a simulation's
        live.render((0, 0, -speed * 0.02), FORWARD, 20_000, "flow")
        flow, valid = live.last_flow, live.last_flow_valid.astype(bool)
    traj = WaypointTrajectory.from_waypoints(
        [0, 1_000_000], [centre + [0, 0, -speed * 0.02], centre + [0, 0, -speed * 1.02]], fronts=[FORWARD, FORWARD])
    with Simulator(scene, _cfg(gt=("flow",))) as sim:
        gt = next(iter(sim.run(traj))).gt
    assert gt.flow_dt_us == 10_000
    both = valid & gt.flow_valid.astype(bool)
    assert both.sum() > 0.5 * both.size and np.abs(gt.flow[both]).max() > 0.01
    assert np.allclose(flow[both], gt.flow[both], atol=1e-4), np.abs(flow[both] - gt.flow[both]).max()


@gl
def test_a_pause_restarts_the_event_model(live):
    live.render((0, 0, 0), FORWARD, 0, "events")
    live.render((0, 0, -0.1), FORWARD, 33_000, "events")
    image = live.render((0.0, 0.0, -1.0), FORWARD, 5_000_000, "events")   # 5 s later: no giant step, no events
    assert (image == 255).all()
    live.render((0.0, 0.0, -1.1), FORWARD, 5_033_000, "events")           # and it carries on afterwards


@gl
def test_non_increasing_time_does_not_crash(live):
    live.render((0, 0, 0), FORWARD, 1000, "events")
    live.render((0, 0, -0.1), FORWARD, 1000, "events")                    # same timestamp
    live.render((0, 0, -0.2), FORWARD, 900, "events")                     # earlier one


def _rotated(vector, heading_deg):
    a = np.radians(heading_deg)
    x, y, z = vector
    return np.array([x * np.cos(a) + z * np.sin(a), y, -x * np.sin(a) + z * np.cos(a)])


@gl
def test_set_origin_places_poses_relative_to_a_point_and_heading(scene, gl_context):
    point, heading = np.array([0.3, 0.2, -0.4]), 90.0
    rel_position, rel_forward = np.array([0.5, 0.0, 0.2]), np.array([0.0, 0.0, -1.0])
    with LiveRenderer(scene, _cfg(), origin="floor") as moved, LiveRenderer(scene, _cfg(), origin="centre") as plain:
        moved.set_origin(point, heading)
        assert np.allclose(moved.origin, point) and moved.heading_deg == heading
        world = point + _rotated(rel_position, heading)
        assert np.allclose(moved.world_pose(rel_position, rel_forward).position, world)
        assert np.allclose(moved.world_pose(rel_position, rel_forward).forward, _rotated(rel_forward, heading))
        # the same view from a plain renderer whose origin is elsewhere: same picture
        a = moved.render(rel_position, rel_forward, 0, "rgb")
        b = plain.render(world - plain.origin, _rotated(rel_forward, heading), 0, "rgb")
        assert np.array_equal(a, b)

        moved.set_origin(None)                                       # back to the one chosen at construction
        assert np.allclose(moved.origin, (np.array(scene.bounds[0]) + scene.bounds[1])[[0, 1, 2]] / 2 * [1, 0, 1]
                           + [0, scene.bounds[0][1] + FLOOR_ORIGIN_HEIGHT, 0]) and moved.heading_deg == 0.0


@gl
def test_a_recalibration_is_not_a_camera_motion(live):
    for k in range(3):
        live.render((0.0, 0.0, -0.05 * k), FORWARD, 33_000 * k, "flow")
    live.set_origin(live.origin + [0.4, 0.0, 0.0], 30.0)             # the camera "jumps"
    events = live.render((0.0, 0.0, 0.0), FORWARD, 133_000, "events")
    assert (events == 255).all(), "no event burst from the jump"
    live.render((0.0, 0.0, 0.0), FORWARD, 166_000, "flow")
    assert np.abs(live.last_flow[live.last_flow_valid.astype(bool)]).max() < 1e-4, "no flow from the jump either"
