"""Config validation and pose maths (no GL needed)."""
from __future__ import annotations

import numpy as np
import pytest

from backend.config import SimConfig
from backend.pose import Pose, interpolate


@pytest.mark.parametrize("bad", [
    {"camera": {"fov_y_deg": 0}}, {"camera": {"fov_y_deg": 200}}, {"camera": {"width": 0}},
    {"camera": {"near": 5.0, "far": 1.0}}, {"camera": {"near": -1.0}}, {"camera": {"stereo_baseline": -0.1}},
    {"events": {"c_pos": 0}}, {"events": {"c_neg": -0.2}}, {"events": {"sigma_c": -1}},
    {"events": {"refractory_us": -5}}, {"events": {"log_eps": 0}}, {"events_right": {"c_pos": 0}},
    {"dt_us": 0}, {"gt_every": 0}, {"gt": "rgb"}, {"gt": ["lidar"]},
])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        SimConfig.from_dict(bad)


def test_defaults_and_partial_dicts_are_valid():
    assert SimConfig().validate() is not None
    cfg = SimConfig.from_dict({"camera": {"width": 8, "height": 6}, "events": {"c_pos": 0.1}})
    assert cfg.camera.width == 8 and cfg.events.c_pos == 0.1 and cfg.events.c_neg == 0.2


def test_zero_or_non_finite_quaternion_is_rejected():
    for q in ([0, 0, 0, 0], [np.nan, 0, 0, 1], [np.inf, 0, 0, 1]):
        with pytest.raises(ValueError):
            Pose([0, 0, 0], q)


def test_from_look_straight_up_and_down_is_a_valid_rotation():
    for front in ([0, 1, 0], [0, -1, 0]):
        pose = Pose.from_look([0, 0, 0], front)
        assert np.allclose(pose.forward, front, atol=1e-9)
        assert np.allclose(pose.rotation() @ pose.rotation().T, np.eye(3), atol=1e-9)


def test_interpolate_takes_the_short_way_and_hits_the_endpoints():
    a, b = Pose.from_look([0, 0, 0], [0, 0, -1]), Pose.from_look([2, 0, 0], [1, 0, 0])
    assert np.allclose(interpolate(a, b, 0.0).as_array(), a.as_array(), atol=1e-12)
    assert np.allclose(interpolate(a, b, 1.0).position, b.position)
    mid = interpolate(a, b, 0.5)
    assert np.allclose(mid.forward, [np.sqrt(0.5), 0, -np.sqrt(0.5)], atol=1e-9)
