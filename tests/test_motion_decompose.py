"""motion: the path / residual split is exact, invariant, and finds what was put in."""
from __future__ import annotations

import numpy as np
import pytest
from motion_helpers import stop_and_turn
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


def test_turning_round_after_a_stop_does_not_spike_the_conditioning():
    path = extract_path(stop_and_turn().positions)
    cond = conditioning(path)
    assert path.held.shape == (900,) and path.held[425:475].all()           # the stationary span (400-500), smoothed
    assert not path.held[:400].any() and not path.held[500:].any()
    assert np.abs(cond[:, 1]).max() < 2.0 and np.abs(cond[:, 2]).max() < 10.0   # the heading jump is not a turn
    assert np.all(path.yaw_rate[path.held] == 0)


def test_a_walk_that_never_stops_holds_no_heading():
    assert not extract_path(walk()[0]).held.any() and not extract_path(walk(turn=0.2)[0]).held.any()
