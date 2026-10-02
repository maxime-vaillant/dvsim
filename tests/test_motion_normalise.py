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
