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
