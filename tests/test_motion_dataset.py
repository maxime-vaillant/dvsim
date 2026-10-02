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
