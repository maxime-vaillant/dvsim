"""motion: turning a planned path into a human walk (needs torch)."""
from __future__ import annotations

import numpy as np
import pytest
from motion_helpers import planner_line

torch = pytest.importorskip("torch")
from backend.motion.dataset import CONTEXT, WINDOW  # noqa: E402
from backend.motion.decompose import split  # noqa: E402
from backend.motion.evaluate import seam_ratio  # noqa: E402
from backend.motion.style import DiffusionMotionStyle  # noqa: E402
from backend.motion.tracks import DT_US  # noqa: E402


@pytest.fixture(scope="module")
def style(motion_checkpoint):
    return DiffusionMotionStyle.load(motion_checkpoint, device="cpu", sample_steps=6)


def test_a_seed_gives_the_same_walk_and_another_seed_a_different_one(style):
    line = planner_line(6)
    a, b, c = style.apply(line, rng=3), style.apply(line, rng=3), style.apply(line, rng=4)
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.quats, b.quats)
    assert not np.allclose(a.positions, c.positions)


def test_the_walk_keeps_the_time_base_and_stays_near_the_planned_path(style):
    line = planner_line(7.03, start_us=5_000_000)
    walk = style.apply(line, rng=0)
    assert walk.start_us == 5_000_000 and 0 <= line.end_us - walk.end_us < DT_US
    assert np.all(np.diff(walk.times_us) == DT_US)
    planned = line.poses_at(walk.times_us)[:, :3]
    assert np.abs(walk.positions - planned).max() < 0.5                      # a person's wobble, not a detour
    assert np.std(walk.quats, axis=0).max() > 1e-3                            # the orientation moves: it is not the planner's


def test_each_window_is_given_the_last_second_of_the_previous_one(style, monkeypatch):
    calls = []
    sample = style.diffusion.sample

    def spy(model, cond, ctx=None, ctx_len=0, steps=50, generator=None):
        out = sample(model, cond, ctx, ctx_len, steps, generator)
        calls.append((None if ctx is None else ctx[0, :ctx_len].clone(), ctx_len, out[0].clone()))
        return out

    monkeypatch.setattr(style.diffusion, "sample", spy)
    walk = style.apply(planner_line(14), rng=1)                               # 701 samples: windows start at 0, 150, ... 600
    assert len(walk) == 701 and len(calls) == 5
    assert calls[0][0] is None and calls[0][1] == 0                           # the first window starts from nothing
    for (_, _, previous), (ctx, ctx_len, out) in zip(calls, calls[1:]):
        assert ctx_len == CONTEXT
        assert torch.equal(ctx, previous[WINDOW - CONTEXT:])                   # its last second ...
        assert torch.equal(out[:CONTEXT], ctx)                                 # ... is the first second of the next


def test_a_short_path_takes_one_window_and_a_tiny_one_is_refused(style):
    assert len(style.apply(planner_line(1.0), rng=0)) == 51
    with pytest.raises(ValueError, match="too short"):
        style.apply(planner_line(0.05), rng=0)


def test_a_speed_the_model_never_saw_is_clamped_with_a_warning(style):
    with pytest.warns(UserWarning, match="outside the .* m/s the model was trained on"):
        walk = style.apply(planner_line(4, speed=6.0), rng=0)
    assert np.isfinite(walk.positions).all()


def test_no_warning_inside_the_trained_range(style, recwarn):
    style.apply(planner_line(4, speed=1.3), rng=0)
    assert not [w for w in recwarn if "trained on" in str(w.message)]


def test_window_seams_stay_small(style):
    out = style.apply(planner_line(14), rng=1)                                # 5 windows: seams at 200, 350, 500, 650
    ratio = seam_ratio(split(out.positions, out.quats)[1])
    # Measured 4.20 with the fixture model (rng=1; 2.2-5.4 over seeds 0-9; 5.05 before the sampler kept the context
    # in its x0 predictions). The bound is loose (more than 2x the measurement) because the fixture is a tiny
    # 120-step model whose seams vary with the seed; it guards against a gross jump where generation takes over.
    assert ratio < 10.0, ratio
