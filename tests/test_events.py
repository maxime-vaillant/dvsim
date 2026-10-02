"""Event model tests: analytic expectations and a scalar reference model, no snapshots."""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.config import EventConfig
from backend.event_processing import EventSimulator
from backend.types import EVENT_DTYPE

EPS = 1e-3


def lum_of(log_l):
    """Linear luminance whose log(L + EPS) is `log_l`."""
    return np.exp(np.asarray(log_l, dtype=np.float64)) - EPS


def make(shape, **kw):
    return EventSimulator(shape, EventConfig(**kw))


def run(sim, log_frames, dt=10_000, t_start=0):
    """Feed log-intensity frames one per step; returns the list of per-step event arrays (first is empty)."""
    return [sim.step(lum_of(f), t_start + (i - 1) * dt, t_start + i * dt) for i, f in enumerate(log_frames)]


def cat(chunks):
    return np.concatenate(chunks) if chunks else np.empty(0, EVENT_DTYPE)


def per_pixel(ev, shape):
    return np.bincount(ev["y"].astype(np.int64) * shape[1] + ev["x"], minlength=shape[0] * shape[1]).reshape(shape)


def random_walk(shape, steps, sigma, seed, lo=-5.0, hi=0.0):
    rng = np.random.default_rng(seed)
    frames = [rng.uniform(-3.5, -1.5, shape)]
    for _ in range(steps):
        frames.append(np.clip(frames[-1] + rng.normal(0.0, sigma, shape), lo, hi))
    return frames


def const_frames(shape, level, steps):
    return [np.full(shape, level)] * steps


# --- ideal model -----------------------------------------------------------------------------------------------

def test_burst_2p2_is_eleven_events_per_pixel():
    """The old model returned 1 event for this jump (16 instead of 176 on a 4x4 image)."""
    shape, dt, c = (4, 4), 10_000, 0.2
    sim = make(shape, c_pos=c, c_neg=c)
    assert sim.step(lum_of(np.full(shape, -3.0)), 0, dt).size == 0
    t_prev, t_now = dt, 2 * dt
    ev = sim.step(lum_of(np.full(shape, -3.0 + 2.2)), t_prev, t_now)
    assert ev.size == 11 * 16
    assert (ev["p"] == 1).all()
    assert (per_pixel(ev, shape) == 11).all()
    assert ev["t"].min() > t_prev and ev["t"].max() <= t_now
    spacing = c / 2.2 * dt
    for y in range(4):
        for x in range(4):
            t = ev["t"][(ev["x"] == x) & (ev["y"] == y)]
            assert (np.diff(t) > 0).all()
            assert np.abs(np.diff(t) - spacing).max() <= 1.0
            assert np.abs(t - (t_prev + spacing * np.arange(1, 12))).max() <= 0.51
    assert ev["t"].max() == t_now                                   # the 11th event sits at the end of the step


def test_float32_input_and_exact_multiple_of_c():
    shape = (5, 7)
    sim = make(shape)
    sim.step(lum_of(np.full(shape, -2.0)).astype(np.float32), 0, 1000)
    ev = sim.step(lum_of(np.full(shape, -2.0 + 2.2)).astype(np.float32), 1000, 2000)
    assert (per_pixel(ev, shape) == 11).all()


def test_off_events_and_asymmetric_thresholds():
    shape, dt = (3, 5), 10_000
    sim = make(shape, c_pos=0.1, c_neg=0.3)
    up, down = -3.0 + 0.55, -3.0 + 0.55 - 0.95
    chunks = run(sim, [np.full(shape, -3.0), np.full(shape, up), np.full(shape, down)], dt)
    on, off = chunks[1], chunks[2]
    assert (on["p"] == 1).all() and (per_pixel(on, shape) == 5).all()      # floor(0.55 / 0.1)
    assert (off["p"] == 0).all() and (per_pixel(off, shape) == 3).all()    # reference is -2.5, d = -0.9 = 3 * 0.3
    assert on["t"].min() > 0 and on["t"].max() <= dt
    assert off["t"].min() > dt and off["t"].max() <= 2 * dt
    spacing = 0.3 / 0.9 * dt
    t = off["t"][(off["x"] == 0) & (off["y"] == 0)]
    assert np.abs(t - (dt + spacing * np.arange(1, 4))).max() <= 0.51


def test_sub_threshold_changes_accumulate_across_steps():
    shape, dt = (2, 2), 1_000
    sim = make(shape, c_pos=0.2, c_neg=0.2)
    frames = [np.full(shape, -2.0 + 0.05 * i) for i in range(11)]      # ten steps of +0.05
    chunks = run(sim, frames, dt)
    fired = [i for i, ev in enumerate(chunks) if ev.size]
    assert fired == [4, 8]                                             # 0.2 reached at step 4, then 0.4 at step 8
    for i in fired:
        assert (per_pixel(chunks[i], shape) == 1).all()
        assert (chunks[i]["t"] == i * dt).all()                        # crossing exactly at the end of the step


def test_constant_input_gives_no_events():
    shape = (6, 9)
    sim = make(shape)
    chunks = run(sim, const_frames(shape, -1.7, 50))
    assert all(ev.size == 0 for ev in chunks)


def test_huge_change_is_not_dropped():
    shape = (3, 3)
    sim = make(shape)
    sim.step(np.zeros(shape), 0, 10)
    ev = sim.step(np.ones(shape), 10, 20)
    n = math.floor((math.log(1.0 + EPS) - math.log(EPS)) / 0.2)         # 34
    assert (per_pixel(ev, shape) == n).all()


def test_reference_consistency_over_random_walk():
    shape, c = (8, 8), 0.2
    frames = random_walk(shape, 300, 0.15, seed=3)
    sim = make(shape, c_pos=c, c_neg=c)
    ev = cat(run(sim, frames))
    net = np.zeros(shape)
    np.add.at(net, (ev["y"], ev["x"]), np.where(ev["p"] == 1, 1.0, -1.0))
    residual = frames[-1] - (frames[0] + c * net)                      # = final log-intensity - reference
    assert np.abs(residual).max() < c * (1 + 1e-3)
    assert ev.size > 500


def test_matches_scalar_reference_model():
    """Per-pixel Python implementation of the documented model, with asymmetric thresholds and refractory."""
    shape, dt = (6, 7), 5_000
    c_pos, c_neg = 0.15, 0.25
    frames = random_walk(shape, 40, 0.5, seed=11)
    for refr in (0, 1_200, 4_000):
        sim = make(shape, c_pos=c_pos, c_neg=c_neg, refractory_us=refr)
        got = run(sim, frames, dt)
        ref = frames[0].copy()
        last = np.full(shape, -(10 ** 15), dtype=np.int64)
        for i in range(1, len(frames)):
            t0 = (i - 1) * dt
            expected = []
            for y in range(shape[0]):
                for x in range(shape[1]):
                    d = frames[i][y, x] - ref[y, x]
                    c = c_pos if d > 0 else c_neg
                    n = int(math.floor(abs(d) / c + 1e-4))
                    kept = 0
                    for k in range(1, n + 1):
                        t = t0 + min(max(math.floor(k * c * dt / abs(d) + 0.5), 1), dt)
                        if t - last[y, x] >= refr:
                            expected.append((y, x, t, int(d > 0)))
                            last[y, x] = t
                            kept += 1
                    ref[y, x] += math.copysign(kept * c, d) if n else 0.0
            ev = got[i]
            expected.sort()
            assert ev.size == len(expected), (refr, i)
            raw = np.lexsort((ev["x"], ev["y"], ev["t"]))                # raw output order must be (t, y, x)
            assert (raw == np.arange(ev.size)).all(), (refr, i)
            ev = ev[np.lexsort((ev["t"], ev["x"], ev["y"]))]             # per-pixel view to compare with `expected`
            exp = np.array(expected, dtype=np.int64).reshape(-1, 4)
            assert (ev["y"] == exp[:, 0]).all() and (ev["x"] == exp[:, 1]).all() and (ev["p"] == exp[:, 3]).all()
            # the model keeps float32 state, this reference float64: times may differ by one rounding step (1 us)
            assert np.abs(ev["t"] - exp[:, 2]).max(initial=0) <= 1


# --- output format ---------------------------------------------------------------------------------------------

def test_output_dtype_sorted_and_empty():
    shape = (12, 10)
    sim = make(shape)
    frames = random_walk(shape, 30, 0.3, seed=5)
    chunks = run(sim, frames, dt=2_000)
    assert chunks[0].dtype == EVENT_DTYPE and chunks[0].shape == (0,)
    assert all(ev.dtype == EVENT_DTYPE and ev.ndim == 1 for ev in chunks)
    for i, ev in enumerate(chunks[1:], start=1):
        if ev.size:
            assert ev["t"].min() > (i - 1) * 2_000 and ev["t"].max() <= i * 2_000
            assert (np.diff(ev["t"]) >= 0).all()
    ev = cat(chunks)
    assert ev.size > 100 and (np.diff(ev["t"]) >= 0).all()
    assert ev["x"].max() < shape[1] and ev["y"].max() < shape[0] and set(np.unique(ev["p"])) <= {0, 1}
    quiet = make(shape).step(lum_of(np.full(shape, -2.0)), 0, 10)
    assert quiet.dtype == EVENT_DTYPE and quiet.shape == (0,)


def test_ties_are_in_raster_order():
    shape = (5, 6)
    sim = make(shape)
    sim.step(lum_of(np.full(shape, -3.0)), 0, 10_000)
    ev = sim.step(lum_of(np.full(shape, -2.0)), 10_000, 20_000)        # 5 events per pixel, all pixels simultaneous
    key = ev["t"] * 64 + ev["y"].astype(np.int64) * shape[1] + ev["x"]
    assert (np.diff(key) > 0).all()
    assert (ev["t"][:30] == ev["t"][0]).all()


def test_keysort_path_matches_radix_path(monkeypatch):
    shape = (9, 11)
    frames = random_walk(shape, 20, 0.4, seed=8)
    fast = cat(run(make(shape, refractory_us=800), frames, 4_000))
    monkeypatch.setattr("backend.event_processing.event._RADIX_MAX_DT", 0)
    slow = cat(run(make(shape, refractory_us=800), frames, 4_000))
    assert fast.size > 200
    assert np.array_equal(fast, slow)


def test_large_step_uses_general_sort():
    shape = (4, 4)
    sim = make(shape)
    sim.step(lum_of(np.full(shape, -3.0)), 0, 1)
    ev = sim.step(lum_of(np.full(shape, -0.8)), 1, 1 + 500_000)
    assert (per_pixel(ev, shape) == 11).all()
    key = ev["t"] * 16 + ev["y"].astype(np.int64) * 4 + ev["x"]
    assert (np.diff(key) > 0).all() and ev["t"].max() == 500_001


def test_input_layout_and_dtype_do_not_matter():
    shape = (6, 9)
    frames = random_walk(shape, 10, 0.4, seed=2)
    base = cat(run(make(shape), frames))
    sim = make(shape)
    chunks = [sim.step(np.asfortranarray(lum_of(f)), (i - 1) * 10_000, i * 10_000) for i, f in enumerate(frames)]
    assert np.array_equal(cat(chunks), base)
    sim = make(shape)
    padded = [np.pad(lum_of(f), ((0, 2), (0, 3)))[:6, :9] for f in frames]     # non-contiguous view
    chunks = [sim.step(f, (i - 1) * 10_000, i * 10_000) for i, f in enumerate(padded)]
    assert np.array_equal(cat(chunks), base)


# --- options ---------------------------------------------------------------------------------------------------

def test_refractory_burst_analytic():
    """d = 2.2, C = 0.2, dt = 10 ms => candidates at k * 909.09 us; refractory 2 ms keeps k = 1, 4, 7, 10."""
    sim = make((1, 1), refractory_us=2_000)
    sim.step(lum_of([[-3.0]]), 0, 10_000)
    ev = sim.step(lum_of([[-0.8]]), 0, 10_000)
    assert ev["t"].tolist() == [909, 3636, 6364, 9091]
    # 4 events consumed 0.8 of the 2.2 change; the remaining 1.4 keeps firing at the refractory rate
    ev2 = sim.step(lum_of([[-0.8]]), 10_000, 20_000)
    assert ev2.size >= 1 and (ev2["t"] - 9091 >= 0).all() and (ev2["t"] > 10_000).all()
    t = np.concatenate([ev["t"], ev2["t"]])
    assert (np.diff(t) >= 2_000).all()


def test_refractory_dropped_events_do_not_advance_reference():
    sim = make((1, 1), refractory_us=15_000)
    chunks = run(sim, [np.array([[-2.0]]), np.array([[-1.8]]), np.array([[-1.6]]),
                       np.array([[-1.6]]), np.array([[-1.6]])])
    assert [ev["t"].tolist() for ev in chunks] == [[], [10_000], [], [30_000], []]
    # the event at 20 ms was dropped (10 ms after the last one) and re-fired at 30 ms; the reference then caught up


def test_refractory_min_gap_holds_across_steps():
    shape, refr = (10, 12), 2_500
    frames = random_walk(shape, 200, 0.5, seed=21)
    ev = cat(run(make(shape, refractory_us=refr), frames, 3_000))
    free = cat(run(make(shape), frames, 3_000))
    assert 0 < ev.size < free.size
    order = np.lexsort((ev["t"], ev["x"], ev["y"]))
    e = ev[order]
    same = (e["x"][1:] == e["x"][:-1]) & (e["y"][1:] == e["y"][:-1])
    assert (np.diff(e["t"])[same] >= refr).all()
    assert same.sum() > 100


def test_refractory_without_effect_when_events_are_far_apart():
    shape = (7, 5)
    frames = random_walk(shape, 60, 0.15, seed=4)
    default = cat(run(make(shape), frames))
    assert np.array_equal(cat(run(make(shape, refractory_us=0), frames)), default)
    assert np.array_equal(cat(run(make(shape, refractory_us=1), frames)), default)
    assert cat(run(make(shape, refractory_us=1), frames)).size > 50


def test_sigma_c_reproducible_and_seed_dependent():
    shape = (16, 16)
    frames = random_walk(shape, 80, 0.3, seed=6)
    a = cat(run(make(shape, sigma_c=0.2, seed=5), frames))
    b = cat(run(make(shape, sigma_c=0.2, seed=5), frames))
    c = cat(run(make(shape, sigma_c=0.2, seed=6), frames))
    ideal = cat(run(make(shape), frames))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, ideal)
    sim = make(shape, sigma_c=0.2, seed=5)
    first = cat(run(sim, frames))
    sim.reset()
    assert np.array_equal(cat(run(sim, frames)), first)


def test_sigma_c_on_and_off_thresholds_are_independent():
    """d = +-2.0 from a fresh reference: the event count is floor(2.0 / C_pix) for ON (up) and OFF (down)."""
    shape, kw = (64, 64), dict(sigma_c=0.3, seed=4)
    up = per_pixel(cat(run(make(shape, **kw), [np.full(shape, -3.0), np.full(shape, -1.0)])), shape).ravel()
    down = per_pixel(cat(run(make(shape, **kw), [np.full(shape, -1.0), np.full(shape, -3.0)])), shape).ravel()
    assert up.std() > 1.0 and down.std() > 1.0                            # thresholds do vary per pixel
    assert abs(up.mean() / down.mean() - 1.0) < 0.05                      # same distribution ...
    assert abs(np.corrcoef(up, down)[0, 1]) < 0.1                         # ... but shared factors would give ~1


def test_sigma_c_spreads_per_pixel_counts_but_not_the_mean():
    shape = (24, 24)
    jump = [np.full(shape, -3.0), np.full(shape, -2.0)]                  # d = 1.0 -> exactly 5 events when C = 0.2
    counts0 = per_pixel(cat(run(make(shape), jump)), shape)
    counts = per_pixel(cat(run(make(shape, sigma_c=0.2, seed=1), jump)), shape)
    assert (counts0 == 5).all()
    assert len(np.unique(counts)) > 2
    frames = random_walk(shape, 100, 0.3, seed=9)
    n0 = cat(run(make(shape), frames)).size
    n1 = cat(run(make(shape, sigma_c=0.1, seed=2), frames)).size
    assert abs(n1 / n0 - 1.0) < 0.05


def test_shot_noise_rate_polarity_and_reference_untouched():
    shape, dt, rate, steps = (32, 32), 10_000, 50.0, 40
    frames = const_frames(shape, -2.0, steps + 1)
    noisy = cat(run(make(shape, shot_noise_hz=rate, seed=3), frames, dt))
    mean = rate * dt * 1e-6 * shape[0] * shape[1] * steps
    assert abs(noisy.size - mean) < 5 * math.sqrt(mean)
    on = (noisy["p"] == 1).sum()
    assert abs(on - noisy.size / 2) < 5 * math.sqrt(noisy.size) / 2
    assert (np.diff(noisy["t"]) >= 0).all() and noisy["t"].min() > 0 and noisy["t"].max() <= steps * dt
    assert noisy["x"].max() < 32 and noisy["y"].max() < 32
    # noise stream is independent of the signal and leaves the reference alone: a later jump of 1.0 adds
    # exactly 5 events per pixel on top of the very same noise events
    with_jump = frames + [np.full(shape, -1.0)]
    a = cat(run(make(shape, shot_noise_hz=rate, seed=3), frames + [frames[-1]], dt)[-1:])
    b = cat(run(make(shape, shot_noise_hz=rate, seed=3), with_jump, dt)[-1:])
    assert b.size - a.size == 5 * 32 * 32
    key = lambda e: (e["t"] * 1024 + e["y"].astype(np.int64) * 32 + e["x"]) * 2 + e["p"]
    assert np.isin(key(a), key(b)).all()
    assert (np.diff(b["t"]) >= 0).all()


def _noise_only(shape, dt, rate, steps, seed):
    """Per-step noise events of a constant image, plus each event's time inside its own step (1..dt)."""
    sim = make(shape, shot_noise_hz=rate, seed=seed)
    chunks = run(sim, const_frames(shape, -2.0, steps + 1), dt)
    for i, ev in enumerate(chunks[1:], start=1):
        assert ((ev["t"] > (i - 1) * dt) & (ev["t"] <= i * dt)).all()      # strictly inside (t_prev, t_now]
    ev = cat(chunks)
    return ev, (ev["t"] - 1) % dt + 1


def test_shot_noise_times_are_uniform_inside_the_step():
    dt = 10_000
    ev, rel = _noise_only((20, 36), dt, 50.0, 50, seed=7)
    assert ev.size > 10_000
    assert rel.min() >= 1 and rel.max() <= dt
    assert (rel == dt).mean() < 1e-3                                       # not piled up at t_now
    grid = np.arange(1, dt + 1)
    ecdf = np.searchsorted(np.sort(rel), grid, side="right") / rel.size
    assert np.abs(ecdf - grid / dt).max() < 1.95 / math.sqrt(rel.size)     # Kolmogorov-Smirnov, p ~ 1e-3
    for q in (0.1, 0.5, 0.9):
        assert abs(np.quantile(rel, q) / dt - q) < 0.02


def test_shot_noise_covers_the_whole_frame_uniformly():
    shape = (20, 36)
    ev, _ = _noise_only(shape, 10_000, 50.0, 50, seed=8)
    assert ev["x"].max() == shape[1] - 1 and ev["y"].max() == shape[0] - 1
    counts = per_pixel(ev, shape)
    expected = ev.size / counts.size
    chi2 = ((counts - expected) ** 2 / expected).sum()                     # ~ chi2(719): mean 719, std 38
    assert chi2 < 719 + 5 * math.sqrt(2 * 719)
    for axis in (0, 1):                                                    # marginals too (catches a swapped x / y)
        m = counts.sum(axis=1 - axis)
        e = ev.size / m.size
        assert ((m - e) ** 2 / e).sum() < (m.size - 1) + 5 * math.sqrt(2 * (m.size - 1))


def test_shot_noise_off_by_default_and_deterministic():
    shape = (10, 10)
    frames = const_frames(shape, -2.0, 20)
    assert cat(run(make(shape), frames)).size == 0
    a = cat(run(make(shape, shot_noise_hz=100.0, seed=1), frames))
    b = cat(run(make(shape, shot_noise_hz=100.0, seed=1), frames))
    c = cat(run(make(shape, shot_noise_hz=100.0, seed=2), frames))
    assert a.size > 0 and np.array_equal(a, b) and not np.array_equal(a, c)


def test_shot_noise_with_signal_stays_sorted_in_raster_order():
    shape = (8, 8)
    sim = make(shape, shot_noise_hz=2_000.0, seed=1, refractory_us=500)
    ev = cat(run(sim, random_walk(shape, 20, 0.5, seed=1), 2_000))
    key = (ev["t"] * 64 + ev["y"].astype(np.int64) * 8 + ev["x"]) * 2 + ev["p"]
    assert ev.size > 100 and (np.diff(key) >= 0).all()


def test_lpf_attenuates_flicker_and_passes_ramp():
    shape, dt = (3, 3), 100
    n = 2_000
    t = np.arange(n + 1) * dt * 1e-6
    flicker = [np.full(shape, -3.0 + 1.0 * np.sin(2 * np.pi * 500.0 * s)) for s in t]
    free = cat(run(make(shape), flicker, dt))
    slow = cat(run(make(shape, lpf_cutoff_hz=20.0), flicker, dt))
    assert free.size > 1_000
    assert slow.size < 0.01 * free.size

    dt = 1_000
    ramp = [np.full(shape, -3.0 + 2.0 * i / 1000) for i in range(1001)]  # +2.0 (10 * C) in one second
    unfiltered = cat(run(make(shape), ramp, dt))
    filtered = cat(run(make(shape, lpf_cutoff_hz=50.0), ramp, dt))
    assert (per_pixel(unfiltered, shape) == 10).all()
    assert (filtered["p"] == 1).all()
    assert set(np.unique(per_pixel(filtered, shape))) <= {9, 10}          # lags the input by ~3 ms => same count


def test_lpf_delays_a_step_and_starts_at_the_first_frame():
    shape, dt = (2, 2), 1_000
    sim = make(shape, lpf_cutoff_hz=100.0)
    chunks = run(sim, [np.full(shape, -3.0)] * 3 + [np.full(shape, -2.0)] * 30, dt)
    assert all(ev.size == 0 for ev in chunks[:3])                          # y is initialised with the first frame
    ev = cat(chunks)
    assert (per_pixel(ev, shape) == 5).all()                               # the filtered step still ends up at d = 1.0
    assert ev["t"].min() > 2 * dt                                          # nothing before the step (2 ms, 3 ms]
    assert ev["t"].max() > 10 * dt                                         # spread over many steps by the filter
    assert make(shape).step(lum_of(np.full(shape, -3.0)), 0, 1).size == 0


# --- validation and determinism --------------------------------------------------------------------------------

def test_first_step_only_initialises_the_reference():
    shape = (4, 4)
    sim = make(shape)
    assert sim.step(lum_of(np.full(shape, -3.0)), 5, 5).size == 0          # t_prev is ignored on the first call
    ev = sim.step(lum_of(np.full(shape, -2.0)), 10, 20)
    assert (per_pixel(ev, shape) == 5).all()
    sim.reset()
    assert sim.step(lum_of(np.full(shape, -1.0)), 30, 40).size == 0        # ... and again after reset()
    assert sim.step(lum_of(np.full(shape, -1.0)), 40, 50).size == 0


def test_invalid_inputs_raise_value_error_and_keep_state():
    shape = (4, 5)
    good = lum_of(np.full(shape, -3.0))
    bad_arrays = {
        "shape": np.zeros((5, 4)), "ndim": np.zeros(20), "nan": np.where(np.eye(4, 5) > 0, np.nan, good),
        "inf": np.where(np.eye(4, 5) > 0, np.inf, good), "-inf": np.where(np.eye(4, 5) > 0, -np.inf, good),
        "dtype": np.full(shape, "a"),
    }
    sim, twin = make(shape), make(shape)
    for name, arr in bad_arrays.items():                                   # invalid first frame: still not initialised
        with pytest.raises(ValueError):
            sim.step(arr, 0, 10)
    assert sim.step(good, 0, 10).size == 0
    twin.step(good, 0, 10)
    for name, arr in bad_arrays.items():
        with pytest.raises(ValueError):
            sim.step(arr, 10, 20)
    for t_prev, t_now in ((20, 20), (30, 20), (0, -5)):
        with pytest.raises(ValueError):
            sim.step(good, t_prev, t_now)
    for t_prev, t_now in ((0.5, 10), (0, float("nan")), (0, float("inf")), ("a", 10), (None, 10)):
        with pytest.raises(ValueError):
            sim.step(good, t_prev, t_now)
    with pytest.raises(ValueError):
        sim.step(good, 0, 1 << 61)                                         # step too long for the sort key
    jump = lum_of(np.full(shape, -2.0))
    assert np.array_equal(sim.step(jump, 10, 20), twin.step(jump, 10, 20))  # failed calls did not touch the state


def test_numpy_integer_and_integral_float_times_are_accepted():
    shape = (2, 2)
    sim = make(shape)
    sim.step(lum_of(np.full(shape, -3.0)), np.int64(0), np.int64(10))
    ev = sim.step(lum_of(np.full(shape, -2.0)), np.int64(10), 20.0)
    assert ev.size == 20


@pytest.mark.parametrize("kw", [
    {"c_pos": 0.0}, {"c_neg": -0.1}, {"log_eps": 0.0}, {"sigma_c": -1.0}, {"refractory_us": -1},
    {"shot_noise_hz": -1.0}, {"lpf_cutoff_hz": -1.0}, {"c_pos": float("nan")},
])
def test_invalid_config_raises(kw):
    with pytest.raises(ValueError):
        make((4, 4), **kw)


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (-1, 4), (4,), (2, 3, 4), (70_000, 4)])
def test_invalid_shape_raises(shape):
    with pytest.raises(ValueError):
        EventSimulator(shape, EventConfig())


def test_full_pipeline_is_deterministic_and_reset_reseeds():
    shape = (12, 12)
    kw = dict(sigma_c=0.1, shot_noise_hz=30.0, refractory_us=700, lpf_cutoff_hz=200.0, seed=13)
    frames = random_walk(shape, 40, 0.4, seed=1)
    sim = make(shape, **kw)
    a = cat(run(sim, frames, 2_000))
    sim.reset()
    b = cat(run(sim, frames, 2_000))
    c = cat(run(make(shape, **kw), frames, 2_000))
    assert a.size > 100 and np.array_equal(a, b) and np.array_equal(a, c)
    key = (a["t"] * 256 + a["y"].astype(np.int64) * 12 + a["x"]) * 2 + a["p"]
    assert (np.diff(key) >= 0).all()
