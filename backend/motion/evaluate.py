"""Realism metrics: generated vs held-out real residual motion, per speed bin. numpy / scipy only (no torch)."""
from __future__ import annotations

from typing import Dict, List, Sequence as Seq, Tuple

import numpy as np
from scipy import signal
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

from backend.motion.dataset import Sequence, assign_splits, build_sequences, recording_key
from backend.motion.decompose import extract_path, split
from backend.motion.tracks import RATE_HZ, Track
from backend.trajectories.trajectory import WaypointTrajectory

WIN = 200
SPEED_BINS = ((0.2, 0.8), (0.8, 1.2), (1.2, 1.6), (1.6, 3.0))
_NPERSEG = 64
_FLOOR = 1e-12


def windows(residual: np.ndarray, speed: np.ndarray, hop: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """((n, WIN, 6) residual windows, (n,) mean path speed of each)."""
    starts = range(0, len(residual) - WIN + 1, hop)
    if not len(starts):
        return np.zeros((0, WIN, residual.shape[1])), np.zeros(0)
    return (np.stack([residual[s:s + WIN] for s in starts]),
            np.array([speed[s:s + WIN].mean() for s in starts]))


def log_psd(w: np.ndarray) -> np.ndarray:
    """(n, F, 6) log10 power spectral density of each window and channel."""
    _, p = signal.welch(w - w.mean(axis=1, keepdims=True), fs=RATE_HZ, nperseg=_NPERSEG, axis=1)
    return np.log10(p + _FLOOR)


def log_spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference (log10 units) between the mean log spectra of two window sets, frequencies 0.3-15 Hz."""
    f = np.fft.rfftfreq(_NPERSEG, 1.0 / RATE_HZ)
    keep = (f >= 0.3) & (f <= 15.0)
    pa, pb = log_psd(a).mean(axis=0)[keep], log_psd(b).mean(axis=0)[keep]
    return float(np.sqrt(np.mean((pa - pb) ** 2)))


def step_frequency(w: np.ndarray) -> float:
    """Frequency (Hz) of the strongest vertical-offset component between 0.8 and 5 Hz (the bob)."""
    f, p = signal.welch(w[:, :, 1] - w[:, :, 1].mean(axis=1, keepdims=True), fs=RATE_HZ, nperseg=128, axis=1)
    band = (f >= 0.8) & (f <= 5.0)
    return float(f[band][np.argmax(p.mean(axis=0)[band])])


def motion_features(w: np.ndarray) -> Dict[str, np.ndarray]:
    """Pooled samples of angular speed (rad/s), linear acceleration (m/s2) and jerk (m/s3) of the residual."""
    pos, rot = w[:, :, :3], w[:, :, 3:]
    return {
        "angular_speed": np.linalg.norm(np.diff(rot, axis=1), axis=2).ravel() * RATE_HZ,
        "acceleration": np.linalg.norm(np.diff(pos, 2, axis=1), axis=2).ravel() * RATE_HZ ** 2,
        "jerk": np.linalg.norm(np.diff(pos, 3, axis=1), axis=2).ravel() * RATE_HZ ** 3,
    }


def _classifier_features(w: np.ndarray) -> np.ndarray:
    return np.concatenate([log_psd(w).reshape(len(w), -1), np.log(w.std(axis=1) + _FLOOR)], axis=1)


def _parity_by_recording(real: Seq[Sequence]) -> List[int]:
    """(n,) parity per sequence: recordings with the same key share a parity, distinct recordings alternate 0/1."""
    seen_keys = {}
    parity_list = []
    next_parity = 0
    for seq in real:
        key = recording_key(seq)
        if key not in seen_keys:
            seen_keys[key] = next_parity
            next_parity = 1 - next_parity
        parity_list.append(seen_keys[key])
    return parity_list


def _logistic_loss(wb: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    z = x @ wb[:-1] + wb[-1]
    return float(np.mean(np.logaddexp(0, z) - y * z) + 1e-2 * wb[:-1] @ wb[:-1])


def classifier_accuracy(a: np.ndarray, b: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Cross-validated accuracy of a logistic regression telling window set ``a`` from ``b`` by their spectra and
    scale. 0.5 = indistinguishable; NaN when there are fewer than ``folds`` windows in a set."""
    if min(len(a), len(b)) < folds:
        return float("nan")
    x = np.concatenate([_classifier_features(a), _classifier_features(b)])
    y = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    order = np.random.default_rng(seed).permutation(len(y))
    correct = 0
    for k in range(folds):
        te = order[k::folds]
        tr = np.setdiff1d(order, te)
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        xt, xe = (x[tr] - mu) / sd, (x[te] - mu) / sd

        wb = minimize(_logistic_loss, np.zeros(x.shape[1] + 1), args=(xt, y[tr]), method="L-BFGS-B",
                      options={"maxiter": 200}).x
        correct += int(np.sum((xe @ wb[:-1] + wb[-1] > 0) == (y[te] > 0.5)))
    return correct / len(y)


def compare(real: Seq[Sequence], generated: Seq[np.ndarray]) -> List[dict]:
    """Per speed bin: metrics of the generated residuals (one array per real sequence, same conditions) against
    the real ones, next to two references: ``plain`` (no motion at all, what ``street`` gives) and ``real_other``
    (even-numbered against odd-numbered real recordings: the noise floor, the best that can be expected)."""
    if len(real) != len(generated):
        raise ValueError(f"{len(real)} real sequences but {len(generated)} generated")
    rec_parity = _parity_by_recording(real)
    w_real, w_gen, speed, parity = [], [], [], []
    for i, (seq, gen) in enumerate(zip(real, generated)):
        w, sp = windows(seq.residual, seq.cond[:, 0])
        g, _ = windows(np.asarray(gen), seq.cond[:, 0])
        w_real.append(w)
        w_gen.append(g)
        speed.append(sp)
        parity.append(np.full(len(w), rec_parity[i]))
    r_all, g_all = np.concatenate(w_real), np.concatenate(w_gen)
    speed, parity = np.concatenate(speed), np.concatenate(parity)
    if len(r_all) == 0:
        raise ValueError("no real sequence is long enough for a 4 s window")
    report = []
    for lo, hi in SPEED_BINS:
        sel = (speed >= lo) & (speed < hi)
        if sel.sum() < 2:
            continue
        r, g = r_all[sel], g_all[sel]
        even, odd = r_all[sel & (parity == 0)], r_all[sel & (parity == 1)]
        has_ref = len(even) >= 1 and len(odd) >= 1
        fr, fg = motion_features(r), motion_features(g)
        fe, fo = (motion_features(even), motion_features(odd)) if has_ref else (None, None)
        nan = float("nan")
        report.append({
            "speed": (lo, hi), "windows": int(sel.sum()),
            "spectral_distance": {"generated": log_spectral_distance(g, r),
                                  "plain": log_spectral_distance(np.zeros_like(r), r),
                                  "real_other": log_spectral_distance(even, odd) if has_ref else nan},
            "step_hz": {"real": step_frequency(r), "generated": step_frequency(g)},
            **{f"{name}_w1": {"generated": float(wasserstein_distance(fr[name], fg[name])),
                              "real_other": float(wasserstein_distance(fe[name], fo[name])) if has_ref else nan}
               for name in fr},
            "classifier_accuracy": classifier_accuracy(r, g),
        })
    return report


def evaluate_model(style, tracks: Seq[Track], max_sequences: int = 0, seed: int = 0) -> List[dict]:
    """``compare`` on the recordings ``style``'s model never trained on (its own split settings, recomputed from the
    corpus ``tracks``): each is replayed along its own smooth path and walked by the model."""
    cfg = style.cfg
    seqs = build_sequences(tracks)
    splits = assign_splits(seqs, cfg.val_frac, cfg.test_frac, cfg.split_seed, cfg.holdout_sources)
    held_out = [i for i, s in enumerate(seqs) if splits[recording_key(s)] == "test"]
    if not held_out:
        raise ValueError("no held-out recording in this corpus: raise --test-frac when training, or add data")
    if max_sequences:
        held_out = held_out[:max_sequences]
    generated = []
    for i in held_out:
        track = tracks[i]
        path = extract_path(track.positions)
        planned = WaypointTrajectory(track.times_us, path.positions, track.quats)
        walk = style.apply(planned, rng=seed + i)
        generated.append(split(walk.positions, walk.quats)[1])
    return compare([seqs[i] for i in held_out], generated)


def format_report(report: List[dict]) -> str:
    lines = []
    for b in report:
        lo, hi = b["speed"]
        sd, st = b["spectral_distance"], b["step_hz"]
        lines.append(f"speed {lo:.1f}-{hi:.1f} m/s ({b['windows']} windows)")
        lines.append(f"  spectral distance  generated {sd['generated']:.2f}  plain {sd['plain']:.2f}  "
                     f"real-other {sd['real_other']:.2f}")
        lines.append(f"  step frequency     real {st['real']:.2f} Hz  generated {st['generated']:.2f} Hz")
        for name in ("angular_speed", "acceleration", "jerk"):
            w = b[f"{name}_w1"]
            lines.append(f"  {name:<18} W1 generated {w['generated']:.3g}  real-other {w['real_other']:.3g}")
        lines.append(f"  classifier accuracy {b['classifier_accuracy']:.2f} (0.50 = indistinguishable)")
    return "\n".join(lines)
