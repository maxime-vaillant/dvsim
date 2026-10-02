# Human-like camera motion (v0.3.0)

Status: design, awaiting review. Date: 2026-10-02.

## Goal

Import real egocentric human trajectories, learn how a person moves, and generate new camera trajectories in DVSim
that move like a person wearing a head-mounted camera (head bob, sway, roll, gaze shifts, speed variation).

Why: the current `street` walks glide at constant speed with no roll. Event cameras respond to exactly the
high-frequency motion that is missing, so event statistics of the simulated data differ from real egocentric
recordings. The generated data is meant to train and test spiking neural networks (SNNs) on more realistic events.

Success criteria:
- Generated residual motion is closer to held-out real motion than the plain `street` path on the metrics of
  "Evaluation", per speed bin.
- Output is a standard `WaypointTrajectory`; simulator, API and dataloader work unchanged.
- Generation is deterministic given a seed.
- Nothing outside `backend/motion/` needs torch. DVSim without torch behaves exactly as in v0.2.0.

Non-goals (deferred):
- Scene-conditioned navigation (the model choosing where to go). The `street` planner keeps choosing the path.
- Style selection (head-mounted vs handheld). Checkpoints record each sample's source, so this is cheap to add.
- Training from the web UI or API. Training is CLI-only.
- Committing pretrained weights to git.

## Decisions taken

| Question | Decision |
|---|---|
| What is generated | Residual human motion on top of a path from the existing planner (`street`) |
| Model | Conditional 1D diffusion (small U-Net), x0-prediction, DDIM sampling |
| Data | Public egocentric datasets for volume, own headset recordings to validate / fine-tune |
| Dependency | torch via the existing `torch` extra (`pip install -e .[torch]`), imported lazily |

## Structure

New package `backend/motion/`; the only code that imports torch.

```
backend/motion/
  importers/     tum.py, aria.py, headset.py, csv.py  -> RawTrack(times_us, positions, quats)
  normalise.py   frame conversion to DVSim's frame, gravity alignment, fixed-rate resampling
  decompose.py   trajectory <-> (smooth path, residual in the path's local frame)
  dataset.py     windows of residual + conditioning, sequence-level split
  model.py       conditional 1D U-Net, DDPM training, DDIM sampling
  train.py       training loop, checkpointing
  style.py       DiffusionMotionStyle(weights).apply(path, rng) -> WaypointTrajectory
  evaluate.py    realism metrics and downstream event report
```

Importers: `tum` (TUM RGB-D and TUM-VI handheld), `aria` (Project Aria / ADT trajectory CSV), `headset`
(DVSim headset JSON, reusing `_parse_records`), generic `csv`. EuRoC is not supported: it is drone motion, which
would teach the model the wrong dynamics.

### Decomposition (single source of truth)

`decompose.py` is the only place that splits a trajectory into path and residual. Training applies it to real
trajectories. Generation applies it to the planner's path to get the conditioning, then composes the sampled
residual back onto that path. Training and sampling therefore see the same definitions.

- **Path:** low-pass of the position (cutoff about 0.5 Hz, below step frequency of about 1.8 Hz). The heading is
  the tangent of the path, gravity is +Y.
- **Residual:** high-frequency position offset (forward, lateral, up) and rotation offset (rotation vector) in the
  path's local frame.
- **Conditioning per sample:** speed, yaw rate, curvature of the path.

### Normalisation

Each importer declares the source frame. `normalise.py` converts to DVSim's convention (+Y up, camera looks
along -Z, metres), aligns gravity, and resamples to 50 Hz (slerp for orientation). Gaps up to 0.2 s are
interpolated; longer gaps split the track into separate segments. Segments shorter than one window are skipped and
counted in the import report.

## Model

- **Representation:** 6 channels per step (3 position offset, 3 rotation vector), normalised per channel. Mean and
  std are stored in the checkpoint.
- **Network:** 1D U-Net, 3 resolution levels (about 64 / 128 / 256 channels, roughly 3-5M parameters). The
  diffusion step enters through FiLM layers. Per-sample conditioning is concatenated as extra input channels.
- **Diffusion:** x0-prediction, cosine schedule, 1000 training steps, DDIM sampling with 50 steps.
- **Windows:** 4 s (200 samples), hop 3 s. The first 1 s of a window is the last 1 s of the previous one, with a
  mask channel. Loss is computed on the generated part only. Context is dropped at random during training so the
  model can also start a clip from nothing.
- **Training:** AdamW with EMA weights. Device is cuda, then mps, then cpu. Train / validation / held-out split is
  by sequence, never by window. Sampling is balanced across sources. A checkpoint stores config, normalisation
  stats, source list, version, and the speed range seen in training.
- **Out-of-range conditioning** (speed outside the trained range) is clamped with a warning.

## Generation and integration

Flow: `street` makes a path -> `decompose` gives the conditioning -> the model samples residual windows, each
conditioned on the tail of the previous window -> residuals are composed onto the path -> a `WaypointTrajectory` is
returned. A seed fixes the whole result. A path shorter than one window generates one window and trims it.

- **Trajectory spec** (`/api/jobs`, `/api/trajectories/generate`): optional field
  `"motion": {"model": "<id>"}` on a generated spec, for example
  `{"type": "street", ..., "motion": {"model": "walk-v1"}}`.
- **CLI:** matching `--motion <id or path>` option on the commands that generate trajectories, plus
  `dvsim motion-import`, `dvsim motion-train --data <dirs> --out <weights>` and `dvsim motion-eval`.
- **Weights** live under `$DVSIM_DATA_DIR/motion/` and are listed by the API (`GET /api/motion-models`).
- **Frontend:** a motion-model selector in the trajectory options when at least one model exists.
- `docs/architecture.md` gets a section for the new package and the new spec field; CHANGELOG gets a 0.3.0 entry.

## Evaluation

`dvsim motion-eval` compares held-out real sequences with generated clips under the same speed and curvature
conditions, per speed bin:

- Power spectra of the six residual channels (log-spectral distance).
- Dominant step / bob frequency from the vertical spectrum peak.
- Distributions of angular velocity, linear acceleration and jerk (Wasserstein distance).
- A small real-vs-generated classifier (accuracy near 50% means hard to tell apart).

Every metric is reported next to two references: plain `street` with no residual (the baseline to beat) and
shuffled real windows (the best case).

A downstream report renders one scene with the plain path and with generated motion, and compares event-rate and
flow-magnitude statistics. It is informational, not pass / fail, because no real event recordings are available to
compare against.

## Errors

- Torch missing: lazy import; using `motion` raises an error naming `pip install -e .[torch]`. API returns a clear
  error with the same hint.
- Malformed input file: error naming the file and line.
- Checkpoint with a different version or config: load fails with a clear message.
- Unknown motion model id: API 404.

## Tests

Without torch (always run): importers on small fixture files; normalisation (frame conventions, gravity
alignment); decompose-then-recompose reproduces the input within tolerance; decompose is invariant to global
translation and yaw; dataset windowing and sequence-level split.

With torch (skipped when missing): a tiny model overfits a fixture; the same seed gives the same clip; no jump
above a threshold at window seams; out-of-range speed clamps and warns.

API: a spec with `motion` validates; unknown id returns 404; torch missing returns the install hint.

The existing test suite stays unchanged and passing.
