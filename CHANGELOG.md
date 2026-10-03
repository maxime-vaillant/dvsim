# Changelog

All notable changes to DVSim. Versions follow [Semantic Versioning](https://semver.org/) (0.x: the interfaces may
still change between minor versions).

## [0.3.0] - Unreleased

Human-like camera motion: learn how people move from recorded trajectories and replay it on planned walks.

### Added
- `backend/motion`: importers for TUM RGB-D, TUM-VI (csv), Project Aria / ADT and DVSim headset trajectories, a
  common 50 Hz track format in DVSim's frame with a heading-agreement check that catches a wrong camera or up-axis
  convention, and a split of every track into a smooth path and a human-motion residual.
- A small conditional 1D diffusion model (x0-prediction, DDIM sampling) of that residual, conditioned on the path's
  speed, yaw rate and curvature; long clips are generated window by window, each given the last second of the
  previous one. Optional dependency: `pip install -e ".[torch]"`.
- `motion` field on `street` trajectory specs (API, `pipeline.build_trajectories`) and `dvsim run --motion MODEL`:
  the planner's path walked with sampled human head motion, reproducible from the seed.
- `dvsim motion-import`, `motion-train` and `motion-eval` (spectra, step frequency, acceleration / jerk distributions
  and a real-vs-generated classifier on held-out recordings, next to the plain-walk baseline; `--scene` adds an
  event-rate and optical-flow comparison).
- `GET /api/motion-models` and a "Human motion" selector in the web interface.

### Changed
- Checkpoints hold tensors and plain data only and are loaded with `weights_only=True`.

## [0.2.0] - 2026-10-02

Rewrite of the first prototype into an installable package with a CLI, a web API and a dataloader.

### Added
- `dvsim` command line: `demo`, `run`, `sample-scene`, `serve`, `info`; `pip install -e .` (`pyproject.toml`).
- Ground truth: RGB, depth, forward optical flow with a validity mask, and camera odometry (HDF5 + TUM text file),
  written incrementally with zstd compression; preview videos (events, RGB, depth, flow in the Middlebury colour code).
- Event model options: per-pixel threshold mismatch, refractory period, background shot noise, photoreceptor low-pass,
  all deterministic given a seed.
- Stereo pair (baseline along the camera's right axis).
- Scene loading: `.obj`, `.glb`, `.gltf`, `.ply`, multi-object `SceneSpec` files with per-object transforms, textured
  OBJ with MTL and several textures.
- Trajectory generators: rectilinear, spline and `street` (eye-height walks on open ground), built on a voxel
  free-space grid.
- FastAPI service with uploads, background jobs (progress, cancel, download), WebSocket status, and live
  endpoints for an AR headset and a phone.
- Web interface: job panel, trajectory preview of generated paths, flow view, selectable view layout, simulation
  request builder, scene and trajectory origin options, trajectory export, phone page with simulator-rendered live
  view.
- Dataloader mode: `ClipStream` (numpy) and `DVSimIterableDataset` (PyTorch) producing event tensors and ground
  truth without writing files.
- Procedural demo scene and sample trajectory (`dvsim demo`).
- Test suite (backend, API, dataloader, frontend helpers) and `docs/architecture.md`.

### Changed
- Event generation emits every event of a large change instead of dropping bursts (a +2.2 log jump gave 1 event
  per pixel instead of 11), timestamps them inside the step by linear interpolation, and works on linear
  luminance.
- Optical flow is exact forward flow with occlusion handling, validated against analytic geometry. Before, it was
  scaled x2, had the wrong sign on dy, let occluded surfaces overwrite visible ones, and lagged one step.
- RGB is in RGB order everywhere (it was BGR); the stereo baseline follows the camera's right axis (it was along
  world Z).
- Trajectory generation takes milliseconds instead of minutes.
- Single-pass renderer (RGB, luminance, depth, flow in one draw call) replaces the per-quantity shader programs.
- Nothing depends on the working directory any more.

### Fixed
- Gzip level 9 output (80x slower than zstd) and holding the whole sequence in RAM: all outputs are now streamed.
- RGB and odometry were never saved.

### Removed
- OpenCV optical-flow processing, `compute.py` and its hard-coded script, `saver.py`, per-pass shaders.

## [0.1.0] - 2025-03-10

First prototype (commits `f161ee4` to `e16cc6b`): dynamic-vision and depth shaders, event model, OBJ loading, `.h5` / `.mp4`
output, trajectory drawing, a first API and AR-headset trajectory capture.
