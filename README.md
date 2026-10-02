# Dynamic Vision Simulator (DVSim)

Simulates an event camera (DVS) moving through 3D scenes and produces **events** plus dense **ground truth**:
RGB, depth, optical flow and odometry (camera pose). It can write datasets to disk, be driven from a web
interface, or stream training clips straight into a PyTorch `DataLoader` (e.g. for spiking neural networks).

## Install

Python 3.9 or newer and an OpenGL 4.1 driver (macOS, Linux with Mesa/EGL, Windows).

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api]"          # core + web server;  add ,torch for the dataloader,  ,dev for the tests
dvsim info                       # checks that an OpenGL context can be created
```

## Try it (60 seconds, no input files)

```sh
dvsim demo                       # procedural room + trajectory -> data/demo/output/demo_000/
dvsim demo --stereo              # same with a stereo pair
```

`demo_000/` contains `left_events.h5`, `ground_truth.h5`, `trajectory_tum.txt`, `meta.json` and preview videos
(`viz_*.mp4`: events, RGB, depth, optical flow in the Middlebury colour code).

## Simulate your own scenes

```sh
# a scene file (.obj/.glb/.gltf/.ply), an optional texture, and recorded camera paths
dvsim run --scene room.obj --texture room.jpg --trajectory walk1.json walk2.json --out out/ \
          --width 346 --height 260 --dt-us 10000 --gt all --visualize

# random trajectories inside the scene (spline or rectilinear)
dvsim run --scene room.obj --texture room.jpg --generate spline --count 20 --duration 10 --out out/

# eye-height walks along streets / open floors (scene +Y up, in metres)
dvsim run --scene town.json --generate street --count 10 --duration 30 --walk-speed 1.4 --eye-height 1.6 --out out/

# several objects with their own transforms: a SceneSpec .json
#   {"objects": [{"path": "room.obj", "texture": "room.png"},
#                {"path": "chair.glb", "position": [1, 0, 2], "rotation_euler_deg": [0, 45, 0], "scale": 0.5}]}
dvsim run --scene scene.json --trajectory walk.json --out out/
```

Useful options: `--stereo`, `--gt rgb,depth,flow,pose|all|none`, `--gt-every N`, `--c-pos/--c-neg` (contrast
thresholds), `--sigma-c`, `--refractory-us`, `--shot-noise-hz`, `--lpf-cutoff-hz` (event-model noise, all off by
default), `--compression zstd|gzip|none`, `--config file.json` (a partial `SimConfig`). `dvsim run -h` lists everything.

**Trajectory files** are JSON lists of `{"timestamp": ms, "x", "y", "z", "rx", "ry", "rz"}` (a forward vector) or
`{..., "qx", "qy", "qz", "qw"}` (a full orientation). They can be recorded with the AR-headset page or drawn in the web
interface. Positions can be offset with `--origin X Y Z`.

## Reading the output

```python
import h5py, hdf5plugin                         # hdf5plugin registers the zstd filter
with h5py.File("left_events.h5") as f:
    t, x, y, p = (f["events"][k][:] for k in "txyp")           # microseconds, pixels, 1 = ON / 0 = OFF
with h5py.File("ground_truth.h5") as f:
    gt = f["ground_truth"]
    depth, flow, rgb, pose = gt["depth"][:], gt["optical_flow"][:], gt["rgb_left"][:], gt["pose"][:]
```

| Quantity | Convention |
|---|---|
| time | microseconds, **relative to the trajectory start** (`t_offset_us` attribute gives the offset) |
| depth | z-depth along the optical axis in scene units, `0` = no surface |
| optical flow | forward flow in pixels `(dx right, dy down)` from frame `t` to `t + flow_dt_us`, on frame `t`'s grid; `flow_valid` marks meaningful pixels |
| RGB | uint8, RGB order |
| pose | `x y z qx qy qz qw`, camera to world, OpenGL camera (looks along -Z, +Y up); also written as `trajectory_tum.txt` |

The full contract is in [`docs/architecture.md`](docs/architecture.md).

## Web interface

```sh
cd frontend && npm install && npm run build && cd ..
dvsim serve                       # http://localhost:8000
```

Load an OBJ and its texture, draw or import a trajectory (or ask for a generated one), set the camera and event
parameters in the *Simulation* panel and press *Run simulation*; progress and the resulting files appear in the
panel. For frontend development run `dvsim serve` and `npm run dev` (port 3000, proxies to the API). The AR-headset
page (`headset.html`) needs HTTPS: put `key.pem` / `cert.pem` in `ssl/` and start `dvsim serve` with
`--ssl-keyfile ssl/key.pem --ssl-certfile ssl/cert.pem`. See [`frontend/README.md`](frontend/README.md).

## Dataloader mode (no files on disk)

```python
from backend.dataloader import DVSimIterableDataset, make_dataloader

ds = DVSimIterableDataset(["room.obj", "office.glb"], {"camera": {"width": 128, "height": 96}},
                          clip_steps=10, gt=("depth", "flow"), samples_per_epoch=2000, seed=0)
loader = make_dataloader(ds, batch_size=8, num_workers=2)      # spawned workers, each owns a simulator
for batch in loader:
    spikes = batch["events"]         # (B, T, 2, H, W): one bin per 10 ms step, ON/OFF channels
    flow = batch["gt"]["flow"]       # (B, G, 2, H, W), aligned with the bins
    if batch["meta"]["is_first_clip"].any():
        ...                          # a new sequence starts: reset your network state
```

Clips are temporally contiguous slices of random trajectories generated inside each scene. Representations:
`count_frames`, `voxel`, `raw`. Without torch use `backend.dataloader.ClipStream` (numpy). Try
`python -m backend.dataloader`.

## Event model

Ideal DVS by default: per pixel, the log of the linear luminance is compared with the level of the last event and
`floor(|change| / C)` events are emitted, timestamped by linear interpolation inside the step (a burst is never
dropped). Optional, all off by default: per-pixel threshold mismatch (`sigma_c`), refractory period, background
shot noise and a photoreceptor low-pass. The step is the rendering step `dt_us` (default 10 ms).

## What changed compared with the first version

| Before | Now |
|---|---|
| bursts dropped: a +2.2 log jump gave 1 event/pixel instead of 11; timestamps after the frame that saw the change | all events emitted, timestamped inside the step; luminance is linear |
| optical flow: x2 scale, wrong dy sign, occluded surfaces overwrote visible ones, lagging one step | exact forward flow with occlusion, validated against analytic geometry |
| RGB rendered as BGR; stereo baseline along world Z | RGB order everywhere; baseline along the camera's right axis |
| RGB and odometry never saved; gzip-9 (80x slower than zstd); everything held in RAM | all ground truth saved incrementally with zstd |
| trajectory generation: minutes per trajectory | voxel free-space grid: milliseconds |
| hard-coded script, run from a specific directory | `dvsim` CLI, web API, dataloader; nothing depends on the working directory |

The old version is kept in git as `refs/backup/pre-dvsim-refactor` (`git show refs/backup/pre-dvsim-refactor:<path>`).

## Development

```sh
pip install -e ".[api,torch,dev]"
python -m pytest -q              # GL tests skip cleanly when no OpenGL context is available
cd frontend && npm test
```

Layout: `backend/scene` (loading), `graphical_computation` (one-pass renderer: RGB, luminance, depth, flow),
`event_processing`, `trajectories`, `simulator.py`, `pipeline.py`, `saving`, `vizualisation`, `api`, `dataloader`,
`samples`, `cli.py`; `frontend/` (Vite + three.js); `tests/`.
