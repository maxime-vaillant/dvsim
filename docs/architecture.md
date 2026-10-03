# DVSim architecture & conventions

DVSim simulates an event camera (DVS) on 3D scenes and produces events plus ground truth
(RGB, depth, optical flow, odometry). This document is the contract between modules.
Python >= 3.9 (do not use `X | Y` in runtime annotations, `match`, etc.; use `from __future__ import annotations`).

## Conventions (the single source of truth)

| Quantity | Convention |
|---|---|
| World frame | frame of the scene files, right-handed, +Y up, units = scene units (assume metres) |
| Camera frame | OpenGL: +X right, +Y up, looks along **-Z** |
| Pose | `backend.pose.Pose`: position (3,) + quaternion (x,y,z,w) camera -> world. Stored as `(7,)` = `x y z qx qy qz qw` (TUM order) |
| Time | integer **microseconds** everywhere in Python. Legacy trajectory JSON uses **milliseconds** (`timestamp`), converted on load. **Outputs are relative to the trajectory start** (step `k` is at `k*dt_us`); the offset is `trajectory.start_us`, stored as `t_offset_us`, so a headset recording taken hours after page load (`performance.now()`) works. Limit: 71.6 min of relative time per sequence (uint32 in HDF5) |
| Images | row 0 = top, column 0 = left; arrays are `(H, W[, C])` |
| RGB | uint8, **RGB channel order** (convert to BGR only at the cv2 boundary) |
| Depth | float32 `(H, W)`, **z-depth** along the optical axis (not Euclidean range), scene units, `0` = no surface |
| Optical flow | float32 `(H, W, 2)`, pixels, `(dx, dy)` with +x right and **+y down** (image coordinates). **Forward flow**: displacement of the surface point seen at pixel `(x, y)` in frame `t` to its position in frame `t + flow_dt_us`, sampled on frame `t`'s pixel grid. `flow_dt_us = gt_every * dt_us` (next GT frame; clamped to the trajectory end) |
| Flow validity | `flow_valid` uint8 `(H, W)`: 1 where a surface is visible in frame `t` and lies in front of the camera at `t + flow_dt_us` |
| Events | structured array `backend.types.EVENT_DTYPE` = `(t:int64 us, x:uint16, y:uint16, p:uint8)`, `p=1` ON (brighter), `p=0` OFF. Events produced at step `t_k` lie in `(t_{k-1}, t_k]` and arrays are sorted by `t` |
| Luminance for events | linear luminance in [0,1] computed **on the GPU** from the sRGB texture (Rec.709 weights on linearised sRGB); the event model uses `log(L + log_eps)` |

Ideal event model per pixel: reference `r` (log-intensity of the last event). Over a step with log-intensity change
`d = logL_now - r`: `n = floor(|d| / C)` events, the k-th (k=1..n) at time `t_prev + (k*C/|d|) * (t_now - t_prev)`
(linear interpolation of log-intensity inside the step), then `r += sign(d) * n * C`. **All n events are emitted**
(bursts must not be dropped). Options (all default off): per-pixel threshold mismatch, refractory period,
background shot noise, first-order low-pass on log-intensity.

## Module map (owner -> files)

```
backend/
  config.py            SimConfig / CameraConfig / EventConfig  (JSON-serializable)      [shared, done]
  pose.py              Pose, interpolate, perspective                                    [shared, done]
  types.py             EVENT_DTYPE, GTFrame, StepOutput                                  [shared, done]
  scene/               scene loading (trimesh): Scene, SceneObject, load_scene, root= confinement for uploads
  graphical_computation/
     shaders/          scene.vert / scene.frag (single MRT pass: rgb, luminance, depth, flow, flow_valid)
     utils/renderer.py Renderer (one GL context, one owning thread)
     utils/image_texture.py, gl_context.py ...
  simulator.py         Simulator: scene + config + trajectory -> iterator of StepOutput (relative time base)
  pipeline.py          build_trajectories(...), run_to_disk(...) used by the CLI and the API worker
  event_processing/    EventSimulator (vectorised numpy; noise/mismatch options)
  trajectories/        WaypointTrajectory, JSON loader, generators, FreeSpaceGrid
  motion/              learned human camera motion: importers, path/residual split, diffusion model, training, style (torch only in model/train/style)
  saving/              streaming HDF5 writers, TUM export, SequenceOutput
  vizualisation/       streaming video sinks
  samples/             procedural demo scene + trajectory (`backend.samples.create_demo_scene`)
  api/                 FastAPI app, job manager (spawned worker processes), static frontend; see api/__init__.py
  dataloader/          ClipStream (numpy) + torch IterableDataset; see dataloader/__init__.py for the training-loop example
  cli.py, __main__.py  `dvsim` command line (demo / run / sample-scene / serve / info / motion-import / motion-train / motion-eval)
  (compute.py removed: its hard-coded __main__ is replaced by the CLI; old version is in refs/backup/pre-dvsim-refactor)
frontend/              three.js previewer + simulation control panel (frontend/README.md)
tests/                 pytest, one file per module (243 backend tests + 64 API tests + dataloader tests)
```

## Interfaces

```python
# backend/scene
class SceneObject:   # one drawable: triangles + material
    vertices: np.ndarray (V,3) float32 ; faces: np.ndarray (F,3) uint32 ; uv: Optional[(V,2) float32] ; colors: Optional[(V,4) float32]
    texture: Optional[np.ndarray] (h,w,3) uint8 ; base_color: (4,) float32 ; model_matrix: (4,4) float64 ; name: str
class Scene:
    objects: List[SceneObject]; bounds: (2,3) float64 (world AABB after model matrices); diagonal: float
    def to_trimesh(self) -> trimesh.Trimesh      # merged, world-space, geometry only (for trajectory generation)
def load_scene(spec, *, root: Optional[Path] = None, max_texture_pixels=16384**2, max_triangles=20_000_000) -> Scene
    # spec: path to .obj/.glb/.gltf/.ply | dict SceneSpec | path to a .json SceneSpec
    # SceneSpec = {"objects": [{"path": str, "texture": str|None, "position":[x,y,z], "rotation_euler_deg":[rx,ry,rz], "scale": float|[sx,sy,sz]}], "texture": optional default}
    # a single OBJ + separate texture image (legacy usage) must keep working: load_scene({"objects":[{"path": "a.obj","texture":"a.jpg"}]})
    # root: CONFINES every file this reads (objects, textures, mtl/glTF references) to inside root; raises
    #   ValueError otherwise. Always pass root= for untrusted (uploaded) scenes; the API does. glTF node
    #   transforms and any offset baked out of far-from-origin geometry live in `SceneObject.model_matrix`
    #   (float64); `vertices` stay mesh-local for precision.

# backend/graphical_computation/utils/renderer.py
@dataclass
class RenderOutput:
    lum: np.ndarray            # (H,W) float32 linear luminance, always filled
    rgb: Optional[np.ndarray]  # (H,W,3) uint8
    depth: Optional[np.ndarray]      # (H,W) float32, 0 = none
    flow: Optional[np.ndarray]       # (H,W,2) float32
    flow_valid: Optional[np.ndarray] # (H,W) uint8
class Renderer:
    def __init__(self, scene: Scene, camera: CameraConfig, context=None, *, camera_positions: Optional[np.ndarray] = None): ...
        # camera_positions (M,3): sizes auto near/far to the actual camera path (only when camera.near/far are None)
    def set_camera_positions(self, positions: np.ndarray) -> None: ...   # refit auto clip planes later (Simulator.run calls this)
    def render(self, pose: Pose, next_pose: Optional[Pose] = None, *, rgb=False, depth=False, flow=False) -> RenderOutput
        # flow requires next_pose (flow from `pose` to `next_pose`)
    def close(self): ...
    # the GL context belongs to the thread that creates the Renderer: render()/close() from another thread raise RuntimeError

# backend/event_processing/event.py
class EventSimulator:
    def __init__(self, shape: Tuple[int,int] (H,W), config: EventConfig): ...   # validates config; sigma_c/refractory/shot noise/LPF are opt-in
    def reset(self): ...     # re-seeds from config.seed; independent ON/OFF threshold draw when sigma_c > 0
    def step(self, lum: np.ndarray, t_prev_us: int, t_now_us: int) -> np.ndarray   # EVENT_DTYPE, sorted by t
        # first call after reset() only initialises the reference and returns no events (t_prev_us is ignored then)

# backend/trajectories
class Trajectory(ABC):      # ONE continuous camera path
    start_us: int; end_us: int
    def pose_at(self, t_us: float) -> Pose        # pure function of time, clamped to [start_us, end_us]
class WaypointTrajectory(Trajectory):  # times_us (N,), positions (N,3), quats (N,4); lerp + slerp
    @classmethod from_waypoints(times_us, positions, quats=None, fronts=None)
    @classmethod from_json(path_or_records, origin=(0,0,0))   # records: {timestamp(ms), x,y,z, rx,ry,rz | qx,qy,qz,qw}; positions += origin
    def to_records(self) -> list[dict]           # legacy JSON records (timestamp ms, x,y,z, rx,ry,rz forward vector) + qx..qw
class FreeSpaceGrid: ...                        # voxel free-space / clearance grid built once per mesh
class RectilinearTrajectoryGenerator(mesh, ...).generate(duration_us, rng) -> WaypointTrajectory
class SplineTrajectoryGenerator(mesh, ...).generate(duration_us, n_key_points, rng) -> WaypointTrajectory

# backend/motion   (only model.py, train.py, style.py import torch; `backend.motion.require_torch()` raises with the install hint)
RATE_HZ = 50; DT_US = 20_000
class RawTrack(name, times_us, positions, quats, frame: FrameSpec)   # a recording in its dataset's own frame
class FrameSpec(up=(x, y, z) of the dataset's world, camera="opengl"|"opencv")
class Track(source, name, times_us, positions, quats)                # one segment at RATE_HZ in DVSim's frame (+Y up, camera -Z)
importers.read(path, fmt, layout="tumvi", up=None, camera=None) -> [RawTrack]      # fmt: tum | aria | headset | csv
normalise.normalise(raw, source, max_gap_s=0.2, min_duration_s=4.0) -> [Track]    # frame, gaps, 50 Hz resampling
decompose.split(positions, quats) -> (PathFrame, residual (N, 6))    # THE definition of path / residual, shared by training and sampling
decompose.compose(path, residual) -> (positions, quats)              # exact inverse of split
decompose.conditioning(path) -> (N, 3)                               # speed m/s, yaw rate rad/s, curvature 1/m
   # path = Gaussian low-pass (-3 dB at 0.5 Hz) of the positions, ends extended by point reflection; heading = its horizontal
   # tangent (held when slower than 0.1 m/s), no pitch or roll. residual = [offset (x right, y up, z back) in the path frame, rotation vector]
dataset: Sequence(source, name, residual, cond), assign_splits (hash of the recording, so segments never straddle splits), Stats, WindowSampler
   # WINDOW = 200 samples (4 s), CONTEXT = 50 (the 1 s shared with the previous window), HOP = 150
model.MotionUNet / Diffusion   # x0-prediction, cosine schedule, DDIM; the loss ignores the given context
train.train(...) -> history; train.load_checkpoint(path) -> (model, TrainConfig, Stats, sources)
   # checkpoint = {"version": "dvsim-motion-1", config/stats/sources as JSON strings, state_dict}: loaded with weights_only=True
style.DiffusionMotionStyle.load(path).apply(trajectory, rng) -> WaypointTrajectory   # the trajectory's positions walked by a person; 50 Hz
registry.resolve_model({"model": id} | {"path": file}) -> Path       # ids are files of $DVSIM_DATA_DIR/motion/<id>.pt (no torch)
evaluate.compare / evaluate_model / events_report.compare_event_statistics            # motion-eval

# backend/saving
class EventWriter:  path, width, height, compression="zstd", t_offset_us=0; write(events); close()   # /events/{t,x,y,p}
class GroundTruthWriter:  path, config, compression="zstd", t_offset_us=0; write(gt: GTFrame); close()  # /ground_truth/{timestamp,flow_dt_us,depth,optical_flow,flow_valid,rgb_left,rgb_right,pose}
def write_tum(path, timestamps_us, poses, t_offset_us=None)                                  # "t[s] x y z qx qy qz qw"
class SequenceOutput:  __init__(output_dir, config, visualize=False, compression="zstd", meta=None, t_offset_us=0)
    def write_step(step_output): ...
    def close(self): ...      # writes meta.json (files listed from the directory at close time)

# backend/simulator.py (integration)
class Simulator:
    def __init__(self, scene: Scene, config: SimConfig, context=None): ...
    def run(self, trajectory: Trajectory, *, event_seed=None) -> Iterator[StepOutput]
        # step k at output time t = k*dt_us (relative to trajectory.start_us) while start_us + t <= end_us
        # event_seed overrides the event models' seed for this run (left = seed, right = seed + 1)
    def close(self): ...
# The OpenGL context belongs to the thread that creates the Simulator/Renderer: create it and consume run()
# in the same thread or process (the API runs every job in its own spawned process).
# load_scene(spec, root=dir) refuses to read any file outside `dir` (use it for uploaded, untrusted scenes;
# pipeline.run_to_disk(..., scene_root=dir) forwards it).
```

## HDF5 output layout (per sequence directory)

```
<out>/<sequence>/left_events.h5    /events/{t: uint32 us, x: uint16, y: uint16, p: uint8}   (+ attrs width, height, t_offset_us)
<out>/<sequence>/right_events.h5   same (stereo only)
<out>/<sequence>/ground_truth.h5   /ground_truth/{timestamp uint32 us (N,), flow_dt_us uint32 (N,), depth f32 (N,H,W), optical_flow f32 (N,H,W,2),
                                    flow_valid u8 (N,H,W), rgb_left u8 (N,H,W,3), rgb_right (stereo), pose f64 (N,7)}  (+ attrs: conventions)
                                   (the last frame has flow_dt_us = 0, zero flow and flow_valid = 0: there is no later frame)
<out>/<sequence>/trajectory_tum.txt   TUM format, relative time; header comment gives t_offset_us
<out>/<sequence>/trajectory.json      the simulated path in the source's own (absolute) time, legacy record format
<out>/<sequence>/meta.json          config dump, conventions, versions, counts, files, t_offset_us, caller meta
<out>/<sequence>/viz_*.mp4          only when visualize=True
```
Compression: `zstd` via `hdf5plugin` by default (files then need `import hdf5plugin` to be read), `gzip` (level 4) or `none` selectable.
Datasets are written **incrementally** (resizable chunked datasets); nothing accumulates in RAM.

## HTTP API contract (backend/api <-> frontend)

All JSON unless noted. Base = same origin as the page (or `VITE_API_URL`).

```
GET    /api/health                      -> {"status":"ok","version":str}
POST   /api/scenes   multipart: obj (file), texture (file, optional), files (file, repeated: more files of a multi-file scene), name (str)
                                           -> 201 {"id","name","files":[...]}.  `files` makes it a bundle: other .obj files, their .mtl files and the PNG / JPEG
                                           textures (no separate `texture`). Stored under generated names (model_N.obj, material_N.mtl, texture_N.png|jpg), the
                                           mtllib / map_* references rewritten to them (matched by file name; directories, spaces and texture options are fine).
                                           A missing .mtl or diffuse texture is a 422 naming the file. A single OBJ with a `mtllib` is still refused.
GET    /api/scenes                      -> [{"id","name","files","created_at"}]
DELETE /api/scenes/{id}
POST   /api/trajectories  json {"name": str, "points": [{"timestamp":ms,"x","y","z","rx","ry","rz"} ...]}  -> 201 {"id","name","num_points","duration_s"}
GET    /api/motion-models               -> [{"id","size_bytes"}]  trained human-motion models ($DVSIM_DATA_DIR/motion/<id>.pt)
GET    /api/trajectories                -> [{"id","name","num_points","duration_s","source":"upload|headset"}]
GET    /api/trajectories/{id}           -> {..., "points":[...]}
DELETE /api/trajectories/{id}           -> 204 (409 if a queued/running job references it)
POST   /api/trajectories/generate  json {"scene_id": str, "trajectory": <a generated spec, as in /api/jobs>, "max_points": int=2000}
                                        -> 200 {"seed": int, "trajectory": <the spec with its seed>,
                                                "trajectories": [{"duration_s", "points": [{"timestamp":ms,"x","y","z","rx","ry","rz"}...]}]}
                                        Runs the generator without simulating (the previewer draws and plays the paths). A random seed is drawn
                                        when the spec has none and returned: a job with the same spec and seed simulates exactly these paths.
                                        Paths are thinned evenly to max_points. 404 unknown scene, 422 when the scene cannot be loaded or has no room.
POST   /api/jobs  json {
          "scene_id": str,
          "trajectory": {"type":"imported","id":str,"origin":[x,y,z]|null}
                      | {"type":"inline","points":[...same as trajectories...],"origin":[x,y,z]|null}
                      | {"type":"rectilinear","duration_s":float,"count":int,"seed":int|null}
                      | {"type":"spline","duration_s":float,"n_key_points":int,"count":int,"seed":int|null}
                      | {"type":"street","duration_s":float,"count":int,"seed":int|null,"speed":m/s,"eye_height":m,
                         "min_clearance":m,"edge_margin":m,"motion":{"model":id}|null}   (eye-height walks on open ground; scene +Y up,
                         metres; `motion` plays the walk with a trained human-motion model: 404 unknown id, 501 when the server has no
                         PyTorch; the job keeps the id, the worker gets the file),
          "config": <partial SimConfig dict>, "visualize": bool }             -> 202 {"id","status":"queued"}
GET    /api/jobs                        -> [job]
GET    /api/jobs/{id}                   -> job = {"id","status":"queued|running|done|failed|cancelled","error":str|null,
                                             "progress":{"sequence":int,"num_sequences":int,"step":int,"total_steps":int,"fraction":float},
                                             "outputs":[{"name","path","size_bytes"}], "created_at","finished_at","config":{...}}
DELETE /api/jobs/{id}                   -> queued/running: 202, cancels, record kept with status "cancelled"
                                            done/failed/cancelled: 204, deletes the record and its output files
GET    /api/jobs/{id}/files/{path}      -> file download (path traversal, symlinks and absolute paths rejected: 400)
WS     /api/jobs/{id}/ws                -> pushes the job JSON every ~0.5 s until it finishes
PUT    /api/live/origin  json {"x","y","z","heading_deg"} (GET / DELETE too): where the live camera's origin is in the scene (world
                                           coordinates) and which way its forward points; the interface publishes its effective Live origin here.
                                           A session started with origin "shared" uses it and follows changes ({"type":"origin"} confirms each).
WS     /api/live                        -> the simulator's exact views of a scene for a live camera (the phone page); one session at a time.
                                           client -> {"type":"start","scene_id","width","height","fov","c_pos","c_neg","origin":"centre|floor","view":"rgb|events|depth|flow|grid","fps"}
                                           server -> {"type":"loading"}, {"type":"ready","width","height","origin":[x,y,z]} | {"type":"error","message"}
                                           client -> {"type":"pose","seq":n,"x","y","z","rx","ry","rz"} (relative to the origin: the bounding-box centre, or the
                                                     middle of the floor 1.5 m up), {"type":"view","view":...}
                                           (origin: "centre", "floor", or "shared" = the one the interface set with PUT /api/live/origin)
                                           server -> binary frames: 4 bytes big-endian seq of the pose it shows + a JPEG; {"type":"stats","fps","render_ms"}.
                                           The renderer (backend/live.py: the same Renderer and EventSimulator as a simulation, so RGB / depth are the
                                           simulation's own and flow has its units) runs in a spawned process; only the newest pose is kept.
WS     /headset, /spectate              -> unchanged (AR headset trajectory relay); recordings are saved as trajectories (source "headset")
GET    /                                -> built frontend (frontend/dist) if it exists
GET    /api/docs                        -> interactive OpenAPI docs
```
Errors are always JSON `{"detail": "<readable string>"}` (422s are a single string, not FastAPI's default list).
`GET|POST /api/jobs` config is validated with `SimConfig.from_dict` (422 on unknown/invalid keys) before a job is
queued. `scene_id` is loaded with `load_scene(..., root=<the scene's own upload directory>)`:
every file the loader opens (object, texture, `.mtl`/glTF references) must resolve inside that directory or it
raises; an uploaded `.gltf`, or an `.obj` whose text contains `mtllib` or an absolute/`..` texture reference, is
also rejected at upload time (422) as defence in depth. `$DVSIM_DATA_DIR` (default `./data`) holds `scenes/`, `trajectories/`,
`jobs/`; the rest of the environment variables and the on-disk layout are documented in `backend/api/__init__.py`.

## Testing rules

* `python -m pytest` from the repo root. GL tests use the `gl_context` fixture (tests/conftest.py) and skip cleanly if no GL context can be created.
* Tests must use **analytic ground truth** (known geometry -> known flow / depth / event counts), not snapshots.
* No test may write outside a `tmp_path`.
