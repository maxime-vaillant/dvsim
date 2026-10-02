# Dynamic Vision Simulator Frontend

Web interface of the Dynamic Vision Simulator: a three.js previewer (overview, camera view, in-browser
event view and depth view) and a control panel that runs the Python simulator through its HTTP API.

## Getting Started

### Prerequisites

- Node.js 18+ and npm.
- The simulator API (`backend/api`, port 8000) to run simulations. Previewing works without it.

### Installation

```sh
npm install
```

### Running the Application

```sh
npm run dev
```

The dev server listens on port 3000 and proxies `/api`, `/spectate` and `/headset` (websockets included)
to the backend on `http://localhost:8000` (override with `DVSIM_BACKEND_URL`). It uses HTTPS, which the AR
headset page needs, when `../ssl/key.pem` and `../ssl/cert.pem` exist, plain http otherwise.

`npm run build` writes `dist/`, which the backend serves at `/`.

### Phone as a live camera

Open `https://<your computer's address>:3000/phone.html` on the phone (same Wi-Fi; the dev server above, with the
`ssl/` certificates; accept the browser's certificate warning once). In the interface, load your scene and choose
Trajectory > Type > Live: the Camera / Events / Depth / Optical flow views follow the phone. (The previews are the
browser's own rendering: exact depth, approximate events and flow; simulate a recorded walk for the exact ones.)

- **Android, Chrome** with ARCore: `Start AR` gives full 6DoF tracking in metres through WebXR.
- **iPhone** (Safari or Chrome: both are WebKit, which has no WebXR): `Sensors + joysticks`. The motion sensors give
  the viewing direction (allow "Motion & Orientation" when asked), the left stick walks (forward / strafe along where
  you look), the right stick goes up and down, the slider sets the walking speed. Position is not measured: the
  accelerometer drifts by metres within seconds.
- **Simulator view on the phone:** pick a scene (the ones uploaded to the server by `Run simulation` or
  `Preview paths`), a view and a resolution before starting. The server renders the scene with the simulator's own
  renderer from the phone's pose and streams JPEG frames back, drawn full screen: `RGB`, `Events` (accumulated
  over 50 ms), `Depth`, `Flow` (in pixels per simulation step, like `viz_optical_flow.mp4`) or `All` (2 x 2). The
  buttons under the readout switch view while you move. These are the simulation's exact images, not the browser's
  approximation, rendered at up to 30 fps; the status chip shows the round-trip latency, the frame rate and the
  render time. Only one live session runs at a time: a second phone replaces the first. Set `Positions relative
  to` on the phone to match the interface's if you use both.
- `Recenter` makes the current pose the origin facing forward (-Z). Hold the phone upright, facing the way you want
  "forward" to be in the scene. Use `Positions relative to` > `Floor centre` for street-scale scenes.
- `Record` saves the walk as a trajectory on the server (`Headset <date>`), listed by `GET /api/trajectories`.

The motion sensors and WebXR only work on HTTPS (or `localhost`): a plain `http://` address shows a warning.

### API address

By default the page talks to the origin that served it. To point it elsewhere (the backend must then allow
CORS), set `VITE_API_URL`, for example `VITE_API_URL=https://my-server:8000 npm run dev`.

## Using it

1. `Loading > Load scene`: **one dialog for everything**: select all the files of the scene at once. They stay in the
   browser cache and are the ones uploaded for simulations, path previews and the phone's live view.
   - an `.obj` with its `.mtl` and texture images (PNG / JPEG): a textured scene, with as many textures as the `.mtl`
     uses. Textures are matched by file name, so `Textures/colormap.png` or a name with spaces is fine; a missing
     `.mtl` or texture is reported by name. Several `.obj` files in the selection are loaded together.
   - an `.obj` with one image (no `.mtl`): the image covers the whole object.
   - an `.obj` alone: grey.
   Pick the files from one folder (a file dialog cannot span folders). Loading again replaces the current scene.
   `Reset scene` forgets the loaded scene in the page and in the browser cache, back to the empty previewer; drawn
   and imported trajectories are kept, previewed generated paths and the live start go with the scene.
2. `Trajectory > Type` (`Generated` is the default; nothing is shown until you `Preview paths`):
   - **Custom**: left-drag on the overview (top-left) to draw a path, right-drag rotates, wheel zooms.
     `Speed (m/s)` sets how fast the camera moves along it; the two `Export JSON` buttons download it in the
     import format (relative to the object centre) or in world coordinates.
   - **Imported**: a JSON file of `{timestamp (ms), x, y, z, rx, ry, rz}` records (`Load Trajectory`),
     relative to the object centre.
   - **Generated**: the simulator generates rectilinear, spline or street trajectories itself (street: walks at
     eye height along the open ground, for outdoor scenes in metres with +Y up). `Preview paths` asks the
     simulator for the paths without simulating them: they are drawn on the overview (the one `Play` follows in
     dark blue, the others lighter, a dot at each start) and played through the camera views like an imported
     file; `Path to play` picks another one when there are several. A random seed is written into the Seed
     field, so `Run simulation` then simulates exactly the paths you saw. Changing the generator, duration,
     count or seed clears the preview. The overview camera follows the size of large scenes (street, campus).
   - **Live**: follow an AR headset (`headset.html`) or a phone (`phone.html`, see below); not simulated.
   - **Live start** (Live type): `Pick live start` then click the scene on the overview (top-left) where the phone
     starts: an orange arrow shows the point (one phone height above the surface you clicked) and the way it faces.
     `Start heading` turns it (the phone's forward after `Recenter` points that way), `Clear live start` goes back to
     `Positions relative to`. To recalibrate: stand where you want to be, press `Recenter` on the phone, pick the point
     here. A drag on the overview orbits it and does not pick; `Esc` cancels. The simulator gets the same start, so the
     phone's rendered view (origin `As in the interface`, the default) starts there too and follows changes live.
   - `Positions relative to` (Imported and Live without a picked start): `Object centre` (the default, right for a room) or `Floor centre`
     (the middle of the floor, 1.5 m up: right for a street or a campus, whose centre is high above the ground).
3. `Views`: the overview always stays top-left; choose what each of the three other quadrants shows (Camera,
   Events, Depth, Optical flow). The flow preview uses the simulator's Middlebury colours (hue = direction,
   white = no motion, full colour = `Flow scale`), in pixels of the simulated image per simulation step (`dt`), so it
   reads like the simulator's `viz_optical_flow.mp4`. The camera position, heading and trajectory time are shown at
   the bottom-right corner of the overview.
4. `Simulation`: image size, field of view, `dt`, contrast thresholds, ground truth, video. Values are kept
   in `localStorage`. `Run simulation` uploads the scene (once per set of loaded files), starts the job and
   shows its progress and output download links in the panel at the top right.

## Tests

```sh
npm test
```

Runs `tests/*.test.mjs` with `node --test`: the trajectory sampler and record helpers, the request builder,
and the API client and job runner against an in-process fake of the HTTP contract
(`docs/architecture.md`, "HTTP API contract").
