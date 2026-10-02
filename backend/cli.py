"""`dvsim` command line: demo, run, sample-scene, serve, info."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from backend.config import GT_FIELDS, SimConfig


def _add_sim_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("simulation (override --config)")
    g.add_argument("--config", type=Path, help="JSON file with a (partial) SimConfig")
    g.add_argument("--width", type=int)
    g.add_argument("--height", type=int)
    g.add_argument("--fov", type=float, help="vertical field of view in degrees")
    g.add_argument("--dt-us", type=int, help="simulation step in microseconds (default 10000)")
    g.add_argument("--gt-every", type=int, help="ground-truth frame every N steps")
    g.add_argument("--gt", help=f"comma list of {','.join(GT_FIELDS)}, or 'all' / 'none'")
    g.add_argument("--stereo", action="store_true")
    g.add_argument("--stereo-baseline", type=float)
    g.add_argument("--c-pos", type=float, help="ON contrast threshold (log units)")
    g.add_argument("--c-neg", type=float, help="OFF contrast threshold (log units)")
    g.add_argument("--sigma-c", type=float, help="relative threshold mismatch")
    g.add_argument("--refractory-us", type=int)
    g.add_argument("--shot-noise-hz", type=float)
    g.add_argument("--lpf-cutoff-hz", type=float)
    g.add_argument("--seed", type=int, help="event-model seed (and trajectory generation seed)")


def _config_from_args(a: argparse.Namespace) -> SimConfig:
    d = json.loads(a.config.read_text()) if getattr(a, "config", None) else {}
    cam, ev = d.setdefault("camera", {}), d.setdefault("events", {})
    cam_args = (("width", a.width), ("height", a.height), ("fov_y_deg", a.fov), ("stereo_baseline", a.stereo_baseline))
    for key, value in cam_args:
        if value is not None:
            cam[key] = value
    if a.stereo:
        cam["stereo"] = True
    for key, value in (
        ("c_pos", a.c_pos), ("c_neg", a.c_neg), ("sigma_c", a.sigma_c), ("refractory_us", a.refractory_us),
        ("shot_noise_hz", a.shot_noise_hz), ("lpf_cutoff_hz", a.lpf_cutoff_hz), ("seed", a.seed),
    ):
        if value is not None:
            ev[key] = value
    if a.dt_us is not None:
        d["dt_us"] = a.dt_us
    if a.gt_every is not None:
        d["gt_every"] = a.gt_every
    if a.gt is not None:
        d["gt"] = [] if a.gt == "none" else list(GT_FIELDS) if a.gt == "all" else [s for s in a.gt.split(",") if s]
    return SimConfig.from_dict(d)


def _scene_spec(scene: str, texture: Optional[str]):
    if texture:
        return {"objects": [{"path": scene, "texture": texture}]}
    return scene


def _cmd_run(a: argparse.Namespace) -> int:
    from backend.pipeline import run_to_disk

    if bool(a.trajectory) == bool(a.generate):
        print("error: give either --trajectory FILE... or --generate rectilinear|spline|street", file=sys.stderr)
        return 2
    if a.trajectory:
        traj = [{"type": "file", "path": str(p), "origin": a.origin} for p in a.trajectory]
    else:
        traj = {
            "type": a.generate, "duration_s": a.duration, "count": a.count,
            "n_key_points": a.n_key_points, "seed": a.seed,
        }
        if a.generate == "street":
            del traj["n_key_points"]
            traj.update(speed=a.walk_speed, eye_height=a.eye_height, edge_margin=a.edge_margin)
    dirs = run_to_disk(
        _scene_spec(a.scene, a.texture), traj, _config_from_args(a), a.out,
        visualize=a.visualize, compression=a.compression, prefix=a.prefix, show_bar=True,
    )
    for d in dirs:
        print(d)
    return 0


def _cmd_demo(a: argparse.Namespace) -> int:
    from backend.pipeline import run_to_disk
    from backend.samples import create_demo_scene

    root = Path(a.out)
    demo = create_demo_scene(root / "scene")
    cfg = SimConfig.from_dict({"camera": {"stereo": a.stereo}, "events": {"seed": 0}})
    dirs = run_to_disk(
        demo.scene_spec, {"type": "file", "path": str(demo.trajectory_json)}, cfg, root / "output",
        visualize=not a.no_visualize, prefix="demo", show_bar=True,
    )
    print(f"\nDemo sequence written to {dirs[0]}:")
    for f in sorted(dirs[0].iterdir()):
        print(f"  {f.name:<28} {f.stat().st_size / 1e6:8.2f} MB")
    return 0


def _cmd_sample_scene(a: argparse.Namespace) -> int:
    from backend.samples import create_demo_scene

    demo = create_demo_scene(a.out)
    print(f"Demo scene written to {demo.root}")
    return 0


def _cmd_serve(a: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The web server needs the API extras:  pip install 'dvsim[api]'", file=sys.stderr)
        return 1
    if a.data_dir:
        os.environ["DVSIM_DATA_DIR"] = str(a.data_dir)
    kwargs = {}
    if a.ssl_keyfile and a.ssl_certfile:
        kwargs.update(ssl_keyfile=a.ssl_keyfile, ssl_certfile=a.ssl_certfile)
    uvicorn.run("backend.api.main:app", host=a.host, port=a.port, **kwargs)
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    import platform

    import numpy

    print(f"python {platform.python_version()} on {platform.platform()}\nnumpy {numpy.__version__}")
    try:
        from backend.graphical_computation.utils.gl_context import create_gl_context

        ctx = create_gl_context()
        print(f"OpenGL: {ctx.info['GL_VERSION']} | renderer: {ctx.info['GL_RENDERER']}")
        ctx.release()
    except Exception as exc:  # diagnostic command: report, don't crash
        print(f"OpenGL: NOT AVAILABLE ({exc})")
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="dvsim", description="Event-camera simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="generate a demo scene and simulate it (no input files needed)")
    p.add_argument("--out", default="data/demo")
    p.add_argument("--stereo", action="store_true")
    p.add_argument("--no-visualize", action="store_true")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("run", help="simulate a scene along one or more trajectories")
    p.add_argument("--scene", required=True, help=".obj/.glb/.gltf/.ply file or a SceneSpec .json")
    p.add_argument("--texture", help="texture image (single OBJ scenes)")
    p.add_argument(
        "--trajectory", nargs="+", type=Path,
        help="trajectory JSON file(s) (ms timestamps, x y z, rx ry rz | qx qy qz qw)",
    )
    p.add_argument(
        "--origin", nargs=3, type=float, metavar=("X", "Y", "Z"), help="offset added to imported trajectories"
    )
    p.add_argument(
        "--generate", choices=["rectilinear", "spline", "street"],
        help="random trajectories inside the scene (street: eye-height walks on open ground, scene +Y up in metres)",
    )
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--duration", type=float, default=5.0, help="seconds per generated trajectory")
    p.add_argument("--n-key-points", type=int, default=10)
    p.add_argument("--walk-speed", type=float, default=1.4, help="street: walking speed, m/s")
    p.add_argument("--eye-height", type=float, default=1.6, help="street: camera height above the ground, m")
    p.add_argument(
        "--edge-margin", type=float, default=None,
        help="street: keep walks this far from where the scene ends, m (default 5; 0 to walk up to the edge)",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--prefix", default="seq")
    p.add_argument("--visualize", action="store_true", help="also write mp4 previews")
    p.add_argument("--compression", choices=["zstd", "gzip", "none"], default="zstd")
    _add_sim_args(p)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("sample-scene", help="write the procedural demo scene")
    p.add_argument("out", type=Path)
    p.set_defaults(func=_cmd_sample_scene)

    p = sub.add_parser("serve", help="start the web API + interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--ssl-keyfile")
    p.add_argument("--ssl-certfile")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("info", help="check the installation (OpenGL context)")
    p.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
