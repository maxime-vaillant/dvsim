"""`dvsim` command line: demo, run, sample-scene, serve, info, motion-import, motion-train, motion-eval."""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from backend.config import GT_FIELDS, SimConfig


# backend.motion.importers.FORMATS, spelled out here so that `dvsim --help` does not import the motion package
MOTION_FORMATS = ("tum", "aria", "headset", "csv")


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
    if a.trajectory and a.motion:
        print("error: --motion needs --generate street", file=sys.stderr)
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
            if a.motion:
                traj["motion"] = _motion_ref(a.motion)
        elif a.motion:
            print("error: --motion needs --generate street", file=sys.stderr)
            return 2
    dirs = run_to_disk(
        _scene_spec(a.scene, a.texture), traj, _config_from_args(a), a.out,
        visualize=a.visualize, compression=a.compression, prefix=a.prefix, show_bar=True,
    )
    for d in dirs:
        print(d)
    return 0


def _reports_errors(func):
    """Commands that read user files and models: a bad input is an `error:` line and exit code 1, not a traceback."""
    @functools.wraps(func)
    def run(a: argparse.Namespace) -> int:
        try:
            return func(a)
        except (ImportError, FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return run


def _motion_ref(value: str) -> dict:
    """A trained model given as a file (``walk.pt``, any path) or as the id of one in $DVSIM_DATA_DIR/motion."""
    path = Path(value)
    return {"path": str(path)} if path.suffix == ".pt" or path.exists() else {"model": value}


@_reports_errors
def _cmd_motion_import(a: argparse.Namespace) -> int:
    from backend.motion.corpus import import_source, load_corpus, npz_path, save_corpus

    tracks, report = import_source(a.name, a.path, a.format, a.layout, a.up, a.camera)
    print(f"{a.name}: {report.files} file(s), {report.segments} segment(s), {report.hours:.2f} h, "
          f"heading agreement {report.agreement:.2f}")
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not tracks:
        print("error: nothing to import", file=sys.stderr)
        return 1
    kept = [t for t in load_corpus(a.out) if t.source != a.name] if npz_path(a.out).exists() else []
    written = save_corpus(a.out, kept + tracks)
    print(f"{written}: {len(kept) + len(tracks)} segment(s) from {len({t.source for t in kept + tracks})} source(s)")
    return 0


def _model_path(name: Optional[str], out: Optional[Path]) -> Path:
    """The checkpoint file to write, its folder made now: a bad name or folder fails before training starts."""
    from backend.motion.registry import SUFFIX, check_model_id, models_dir

    if bool(name) == bool(out):
        raise SystemExit("error: give either --name ID (stored in $DVSIM_DATA_DIR/motion) or --out FILE")
    path = out if out else models_dir() / f"{check_model_id(name)}{SUFFIX}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"cannot create the folder of {path}: {exc}") from None
    if not os.access(path.parent, os.W_OK):
        raise ValueError(f"cannot write to {path.parent}")
    return path


@_reports_errors
def _cmd_motion_train(a: argparse.Namespace) -> int:
    from backend.motion.corpus import load_corpus
    from backend.motion.dataset import Stats, assign_splits, build_sequences, select
    from backend.motion.train import TrainConfig, train

    out = _model_path(a.name, a.out)
    seqs = build_sequences(load_corpus(a.corpus))
    cfg = TrainConfig(
        steps=a.steps, batch=a.batch, lr=a.lr, base=a.base, mults=tuple(a.mults), diffusion_steps=a.diffusion_steps,
        seed=a.seed, val_frac=a.val_frac, test_frac=a.test_frac, split_seed=a.seed,
        holdout_sources=tuple(a.holdout_source or ()),
    )
    splits = assign_splits(seqs, cfg.val_frac, cfg.test_frac, cfg.split_seed, cfg.holdout_sources)
    train_seqs, val_seqs = select(seqs, splits, "train"), select(seqs, splits, "val")
    if not train_seqs:
        print("error: no training recording left after the split", file=sys.stderr)
        return 1
    print(f"{len(train_seqs)} training, {len(val_seqs)} validation, {len(select(seqs, splits, 'test'))} held-out "
          f"segment(s)")
    train(train_seqs, val_seqs, Stats.fit(train_seqs), cfg, out, device=a.device, init=a.init)
    print(out)
    return 0


@_reports_errors
def _cmd_motion_eval(a: argparse.Namespace) -> int:
    from backend.motion.corpus import load_corpus
    from backend.motion.evaluate import evaluate_model, format_report
    from backend.motion.registry import resolve_model
    from backend.motion.style import DiffusionMotionStyle

    style = DiffusionMotionStyle.load(resolve_model(_motion_ref(a.model)), device=a.device, sample_steps=a.sample_steps)
    report = evaluate_model(style, load_corpus(a.corpus), a.max_sequences, a.seed)
    print(format_report(report))
    if a.scene:
        from backend.motion.events_report import compare_event_statistics, format_event_report
        from backend.scene import load_scene

        street = {"duration_s": a.events_duration, "seed": a.seed}
        for key in ("min_clearance", "edge_margin"):
            if getattr(a, key) is not None:
                street[key] = getattr(a, key)
        print("\nevents on the scene (same planned walk, without and with the model):")
        print(format_event_report(compare_event_statistics(load_scene(a.scene), _motion_ref(a.model), street)))
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
    p.add_argument(
        "--motion", metavar="MODEL",
        help="street: walk like a person: a trained model (file.pt, or the id of one in $DVSIM_DATA_DIR/motion)",
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

    p = sub.add_parser("motion-import", help="add a dataset of human trajectories to a corpus file")
    p.add_argument("name", help="source name (re-importing a name replaces that source)")
    p.add_argument("format", choices=MOTION_FORMATS)
    p.add_argument("path", type=Path, help="a trajectory file, or a folder searched recursively")
    p.add_argument("--out", type=Path, default=Path("corpus.npz"))
    p.add_argument("--layout", default="tumvi", help="csv: column layout (tumvi | dvsim)")
    p.add_argument("--up", nargs=3, type=float, metavar=("X", "Y", "Z"), help="up direction of the dataset's world")
    p.add_argument("--camera", choices=["opengl", "opencv"], help="camera axes (opencv: Y down, looks along +Z)")
    p.set_defaults(func=_cmd_motion_import)

    p = sub.add_parser("motion-train", help="train a human-motion model on a corpus (needs torch)")
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--name", help="model id: stored as $DVSIM_DATA_DIR/motion/<id>.pt (what the API lists)")
    p.add_argument("--out", type=Path, help="checkpoint file (instead of --name)")
    p.add_argument("--init", help="continue from this checkpoint (same --base / --mults)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64, help="U-Net width (a multiple of 8)")
    p.add_argument("--mults", nargs="+", type=int, default=[1, 2, 4], help="U-Net width multipliers per level")
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="training and split seed")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--holdout-source", action="append", help="keep every recording of this source for evaluation")
    p.add_argument("--device", help="cuda | mps | cpu (default: the best available)")
    p.set_defaults(func=_cmd_motion_train)

    p = sub.add_parser("motion-eval", help="compare a model's motion with held-out real motion (needs torch)")
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--model", required=True, help="file.pt, or a model id")
    p.add_argument("--scene", help="also simulate this scene (.obj/.glb/.json) to compare event statistics")
    p.add_argument("--events-duration", type=float, default=10.0, help="seconds of walk for the event comparison")
    p.add_argument("--min-clearance", type=float, help="street walk of the event comparison, as in `run`")
    p.add_argument("--edge-margin", type=float, help="street walk of the event comparison, as in `run`")
    p.add_argument("--max-sequences", type=int, default=0, help="evaluate at most this many held-out segments")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    p.set_defaults(func=_cmd_motion_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
