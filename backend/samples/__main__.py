"""`python -m backend.samples <out_dir>`: write the procedural demo scene + trajectory."""
from __future__ import annotations

import argparse

from .demo import DEMO_SEED, create_demo_scene


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.samples", description=__doc__)
    ap.add_argument("out_dir", help="directory to write the demo files into (created if missing)")
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--force", action="store_true", help="regenerate even if up-to-date files already exist")
    args = ap.parse_args(argv)
    scene = create_demo_scene(args.out_dir, seed=args.seed, force=args.force)
    for p in (scene.room_obj, scene.room_texture, scene.objects_glb, scene.scene_spec, scene.trajectory_json):
        print(f"{p}  ({p.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
