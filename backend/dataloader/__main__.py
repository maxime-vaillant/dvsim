"""``python -m backend.dataloader``: build a dataloader, print one batch's structure and the measured throughput."""
from __future__ import annotations

import argparse
import tempfile
import time
from typing import Any, List, Optional


def _describe(value: Any, indent: int = 0) -> List[str]:
    pad = "  " * indent
    if isinstance(value, dict):
        lines: List[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines += [f"{pad}{key}:"] + _describe(item, indent + 1)
            else:
                lines.append(f"{pad}{key}: {_describe(item)[0].strip()}")
        return lines
    return [f"{pad}{tuple(value.shape)} {str(value.dtype).replace('torch.', '')}"]


def _num_events(batch: dict, representation: str) -> int:
    """Events in a batch (0 for the voxel grid, whose bins are interpolated weights, not counts)."""
    if representation == "raw":
        return int(len(batch["events"]["t"]))
    return int(batch["events"].sum().item()) if representation == "count_frames" else 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m backend.dataloader", description=__doc__)
    p.add_argument("--scene", action="append", help="scene model / SceneSpec .json (repeatable); default: demo scene")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (spawn); 0 = in-process")
    p.add_argument("--batches", type=int, default=10, help="batches to time (after one warm-up batch)")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=48)
    p.add_argument("--clip-steps", type=int, default=10, help="simulation steps (dt = 10 ms) per clip")
    p.add_argument("--representation", choices=("count_frames", "voxel", "raw"), default="count_frames")
    p.add_argument("--gt", default="depth,flow", help="comma list of rgb,depth,flow,pose or 'none'")
    p.add_argument("--duration", type=float, default=2.0, help="generated trajectory length in seconds")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    try:
        from backend.dataloader import DVSimIterableDataset, make_dataloader
    except ImportError as exc:
        raise SystemExit(f"error: {exc}") from None
    from backend.config import SimConfig

    with tempfile.TemporaryDirectory(prefix="dvsim_demo_") as tmp:
        scenes = a.scene
        if not scenes:
            from backend.samples import create_demo_scene

            scenes = [str(create_demo_scene(tmp).scene_spec)]
            print(f"no --scene given: using the demo scene ({scenes[0]})")
        gt = () if a.gt == "none" else tuple(s for s in a.gt.split(",") if s)
        config = SimConfig.from_dict({"camera": {"width": a.width, "height": a.height}})
        dataset = DVSimIterableDataset(
            scenes, config, clip_steps=a.clip_steps, representation=a.representation, gt=gt,
            trajectory_duration_s=a.duration, seed=a.seed,
        )
        loader = make_dataloader(dataset, a.batch_size, a.workers)
        print(f"{a.width}x{a.height}, {a.clip_steps} steps/clip, batch {a.batch_size}, "
              f"{a.workers} worker(s), representation {a.representation}, gt {list(gt)}")
        it = iter(loader)
        start = time.perf_counter()
        first = next(it)                                        # includes worker start-up and scene loading
        print(f"first batch after {time.perf_counter() - start:.1f} s; structure:")
        print("\n".join("  " + line for line in _describe(first)))
        clips = events = 0
        t0 = time.perf_counter()
        for _ in range(a.batches):
            batch = next(it)
            clips += a.batch_size
            events += _num_events(batch, a.representation)
        dt = time.perf_counter() - t0
        rate = f"{clips / dt:.1f} clips/s"
        if events:
            rate += f", {events / dt / 1e6:.2f} Mevents/s"
        print(f"throughput over {a.batches} batches: {rate} ({a.batch_size * a.batches} clips in {dt:.1f} s)")
        del it, loader
        dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
