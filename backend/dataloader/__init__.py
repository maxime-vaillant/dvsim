"""Dataloader mode: stream (events, ground truth) training clips straight from 3D scenes, no files on disk.

Training-loop skeleton (``pip install 'dvsim[torch]'``)::

    from backend.dataloader import DVSimIterableDataset, make_dataloader

    ds = DVSimIterableDataset(["room.obj", "office.glb"], {"camera": {"width": 128, "height": 96}},
                              clip_steps=10, gt=("depth", "flow"), samples_per_epoch=2000, seed=0)
    loader = make_dataloader(ds, batch_size=8, num_workers=2)      # spawn workers, each owns a simulator
    for epoch in range(10):
        ds.set_epoch(epoch)
        for batch in loader:
            spikes = batch["events"]                               # (B, T, 2, H, W) float32, one bin per 10 ms step
            if batch["meta"]["is_first_clip"].any(): snn.reset()   # sequence start: reset the network state
            loss = criterion(snn(spikes), batch["gt"]["flow"])    # (B, G, 2, H, W), aligned with the bins
            loss.backward(); optimizer.step(); optimizer.zero_grad()

Without torch use :class:`ClipStream` directly (an iterator of dicts of numpy arrays). The torch names
(``DVSimIterableDataset``, ``make_dataloader``, ``collate_raw``) are imported on first use.
"""
from __future__ import annotations

from .representations import bin_end_times_us, events_to_count_frames, events_to_voxel_grid, rebase_events
from .stream import REPRESENTATIONS, ClipStream

__all__ = [
    "REPRESENTATIONS", "ClipStream", "bin_end_times_us", "events_to_count_frames", "events_to_voxel_grid",
    "rebase_events",
]
_TORCH_NAMES = ("DVSimIterableDataset", "collate_raw", "make_dataloader")


def __getattr__(name: str):
    if name in _TORCH_NAMES:
        try:
            from . import dataset
        except ImportError as exc:
            if (exc.name or "").split(".")[0] != "torch":
                raise
            raise ImportError(f"backend.dataloader.{name} needs PyTorch: pip install 'dvsim[torch]'") from exc
        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_TORCH_NAMES))
