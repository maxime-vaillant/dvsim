"""PyTorch side of the dataloader mode: an IterableDataset over ClipStream and DataLoader helpers (needs torch)."""
from __future__ import annotations

import itertools
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.dataloader import default_collate

from backend.dataloader.stream import ClipStream

__all__ = ["DVSimIterableDataset", "collate_raw", "make_dataloader"]

_RAW_KEYS = ("events", "events_right")


def _raw_to_torch(events: np.ndarray) -> Dict[str, torch.Tensor]:
    """EVENT_DTYPE array -> ``{"t": int64, "x": int32, "y": int32, "p": uint8}`` (torch has no usable uint16)."""
    return {
        "t": torch.from_numpy(np.ascontiguousarray(events["t"])),
        "x": torch.from_numpy(events["x"].astype(np.int32)),
        "y": torch.from_numpy(events["y"].astype(np.int32)),
        "p": torch.from_numpy(np.ascontiguousarray(events["p"])),
    }


def _to_torch(clip: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in clip.items():
        if key in _RAW_KEYS and value.dtype.names:
            out[key] = _raw_to_torch(value)
        elif key == "gt":
            out[key] = {k: torch.from_numpy(v) for k, v in value.items()}
        elif key == "meta":
            out[key] = dict(value)
        else:
            out[key] = torch.from_numpy(value)
    return out


class DVSimIterableDataset(IterableDataset):
    """Endless (or ``samples_per_epoch`` long) stream of simulated clips as dicts of tensors.

    The constructor only stores picklable specs; the ClipStream (and its GL context) is created lazily inside the
    process that iterates, so DataLoader workers started with ``spawn`` each own their simulator (see
    :func:`make_dataloader`). ``scenes``, ``config``, ``trajectories`` and ``**stream_kwargs`` are those of
    :class:`~backend.dataloader.ClipStream`. Clips are the numpy clips as tensors, ``meta`` values are python ints /
    bools (default_collate makes tensors of them) and raw events are a dict of tensors ``t, x, y, p``.

    Seeding: stream seed = f(``seed``, epoch, worker id). ``seed=None`` takes torch's per-iterator base seed
    (``torch.manual_seed`` controls it), an int is fully reproducible. Every ``iter()`` on a dataset object advances its
    ``epoch`` (so a persistent worker sees new data each epoch); ``set_epoch(e)`` pins it (needed with non-persistent
    workers, which start from the epoch the main process holds). With ``samples_per_epoch`` the clips are split across
    workers (worker ``w`` of ``W`` yields ``N // W + (w < N % W)``); without it the stream never ends.
    One active iterator per dataset object and process.
    """

    def __init__(self, scenes, config=None, trajectories=None, *, samples_per_epoch: Optional[int] = None,
                 seed: Optional[int] = 0, **stream_kwargs):
        if samples_per_epoch is not None and samples_per_epoch < 0:
            raise ValueError("samples_per_epoch must be >= 0")
        if "context" in stream_kwargs:
            raise ValueError("a GL context cannot be shared with a dataset (it must be created in the worker)")
        self._args = (scenes, config, trajectories)
        self._stream_kwargs = stream_kwargs
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        self.representation = stream_kwargs.get("representation", "count_frames")
        self._stream: Optional[ClipStream] = None
        ClipStream(scenes, config, trajectories, **stream_kwargs)      # fail early on bad arguments (nothing is loaded)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.samples_per_epoch is None:
            raise TypeError("an endless DVSimIterableDataset has no len(); pass samples_per_epoch")
        return self.samples_per_epoch

    def _stream_seed(self, worker_id: int, info) -> int:
        if self.seed is not None:
            base = int(self.seed)
        else:
            base = int(info.seed - info.id) if info is not None else int(torch.initial_seed())
        return int(np.random.SeedSequence([base, self.epoch, worker_id]).generate_state(1)[0])

    def __iter__(self):
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        seed = self._stream_seed(worker_id, info)
        self.epoch += 1
        quota = None
        if self.samples_per_epoch is not None:
            quota = self.samples_per_epoch // num_workers + (worker_id < self.samples_per_epoch % num_workers)
            if quota == 0:
                return
        if self._stream is None:
            self._stream = ClipStream(*self._args, seed=seed, **self._stream_kwargs)
        else:
            self._stream.reset(seed)
        for clip in itertools.islice(self._stream, quota):
            yield _to_torch(clip)

    def close(self) -> None:
        """Release the simulator of this process (also done on garbage collection)."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stream"] = None                                  # GL objects never cross a process boundary
        return state


def collate_raw(batch):
    """default_collate for ``representation="raw"``: ragged events are concatenated over the batch.

    ``events`` (and ``events_right``) become ``{"t", "x", "y", "p", "batch"}`` flat tensors where ``batch[i]`` is the
    index in the batch of event ``i``; every other entry is collated as usual.
    """
    keys = [k for k in _RAW_KEYS if k in batch[0]]
    out = default_collate([{k: v for k, v in item.items() if k not in keys} for item in batch])
    for key in keys:
        parts = [item[key] for item in batch]
        merged = {f: torch.cat([p[f] for p in parts]) for f in ("t", "x", "y", "p")}
        counts = torch.tensor([len(p["t"]) for p in parts])
        merged["batch"] = torch.repeat_interleave(torch.arange(len(parts)), counts)
        out[key] = merged
    return out


def make_dataloader(dataset: DVSimIterableDataset, batch_size: int, num_workers: int = 0, **kwargs) -> DataLoader:
    """DataLoader for a :class:`DVSimIterableDataset`: workers use the ``spawn`` context (GL contexts must not cross
    a fork) and stay alive between epochs (``persistent_workers``, so each keeps its cached simulators); raw events get
    :func:`collate_raw`. Extra keyword arguments go to ``torch.utils.data.DataLoader``."""
    if num_workers > 0:
        kwargs.setdefault("multiprocessing_context", "spawn")
        kwargs.setdefault("persistent_workers", True)
    if dataset.representation == "raw":
        kwargs.setdefault("collate_fn", collate_raw)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)
