"""Streaming HDF5 event writer."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from backend.saving.common import CONVENTIONS, check_uint32, dataset_filters, normalize_compression

_COLUMNS = (("t", np.uint32), ("x", np.uint16), ("y", np.uint16), ("p", np.uint8))


class EventWriter:
    """Appends events to `/events/{t,x,y,p}`; small writes are buffered so each chunk is compressed and written once.

    Timestamps are stored as given (uint32, relative to the sequence start); `t_offset_us` is recorded as an int64
    attribute so that absolute time = stored t + t_offset_us.
    """

    def __init__(self, path, width: int, height: int, compression: str = "zstd", chunk_events: int = 1 << 18,
                 t_offset_us: int = 0):
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        self.t_offset_us = int(t_offset_us)
        self.compression = normalize_compression(compression)
        self._chunk = int(chunk_events)
        self._buf = {name: np.empty(self._chunk, dtype) for name, dtype in _COLUMNS}
        self._fill = 0
        self._written = 0
        self._last_t = 0
        self._file = h5py.File(self.path, "w")
        try:
            group = self._file.create_group("events")
            self._ds = {
                name: group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(self._chunk,),
                                           **dataset_filters(self.compression, np.dtype(dtype).itemsize))
                for name, dtype in _COLUMNS
            }
            group.attrs.update(width=self.width, height=self.height, time_unit="us",
                               polarity="1 = ON (brighter), 0 = OFF", description=CONVENTIONS["events"],
                               compression=self.compression, num_events=0, t_offset_us=np.int64(self.t_offset_us))
        except BaseException:
            self._file.close()
            raise

    @property
    def num_events(self) -> int:
        return self._written + self._fill

    def write(self, events: np.ndarray) -> None:
        if self._file is None:
            raise ValueError(f"EventWriter({self.path.name}) is closed")
        n = len(events)
        if n == 0:
            return
        t, x, y, p = (events[name] for name, _ in _COLUMNS)
        if n > 1 and np.any(t[1:] < t[:-1]):
            raise ValueError("events must be sorted by timestamp")
        check_uint32(int(t[0]), int(t[-1]), "event timestamp")
        if int(t[0]) < self._last_t:
            raise ValueError(f"events go back in time: t={int(t[0])} us after t={self._last_t} us")
        if int(x.max()) >= self.width or int(y.max()) >= self.height:
            raise ValueError(f"event outside the {self.width}x{self.height} sensor")
        self._last_t = int(t[-1])
        cols = (t, x, y, p)
        i = 0
        while i < n:
            k = min(self._chunk - self._fill, n - i)
            for (name, _), col in zip(_COLUMNS, cols):
                self._buf[name][self._fill:self._fill + k] = col[i:i + k]
            self._fill += k
            i += k
            if self._fill == self._chunk:
                self._flush()

    def _flush(self) -> None:
        k = self._fill
        if k == 0:
            return
        lo, hi = self._written, self._written + k
        for name, ds in self._ds.items():
            ds.resize((hi,))
            ds[lo:hi] = self._buf[name][:k]
        self._written, self._fill = hi, 0

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._flush()
            self._file["events"].attrs["num_events"] = self._written
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
