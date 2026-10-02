"""TUM RGB-D trajectory export."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from backend.pose import Pose


def write_tum(path, timestamps_us: Sequence[int], poses: Union[np.ndarray, Sequence[Pose]],
              t_offset_us: Optional[int] = None) -> None:
    """Write `timestamp[s] x y z qx qy qz qw` lines; poses = (N,7) array or Pose objects (camera -> world).

    If `t_offset_us` is given it is recorded in a comment line; the timestamps written are the ones passed in.
    """
    t = np.asarray(timestamps_us).astype(np.int64).reshape(-1)
    rows = [q.as_array() if isinstance(q, Pose) else q for q in poses]
    p = np.asarray(rows, dtype=np.float64).reshape(len(rows), -1) if rows else np.empty((0, 7))
    if p.shape != (len(t), 7):
        raise ValueError(f"expected {len(t)} poses of 7 values (x y z qx qy qz qw), got shape {p.shape}")
    lines = ["# timestamp[s] x y z qx qy qz qw", "# camera -> world pose of the left camera, OpenGL camera frame, world +Y up"]
    if t_offset_us is not None:
        lines.append(f"# t_offset_us = {int(t_offset_us)}; timestamps below are relative to it")
    for ti, row in zip(t.tolist(), p.tolist()):
        sec, us = divmod(abs(ti), 1_000_000)  # integer arithmetic: the timestamp is exact to the microsecond
        lines.append(f"{'-' if ti < 0 else ''}{sec}.{us:06d} " + " ".join(f"{v:.9f}" for v in row))
    Path(path).write_text("\n".join(lines) + "\n")
