"""Job worker: runs in its own *spawned* process (GL contexts are never shared or forked) and reports through files.

Layout of ``jobs/<id>/``: ``request.json`` (input), ``job.json`` (state, rewritten atomically at most every 0.25 s),
``cancel`` (flag file set by the API), ``error.log`` (traceback of a failure), ``output/`` (the sequences).
Only import-light modules at the top: the API process imports this module too.
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from backend.api.util import list_outputs, now_iso, read_json, short_error, write_json_atomic

logger = logging.getLogger("dvsim.worker")
WRITE_INTERVAL_S = 0.25
FINAL = ("done", "failed", "cancelled")


class _State:
    """The job JSON as seen by the worker: keeps the dict in memory, throttles the file writes."""

    def __init__(self, job_dir: Path):
        self.path = job_dir / "job.json"
        self.data: Dict[str, Any] = read_json(self.path) or {}
        self._last = 0.0

    def write(self) -> None:
        write_json_atomic(self.path, self.data)
        self._last = time.monotonic()

    def progress(self, sequence: int, num_sequences: int, step: int, total: int) -> None:
        fraction = (sequence + (step / total if total else 1.0)) / max(num_sequences, 1)
        self.data["progress"] = {
            "sequence": sequence, "num_sequences": num_sequences, "step": step, "total_steps": total,
            "fraction": min(1.0, fraction),
        }
        if time.monotonic() - self._last >= WRITE_INTERVAL_S:
            self.write()

    def finish(self, status: str, error: Any, outputs_dir: Path) -> None:
        if (read_json(self.path) or {}).get("status") in FINAL:
            return  # already settled by the API (e.g. "server restarted" while this orphan was still running)
        self.data.update(status=status, error=error, finished_at=now_iso(), outputs=list_outputs(outputs_dir))
        if status == "done":
            progress = self.data.get("progress") or {}
            total = progress.get("total_steps", 0)
            self.data["progress"] = {
                "sequence": max(progress.get("num_sequences", 1) - 1, 0),
                "num_sequences": progress.get("num_sequences", 1),
                "step": total, "total_steps": total, "fraction": 1.0,
            }
        self.write()


def _execute(job_dir: Path, state: _State) -> str:
    """Returns "done" or "cancelled"; anything else is an exception."""
    from backend.config import SimConfig
    from backend.pipeline import Cancelled, run_to_disk

    request = json.loads((job_dir / "request.json").read_text())
    flag, parent = job_dir / "cancel", os.getppid()

    def cancel() -> bool:  # the flag file, or the server is gone (killed): stop instead of running as an orphan
        return flag.exists() or os.getppid() != parent

    # the loader may only read the scene's own upload directory
    extra = {"scene_root": request["scene_root"]} if request.get("scene_root") else {}
    try:
        run_to_disk(
            request["scene"], request["trajectory"], SimConfig.from_dict(request["config"]), job_dir / "output",
            visualize=bool(request.get("visualize")), progress=state.progress, cancel=cancel, **extra,
        )
    except Cancelled:
        return "cancelled"
    return "done"


def run_job(job_dir: str) -> None:
    """Process entry point: ``multiprocessing.get_context("spawn").Process(target=run_job, args=(dir,))``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = Path(job_dir)
    state = _State(path)
    state.data.update(status="running", started_at=state.data.get("started_at") or now_iso())
    state.write()
    hide = str(path.parent.parent)  # the data directory: not shown to clients
    try:
        status = "cancelled" if (path / "cancel").exists() else _execute(path, state)
        error = None
    except BaseException as exc:  # includes SystemExit / KeyboardInterrupt: the job must always end in a final state
        status, error = "failed", short_error(exc, hide)
        logger.error("job %s failed: %s", path.name, error)
        text = traceback.format_exc()
        logger.error(text)
        try:
            (path / "error.log").write_text(text)
        except OSError:
            pass
    state.finish(status, error, path / "output")
