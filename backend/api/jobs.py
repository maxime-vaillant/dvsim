"""Job manager: FIFO queue, at most ``max_running`` spawned worker processes, cancellation, restart recovery.

The worker (backend.api.worker) owns ``job.json`` while it runs; the manager writes it only before a worker is
started and after it has exited (for jobs that were cancelled while queued, killed, or crashed).
"""
from __future__ import annotations

import logging
import multiprocessing
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Set

from backend.api.util import NotFound, list_outputs, new_id, now_iso, read_json, valid_id, write_json_atomic
from backend.api.worker import run_job

if TYPE_CHECKING:
    from backend.api.planner import JobPlan

logger = logging.getLogger("dvsim.jobs")
TERMINAL = ("done", "failed", "cancelled")
_KILL_DELAY_S = 2.0  # after terminate(): how long before kill()


class QueueFull(Exception):
    """Too many queued + running jobs."""


@dataclass
class _Active:
    process: Any
    terminate_at: Optional[float] = None
    kill_at: Optional[float] = None


class JobManager:
    def __init__(self, root: Path, max_running: int = 1, max_pending: int = 32, cancel_grace_s: float = 5.0):
        self.root = root
        self.max_running = max_running
        self.max_pending = max_pending
        self.cancel_grace_s = cancel_grace_s
        self._ctx = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._pending: Deque[str] = deque()
        self._active: Dict[str, _Active] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recovered = False

    # ---- reading -----------------------------------------------------------------------------------------------
    def path(self, job_id: str) -> Path:
        if not valid_id(job_id):
            raise NotFound("job not found")
        return self.root / job_id

    def get(self, job_id: str) -> Dict[str, Any]:
        self._recover()
        job = read_json(self.path(job_id) / "job.json")
        if not isinstance(job, dict):
            raise NotFound("job not found")
        return job

    def list(self) -> List[Dict[str, Any]]:
        self._recover()
        jobs = []
        if self.root.is_dir():
            for entry in self.root.iterdir():
                job = read_json(entry / "job.json") if valid_id(entry.name) else None
                if isinstance(job, dict):
                    jobs.append(job)
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jobs

    def references(self) -> Set[str]:
        """Scene / imported-trajectory ids used by queued or running jobs (they must not be deleted meanwhile)."""
        self._recover()
        with self._lock:
            ids = list(self._pending) + list(self._active)
        used: Set[str] = set()
        for job_id in ids:
            job = read_json(self.path(job_id) / "job.json") or {}
            if job.get("status") not in TERMINAL:
                used.add(job.get("scene_id"))
                used.add((job.get("trajectory") or {}).get("id"))
        used.discard(None)
        return used

    # ---- lifecycle ---------------------------------------------------------------------------------------------
    def submit(self, plan: JobPlan) -> str:
        self._recover()
        with self._lock:
            if len(self._pending) + len(self._active) >= self.max_pending:
                raise QueueFull(f"too many unfinished jobs (limit {self.max_pending})")
            self.root.mkdir(parents=True, exist_ok=True)
            while True:
                job_id = new_id()
                try:
                    (self.root / job_id).mkdir()
                except FileExistsError:
                    continue
                break
            job_dir = self.root / job_id
            (job_dir / "output").mkdir()
            write_json_atomic(job_dir / "request.json", {
                "scene": plan.scene, "scene_root": plan.scene_root, "trajectory": plan.trajectory,
                "config": plan.config, "visualize": plan.visualize,
            })
            write_json_atomic(job_dir / "job.json", {
                "id": job_id, "status": "queued", "error": None,
                "progress": {
                    "sequence": 0, "num_sequences": plan.num_sequences, "step": 0, "total_steps": 0, "fraction": 0.0,
                },
                "outputs": [], "created_at": now_iso(), "started_at": None, "finished_at": None,
                "config": plan.config, "scene_id": plan.scene_id, "visualize": plan.visualize,
                "trajectory": plan.summary,
            })
            self._pending.append(job_id)
        self.start()
        self._wake.set()
        return job_id

    def cancel_or_delete(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Queued -> cancelled. Running -> cancel requested (job JSON returned). Finished -> deleted (None)."""
        self._recover()
        with self._lock:
            job = self.get(job_id)
            job_dir = self.path(job_id)
            if job_id in self._pending:
                self._pending.remove(job_id)
                self._finish(job_dir, job, "cancelled", None)
                return self.get(job_id)
            active = self._active.get(job_id)
            if active is not None and job.get("status") not in TERMINAL:
                (job_dir / "cancel").touch()
                if active.terminate_at is None:
                    active.terminate_at = time.monotonic() + self.cancel_grace_s
                self._wake.set()
                return job
            shutil.rmtree(job_dir)
            return None

    def start(self) -> None:
        self._recover()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="dvsim-jobs", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the manager thread and every worker; unfinished jobs are marked failed."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        with self._lock:
            while self._pending:
                job_id = self._pending.popleft()
                self._finish_unless_done(job_id, "failed", "server stopped")
            for job_id, active in list(self._active.items()):
                process = active.process
                process.terminate()
                process.join(_KILL_DELAY_S)
                if process.is_alive():
                    process.kill()
                    process.join()
                self._finish_unless_done(job_id, "failed", "server stopped")
            self._active.clear()

    # ---- internals ---------------------------------------------------------------------------------------------
    def _recover(self) -> None:
        """Once, on first use (not at import): jobs left queued / running by a previous server run are marked failed."""
        if self._recovered:
            return
        with self._lock:
            if self._recovered:
                return
            self._recovered = True
            if not self.root.is_dir():
                return
            for entry in self.root.iterdir():
                job = read_json(entry / "job.json") if valid_id(entry.name) else None
                if isinstance(job, dict) and job.get("status") not in TERMINAL:
                    logger.warning("job %s was %s at server stop: marked failed", entry.name, job.get("status"))
                    self._finish(entry, job, "failed", "server restarted")

    def _finish(self, job_dir: Path, job: Dict[str, Any], status: str, error: Optional[str]) -> None:
        job.update(status=status, error=error, finished_at=now_iso(), outputs=list_outputs(job_dir / "output"))
        write_json_atomic(job_dir / "job.json", job)

    def _finish_unless_done(self, job_id: str, status: str, error: Optional[str]) -> None:
        job = read_json(self.path(job_id) / "job.json")
        if isinstance(job, dict) and job.get("status") not in TERMINAL:
            self._finish(self.path(job_id), job, status, error)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.2)
            self._wake.clear()
            try:
                with self._lock:
                    self._tick()
            except Exception:
                logger.exception("job manager error")

    def _tick(self) -> None:
        now = time.monotonic()
        for job_id, active in list(self._active.items()):
            process = active.process
            if process.is_alive():
                if active.kill_at is not None and now >= active.kill_at:
                    process.kill()
                elif active.terminate_at is not None and active.kill_at is None and now >= active.terminate_at:
                    logger.warning("job %s ignored cancel for %.1f s: terminating", job_id, self.cancel_grace_s)
                    process.terminate()
                    active.kill_at = now + _KILL_DELAY_S
                continue
            process.join()
            del self._active[job_id]
            self._settle(job_id, process.exitcode)
            try:
                process.close()
            except ValueError:
                pass
        while self._pending and len(self._active) < self.max_running:
            self._launch(self._pending.popleft())

    def _settle(self, job_id: str, exitcode: Optional[int]) -> None:
        """The worker has exited: make sure the job JSON is in a final state."""
        job_dir = self.path(job_id)
        job = read_json(job_dir / "job.json")
        if not isinstance(job, dict) or job.get("status") in TERMINAL:
            return
        if (job_dir / "cancel").exists():
            self._finish(job_dir, job, "cancelled", None)
        else:
            logger.error("job %s: worker exited with code %s without a final state", job_id, exitcode)
            self._finish(job_dir, job, "failed", f"worker process exited unexpectedly (exit code {exitcode})")

    def _launch(self, job_id: str) -> None:
        job_dir = self.path(job_id)
        job = read_json(job_dir / "job.json")
        if not isinstance(job, dict):
            return
        job.update(status="running", started_at=now_iso())
        write_json_atomic(job_dir / "job.json", job)
        try:
            process = self._ctx.Process(target=run_job, args=(str(job_dir),), name=f"dvsim-job-{job_id}", daemon=True)
            process.start()
        except Exception as exc:
            logger.exception("job %s: cannot start the worker", job_id)
            self._finish(job_dir, job, "failed", f"cannot start the worker process: {exc}")
            return
        self._active[job_id] = _Active(process)
