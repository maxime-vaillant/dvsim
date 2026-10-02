"""``/api/live``: the simulator's exact views of a scene, rendered for a live camera (a phone, a headset).

Websocket protocol (text = JSON, binary = frames):
  client -> {"type":"start","scene_id":..., "width","height","fov","c_pos","c_neg","origin":"centre|floor","view","fps"}
  server -> {"type":"loading"}, then {"type":"ready","width","height","origin":[x,y,z]} or {"type":"error","message"}
  client -> {"type":"pose","seq":n,"x","y","z","rx","ry","rz"}   many times a second: position relative to the origin
            (see backend/live.py) and the forward vector; {"type":"view","view":"rgb|events|depth|flow|grid"}
  server -> binary: 4 bytes big-endian ``seq`` of the pose the frame was rendered from, then a JPEG;
            {"type":"stats","fps","render_ms"} about once a second; {"type":"error","message"} when it stops.
``PUT /api/live/origin`` (json x, y, z, heading_deg), ``DELETE`` and ``GET`` hold the origin the interface picked; a
session started with ``"origin":"shared"`` uses it and follows later changes (``{"type":"origin"}`` is sent when it
applies one).
One session at a time: a new ``start`` replaces the running one. The renderer lives in its own spawned process
(backend/api/live_worker.py); only the newest pose is kept, so a slow renderer or link drops frames, never lags.
"""
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import struct
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request, Response, WebSocket
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from backend.api import live_worker
from backend.api.schemas import LiveOrigin, LivePose, LiveStart, LiveView
from backend.api.util import NotFound

logger = logging.getLogger("dvsim.live")
router = APIRouter(prefix="/api")
START_TIMEOUT_S = 10.0
_STOP_GRACE_S = 2.0


class LiveSession:
    """A worker process, its pipe and its pose slot. A thread reads the pipe and keeps the newest frame."""

    def __init__(self, process, conn, slot):
        self.process, self.conn, self.slot = process, conn, slot
        self._lock = threading.Lock()
        self._messages: Deque[Tuple[Any, ...]] = deque()
        self._frame: Optional[Tuple[Any, ...]] = None
        self._wake = lambda: None
        self.uses_shared_origin = False
        self._reader = threading.Thread(target=self._read, name="dvsim-live-reader", daemon=True)

    def start_reader(self, wake) -> None:
        self._wake = wake
        self._reader.start()

    def _read(self) -> None:
        while True:
            try:
                message = self.conn.recv()
            except (EOFError, OSError):
                break
            with self._lock:
                if message[0] == "frame":
                    self._frame = message            # only the newest frame is worth sending
                else:
                    self._messages.append(message)
            self._wake()
        with self._lock:
            self._messages.append(("closed",))
        self._wake()

    def take(self) -> Tuple[List[Tuple[Any, ...]], Optional[Tuple[Any, ...]]]:
        with self._lock:
            messages, frame = list(self._messages), self._frame
            self._messages.clear()
            self._frame = None
        return messages, frame

    def set_pose(self, pose: LivePose) -> None:
        with self.slot.get_lock():
            self.slot[:] = [float(pose.seq), pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz, 0.0]

    def command(self, message: Tuple[Any, ...]) -> None:
        try:
            self.conn.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass


class LiveManager:
    """At most one live session; starting another stops the first."""

    def __init__(self) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._current: Optional[LiveSession] = None
        self.origin: Optional[Tuple[float, float, float, float]] = None   # picked in the interface: x, y, z, heading

    def set_origin(self, origin: Optional[Tuple[float, float, float, float]]) -> None:
        """Remember the picked origin and move the running session (if it uses it) there."""
        with self._lock:
            self.origin = origin
            if self._current is not None and self._current.uses_shared_origin:
                self._current.command(("origin", origin))

    def start(self, request: Dict[str, Any], wake) -> LiveSession:
        with self._lock:
            if self._current is not None:
                self._stop(self._current)
            request = {**request, "shared_origin": list(self.origin) if self.origin else None}
            parent, child = self._ctx.Pipe(duplex=True)
            slot = self._ctx.Array("d", [-1.0] + [0.0] * (live_worker.POSE_FIELDS - 1))
            process = self._ctx.Process(
                target=live_worker.run, args=(child, slot, request), name="dvsim-live", daemon=True)
            process.start()
            child.close()
            session = LiveSession(process, parent, slot)
            session.uses_shared_origin = request["origin"] == "shared"
            session.start_reader(wake)
            self._current = session
            return session

    def stop(self, session: LiveSession) -> None:
        with self._lock:
            self._stop(session)

    def stop_all(self) -> None:
        with self._lock:
            if self._current is not None:
                self._stop(self._current)

    def _stop(self, session: LiveSession) -> None:
        session.command(("stop",))
        session.process.join(_STOP_GRACE_S)
        if session.process.is_alive():
            session.process.terminate()
            session.process.join(1.0)
            if session.process.is_alive():
                session.process.kill()
        try:
            session.conn.close()
        except OSError:
            pass
        if self._current is session:
            self._current = None


def _config(start: LiveStart) -> Dict[str, Any]:
    return {
        "camera": {"width": start.width, "height": start.height, "fov_y_deg": start.fov},
        "events": {"c_pos": start.c_pos, "c_neg": start.c_neg},
    }


async def _error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "message": message})
    except Exception:  # the client is already gone
        pass


async def _receive(websocket: WebSocket, session: LiveSession) -> None:
    """Poses and view changes from the client, until it disconnects."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        text = message.get("text")
        if text is None:
            continue
        try:
            data = json.loads(text)
            kind = data.get("type") if isinstance(data, dict) else None
            if kind == "pose":
                session.set_pose(LivePose.model_validate(data))
            elif kind == "view":
                session.command(("view", LiveView.model_validate(data).view))
            else:
                raise ValueError(f"unknown message type {kind!r}")
        except (ValueError, ValidationError) as exc:
            await websocket.send_json({"type": "warning", "message": f"ignored a message: {str(exc)[:200]}"})


async def _pump(websocket: WebSocket, session: LiveSession, wake: asyncio.Event) -> None:
    """Worker -> client: statuses as JSON, the newest frame as binary."""
    while True:
        await wake.wait()
        wake.clear()
        messages, frame = session.take()
        for message in messages:
            kind = message[0]
            if kind in ("ready", "stats", "origin"):
                await websocket.send_json({"type": kind, **message[1]})
            elif kind == "error":
                await _error(websocket, message[1])
                return
            elif kind == "closed":
                await _error(websocket, "the live renderer stopped (replaced by another session, or it crashed)")
                return
        if frame is not None:
            await websocket.send_bytes(struct.pack(">I", frame[1] & 0xFFFFFFFF) + frame[2])


@router.put("/live/origin", summary="Pick where the live camera's origin is in the scene")
def put_live_origin(body: LiveOrigin, request: Request) -> Dict[str, float]:
    request.app.state.live.set_origin((body.x, body.y, body.z, body.heading_deg))
    return body.model_dump()


@router.get("/live/origin", summary="The picked live origin, or null")
def get_live_origin(request: Request) -> Optional[Dict[str, float]]:
    origin = request.app.state.live.origin
    return None if origin is None else dict(zip(("x", "y", "z", "heading_deg"), origin))


@router.delete("/live/origin", status_code=204, summary="Forget the picked live origin")
def delete_live_origin(request: Request) -> Response:
    request.app.state.live.set_origin(None)
    return Response(status_code=204)


@router.websocket("/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    state = websocket.app.state
    try:
        start = LiveStart.model_validate(await asyncio.wait_for(websocket.receive_json(), START_TIMEOUT_S))
    except (ValidationError, ValueError, asyncio.TimeoutError, KeyError) as exc:
        await _error(websocket, f"the first message must be a valid start message: {str(exc)[:300]}")
        await websocket.close(code=1008)
        return
    except Exception:  # disconnected before starting
        return
    try:
        scene, scene_root = state.scenes.spec(start.scene_id), str(state.scenes.path(start.scene_id))
    except NotFound:
        await _error(websocket, "unknown scene: upload it first")
        await websocket.close(code=1008)
        return

    await websocket.send_json({"type": "loading"})
    loop = asyncio.get_running_loop()
    wake = asyncio.Event()
    request = {"scene": scene, "scene_root": scene_root, "config": _config(start), "origin": start.origin,
               "view": start.view, "fps": start.fps}
    session = await run_in_threadpool(state.live.start, request, lambda: loop.call_soon_threadsafe(wake.set))
    tasks = [asyncio.create_task(_receive(websocket, session)), asyncio.create_task(_pump(websocket, session, wake))]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # Stop the renderer first and without awaiting: this block must finish even when the connection task is
        # cancelled (a test client, a server shutdown). The thread takes the manager's lock, so it also orders
        # itself before the next session.
        threading.Thread(target=state.live.stop, args=(session,), name="dvsim-live-stop", daemon=True).start()
        for task in tasks:
            if task.done() and not task.cancelled() and task.exception() is not None:
                logger.warning("live connection task failed: %r", task.exception())
            task.cancel()
        try:
            await websocket.close()
        except Exception:  # already closed
            pass
