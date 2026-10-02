"""AR headset relay: ``/headset`` receives camera samples, ``/spectate`` clients get them; recordings -> trajectories.

Protocol (unchanged): ``{"action": "start_recording" | "stream_data" | "stop_recording", "data": {timestamp, x, y, z,
rx, ry, rz}}`` (timestamp in ms, (rx, ry, rz) = forward vector). Spectators receive the ``data`` dict of every
``stream_data`` message. A recording (the samples streamed between start and stop) is stored as a trajectory with
``source: "headset"``; its timestamps are rebased to start at 0 ms.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from backend.api.schemas import HeadsetMessage
from backend.api.settings import MAX_INLINE_POINTS
from backend.api.stores import TrajectoryStore

logger = logging.getLogger("dvsim.headset")
router = APIRouter()
_FIELDS = ("x", "y", "z", "rx", "ry", "rz")


class SpectatorHub:
    """Connected spectators. Only touched from the event loop, so no lock is needed."""

    def __init__(self) -> None:
        self._sockets: Dict[uuid.UUID, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> uuid.UUID:
        await websocket.accept()
        spectator_id = uuid.uuid4()
        self._sockets[spectator_id] = websocket
        return spectator_id

    def disconnect(self, spectator_id: uuid.UUID) -> None:
        self._sockets.pop(spectator_id, None)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        text = json.dumps(message)
        targets = list(self._sockets.items())
        results = await asyncio.gather(*(ws.send_text(text) for _, ws in targets), return_exceptions=True)
        for (spectator_id, _), result in zip(targets, results):
            if isinstance(result, BaseException):  # gone (or broken): forget it, keep the others
                self.disconnect(spectator_id)


class Recording:
    """Samples of one start_recording .. stop_recording span."""

    def __init__(self) -> None:
        self.active = False
        self.start_ms = 0.0
        self.samples: List[Dict[str, float]] = []
        self.overflow = False

    def start(self, timestamp: float) -> None:
        self.active, self.start_ms, self.samples, self.overflow = True, timestamp, [], False

    def add(self, sample: Dict[str, Any]) -> None:
        if any(sample.get(k) is None for k in _FIELDS):
            return
        if len(self.samples) >= MAX_INLINE_POINTS:
            if not self.overflow:
                logger.warning("recording longer than %d samples: the rest is not recorded", MAX_INLINE_POINTS)
                self.overflow = True
            return
        self.samples.append(sample)

    def records(self) -> List[Dict[str, float]]:
        """Legacy records with strictly increasing timestamps (at us resolution), starting at 0 ms."""
        out: List[Dict[str, float]] = []
        last_us: Optional[int] = None
        t0 = self.samples[0]["timestamp"] if self.samples else 0.0
        for s in self.samples:
            us = int(round((s["timestamp"] - t0) * 1000))
            if last_us is not None and us <= last_us:
                continue
            last_us = us
            out.append({"timestamp": (s["timestamp"] - t0), **{k: s[k] for k in _FIELDS}})
        return out


async def _stop(store: TrajectoryStore, recording: Recording, end_ms: float) -> None:
    recording.active = False
    duration = end_ms - recording.start_ms
    count = len(recording.samples)
    rate = f"{1000 * count / duration:.1f} Hz" if duration > 0 else "n/a"
    logger.info("Recording stopped: %d samples, %.1f ms, %s", count, duration, rate)
    records = recording.records()
    recording.samples = []
    if len(records) < 2:
        logger.warning("Recording discarded: fewer than 2 usable samples")
        return
    name = "Headset " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        meta = await run_in_threadpool(store.create, name, records, "headset")
    except ValueError as exc:
        logger.warning("Recording discarded: %s", exc)
        return
    logger.info("Saved headset recording as trajectory %s (%d points)", meta["id"], meta["num_points"])


async def _handle(websocket: WebSocket, recording: Recording, message: HeadsetMessage) -> None:
    state = websocket.app.state
    data = message.data
    if message.action == "start_recording":
        if recording.active:
            logger.warning("start_recording while recording: saving the previous recording first")
            await _stop(state.trajectories, recording, data.timestamp)
        recording.start(data.timestamp)
        logger.info("Started recording")
    elif message.action == "stream_data":
        sample = data.model_dump()
        await state.hub.broadcast(sample)
        if recording.active:
            recording.add(sample)
    elif message.action == "stop_recording":
        if not recording.active:
            logger.warning("stop_recording received while not recording: ignored")
            return
        await _stop(state.trajectories, recording, data.timestamp)
    else:
        logger.warning("Unknown action received: %r", message.action)


@router.websocket("/spectate")
async def spectate(websocket: WebSocket) -> None:
    hub: SpectatorHub = websocket.app.state.hub
    spectator_id = await hub.connect(websocket)
    logger.info("Spectator %s connected", spectator_id)
    try:
        while True:
            if (await websocket.receive())["type"] == "websocket.disconnect":
                break
    finally:
        hub.disconnect(spectator_id)
        logger.info("Spectator %s disconnected", spectator_id)


@router.websocket("/headset")
async def headset(websocket: WebSocket) -> None:
    await websocket.accept()
    recording = Recording()
    try:
        while True:
            event = await websocket.receive()
            if event["type"] == "websocket.disconnect":
                break
            raw = event.get("text") if event.get("text") is not None else event.get("bytes")
            try:
                message = HeadsetMessage.model_validate_json(raw or b"")
            except ValidationError as exc:
                logger.warning("Invalid headset message ignored: %s", exc.errors()[0]["msg"] if exc.errors() else exc)
                continue
            try:
                await _handle(websocket, recording, message)
            except Exception:
                logger.exception("Error while processing a headset message")
    finally:
        if recording.active:
            logger.info("Headset disconnected during a recording: %d samples discarded", len(recording.samples))
        else:
            logger.info("Headset disconnected")
