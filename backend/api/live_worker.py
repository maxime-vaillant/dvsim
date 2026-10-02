"""Live session worker: a *spawned* process (a GL context is never shared or forked) that renders the newest pose.

Parent -> worker: the latest pose in a shared array (a slot, so poses never queue up: only the newest matters) and,
rarely, commands over the pipe: ``("view", name)``, ``("origin", None | (x, y, z, heading_deg))``, ``("stop",)``.
Worker -> parent, over the pipe: ``("ready", info)``, ``("origin", info)`` (after a recalibration),
``("frame", seq, jpeg_bytes)``, ``("stats", dict)``, ``("error", message)``. ``seq`` is the sequence number of the
pose the frame was rendered from.
Only import-light modules at the top: the API process imports this module too.
"""
from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

from backend.api.util import short_error

logger = logging.getLogger("dvsim.live")
POSE_FIELDS = 8              # seq, x, y, z, rx, ry, rz, t
STATS_INTERVAL_S = 1.0
STILL_FRAMES = 3             # frames re-rendered after the last pose change (lets the events image fade)


def run(conn, slot, request: Dict[str, Any]) -> None:
    """Entry point of the process. ``slot``: shared array [seq, x, y, z, rx, ry, rz, t] (seq < 0: no pose yet).
    ``request``: scene (SceneSpec dict), scene_root, config (SimConfig dict), origin ("centre", "floor" or "shared": the
    point the interface picked, ``shared_origin`` = [x, y, z, heading_deg] or None), view, fps."""
    parent = os.getppid()
    try:
        from backend.config import SimConfig
        from backend.live import LiveRenderer
        from backend.scene import load_scene

        scene = load_scene(request["scene"], root=Path(request["scene_root"]))
        live = LiveRenderer(scene, SimConfig.from_dict(request["config"]),
                            origin="floor" if request["origin"] == "shared" else request["origin"])
        if request["origin"] == "shared" and request.get("shared_origin"):
            live.set_origin(request["shared_origin"][:3], request["shared_origin"][3])
    except Exception as exc:  # reported to the client; the traceback goes to the log
        logger.exception("live session could not start")
        _send(conn, ("error", short_error(exc, hide=str(request.get("scene_root", "")))))
        return
    try:
        _send(conn, ("ready", {
            "width": live.config.camera.width, "height": live.config.camera.height,
            "origin": [float(v) for v in live.origin], "heading_deg": live.heading_deg,
        }))
        _loop(conn, slot, live, request, parent)
    except Exception as exc:
        logger.exception("live session failed")
        _send(conn, ("error", short_error(exc)))
    finally:
        live.close()


def _send(conn, message) -> bool:
    try:
        conn.send(message)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _loop(conn, slot, live, request: Dict[str, Any], parent: int) -> None:
    view = request["view"]
    period = 1.0 / float(request["fps"])
    started = time.monotonic()
    next_at = started
    last_seq, still = -1.0, 0
    frames, render_s, stats_at = 0, 0.0, started
    while True:
        while conn.poll():
            try:
                message = conn.recv()
            except EOFError:
                return
            if message[0] == "stop":
                return
            if message[0] == "view" and message[1] in _views():
                view = message[1]
            elif message[0] == "origin":
                value = message[1]
                live.set_origin(None if value is None else value[:3], 0.0 if value is None else value[3])
                _send(conn, ("origin", {"origin": [float(v) for v in live.origin], "heading_deg": live.heading_deg}))
        if os.getppid() != parent:   # the server is gone: do not run as an orphan
            return
        with slot.get_lock():
            seq, x, y, z, rx, ry, rz, _ = slot[:POSE_FIELDS]
        now = time.monotonic()
        if seq < 0 or now < next_at:
            time.sleep(0.002 if seq >= 0 else 0.01)
            continue
        next_at = max(next_at + period, now - period)   # never try to catch up after a slow frame
        still = still + 1 if seq == last_seq else 0
        if still > STILL_FRAMES:                         # the camera has not moved: nothing new to render
            continue
        if not all(math.isfinite(v) for v in (x, y, z, rx, ry, rz)) or math.hypot(rx, ry, rz) < 1e-6:
            continue                                     # an unusable pose: wait for the next one
        began = time.monotonic()
        jpeg = live.encode(live.render((x, y, z), (rx, ry, rz), int((began - started) * 1e6), view))
        render_s += time.monotonic() - began
        frames += 1
        if not _send(conn, ("frame", int(seq), jpeg)):
            return
        last_seq = seq
        if time.monotonic() - stats_at >= STATS_INTERVAL_S:
            span = time.monotonic() - stats_at
            _send(conn, ("stats", {"fps": round(frames / span, 1), "render_ms": round(1000 * render_s / frames, 1)}))
            frames, render_s, stats_at = 0, 0.0, time.monotonic()


def _views():
    from backend.live import VIEWS

    return VIEWS
