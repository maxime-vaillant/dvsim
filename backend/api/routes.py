"""REST + websocket endpoints under /api (see docs/architecture.md, "HTTP API contract")."""
from __future__ import annotations

import asyncio
import mimetypes
from importlib import metadata
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse

from backend.api.jobs import TERMINAL, QueueFull
from backend.api.planner import plan_job
from backend.api.schemas import JobCreate, TrajectoryCreate, TrajectoryGenerate
from backend.api.util import NotFound, UnsafePath, resolve_below

router = APIRouter(prefix="/api")


def package_version() -> str:
    try:
        return metadata.version("dvsim")
    except metadata.PackageNotFoundError:
        return "0+unknown"


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "version": package_version()}


# ---- scenes ----------------------------------------------------------------------------------------------------
@router.post("/scenes", status_code=201, summary="Upload a scene")
def create_scene(
    request: Request,
    obj: Annotated[UploadFile, File(description="scene model: .obj, .glb or .ply (never .gltf; see backend/api docs)")],
    texture: Annotated[Optional[UploadFile], File(description="texture image (.png/.jpg), .obj scenes only")] = None,
    files: Annotated[Optional[List[UploadFile]], File(
        description="more files of a multi-file scene: other .obj files, the .mtl files and their textures"
    )] = None,
    name: Annotated[str, Form()] = "",
) -> Dict[str, Any]:
    state = request.app.state
    if texture is not None and not texture.filename and not texture.size:
        texture = None  # a browser form with no file chosen
    extras = [(f.filename or "", f.file) for f in files or [] if f.filename]
    return state.scenes.create(
        name, obj.file, obj.filename, texture.file if texture else None,
        texture.filename if texture else None, state.settings.max_upload_bytes, extras or None,
    )


@router.get("/scenes")
def list_scenes(request: Request) -> List[Dict[str, Any]]:
    return request.app.state.scenes.list()


@router.delete("/scenes/{scene_id}", status_code=204)
def delete_scene(scene_id: str, request: Request) -> Response:
    state = request.app.state
    state.scenes.meta(scene_id)
    if scene_id in state.jobs.references():
        raise HTTPException(status_code=409, detail="scene is used by a queued or running job")
    state.scenes.delete(scene_id)
    return Response(status_code=204)


# ---- trajectories ----------------------------------------------------------------------------------------------
@router.post("/trajectories", status_code=201, summary="Store a trajectory (legacy records, timestamps in ms)")
def create_trajectory(body: TrajectoryCreate, request: Request) -> Dict[str, Any]:
    try:
        meta = request.app.state.trajectories.create(body.name, body.points, "upload")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid trajectory: {exc}") from None
    return {key: meta[key] for key in ("id", "name", "num_points", "duration_s")}


@router.post("/trajectories/generate", summary="Generate trajectories on a scene without simulating them")
async def generate_trajectories(body: TrajectoryGenerate, request: Request) -> Dict[str, Any]:
    state = request.app.state
    return await run_in_threadpool(state.previewer.generate, body, state.scenes)


@router.get("/trajectories")
def list_trajectories(request: Request) -> List[Dict[str, Any]]:
    return request.app.state.trajectories.list()


@router.get("/trajectories/{trajectory_id}")
def get_trajectory(trajectory_id: str, request: Request) -> Dict[str, Any]:
    return request.app.state.trajectories.get(trajectory_id)


@router.delete("/trajectories/{trajectory_id}", status_code=204)
def delete_trajectory(trajectory_id: str, request: Request) -> Response:
    state = request.app.state
    state.trajectories.meta(trajectory_id)
    if trajectory_id in state.jobs.references():
        raise HTTPException(status_code=409, detail="trajectory is used by a queued or running job")
    state.trajectories.delete(trajectory_id)
    return Response(status_code=204)


# ---- jobs ------------------------------------------------------------------------------------------------------
@router.post("/jobs", status_code=202, summary="Queue a simulation job")
def create_job(body: JobCreate, request: Request) -> Dict[str, str]:
    state = request.app.state
    plan = plan_job(body, state.scenes, state.trajectories)
    try:
        job_id = state.jobs.submit(plan)
    except QueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from None
    return {"id": job_id, "status": "queued"}


@router.get("/jobs")
def list_jobs(request: Request) -> List[Dict[str, Any]]:
    return request.app.state.jobs.list()


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> Dict[str, Any]:
    return request.app.state.jobs.get(job_id)


@router.delete("/jobs/{job_id}", summary="Cancel a queued / running job, or delete a finished one")
def delete_job(job_id: str, request: Request) -> Response:
    job = request.app.state.jobs.cancel_or_delete(job_id)
    if job is None:
        return Response(status_code=204)
    return JSONResponse(job, status_code=202)


@router.get("/jobs/{job_id}/files/{file_path:path}", summary="Download an output file")
def download(job_id: str, file_path: str, request: Request) -> FileResponse:
    root = request.app.state.jobs.path(job_id) / "output"
    if not root.is_dir():
        raise NotFound("job not found")
    try:
        target = resolve_below(root, file_path)
    except UnsafePath:
        raise HTTPException(status_code=400, detail="invalid file path") from None
    if not target.is_file():
        raise NotFound("file not found")
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(
        target, media_type=media_type, filename=target.name, content_disposition_type="attachment",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.websocket("/jobs/{job_id}/ws")
async def job_updates(websocket: WebSocket, job_id: str) -> None:
    """Pushes the job JSON every 0.5 s until the job reaches a final state."""
    jobs = websocket.app.state.jobs
    await websocket.accept()
    try:
        while True:
            try:
                job = await run_in_threadpool(jobs.get, job_id)
            except NotFound:
                await websocket.close(code=4404, reason="job not found")
                return
            await websocket.send_json(job)
            if job["status"] in TERMINAL:
                await websocket.close(code=1000)
                return
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return  # the client went away
