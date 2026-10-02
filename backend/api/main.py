"""FastAPI app: ``uvicorn backend.api.main:app`` (that is what ``dvsim serve`` runs)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from backend.api import headset, live, routes
from backend.api.jobs import JobManager
from backend.api.limits import BodyLimitMiddleware
from backend.api.live import LiveManager
from backend.api.preview import TrajectoryPreviewer
from backend.api.settings import MAX_JSON_BYTES, Settings
from backend.api.stores import SceneStore, TrajectoryStore
from backend.api.util import NotFound


def _configure_logging() -> None:
    log = logging.getLogger("dvsim")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False


def describe_errors(errors: List[Dict[str, Any]], limit: int = 5) -> str:
    """Pydantic error list -> one readable string ("trajectory.rectilinear.duration_s: Input should be > 0")."""
    parts = []
    for error in errors[:limit]:
        kind = error.get("type")
        loc = [str(item) for item in error.get("loc", ())]
        loc = [] if kind == "json_invalid" else loc[1:] if loc[:1] == ["body"] else loc
        message = {
            "extra_forbidden": "unknown key", "missing": "field required", "json_invalid": "invalid JSON body",
        }.get(kind, error.get("msg"))
        parts.append(f"{'.'.join(loc)}: {message}" if loc else str(message))
    if len(errors) > limit:
        parts.append(f"(+{len(errors) - limit} more)")
    return "; ".join(parts)


def create_app(data_dir: Optional[Path] = None) -> FastAPI:
    """Build the app. Data root: ``data_dir`` or ``$DVSIM_DATA_DIR`` (default ./data); subdirectories appear on use."""
    settings = Settings.from_env(data_dir)
    jobs = JobManager(
        settings.jobs_dir, settings.max_running_jobs, settings.max_pending_jobs, settings.cancel_grace_s
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        _configure_logging()
        jobs.start()
        try:
            yield
        finally:
            app.state.live.stop_all()
            jobs.stop()

    app = FastAPI(
        title="DVSim", version=routes.package_version(), lifespan=lifespan,
        docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.scenes = SceneStore(settings.scenes_dir)
    app.state.trajectories = TrajectoryStore(settings.trajectories_dir)
    app.state.jobs = jobs
    app.state.hub = headset.SpectatorHub()
    app.state.previewer = TrajectoryPreviewer()
    app.state.live = LiveManager()

    @app.exception_handler(NotFound)
    async def not_found(_: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": describe_errors(exc.errors())}, status_code=422)

    app.add_middleware(BodyLimitMiddleware, upload_limit=settings.max_upload_bytes,
                       default_limit=min(settings.max_upload_bytes, MAX_JSON_BYTES))
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"],
        )
    app.include_router(routes.router)
    app.include_router(headset.router)
    app.include_router(live.router)

    if settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def index() -> dict:
            return {"name": "DVSim API", "docs": "/api/docs", "frontend": "not built (npm run build in frontend/)"}

    return app


app = create_app()
