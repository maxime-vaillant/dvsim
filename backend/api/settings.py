"""Server settings, read from plain environment variables, and the hard limits of the API."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from backend.config import data_dir as default_data_dir

MAX_INLINE_POINTS = 200_000       # points of one uploaded / inline trajectory
MAX_SEQUENCES = 100               # `count` of a generated trajectory spec
MAX_DURATION_S = 3600.0           # duration of a generated trajectory
MAX_STEPS = 2_000_000             # simulation steps of one sequence
MAX_RESOLUTION = 8192             # camera width / height
MAX_JSON_BYTES = 64 * 1024 * 1024  # body of every request except the scene upload

REPO_ROOT = Path(__file__).resolve().parents[2]


def _number(name: str, default: float, minimum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"${name} must be a number, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"${name} must be >= {minimum:g}, got {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    frontend_dir: Path
    max_upload_bytes: int
    max_running_jobs: int
    max_pending_jobs: int
    cancel_grace_s: float
    cors_origins: Tuple[str, ...]

    @property
    def scenes_dir(self) -> Path:
        return self.data_dir / "scenes"

    @property
    def trajectories_dir(self) -> Path:
        return self.data_dir / "trajectories"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @classmethod
    def from_env(cls, data_dir: Optional[Path] = None) -> "Settings":
        frontend = os.environ.get("DVSIM_FRONTEND_DIR", "").strip()
        origins = os.environ.get("DVSIM_CORS_ORIGINS", "")
        return cls(
            data_dir=Path(data_dir).resolve() if data_dir is not None else default_data_dir(),
            frontend_dir=Path(frontend).resolve() if frontend else REPO_ROOT / "frontend" / "dist",
            max_upload_bytes=int(_number("DVSIM_MAX_UPLOAD_MB", 512, 0.001) * 1024 * 1024),
            max_running_jobs=int(_number("DVSIM_MAX_JOBS", 1, 1)),
            max_pending_jobs=int(_number("DVSIM_MAX_PENDING", 32, 1)),
            cancel_grace_s=_number("DVSIM_CANCEL_GRACE_S", 5.0, 0.0),
            cors_origins=tuple(o.strip().rstrip("/") for o in origins.split(",") if o.strip()),
        )
