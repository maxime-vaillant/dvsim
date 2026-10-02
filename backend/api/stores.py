"""File-backed stores for scenes and trajectories: ``<root>/<id>/meta.json`` plus the payload files.

``meta.json`` is written last, so an entry that is missing it (crash while uploading) is invisible.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

from fastapi import HTTPException

from backend.api import uploads
from backend.api.schemas import POINT_KEYS
from backend.api.util import NotFound, new_id, now_iso, read_json, valid_id, write_json_atomic

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def clean_name(name: Optional[str], fallback: str) -> str:
    return _CONTROL.sub("", name or "").strip()[:200] or fallback


def clean_points(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Keep only the known record keys (nothing else of a client payload is stored)."""
    return [{key: p[key] for key in POINT_KEYS if key in p} for p in points]


class _Store:
    kind = "item"
    public_keys: Tuple[str, ...] = ("id", "name", "created_at")

    def __init__(self, root: Path):
        self.root = root

    def path(self, item_id: str) -> Path:
        if not valid_id(item_id):
            raise NotFound(f"{self.kind} not found")
        return self.root / item_id

    def meta(self, item_id: str) -> Dict[str, Any]:
        meta = read_json(self.path(item_id) / "meta.json")
        if not isinstance(meta, dict):
            raise NotFound(f"{self.kind} not found")
        return meta

    def public(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {key: meta[key] for key in self.public_keys if key in meta}

    def list(self) -> List[Dict[str, Any]]:
        metas = []
        if self.root.is_dir():
            for entry in self.root.iterdir():
                meta = read_json(entry / "meta.json") if valid_id(entry.name) else None
                if isinstance(meta, dict):
                    metas.append(meta)
        metas.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return [self.public(m) for m in metas]

    def delete(self, item_id: str) -> None:
        self.meta(item_id)
        shutil.rmtree(self.path(item_id))

    def _new_dir(self) -> Tuple[str, Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            item_id = new_id()
            try:
                (self.root / item_id).mkdir()
            except FileExistsError:
                continue
            return item_id, self.root / item_id


class SceneStore(_Store):
    kind = "scene"
    public_keys = ("id", "name", "files", "created_at", "size_bytes")

    def create(
        self,
        name: Optional[str],
        model: BinaryIO,
        model_name: Optional[str],
        texture: Optional[BinaryIO],
        texture_name: Optional[str],
        max_bytes: int,
        extras: Optional[List[Tuple[str, BinaryIO]]] = None,
    ) -> Dict[str, Any]:
        """``extras``: (name, file) pairs of a multi-file scene (more OBJ files, MTL files, textures): then ``model``
        is its first OBJ and the upload is stored as a bundle (see ``uploads.store_scene_bundle``)."""
        item_id, path = self._new_dir()
        try:
            if extras:
                if texture is not None:
                    raise HTTPException(
                        status_code=422,
                        detail="a scene bundle takes its textures from the .mtl files: do not send a separate texture")
                stored = uploads.store_scene_bundle(path, [(model_name or "", model)] + list(extras), max_bytes)
            else:
                stored = uploads.store_scene_files(path, model, model_name, texture, texture_name, max_bytes)
                stored["files"] = [stored[k] for k in ("model", "texture") if stored[k]]
            meta = {
                "id": item_id,
                "name": clean_name(name, clean_name(Path(model_name or "").name, "scene")),
                "files": stored["files"],
                "model": stored["model"],
                "models": stored.get("models") or [stored["model"]],
                "texture": stored["texture"],
                "size_bytes": stored["size_bytes"],
                "created_at": now_iso(),
            }
            write_json_atomic(path / "meta.json", meta)
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise
        return self.public(meta)

    def spec(self, item_id: str) -> Dict[str, Any]:
        """SceneSpec for ``load_scene``: the stored model and, for OBJ, its texture."""
        meta, path = self.meta(item_id), self.path(item_id)
        entries = [{"path": str(path / model)} for model in meta.get("models") or [meta["model"]]]
        if meta.get("texture"):
            entries[0]["texture"] = str(path / meta["texture"])
        return {"objects": entries}


class TrajectoryStore(_Store):
    kind = "trajectory"
    public_keys = ("id", "name", "num_points", "duration_s", "source", "created_at")

    def create(self, name: Optional[str], points: List[Dict[str, float]], source: str = "upload") -> Dict[str, Any]:
        """Validate legacy records (ms timestamps) and store them; raises ValueError for unusable points."""
        from backend.trajectories import WaypointTrajectory

        records = clean_points(points)
        traj = WaypointTrajectory.from_json(records)
        item_id, path = self._new_dir()
        try:
            meta = {
                "id": item_id,
                "name": clean_name(name, "trajectory"),
                "num_points": len(records),
                "duration_s": (traj.end_us - traj.start_us) / 1e6,
                "source": source,
                "created_at": now_iso(),
            }
            write_json_atomic(path / "trajectory.json", records)
            write_json_atomic(path / "meta.json", meta)
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise
        return self.public(meta)

    def file(self, item_id: str) -> Path:
        self.meta(item_id)
        return self.path(item_id) / "trajectory.json"

    def get(self, item_id: str) -> Dict[str, Any]:
        records = read_json(self.file(item_id))
        return {**self.public(self.meta(item_id)), "points": records if isinstance(records, list) else []}
