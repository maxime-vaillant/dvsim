"""Helpers shared by the API modules and the job worker (import-light: the spawned worker imports this)."""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ID_RE = re.compile(r"[0-9a-f]{16}")


class NotFound(LookupError):
    """Unknown (or malformed) id; mapped to HTTP 404."""


class UnsafePath(ValueError):
    """A client-supplied relative path that is not a plain path below the expected root."""


def new_id() -> str:
    return secrets.token_hex(8)


def valid_id(value: Any) -> bool:
    return isinstance(value, str) and _ID_RE.fullmatch(value) is not None


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_json_atomic(path: Path, data: Any) -> None:
    """Write next to the target, then ``os.replace``: readers see the old or the new file, never half of one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def list_outputs(root: Path) -> List[Dict[str, Any]]:
    """Every regular file below ``root`` (symlinks are neither followed nor listed), sequences in subdirectories."""
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            out.append({"name": filename, "path": path.relative_to(root).as_posix(), "size_bytes": size})
    return out


def resolve_below(root: Path, rel: str) -> Path:
    """``root / rel`` for a plain relative POSIX path; raises UnsafePath on `..`, absolute, NUL, backslash, symlink."""
    if not rel or "\x00" in rel or "\\" in rel or rel.startswith("/"):
        raise UnsafePath(rel)
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise UnsafePath(rel)
    target = root
    for part in parts:
        target = target / part
        if target.is_symlink():
            raise UnsafePath(rel)
    if not target.resolve().is_relative_to(root.resolve()):
        raise UnsafePath(rel)
    return target


def short_error(exc: BaseException, hide: str = "") -> str:
    """One-line, bounded error text for API clients (the traceback goes to the server log)."""
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    if hide:
        text = text.replace(hide, "<data>")
    return " ".join(text.split())[:500]
