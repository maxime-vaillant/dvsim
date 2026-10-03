"""Where trained motion models live and how a spec names one. No torch: the API uses it to validate requests."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from backend.config import data_dir as default_data_dir

MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SUFFIX = ".pt"


def models_dir(data_dir: Optional[Path] = None) -> Path:
    return Path(data_dir or default_data_dir()) / "motion"


def check_model_id(model: Any) -> str:
    """``model`` if it is a valid model id (a file name of the models directory), else ValueError."""
    if not isinstance(model, str) or not MODEL_ID.fullmatch(model):
        raise ValueError(f"invalid motion model id {model!r}: use 1-64 letters, digits, '.', '_' or '-'")
    return model


def resolve_model(ref: Mapping[str, Any], data_dir: Optional[Path] = None) -> Path:
    """The checkpoint file a ``motion`` spec points at: ``{"model": id}`` (a file of the models directory) or
    ``{"path": file}`` (CLI and trusted callers). Raises ValueError for a malformed reference and
    FileNotFoundError for a missing file."""
    if "path" in ref:
        path = Path(str(ref["path"])).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"motion model file not found: {path}")
        return path
    model = check_model_id(ref.get("model"))
    path = models_dir(data_dir) / f"{model}{SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(f"motion model {model!r} not found in {path.parent}")
    return path


def list_models(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    folder = models_dir(data_dir)
    if not folder.is_dir():
        return []
    return [{"id": p.stem, "size_bytes": p.stat().st_size}
            for p in sorted(folder.glob(f"*{SUFFIX}")) if MODEL_ID.fullmatch(p.stem)]
