from __future__ import annotations

import os
import sys
from typing import List, Optional

import moderngl

MIN_GL_VERSION = 410


def create_gl_context(backend: Optional[str] = None) -> moderngl.Context:
    """Standalone (headless) OpenGL >= 4.1 context.

    Tries the platform default; on Linux falls back to EGL (no display needed). ``backend`` or
    ``$DVSIM_GL_BACKEND`` force a moderngl backend ("egl", "x11", ...). Raises RuntimeError if none works.
    """
    backend = backend or os.environ.get("DVSIM_GL_BACKEND") or None
    attempts: List[Optional[str]] = [backend]
    if backend is None and sys.platform.startswith("linux"):
        attempts.append("egl")
    errors = []
    for name in attempts:
        kwargs = {"require": MIN_GL_VERSION}
        if name:
            kwargs["backend"] = name
        try:
            return moderngl.create_standalone_context(**kwargs)
        except Exception as exc:  # moderngl raises plain Error / ValueError / ImportError depending on the backend
            errors.append(f"{name or 'default'}: {exc}")
    raise RuntimeError(
        "could not create an OpenGL >= 4.1 context (" + "; ".join(errors) + "). "
        "On a headless Linux machine install EGL/Mesa (e.g. libegl1 libgl1-mesa-dri) or set DVSIM_GL_BACKEND=egl."
    )
