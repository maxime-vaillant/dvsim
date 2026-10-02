"""Shared fixtures: an offscreen GL context (skips cleanly when none is available) and the procedural demo scene."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items):
    """Tests using `gl_context` are marked `gl` automatically (`pytest -m "not gl"` deselects them)."""
    for item in items:
        if "gl_context" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.gl)


@pytest.fixture
def gl_context():
    """Standalone moderngl context: default backend first, then EGL (headless Linux); skips if both fail."""
    moderngl = pytest.importorskip("moderngl")
    errors = []
    ctx = None
    for kwargs in ({}, {"backend": "egl"}):
        try:
            ctx = moderngl.create_standalone_context(**kwargs)
            break
        except Exception as exc:  # no display / driver: try the next backend
            errors.append(f"{kwargs or 'default'}: {exc}")
    if ctx is None:
        pytest.skip("no OpenGL context available (" + "; ".join(errors) + ")")
    try:
        yield ctx
    finally:
        ctx.release()


@pytest.fixture(scope="session")
def demo_scene(tmp_path_factory):
    """`backend.samples.DemoScene` built once per session in a temp dir."""
    from backend.samples import create_demo_scene

    return create_demo_scene(tmp_path_factory.mktemp("demo_scene"))


@pytest.fixture(scope="session")
def demo_scene_dir(demo_scene):
    """Directory (Path) holding demo_room.obj/.png, demo_objects.glb, demo_scene.json, demo_trajectory.json."""
    return demo_scene.root
