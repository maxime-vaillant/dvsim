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


@pytest.fixture(scope="session")
def motion_checkpoint(tmp_path_factory):
    """A tiny trained motion model (a few seconds on the CPU): enough to exercise loading and sampling."""
    pytest.importorskip("torch")
    from motion_helpers import synthetic_walk

    from backend.motion.dataset import Stats, build_sequences
    from backend.motion.train import TrainConfig, train

    seqs = build_sequences([synthetic_walk(i) for i in range(6)])
    path = tmp_path_factory.mktemp("motion") / "tiny.pt"
    cfg = TrainConfig(steps=120, batch=8, base=16, mults=(1, 2), diffusion_steps=200, val_every=60, val_batches=2,
                      val_frac=0.0, test_frac=0.5)
    train(seqs, [], Stats.fit(seqs), cfg, path, device="cpu", log=lambda message: None)
    return path
