"""HTTP API (backend.api): stores, uploads, real spawned-worker jobs, files endpoint, websockets, headset relay."""
from __future__ import annotations

import io
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
pytest.importorskip("httpx")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from backend.api import uploads  # noqa: E402
from backend.api.main import create_app  # noqa: E402

FINAL = ("done", "failed", "cancelled")
SMALL = {"camera": {"width": 32, "height": 24}}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("DVSIM_"):
            monkeypatch.delenv(key)


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def client(data_dir):
    with TestClient(create_app(data_dir)) as c:
        yield c


def _client(data_dir, monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return TestClient(create_app(data_dir))


def _upload(client, demo, texture=True, name="room"):
    files = {"obj": ("room.obj", demo.room_obj.read_bytes())}
    if texture:
        files["texture"] = ("room.png", demo.room_texture.read_bytes())
    r = client.post("/api/scenes", files=files, data={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


def _points(demo, n=21):
    return json.loads(demo.trajectory_json.read_text())[:n]


def _line(seconds, y=1.5):
    """Two-point straight path inside the demo room (the trajectory interpolates: `seconds` of simulation)."""
    return [
        {"timestamp": 0.0, "x": -2.0, "y": y, "z": 0.0, "rx": 1.0, "ry": 0.0, "rz": 0.0},
        {"timestamp": seconds * 1000.0, "x": 2.0, "y": y, "z": 0.0, "rx": 1.0, "ry": 0.0, "rz": 0.0},
    ]


def _job(client, scene_id, trajectory, config=SMALL, **extra):
    r = client.post("/api/jobs", json={"scene_id": scene_id, "trajectory": trajectory, "config": config, **extra})
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "queued"
    return r.json()["id"]


def _wait(client, job_id, until=FINAL, timeout=90.0):
    deadline = time.monotonic() + timeout
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in until:
            return job
        if time.monotonic() > deadline:
            pytest.fail(f"job {job_id} still {job['status']} after {timeout} s: {job}")
        time.sleep(0.1)


def _wait_progress(client, job_id, timeout=60.0):
    """Until the worker is simulating (at least one step done)."""
    deadline = time.monotonic() + timeout
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] in ("queued", "running"), job
        if job["progress"]["step"] >= 1:
            return job
        if time.monotonic() > deadline:
            pytest.fail(f"job {job_id} made no progress: {job}")
        time.sleep(0.1)


def _fake_job(data_dir: Path, status="done", job_id="0123456789abcdef", files=None):
    """A job directory as the worker would leave it (no process involved)."""
    root = data_dir / "jobs" / job_id
    (root / "output" / "seq_000").mkdir(parents=True)
    for rel, content in (files or {"seq_000/a.txt": b"hello"}).items():
        (root / "output" / rel).write_bytes(content)
    job = {
        "id": job_id, "status": status, "error": None,
        "progress": {"sequence": 0, "num_sequences": 1, "step": 0, "total_steps": 0, "fraction": 0.0},
        "outputs": [], "created_at": "2026-01-01T00:00:00.000000Z", "started_at": None, "finished_at": None,
        "config": {},
    }
    (root / "job.json").write_text(json.dumps(job))
    return root


# ---- basics -----------------------------------------------------------------------------------------------------
def test_health_and_lazy_data_dir(client, data_dir):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["version"]
    assert client.get("/api/scenes").json() == []
    assert client.get("/api/trajectories").json() == []
    assert client.get("/api/jobs").json() == []
    assert not data_dir.exists()  # scenes/ trajectories/ jobs/ are created on demand


def test_start_method_is_spawn(data_dir):
    assert create_app(data_dir).state.jobs._ctx.get_start_method() == "spawn"


# ---- scenes -----------------------------------------------------------------------------------------------------
def test_scene_upload_list_delete(client, data_dir, demo_scene, tmp_path):
    scene = _upload(client, demo_scene)
    assert scene["name"] == "room" and scene["files"] == ["scene.obj", "texture.png"]
    stored = data_dir / "scenes" / scene["id"]
    assert (stored / "scene.obj").read_bytes() == demo_scene.room_obj.read_bytes()
    assert (stored / "texture.png").read_bytes() == demo_scene.room_texture.read_bytes()

    listed = client.get("/api/scenes").json()
    assert [s["id"] for s in listed] == [scene["id"]]
    assert set(listed[0]) >= {"id", "name", "files", "created_at"}

    assert client.delete(f"/api/scenes/{scene['id']}").status_code == 204
    assert client.get("/api/scenes").json() == []
    assert not stored.exists()
    assert client.delete(f"/api/scenes/{scene['id']}").status_code == 404


def test_scene_without_texture_and_glb(client, demo_scene):
    obj = _upload(client, demo_scene, texture=False)
    assert obj["files"] == ["scene.obj"] and obj["name"] == "room"
    glb = client.post("/api/scenes", files={"obj": ("x.GLB", demo_scene.objects_glb.read_bytes())})
    assert glb.status_code == 201 and glb.json()["files"] == ["scene.glb"] and glb.json()["name"] == "x.GLB"


def test_client_filenames_never_reach_the_disk(client, data_dir, demo_scene, tmp_path):
    r = client.post(
        "/api/scenes",
        files={"obj": ("../../../evil.obj", demo_scene.room_obj.read_bytes()),
               "texture": ("..\\..\\evil.png", demo_scene.room_texture.read_bytes())},
        data={"name": "x\x00\x1f/../y"},
    )
    assert r.status_code == 201, r.text
    stored = data_dir / "scenes" / r.json()["id"]
    assert sorted(p.name for p in stored.iterdir()) == ["meta.json", "scene.obj", "texture.png"]
    assert not [p for p in tmp_path.rglob("*") if "evil" in p.name]
    assert "\x00" not in r.json()["name"]


def _glb(document):
    body = json.dumps(document).encode()
    body += b" " * (-len(body) % 4)
    return b"glTF" + struct.pack("<II", 2, 20 + len(body)) + struct.pack("<II", len(body), 0x4E4F534A) + body


@pytest.mark.parametrize("filename, content, extra", [
    ("a.gltf", b'{"asset": {"version": "2.0"}}', {}),
    ("a.exe", b"MZ", {}),
    ("a", b"v 0 0 0\n", {}),
    ("a.json", b'{"objects": [{"path": "/etc/passwd"}]}', {}),
    ("a.obj", b"mtllib ../../secret.mtl\nv 0 0 0\n", {}),
    ("a.obj", b"# note: MTLLIB /etc/passwd\nv 0 0 0\n", {}),
    ("a.obj", b"v 0 0 0\n  map_Kd /etc/passwd\n", {}),
    ("a.ply", b"ply\nformat ascii 1.0\ncomment TextureFile ../../x.png\nend_header\n", {}),
    ("a.glb", _glb({"asset": {"version": "2.0"}, "buffers": [{"uri": "../../secret.bin", "byteLength": 4}]}), {}),
    ("a.glb", _glb({"asset": {"version": "2.0"}, "images": [{"uri": "file:///etc/passwd"}]}), {}),
    ("a.glb", b"not a glb at all, not at all", {}),
    ("a.obj", b"", {}),
    ("a.obj", b"v 0 0 0\n", {"texture": ("t.png", b"definitely not an image")}),
    ("a.obj", b"v 0 0 0\n", {"texture": ("t.exe", b"\x89PNG\r\n\x1a\n")}),
    ("a.glb", _glb({"asset": {"version": "2.0"}}), {"texture": ("t.png", b"\x89PNG")}),
])
def test_scene_upload_rejected(client, data_dir, filename, content, extra):
    files = {"obj": (filename, content), **extra}
    r = client.post("/api/scenes", files=files)
    assert r.status_code == 422, r.text
    assert client.get("/api/scenes").json() == []
    scenes = data_dir / "scenes"
    assert not scenes.exists() or list(scenes.iterdir()) == []  # nothing left behind


def test_self_contained_glb_accepted(client):
    doc = {"asset": {"version": "2.0"}, "buffers": [{"uri": "data:application/octet-stream;base64,AAAA", "byteLength": 3}]}
    assert client.post("/api/scenes", files={"obj": ("a.glb", _glb(doc))}).status_code == 201


def test_upload_requires_the_obj_field(client):
    r = client.post("/api/scenes", files={"texture": ("t.png", b"x")})
    assert r.status_code == 422 and r.json()["detail"] == "obj: field required"


def test_forbidden_token_split_across_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "_CHUNK", 16)
    data = b"v 0 0 0\n" + b"#" * 9 + b"mtl" + b"lib x\n" + b"v 1 1 1\n"
    with pytest.raises(HTTPException) as info:
        uploads._copy(io.BytesIO(data), tmp_path / "o.obj", 1 << 20, ".obj")
    assert info.value.status_code == 422


def test_oversize_upload_rejected(data_dir, monkeypatch, demo_scene):
    big = b"v 0 0 0\n" * 300_000  # 2.4 MB
    with _client(data_dir, monkeypatch, DVSIM_MAX_UPLOAD_MB=1) as client:
        r = client.post("/api/scenes", files={"obj": ("a.obj", big)})
        assert r.status_code == 413
        boundary = "xxBOUNDARYxx"

        def body():  # no Content-Length: the limit must be applied while the body streams in
            yield f'--{boundary}\r\nContent-Disposition: form-data; name="obj"; filename="a.obj"\r\n\r\n'.encode()
            for i in range(0, len(big), 65536):
                yield big[i:i + 65536]
            yield f"\r\n--{boundary}--\r\n".encode()

        r = client.post("/api/scenes", content=body(), headers={"content-type": f"multipart/form-data; boundary={boundary}"})
        assert r.status_code == 413
        assert client.get("/api/scenes").json() == []
        small = client.post("/api/scenes", files={"obj": ("a.obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")})
        assert small.status_code == 201  # the server is fine afterwards
        assert client.post("/api/trajectories", content=big * 2, headers={"content-type": "application/json"}).status_code == 413


# ---- trajectories -----------------------------------------------------------------------------------------------
def test_trajectory_create_list_get_delete(client, data_dir, demo_scene):
    points = _points(demo_scene, 51)
    r = client.post("/api/trajectories", json={"name": "walk", "points": points})
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["name"] == "walk" and created["num_points"] == 51 and created["duration_s"] == pytest.approx(0.5)

    listed = client.get("/api/trajectories").json()
    assert [t["id"] for t in listed] == [created["id"]] and listed[0]["source"] == "upload"
    detail = client.get(f"/api/trajectories/{created['id']}").json()
    assert detail["points"] == points
    assert (data_dir / "trajectories" / created["id"] / "trajectory.json").is_file()

    assert client.delete(f"/api/trajectories/{created['id']}").status_code == 204
    assert client.get("/api/trajectories").json() == []
    assert client.get(f"/api/trajectories/{created['id']}").status_code == 404
    assert client.delete(f"/api/trajectories/{created['id']}").status_code == 404


def test_trajectory_validation(client):
    p = {"timestamp": 0.0, "x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": -1}
    q = {**p, "timestamp": 10.0}
    assert client.post("/api/trajectories", json={"name": "a", "points": [p]}).status_code == 422
    assert client.post("/api/trajectories", json={"name": "a", "points": [q, p]}).status_code == 422  # goes backwards
    assert client.post("/api/trajectories", json={"name": "a", "points": [p, p]}).status_code == 422  # duplicate time
    assert client.post("/api/trajectories", json={"name": "a", "points": [p, {"timestamp": 5.0}]}).status_code == 422
    assert client.post("/api/trajectories", json={"name": "a", "points": [p, q], "extra": 1}).status_code == 422
    assert client.post("/api/trajectories", json={"name": "a", "points": [p, {**q, "x": "abc"}]}).status_code == 422
    r = client.post("/api/trajectories", content='{"name": "a", "points": [{"timestamp": 0, "x": NaN}, {"timestamp": 1}]}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert client.get("/api/trajectories").json() == []
    ok = client.post("/api/trajectories", json={"name": "a", "points": [p, {**q, "junk": 1}]})
    assert ok.status_code == 201  # unknown per-point keys are dropped, not stored
    assert "junk" not in client.get(f"/api/trajectories/{ok.json()['id']}").json()["points"][1]


def test_too_many_trajectory_points(client):
    point = {"timestamp": 0.0, "x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": -1}
    r = client.post("/api/trajectories", json={"name": "a", "points": [point] * 200_001})
    assert r.status_code == 422


# ---- job requests: validation before anything is spawned ----------------------------------------------------------
def test_job_validation(client, demo_scene):
    scene = _upload(client, demo_scene, texture=False)["id"]
    traj = client.post("/api/trajectories", json={"name": "t", "points": _points(demo_scene, 11)}).json()["id"]
    imported = {"type": "imported", "id": traj}

    def post(**body):
        r = client.post("/api/jobs", json=body)
        if r.status_code in (404, 422):
            assert isinstance(r.json()["detail"], str) and r.json()["detail"], r.text  # always one readable string
        return r


    assert post(scene_id="0" * 16, trajectory=imported).status_code == 404
    assert post(scene_id="../../etc", trajectory=imported).status_code == 404
    assert post(scene_id=scene, trajectory={"type": "imported", "id": "f" * 16}).status_code == 404
    assert post(scene_id=scene, trajectory={"type": "imported", "id": "../x"}).status_code == 404
    r = post(scene_id=scene, trajectory=imported, config={"camera": {"wdith": 5}})
    assert r.status_code == 422 and "wdith" in r.text
    assert post(scene_id=scene, trajectory=imported, config={"nope": 1}).status_code == 422
    assert post(scene_id=scene, trajectory=imported, config={"camera": {"width": "big"}}).status_code == 422
    assert post(scene_id=scene, trajectory=imported, config={"camera": {"width": 1.5}}).status_code == 422
    assert post(scene_id=scene, trajectory=imported, config={"camera": {"width": 100000}}).status_code == 422
    assert post(scene_id=scene, trajectory=imported, config={"dt_us": 0}).status_code == 422
    assert post(scene_id=scene, trajectory=imported, config={"events": {"c_pos": "x"}}).status_code == 422
    assert post(scene_id=scene, trajectory=imported, config={"gt": ["nope"]}).status_code == 422
    assert post(scene_id=scene, trajectory=imported, config="rgb").status_code == 422
    assert post(scene_id=scene, trajectory=imported, extra=1).status_code == 422
    assert post(scene_id=scene).json()["detail"] == "trajectory: field required"
    r = client.post("/api/jobs", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["detail"] == "invalid JSON body"
    # trajectory specs
    assert post(scene_id=scene, trajectory={"type": "file", "path": "/etc/passwd"}).status_code == 422
    assert post(scene_id=scene, trajectory={"type": "nope"}).status_code == 422
    r = post(scene_id=scene, trajectory={"type": "rectilinear", "duration_s": -1})
    assert r.status_code == 422 and r.json()["detail"].startswith("trajectory.rectilinear.duration_s: ")
    assert post(scene_id=scene, trajectory={"type": "rectilinear", "duration_s": 1, "count": 0}).status_code == 422
    assert post(scene_id=scene, trajectory={"type": "rectilinear", "duration_s": 1, "count": 10 ** 6}).status_code == 422
    assert post(scene_id=scene, trajectory={"type": "rectilinear", "duration_s": 1, "seed": -3}).status_code == 422
    r = post(scene_id=scene, trajectory={"type": "rectilinear", "duration_s": 1, "bogus": 1})
    assert r.status_code == 422 and r.json()["detail"] == "trajectory.rectilinear.bogus: unknown key"
    r = post(scene_id=scene, trajectory={"type": "nope"})
    assert r.status_code == 422 and "imported" in r.json()["detail"]  # lists the accepted types
    assert post(scene_id=scene, trajectory=imported, visualize="maybe").json()["detail"].startswith("visualize: ")
    assert post(scene_id=scene, trajectory={"type": "spline", "duration_s": 1, "n_key_points": 1}).status_code == 422
    assert post(scene_id=scene, trajectory={"type": "street", "duration_s": 1, "speed": 0}).status_code == 422
    assert post(scene_id=scene, trajectory={"type": "street", "duration_s": 1, "eye_height": -1}).status_code == 422
    r = post(scene_id=scene, trajectory={"type": "street", "duration_s": 1, "n_key_points": 3})
    assert r.status_code == 422 and r.json()["detail"] == "trajectory.street.n_key_points: unknown key"
    assert post(scene_id=scene, trajectory={"type": "imported", "id": traj, "origin": [1, 2]}).status_code == 422
    assert post(scene_id=scene, trajectory={"type": "inline", "points": _line(1)[:1]}).status_code == 422
    bad = [_line(1)[1], _line(1)[0]]
    r = post(scene_id=scene, trajectory={"type": "inline", "points": bad})
    assert r.status_code == 422 and "backwards" in r.text
    r = post(scene_id=scene, trajectory={"type": "inline", "points": _line(10)}, config={"dt_us": 1})
    assert r.status_code == 422 and "steps" in r.text
    assert client.get("/api/jobs").json() == []  # nothing was queued

    assert client.get("/api/jobs/0123456789abcdef").status_code == 404
    assert client.get("/api/jobs/..%2f..%2fetc").status_code == 404
    assert client.get("/api/jobs/NOT-AN-ID").status_code == 404
    assert client.delete("/api/jobs/0123456789abcdef").status_code == 404


def test_worker_confines_the_loader_to_the_scene_directory(data_dir, monkeypatch):
    import backend.pipeline
    from backend.api import worker

    job_dir = _fake_job(data_dir)
    (job_dir / "request.json").write_text(json.dumps({
        "scene": {"objects": [{"path": "/x/scene.obj"}]}, "scene_root": "/x", "trajectory": {"type": "inline"},
        "config": {}, "visualize": False,
    }))
    calls = []

    def fake(scene, trajectories, config, output_dir, *, visualize=False, progress=None, cancel=None, scene_root=None):
        calls.append(scene_root)

    monkeypatch.setattr(backend.pipeline, "run_to_disk", fake)
    assert worker._execute(job_dir, worker._State(job_dir)) == "done"
    assert calls == ["/x"]


def test_worker_stops_when_cancelled_or_orphaned(data_dir, monkeypatch):
    import backend.pipeline
    from backend.api import worker

    job_dir = _fake_job(data_dir)
    (job_dir / "request.json").write_text(json.dumps({
        "scene": {}, "scene_root": "/x", "trajectory": {}, "config": {}, "visualize": False,
    }))
    seen = []
    monkeypatch.setattr(backend.pipeline, "run_to_disk", lambda *a, cancel=None, **k: seen.append(cancel))
    worker._execute(job_dir, worker._State(job_dir))
    cancel = seen[0]
    assert cancel() is False
    (job_dir / "cancel").touch()
    assert cancel() is True
    (job_dir / "cancel").unlink()
    monkeypatch.setattr(worker.os, "getppid", lambda: 1)  # the server died: the worker was re-parented
    assert cancel() is True


def test_worker_does_not_overwrite_a_settled_job(data_dir):
    from backend.api import worker

    job_dir = _fake_job(data_dir, "failed")
    state = worker._State(job_dir)
    state.finish("done", None, job_dir / "output")
    assert json.loads((job_dir / "job.json").read_text())["status"] == "failed"


# ---- real jobs (spawned worker + GL) ----------------------------------------------------------------------------
@pytest.mark.usefixtures("gl_context")
def test_job_end_to_end(client, data_dir, demo_scene):
    import h5py
    import hdf5plugin  # noqa: F401  (registers the zstd filter)

    scene = _upload(client, demo_scene)
    job_id = _job(client, scene["id"], {"type": "inline", "points": _points(demo_scene, 21), "origin": None})
    assert client.get("/api/jobs").json()[0]["id"] == job_id

    seen = []
    deadline = time.monotonic() + 90
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        seen.append((job["status"], job["progress"]["fraction"]))
        if job["status"] in FINAL or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["error"] is None and job["finished_at"] and job["started_at"] and job["created_at"]
    assert job["progress"] == {"sequence": 0, "num_sequences": 1, "step": 21, "total_steps": 21, "fraction": 1.0}
    assert [f for _, f in seen] == sorted(f for _, f in seen)  # progress never goes back
    assert job["config"]["camera"]["width"] == 32 and job["config"]["camera"]["height"] == 24
    assert job["scene_id"] == scene["id"] and job["trajectory"]["type"] == "inline"
    request = json.loads((data_dir / "jobs" / job_id / "request.json").read_text())
    assert request["scene_root"] == str(data_dir / "scenes" / scene["id"])
    assert request["scene"]["objects"][0]["path"] == str(data_dir / "scenes" / scene["id"] / "scene.obj")
    assert request["scene"]["objects"][0]["texture"] == str(data_dir / "scenes" / scene["id"] / "texture.png")

    outputs = {o["path"]: o for o in job["outputs"]}
    output_dir = data_dir / "jobs" / job_id / "output"
    on_disk = sorted(p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*") if p.is_file())
    assert sorted(outputs) == on_disk
    assert {"seq_000/left_events.h5", "seq_000/ground_truth.h5", "seq_000/meta.json"} <= set(outputs)
    for path, out in outputs.items():
        assert out["name"] == Path(path).name and out["size_bytes"] == (output_dir / path).stat().st_size

    r = client.get(f"/api/jobs/{job_id}/files/seq_000/left_events.h5")
    assert r.status_code == 200
    assert r.content == (output_dir / "seq_000" / "left_events.h5").read_bytes()
    assert r.headers["content-disposition"].startswith("attachment")
    downloaded = data_dir / "downloaded.h5"
    downloaded.write_bytes(r.content)
    with h5py.File(downloaded, "r") as f:
        assert f["events/t"].shape == f["events/x"].shape == f["events/p"].shape
        assert f["events"].attrs["width"] == 32 and f["events"].attrs["height"] == 24
        assert f["events/t"].shape[0] == f["events"].attrs["num_events"]
    with h5py.File(output_dir / "seq_000" / "ground_truth.h5", "r") as f:
        assert f["ground_truth/depth"].shape == (21, 24, 32)

    assert client.get(f"/api/jobs/{job_id}/files/seq_000/nope.h5").status_code == 404
    # finished job: DELETE removes the record and its files
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not (data_dir / "jobs" / job_id).exists()


@pytest.mark.usefixtures("gl_context")
def test_generated_trajectories_write_one_directory_per_sequence(client, demo_scene):
    scene = _upload(client, demo_scene)["id"]
    spec = {"type": "spline", "duration_s": 0.1, "n_key_points": 4, "count": 2, "seed": 3}
    job = _wait(client, _job(client, scene, spec, config={**SMALL, "gt": ["depth"]}, visualize=False))
    assert job["status"] == "done", job
    assert job["progress"]["num_sequences"] == 2 and job["progress"]["fraction"] == 1.0
    dirs = {o["path"].split("/")[0] for o in job["outputs"]}
    assert dirs == {"seq_000", "seq_001"}


@pytest.mark.usefixtures("gl_context")
def test_job_progress_websocket(client, demo_scene):
    scene = _upload(client, demo_scene)["id"]
    job_id = _job(client, scene, {"type": "inline", "points": _points(demo_scene, 21), "origin": None})
    messages = []
    with client.websocket_connect(f"/api/jobs/{job_id}/ws") as ws:
        while True:
            message = ws.receive_json()
            messages.append(message)
            if message["status"] in FINAL:
                break
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1000
    assert messages[0]["id"] == job_id and messages[-1]["status"] == "done"
    assert messages[-1]["progress"]["fraction"] == 1.0 and messages[-1]["outputs"]
    fractions = [m["progress"]["fraction"] for m in messages]
    assert fractions == sorted(fractions)


def test_job_websocket_unknown_and_finished(client, data_dir):
    with client.websocket_connect("/api/jobs/ffffffffffffffff/ws") as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 4404
    _fake_job(data_dir, "failed")
    with client.websocket_connect("/api/jobs/0123456789abcdef/ws") as ws:
        assert ws.receive_json()["status"] == "failed"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


@pytest.mark.usefixtures("gl_context")
def test_cancel_running_job(client, data_dir, demo_scene):
    scene = _upload(client, demo_scene)["id"]
    job_id = _job(client, scene, {"type": "inline", "points": _line(60), "origin": None})
    running = _wait_progress(client, job_id)
    assert running["status"] == "running"

    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 202 and r.json()["id"] == job_id
    job = _wait(client, job_id, timeout=30)
    assert job["status"] == "cancelled" and job["error"] is None and job["finished_at"]
    assert job["progress"]["fraction"] < 1.0
    assert (data_dir / "jobs" / job_id / "output").is_dir()
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204  # now it is a finished job: deleted
    assert not (data_dir / "jobs" / job_id).exists()


@pytest.mark.usefixtures("gl_context")
def test_cancel_terminates_a_worker_that_ignores_the_flag(data_dir, monkeypatch, demo_scene):
    with _client(data_dir, monkeypatch, DVSIM_CANCEL_GRACE_S=0) as client:
        scene = _upload(client, demo_scene)["id"]
        job_id = _job(client, scene, {"type": "inline", "points": _line(60), "origin": None})
        assert client.get(f"/api/jobs/{job_id}").json()["status"] in ("queued", "running")
        _wait(client, job_id, until=("running",))
        assert client.delete(f"/api/jobs/{job_id}").status_code == 202  # may hit the worker at any point of its start
        job = _wait(client, job_id, timeout=30)
        assert job["status"] == "cancelled" and job["error"] is None
        assert client.app.state.jobs._active == {}


def test_failing_job_reports_a_short_error_and_the_server_survives(client, data_dir):
    r = client.post("/api/scenes", files={"obj": ("bad.obj", b"this is not a wavefront file\n\x00\x01\x02")})
    assert r.status_code == 201  # accepted: nothing suspicious in it, it just is not a model
    job_id = _job(client, r.json()["id"], {"type": "inline", "points": _line(0.1), "origin": None})
    job = _wait(client, job_id)
    assert job["status"] == "failed", job
    assert job["error"] and "Traceback" not in job["error"] and len(job["error"]) <= 500
    assert str(data_dir) not in job["error"]
    assert job["finished_at"]
    assert "Traceback" in (data_dir / "jobs" / job_id / "error.log").read_text()
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "failed"


@pytest.mark.usefixtures("gl_context")
def test_concurrency_limit_fifo_and_queue_limit(data_dir, monkeypatch, demo_scene):
    with _client(data_dir, monkeypatch, DVSIM_MAX_JOBS=1, DVSIM_MAX_PENDING=2) as client:
        scene = _upload(client, demo_scene)["id"]
        long_run = {"type": "inline", "points": _line(60), "origin": None}
        first = _job(client, scene, long_run)
        second = _job(client, scene, long_run)
        assert _wait_progress(client, first)["status"] == "running"
        assert client.get(f"/api/jobs/{second}").json()["status"] == "queued"
        third = client.post("/api/jobs", json={"scene_id": scene, "trajectory": long_run, "config": SMALL})
        assert third.status_code == 429  # 1 running + 1 queued = limit
        assert client.delete(f"/api/scenes/{scene}").status_code == 409  # in use
        time.sleep(1.0)
        assert client.get(f"/api/jobs/{second}").json()["status"] == "queued"  # still waiting for the slot

        assert client.delete(f"/api/jobs/{first}").status_code == 202
        assert _wait(client, first, timeout=30)["status"] == "cancelled"
        assert _wait(client, second, until=("running",), timeout=30)["status"] == "running"  # FIFO: next one starts
        assert client.delete(f"/api/jobs/{second}").status_code == 202
        assert _wait(client, second, timeout=30)["status"] == "cancelled"
        assert [j["id"] for j in client.get("/api/jobs").json()] == [second, first]  # newest first
        assert client.delete(f"/api/scenes/{scene}").status_code == 204


@pytest.mark.usefixtures("gl_context")
def test_server_shutdown_stops_the_workers(data_dir, demo_scene):
    app = create_app(data_dir)
    with TestClient(app) as client:
        scene = _upload(client, demo_scene)["id"]
        running = _job(client, scene, {"type": "inline", "points": _line(60), "origin": None})
        queued = _job(client, scene, {"type": "inline", "points": _line(60), "origin": None})
        _wait_progress(client, running)
        process = app.state.jobs._active[running].process
    assert not process.is_alive()
    for job_id in (running, queued):
        job = json.loads((data_dir / "jobs" / job_id / "job.json").read_text())
        assert job["status"] == "failed" and job["error"] == "server stopped"


@pytest.mark.usefixtures("gl_context")
def test_cancel_queued_job_never_runs(data_dir, monkeypatch, demo_scene):
    with _client(data_dir, monkeypatch, DVSIM_MAX_JOBS=1) as client:
        scene = _upload(client, demo_scene)["id"]
        long_run = {"type": "inline", "points": _line(60), "origin": None}
        first, second = _job(client, scene, long_run), _job(client, scene, long_run)
        r = client.delete(f"/api/jobs/{second}")
        assert r.status_code == 202 and r.json()["status"] == "cancelled"
        job = client.get(f"/api/jobs/{second}").json()
        assert job["status"] == "cancelled" and job["started_at"] is None
        client.delete(f"/api/jobs/{first}")
        assert _wait(client, first, timeout=30)["status"] == "cancelled"


# ---- files endpoint ---------------------------------------------------------------------------------------------
def test_download_and_path_traversal(client, data_dir):
    secret = "TOP-SECRET-MARKER"
    root = _fake_job(data_dir, files={"seq_000/a.txt": b"hello", "top.bin": b"\x00\x01"})
    (data_dir / "secret.txt").write_text(secret)
    (root / "secret.txt").write_text(secret)
    (root / "output" / "seq_000" / "dir with space.txt").write_bytes(b"spaces")
    try:
        os.symlink(root / "job.json", root / "output" / "link_to_job")
        os.symlink(data_dir / "secret.txt", root / "output" / "seq_000" / "link_to_secret")
        os.symlink(data_dir, root / "output" / "link_to_dir")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    base = "/api/jobs/0123456789abcdef/files/"

    ok = client.get(base + "seq_000/a.txt")
    assert ok.status_code == 200 and ok.content == b"hello"
    assert ok.headers["content-disposition"].startswith("attachment") and 'a.txt' in ok.headers["content-disposition"]
    assert client.get(base + "top.bin").content == b"\x00\x01"
    assert client.get(base + "seq_000/dir%20with%20space.txt").content == b"spaces"

    attacks = [
        "..%2fsecret.txt", "..%2f..%2fsecret.txt", "%2e%2e/secret.txt", "%2e%2e%2f%2e%2e%2fsecret.txt",
        "seq_000/..%2f..%2fsecret.txt", "seq_000/../../secret.txt", "seq_000/%2e%2e/%2e%2e/secret.txt",
        "..%5csecret.txt", "..%252fsecret.txt", "seq_000%5c..%5c..%5csecret.txt", "%2fetc%2fpasswd", "/etc/passwd",
        f"/{data_dir}/secret.txt".lstrip("/"), "seq_000/a.txt%00.png", "seq_000/", "seq_000", "", "seq_000//a.txt", "link_to_job", "link_to_dir/secret.txt", "seq_000/link_to_secret",
    ]
    for attack in attacks:
        r = client.get(base + attack)
        assert r.status_code in (400, 404), (attack, r.status_code)
        assert secret not in r.text and b'"status"' not in r.content, attack
    for job_id in ("..", "%2e%2e", "0123456789abcde", "0123456789ABCDEF", "0123456789abcdef%00", "..%2f"):
        r = client.get(f"/api/jobs/{job_id}/files/seq_000/a.txt")
        assert r.status_code in (400, 404), job_id
    # there is no directory listing
    assert client.get("/api/jobs/0123456789abcdef/files/").status_code in (400, 404, 405)
    assert client.get("/api/jobs/0123456789abcdef/files").status_code in (400, 404, 405)


@pytest.mark.parametrize("rel", ["", ".", "..", "../x", "a/../x", "a/./b", "a//b", "a/", "/a", "/etc/passwd", "a\\b", "a\x00b"])
def test_resolve_below_rejects_unsafe_paths(tmp_path, rel):
    from backend.api.util import UnsafePath, resolve_below

    with pytest.raises(UnsafePath):
        resolve_below(tmp_path, rel)
    assert resolve_below(tmp_path, "a/b.txt") == tmp_path / "a" / "b.txt"


def test_finished_job_outputs_skip_symlinks(data_dir):
    from backend.api.util import list_outputs

    root = _fake_job(data_dir)
    (data_dir / "secret.txt").write_text("x")
    try:
        os.symlink(data_dir / "secret.txt", root / "output" / "seq_000" / "l")
        os.symlink(data_dir, root / "output" / "d")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    assert [o["path"] for o in list_outputs(root / "output")] == ["seq_000/a.txt"]


# ---- restart recovery -------------------------------------------------------------------------------------------
def test_stale_jobs_are_failed_on_startup(data_dir):
    _fake_job(data_dir, "running", "aaaaaaaaaaaaaaaa")
    _fake_job(data_dir, "queued", "bbbbbbbbbbbbbbbb")
    _fake_job(data_dir, "done", "cccccccccccccccc")
    (data_dir / "jobs" / "not-a-job").mkdir()
    with TestClient(create_app(data_dir)) as client:
        jobs = {j["id"]: j for j in client.get("/api/jobs").json()}
        assert jobs["aaaaaaaaaaaaaaaa"]["status"] == "failed" and jobs["aaaaaaaaaaaaaaaa"]["error"] == "server restarted"
        assert jobs["bbbbbbbbbbbbbbbb"]["status"] == "failed" and jobs["bbbbbbbbbbbbbbbb"]["finished_at"]
        assert jobs["cccccccccccccccc"]["status"] == "done"
        assert jobs["aaaaaaaaaaaaaaaa"]["outputs"] == [
            {"name": "a.txt", "path": "seq_000/a.txt", "size_bytes": 5}
        ]
    # delete works on them, and a second start changes nothing
    with TestClient(create_app(data_dir)) as client:
        assert client.get("/api/jobs/aaaaaaaaaaaaaaaa").json()["error"] == "server restarted"
        assert client.delete("/api/jobs/aaaaaaaaaaaaaaaa").status_code == 204


# ---- headset websockets -----------------------------------------------------------------------------------------
def _sample(i, t0=5000.0):
    return {"timestamp": t0 + 10.0 * i, "x": 0.1 * i, "y": 1.5, "z": -1.0, "rx": 0.0, "ry": 0.0, "rz": -1.0}


def test_headset_recording_becomes_a_trajectory(client, data_dir):
    with client.websocket_connect("/spectate") as spectator, client.websocket_connect("/headset") as headset:
        headset.send_json({"action": "stream_data", "data": _sample(0, 1000.0)})  # not recording yet: only relayed
        assert spectator.receive_json() == _sample(0, 1000.0)
        headset.send_json({"action": "start_recording", "data": {"timestamp": 5000.0}})
        for i in range(30):
            headset.send_json({"action": "stream_data", "data": _sample(i)})
            assert spectator.receive_json() == _sample(i)
        headset.send_json({"action": "stop_recording", "data": {"timestamp": 5300.0}})
        headset.send_json({"action": "stream_data", "data": _sample(99)})  # after the stop: not recorded
        assert spectator.receive_json() == _sample(99)

    listed = client.get("/api/trajectories").json()
    assert len(listed) == 1 and listed[0]["source"] == "headset"
    assert listed[0]["num_points"] == 30 and listed[0]["duration_s"] == pytest.approx(0.29)
    points = client.get(f"/api/trajectories/{listed[0]['id']}").json()["points"]
    assert points[0]["timestamp"] == 0.0 and points[-1]["timestamp"] == pytest.approx(290.0)
    assert points[3]["x"] == pytest.approx(0.3) and set(points[0]) == {"timestamp", "x", "y", "z", "rx", "ry", "rz"}
    assert (data_dir / "trajectories" / listed[0]["id"] / "trajectory.json").is_file()


def test_headset_bad_input_does_not_kill_the_connection(client):
    with client.websocket_connect("/spectate") as spectator, client.websocket_connect("/headset") as headset:
        headset.send_text("this is not json")
        headset.send_text(json.dumps({"action": "stream_data"}))
        headset.send_text(json.dumps({"action": "stream_data", "data": {"timestamp": "yesterday"}}))
        headset.send_text(json.dumps([1, 2, 3]))
        headset.send_bytes(b"\x00\x01")
        headset.send_json({"action": "stop_recording", "data": {"timestamp": 1.0}})  # not recording: ignored
        headset.send_json({"action": "dance", "data": {"timestamp": 1.0}})  # unknown action
        headset.send_json({"action": "start_recording", "data": {"timestamp": 7.0}})
        headset.send_json({"action": "stop_recording", "data": {"timestamp": 7.0}})  # zero duration, no samples
        headset.send_json({"action": "start_recording", "data": {"timestamp": 8.0}})
        headset.send_json({"action": "stream_data", "data": {"timestamp": 8.0, "x": 1.0}})  # incomplete sample
        headset.send_json({"action": "stop_recording", "data": {"timestamp": 8.0}})
        headset.send_json({"action": "stream_data", "data": _sample(1)})
        got = spectator.receive_json()
        assert got["timestamp"] == 8.0 and got["x"] == 1.0 and got["y"] is None  # relayed as sent, in order
        assert spectator.receive_json() == _sample(1)
    assert client.get("/api/trajectories").json() == []  # nothing usable was recorded, nothing crashed


def test_headset_restart_without_stop_and_disconnect(client):
    with client.websocket_connect("/spectate") as spectator, client.websocket_connect("/headset") as headset:
        for start in (0.0, 1000.0):  # second start_recording saves the first recording
            headset.send_json({"action": "start_recording", "data": {"timestamp": start}})
            for i in range(5):
                headset.send_json({"action": "stream_data", "data": _sample(i, start)})
                spectator.receive_json()
        headset.send_json({"action": "stream_data", "data": _sample(9)})
        spectator.receive_json()
    assert [t["num_points"] for t in client.get("/api/trajectories").json()] == [5]  # the open one is dropped


def test_spectator_disconnect_does_not_break_the_relay(client):
    with client.websocket_connect("/headset") as headset:
        with client.websocket_connect("/spectate") as first:
            with client.websocket_connect("/spectate") as second:
                headset.send_json({"action": "stream_data", "data": _sample(0)})
                assert first.receive_json() == second.receive_json() == _sample(0)
            headset.send_json({"action": "stream_data", "data": _sample(1)})  # `second` is gone
            assert first.receive_json() == _sample(1)
        headset.send_json({"action": "stream_data", "data": _sample(2)})  # nobody is listening
        headset.send_json({"action": "stop_recording", "data": {"timestamp": 1.0}})


# ---- static frontend, CORS --------------------------------------------------------------------------------------
def test_frontend_is_served_without_shadowing_the_api(data_dir, monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>INDEX-PAGE</html>")
    (dist / "headset.html").write_text("<html>HEADSET-PAGE</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    with _client(data_dir, monkeypatch, DVSIM_FRONTEND_DIR=dist) as client:
        assert "INDEX-PAGE" in client.get("/").text
        assert "HEADSET-PAGE" in client.get("/headset.html").text
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/nope").status_code == 404
        assert client.get("/../secret").status_code in (400, 404)
        assert client.get("/%2e%2e/%2e%2e/etc/passwd").status_code in (400, 404)
        with client.websocket_connect("/headset") as headset, client.websocket_connect("/spectate") as spectator:
            headset.send_json({"action": "stream_data", "data": _sample(0)})
            assert spectator.receive_json() == _sample(0)


def test_without_a_frontend_build_root_is_a_json_hint(data_dir, monkeypatch, tmp_path):
    with _client(data_dir, monkeypatch, DVSIM_FRONTEND_DIR=tmp_path / "missing") as client:
        assert client.get("/").json()["docs"] == "/api/docs"


def test_cors_only_for_configured_origins(data_dir, monkeypatch):
    origin = "http://localhost:3000"
    with TestClient(create_app(data_dir)) as client:
        assert "access-control-allow-origin" not in client.get("/api/health", headers={"Origin": origin}).headers
    with _client(data_dir, monkeypatch, DVSIM_CORS_ORIGINS=f"{origin}, http://other:1/") as client:
        assert client.get("/api/health", headers={"Origin": origin}).headers["access-control-allow-origin"] == origin
        assert "access-control-allow-origin" not in client.get("/api/health", headers={"Origin": "http://evil"}).headers
        pre = client.options("/api/jobs", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == origin
        big = client.post("/api/scenes", files={"obj": ("a.obj", b"v 0 0 0\n")}, headers={"Origin": origin})
        assert big.headers["access-control-allow-origin"] == origin


def test_invalid_environment_is_reported(data_dir, monkeypatch):
    monkeypatch.setenv("DVSIM_MAX_JOBS", "many")
    with pytest.raises(ValueError, match="DVSIM_MAX_JOBS"):
        create_app(data_dir)


def test_generate_trajectories_preview(client, data_dir, demo_scene):
    from backend.pipeline import build_trajectories
    from backend.scene import load_scene

    scene = _upload(client, demo_scene)["id"]
    spec = {"type": "spline", "duration_s": 2.0, "n_key_points": 4, "count": 2, "seed": 3}
    r = client.post("/api/trajectories/generate", json={"scene_id": scene, "trajectory": spec})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seed"] == 3 and len(body["trajectories"]) == 2
    first = body["trajectories"][0]
    assert first["duration_s"] == 2.0 and set(first["points"][0]) == {"timestamp", "x", "y", "z", "rx", "ry", "rz"}
    # the same spec simulated by a job gives the same paths (what the preview promises)
    store_spec = client.app.state.scenes.spec(scene)
    expected = build_trajectories(spec, load_scene(store_spec, root=data_dir / "scenes" / scene))
    for got, exp in zip(body["trajectories"], expected):
        pos = np.array([[p["x"], p["y"], p["z"]] for p in got["points"]])
        assert np.allclose(pos, exp.positions, atol=1e-5)

    street = {"type": "street", "duration_s": 3.0, "min_clearance": 0.5, "edge_margin": 0}
    a = client.post("/api/trajectories/generate", json={"scene_id": scene, "trajectory": street}).json()
    assert isinstance(a["seed"], int) and a["trajectory"]["seed"] == a["seed"]       # a random seed, returned
    b = client.post("/api/trajectories/generate",
                    json={"scene_id": scene, "trajectory": {**street, "seed": a["seed"]}, "max_points": 5}).json()
    assert len(b["trajectories"][0]["points"]) == 5
    assert b["trajectories"][0]["points"][0] == a["trajectories"][0]["points"][0]
    assert b["trajectories"][0]["points"][-1] == a["trajectories"][0]["points"][-1]

    assert client.post("/api/trajectories/generate",
                       json={"scene_id": "0" * 16, "trajectory": spec}).status_code == 404
    r = client.post("/api/trajectories/generate", json={"scene_id": scene, "trajectory": {"type": "inline"}})
    assert r.status_code == 422
    r = client.post("/api/trajectories/generate",
                    json={"scene_id": scene, "trajectory": {**street, "min_clearance": 50.0}})
    assert r.status_code == 422 and "no walkable ground" in r.json()["detail"]


# ---- live rendering (/api/live) ---------------------------------------------------------------------------------
def _recv_until(ws, predicate, limit=300):
    """Messages of the live websocket until `predicate(message)`; text messages are decoded, frames are (seq, jpeg)."""
    seen = []
    for _ in range(limit):
        raw = ws.receive()
        if raw["type"] == "websocket.disconnect":
            raise AssertionError(f"closed before the expected message; saw {seen}")
        message = ((int.from_bytes(raw["bytes"][:4], "big"), raw["bytes"][4:]) if raw.get("bytes") is not None
                   else json.loads(raw["text"]))
        seen.append(message if isinstance(message, dict) else ("frame", message[0]))
        if predicate(message):
            return message, seen
    raise AssertionError(f"expected message not received; saw {seen}")


def _pose(seq, z=0.0):
    return {"type": "pose", "seq": seq, "x": 0.0, "y": 0.0, "z": z, "rx": 0.0, "ry": 0.0, "rz": -1.0}


def test_live_rejects_bad_starts(client, demo_scene):
    scene = _upload(client, demo_scene)["id"]
    for first, expected in (({"type": "start"}, "valid start message"),
                            ({"type": "start", "scene_id": scene, "view": "thermal"}, "valid start message"),
                            ({"type": "start", "scene_id": scene, "width": 1, "height": 1}, "valid start message"),
                            ({"type": "start", "scene_id": "0" * 16}, "unknown scene"),
                            ({"type": "start", "scene_id": "../../etc"}, "unknown scene")):
        with client.websocket_connect("/api/live") as ws:
            ws.send_json(first)
            message = ws.receive_json()
            assert message["type"] == "error" and expected in message["message"], (first, message)


@pytest.mark.usefixtures("gl_context")
def test_live_session_streams_exact_frames(client, demo_scene):
    import cv2

    scene = _upload(client, demo_scene)["id"]
    with client.websocket_connect("/api/live") as ws:
        ws.send_json({"type": "start", "scene_id": scene, "width": 48, "height": 36, "origin": "centre",
                      "view": "rgb", "fps": 30})
        assert ws.receive_json() == {"type": "loading"}
        ready, _ = _recv_until(ws, lambda m: isinstance(m, dict) and m.get("type") in ("ready", "error"))
        assert ready["type"] == "ready", ready
        assert (ready["width"], ready["height"]) == (48, 36) and len(ready["origin"]) == 3

        ws.send_json({"type": "pose", "seq": 7, "x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0})     # missing rz
        warning, _ = _recv_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "warning")
        assert "ignored" in warning["message"]
        ws.send_json({"type": "dance"})
        assert _recv_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "warning")[0]["type"] == "warning"

        ws.send_json(_pose(41))
        (seq, jpeg), _ = _recv_until(ws, lambda m: isinstance(m, tuple))
        assert seq == 41                                              # the frame says which pose it shows
        assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (36, 48, 3)

        ws.send_json({"type": "view", "view": "grid"})
        for k in range(30):                                           # a few frames later the grid arrives
            ws.send_json(_pose(100 + k, z=-0.05 * k))
            (seq, jpeg), _ = _recv_until(ws, lambda m: isinstance(m, tuple))
            if cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (72, 96, 3):
                break
        else:
            raise AssertionError("the view change was not applied")
        assert seq >= 100


@pytest.mark.usefixtures("gl_context")
def test_live_new_session_replaces_the_old_one(client, demo_scene):
    scene = _upload(client, demo_scene)["id"]
    start = {"type": "start", "scene_id": scene, "width": 32, "height": 24}
    with client.websocket_connect("/api/live") as first:
        first.send_json(start)
        _recv_until(first, lambda m: isinstance(m, dict) and m.get("type") == "ready")
        with client.websocket_connect("/api/live") as second:
            second.send_json(start)
            _recv_until(second, lambda m: isinstance(m, dict) and m.get("type") == "ready")
            message, _ = _recv_until(first, lambda m: isinstance(m, dict) and m.get("type") == "error")
            assert "replaced" in message["message"]
            second.send_json(_pose(1))
            assert _recv_until(second, lambda m: isinstance(m, tuple))[0][0] == 1


# ---- multi-file scenes: OBJ + MTL + several textures ---------------------------------------------------------------
def _image(color, fmt="PNG"):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format=fmt)
    return buffer.getvalue()


def _quad(x0, material, mtllib="scene.mtl", comment=""):
    """An OBJ with one textured quad of `material` at x0..x0+1 (and a texture-coordinate square)."""
    return (f"{comment}mtllib {mtllib}\nusemtl {material}\n"
            f"v {x0} 0 0\nv {x0 + 1} 0 0\nv {x0 + 1} 1 0\nv {x0} 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n").encode()


MTL = (b"newmtl red\nKd 1 1 1\nmap_Kd Textures\\red tex.png\n"
       b"newmtl blue\nKd 1 1 1\nmap_Kd -s 1 1 1 C:/somewhere/blue.jpg\nmap_Ks missing_specular.png\n")


def _bundle(**overrides):
    parts = {
        "obj": ("a.obj", _quad(0, "red") + b"usemtl blue\nv 5 0 0\nv 6 0 0\nv 6 1 0\nf 5/1 6/2 7/3\n"),
        "files": [("scene.mtl", MTL), ("Textures/red tex.png", _image((255, 0, 0))), ("blue.jpg", _image((0, 0, 255), "JPEG"))],
    }
    parts.update(overrides)
    return parts


def _post_bundle(client, parts, **form):
    files = [("obj", (parts["obj"][0], parts["obj"][1]))] + [("files", (n, c)) for n, c in parts["files"]]
    return client.post("/api/scenes", files=files, data={"name": "bundle", **form})


def test_bundle_upload_stores_generated_names_and_rewrites_references(client, data_dir):
    r = _post_bundle(client, _bundle())
    assert r.status_code == 201, r.text
    scene = r.json()
    stored = sorted(p.name for p in (data_dir / "scenes" / scene["id"]).iterdir())
    assert stored == ["material_0.mtl", "meta.json", "model_0.obj", "texture_0.png", "texture_1.jpg"]
    assert sorted(scene["files"]) == ["material_0.mtl", "model_0.obj", "texture_0.png", "texture_1.jpg"]
    folder = data_dir / "scenes" / scene["id"]
    assert (folder / "model_0.obj").read_text().startswith("mtllib material_0.mtl\n")
    mtl = (folder / "material_0.mtl").read_text()
    assert "map_Kd texture_0.png" in mtl and "map_Kd texture_1.jpg" in mtl
    assert "missing_specular" not in mtl and "somewhere" not in mtl and "tex.png" not in mtl


def test_bundle_loads_with_its_textures_inside_the_scene_directory(client, data_dir):
    from backend.scene import load_scene

    scene = _post_bundle(client, _bundle()).json()["id"]
    loaded = load_scene(client.app.state.scenes.spec(scene), root=data_dir / "scenes" / scene)
    colours = sorted(tuple(int(c) for c in obj.texture.reshape(-1, 3).mean(axis=0).round()) for obj in loaded.objects
                     if obj.texture is not None)
    assert len(loaded.objects) == 2
    assert colours[0][2] > 200 and colours[1][0] > 200           # one blue (JPEG) and one red (PNG) texture


def test_bundle_with_several_obj_files(client, data_dir):
    from backend.scene import load_scene

    other = MTL.replace(b"newmtl red", b"newmtl red2").replace(b"newmtl blue", b"newmtl blue2")
    parts = _bundle(files=[("scene.mtl", MTL), ("other.mtl", other),
                           ("Textures/red tex.png", _image((255, 0, 0))), ("blue.jpg", _image((0, 0, 255), "JPEG")),
                           ("b.obj", _quad(10, "red2", "other.mtl"))])
    scene = _post_bundle(client, parts).json()
    assert [f for f in scene["files"] if f.startswith("model_")] == ["model_0.obj", "model_1.obj"]
    loaded = load_scene(client.app.state.scenes.spec(scene["id"]), root=data_dir / "scenes" / scene["id"])
    assert loaded.bounds[1][0] >= 10.9 and len(loaded.objects) == 3


def _rejected(client, parts, expected, **form):
    r = _post_bundle(client, parts, **form)
    assert r.status_code in (413, 422) and expected in r.json()["detail"], (r.status_code, r.text)


def test_bundle_rejections(client, data_dir):
    good = _bundle()
    _rejected(client, _bundle(files=good["files"][1:]), "material file 'scene.mtl'")           # mtl not uploaded
    _rejected(client, _bundle(files=[good["files"][0], good["files"][2]]), "'red tex.png'")    # diffuse map missing
    _rejected(client, _bundle(obj=("a.obj", _quad(0, "red", "../../evil.mtl"))), "'evil.mtl'")  # traversal: no such upload
    _rejected(client, _bundle(files=good["files"] + [("run.exe", b"MZ")]), "unsupported file 'run.exe'")
    _rejected(client, _bundle(files=good["files"] + [("scene.mtl", MTL)]), "two uploaded files")
    _rejected(client, _bundle(files=good["files"] + [("Other/blue.jpg", b"x")]), "two uploaded files")
    _rejected(client, _bundle(files=good["files"][:2] + [("blue.jpg", b"not an image")]), "not a valid PNG or JPEG")
    _rejected(client, _bundle(files=[("scene.mtl", MTL)] + [(f"t{i}.png", _image((i, 0, 0))) for i in range(70)]), "too many files")
    r = _post_bundle(client, good)
    assert r.status_code == 201
    r = client.post("/api/scenes", files=[("obj", ("a.obj", good["obj"][1])), ("texture", ("t.png", _image((1, 2, 3)))),
                                          ("files", ("scene.mtl", MTL))], data={"name": "x"})
    assert r.status_code == 422 and "separate texture" in r.json()["detail"]
    # nothing half-written is left behind by the failures
    assert len(list((data_dir / "scenes").iterdir())) == 1


def test_bundle_drops_mtllib_hidden_in_a_comment(client, data_dir):
    parts = _bundle(obj=("a.obj", _quad(0, "red", comment="# mtllib ../../../etc/passwd\n")))
    scene = _post_bundle(client, parts).json()["id"]
    text = (data_dir / "scenes" / scene / "model_0.obj").read_text()
    assert text.count("mtllib") == 1 and text.startswith("#\nmtllib material_0.mtl")


def test_single_file_upload_still_refuses_material_libraries(client):
    r = client.post("/api/scenes", files={"obj": ("a.obj", _quad(0, "red"))}, data={"name": "x"})
    assert r.status_code == 422 and "material libraries" in r.json()["detail"]


@pytest.mark.usefixtures("gl_context")
def test_job_on_a_bundle_renders_its_textures(client):
    scene = _post_bundle(client, _bundle()).json()["id"]
    spec = {"type": "inline", "points": [
        {"timestamp": 0, "x": 0.5, "y": 0.5, "z": 2, "rx": 0, "ry": 0, "rz": -1},
        {"timestamp": 20, "x": 0.5, "y": 0.5, "z": 1.9, "rx": 0, "ry": 0, "rz": -1}]}
    job = _wait(client, _job(client, scene, spec, config={**SMALL, "gt": ["rgb"]}))
    assert job["status"] == "done", job


def test_live_origin_endpoint(client):
    assert client.get("/api/live/origin").json() is None
    r = client.put("/api/live/origin", json={"x": 1.5, "y": 2, "z": -3, "heading_deg": 45})
    assert r.status_code == 200 and r.json() == {"x": 1.5, "y": 2.0, "z": -3.0, "heading_deg": 45.0}
    assert client.get("/api/live/origin").json() == r.json()
    assert client.put("/api/live/origin", json={"x": 0, "y": 0, "z": 0}).json()["heading_deg"] == 0.0
    for bad in ({"x": "a", "y": 0, "z": 0}, {"x": 0, "y": 0}, {"x": 0, "y": 0, "z": 0, "heading_deg": 1000},
                {"x": 0, "y": 0, "z": 0, "extra": 1}):
        assert client.put("/api/live/origin", json=bad).status_code == 422
    assert client.delete("/api/live/origin").status_code == 204
    assert client.get("/api/live/origin").json() is None


@pytest.mark.usefixtures("gl_context")
def test_live_session_uses_and_follows_the_shared_origin(client, demo_scene):
    scene = _upload(client, demo_scene)["id"]
    start = {"type": "start", "scene_id": scene, "width": 32, "height": 24, "origin": "shared"}

    with client.websocket_connect("/api/live") as ws:                 # nothing picked yet: the floor origin
        ws.send_json(start)
        ready, _ = _recv_until(ws, lambda m: isinstance(m, dict) and m.get("type") in ("ready", "error"))
        assert ready["type"] == "ready" and ready["heading_deg"] == 0.0
        floor_origin = ready["origin"]

    client.put("/api/live/origin", json={"x": 0.25, "y": 1.0, "z": -0.5, "heading_deg": 30})
    with client.websocket_connect("/api/live") as ws:                 # picked before the start: used from the start
        ws.send_json(start)
        ready, _ = _recv_until(ws, lambda m: isinstance(m, dict) and m.get("type") in ("ready", "error"))
        assert ready["origin"] == [0.25, 1.0, -0.5] and ready["heading_deg"] == 30.0 and ready["origin"] != floor_origin
        ws.send_json(_pose(1))
        assert _recv_until(ws, lambda m: isinstance(m, tuple))[0][0] == 1

        client.put("/api/live/origin", json={"x": -1, "y": 0.5, "z": 0.1, "heading_deg": -90})   # recalibrate live
        moved, _ = _recv_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "origin")
        assert moved["origin"] == [-1.0, 0.5, 0.1] and moved["heading_deg"] == -90.0
        ws.send_json(_pose(2))
        assert _recv_until(ws, lambda m: isinstance(m, tuple))[0][0] == 2

    with client.websocket_connect("/api/live") as ws:                 # a session on a fixed origin ignores the pick
        ws.send_json({**start, "origin": "floor"})
        ready, _ = _recv_until(ws, lambda m: isinstance(m, dict) and m.get("type") in ("ready", "error"))
        assert ready["origin"] == floor_origin
        client.put("/api/live/origin", json={"x": 9, "y": 9, "z": 9})
        ws.send_json(_pose(3))
        _, seen = _recv_until(ws, lambda m: isinstance(m, tuple))
        assert not any(isinstance(m, dict) and m.get("type") == "origin" for m in seen)


# ---- human motion models (backend.motion) ---------------------------------------------------------------------------
STREET = {"type": "street", "duration_s": 3.0, "min_clearance": 0.5, "edge_margin": 0}


def _install_motion_model(data_dir, checkpoint, name="tiny"):
    folder = data_dir / "motion"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.pt").write_bytes(Path(checkpoint).read_bytes())


def test_motion_models_are_listed(client, data_dir, motion_checkpoint):
    assert client.get("/api/motion-models").json() == []
    _install_motion_model(data_dir, motion_checkpoint)
    (data_dir / "motion" / "readme.txt").write_text("not a model")
    (listed,) = client.get("/api/motion-models").json()
    assert listed["id"] == "tiny" and listed["size_bytes"] == Path(motion_checkpoint).stat().st_size


def test_street_preview_with_a_motion_model(client, data_dir, demo_scene, motion_checkpoint):
    _install_motion_model(data_dir, motion_checkpoint)
    scene = _upload(client, demo_scene)["id"]
    body = {"scene_id": scene, "trajectory": {**STREET, "seed": 5, "motion": {"model": "tiny"}}}
    r = client.post("/api/trajectories/generate", json=body)
    assert r.status_code == 200, r.text
    first = r.json()
    assert first["trajectory"]["motion"] == {"model": "tiny"} and first["seed"] == 5      # the id, never a server path
    assert len(first["trajectories"][0]["points"]) == 151                                 # 3 s at 50 Hz
    again = client.post("/api/trajectories/generate", json=body).json()
    assert again["trajectories"] == first["trajectories"]                                 # same seed, same walk
    plain = client.post("/api/trajectories/generate", json={**body, "trajectory": {**STREET, "seed": 5}}).json()
    assert len(plain["trajectories"][0]["points"]) < 151                                  # the planner's own waypoints


def test_motion_model_errors(client, data_dir, demo_scene, motion_checkpoint, monkeypatch):
    scene = _upload(client, demo_scene)["id"]

    def generate(motion):
        return client.post("/api/trajectories/generate",
                           json={"scene_id": scene, "trajectory": {**STREET, "motion": motion}})

    r = generate({"model": "missing"})
    assert r.status_code == 404 and "'missing' not found" in r.json()["detail"]
    for bad in ({"model": "../etc/passwd"}, {"model": ""}, {"path": "/etc/passwd"}, {}, {"model": "a", "extra": 1}):
        assert generate(bad).status_code == 422
    r = client.post("/api/jobs", json={"scene_id": scene, "trajectory": {**STREET, "motion": {"model": "missing"}}})
    assert r.status_code == 404 and client.get("/api/jobs").json() == []                  # nothing was queued
    _install_motion_model(data_dir, motion_checkpoint)
    monkeypatch.setattr("backend.api.planner._torch_available", lambda: False)
    for r in (generate({"model": "tiny"}),
              client.post("/api/jobs", json={"scene_id": scene, "trajectory": {**STREET, "motion": {"model": "tiny"}}})):
        assert r.status_code == 501 and "pip install -e .[torch]" in r.json()["detail"]
    assert client.post("/api/trajectories/generate",
                       json={"scene_id": scene, "trajectory": {"type": "spline", "duration_s": 1, "motion": {"model": "tiny"}}}
                       ).status_code == 422                                              # only street walks take motion


def test_a_corrupt_motion_model_is_a_clean_error(client, data_dir, demo_scene):
    (data_dir / "motion").mkdir(parents=True)
    (data_dir / "motion" / "broken.pt").write_bytes(b"not a checkpoint")
    scene = _upload(client, demo_scene)["id"]
    r = client.post("/api/trajectories/generate",
                    json={"scene_id": scene, "trajectory": {**STREET, "motion": {"model": "broken"}}})
    assert r.status_code == 422 and "cannot read this file" in r.json()["detail"]


def test_job_with_a_motion_model(client, data_dir, demo_scene, motion_checkpoint):
    _install_motion_model(data_dir, motion_checkpoint)
    scene = _upload(client, demo_scene)["id"]
    job_id = _job(client, scene, {**STREET, "duration_s": 1.0, "seed": 2, "motion": {"model": "tiny"}})
    job = _wait(client, job_id)
    assert job["status"] == "done", job
    assert job["trajectory"]["motion"] == {"model": "tiny"}                               # the job JSON keeps the id
    request = json.loads((data_dir / "jobs" / job_id / "request.json").read_text())
    assert request["trajectory"]["motion"] == {"path": str(data_dir / "motion" / "tiny.pt")}   # the worker gets the file
    assert job["progress"]["total_steps"] == 101                                          # 1 s at 10 ms


def test_a_long_motion_preview_is_refused(client, demo_scene):
    scene = _upload(client, demo_scene)["id"]
    for duration, count in ((301.0, 1), (100.0, 4)):
        r = client.post("/api/trajectories/generate", json={
            "scene_id": scene, "trajectory": {**STREET, "duration_s": duration, "count": count, "motion": {"model": "any"}}})
        assert r.status_code == 422 and "limited to 300 s in total" in r.json()["detail"]
