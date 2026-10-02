"""Procedural demo scene (backend.samples): closed room, GLB objects, textures, walk and determinism."""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.transformations import euler_matrix, translation_matrix

from backend.samples import ROOM_MAX, ROOM_MIN, create_demo_scene, make_trajectory_records
from backend.samples.demo import FILES, PX_PER_M, TEX_SIZE

REPO = Path(__file__).resolve().parents[1]
KEYS = ("timestamp", "x", "y", "z", "rx", "ry", "rz")
MIN_CLEARANCE = 0.4


def _load_obj_objects(path):
    """{object name: Trimesh} for a multi-object OBJ (`o` lines split).

    Built from our own parser: trimesh's `split_objects=True` behaves differently across versions
    (4.6.x returns only the last object)."""
    out = {}
    for name, o in _parse_obj(path).items():
        mesh = trimesh.Trimesh(vertices=o["v"], faces=o["f"][:, :, 0], process=False)
        mesh.remove_unreferenced_vertices()
        out[name] = mesh
    return out


def _closed(mesh):
    """Copy with vertices merged by position only (UV / normal seams collapsed)."""
    m = mesh.copy()
    m.merge_vertices(merge_tex=True, merge_norm=True)
    return m


def _spec_matrix(obj):
    scale = obj.get("scale", 1.0)
    scale = np.full(3, scale) if np.isscalar(scale) else np.asarray(scale, float)
    s = np.eye(4)
    s[:3, :3] = np.diag(scale)
    r = euler_matrix(*np.radians(obj.get("rotation_euler_deg", (0, 0, 0))), axes="sxyz")
    return translation_matrix(obj.get("position", (0, 0, 0))) @ r @ s


def _world_boxes(scene):
    """(room shell bounds (2,3), [(name, lo, hi)] AABBs of every other geometry) from the files and demo_scene.json."""
    spec = json.loads(scene.scene_spec.read_text())
    shell, boxes = None, []
    for obj in spec["objects"]:
        path = scene.root / obj["path"]
        if path.suffix.lower() == ".obj":
            items = [(name, mesh, np.eye(4)) for name, mesh in _load_obj_objects(path).items()]
        else:
            sc = trimesh.load(str(path), force="scene")
            items = []
            for node in sc.graph.nodes_geometry:
                transform, name = sc.graph[node]
                items.append((name, sc.geometry[name], transform))
        for name, geom, transform in items:
            m = geom.copy()
            m.apply_transform(_spec_matrix(obj) @ transform)
            if name == "room_shell":
                shell = m.bounds.copy()
            else:
                boxes.append((f"{Path(obj['path']).stem}:{name}", m.bounds[0].copy(), m.bounds[1].copy()))
    return shell, boxes


def _point_box_distance(p, lo, hi):
    return np.linalg.norm(np.maximum(np.maximum(lo - p, 0.0), p - hi), axis=-1)


def _import_or_skip(module):
    """Import a module owned by another agent; skip (not fail) while it is missing or half-written."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        pytest.skip(f"{module} not importable yet: {exc}")


def _records(scene):
    return json.loads(scene.trajectory_json.read_text())


def _array(records):
    return np.array([[r[k] for k in KEYS] for r in records], np.float64)


def _parse_obj(path):
    """Raw OBJ parse -> {object: dict(v, vt, vn, f)} with per-object 0-based local indices."""
    v, vt, vn = [], [], []
    objs, cur = {}, None
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if not tok or tok[0].startswith("#"):
            continue
        if tok[0] == "o":
            cur = objs.setdefault(tok[1], {"faces": []})
        elif tok[0] == "v":
            v.append([float(a) for a in tok[1:4]])
        elif tok[0] == "vt":
            vt.append([float(a) for a in tok[1:3]])
        elif tok[0] == "vn":
            vn.append([float(a) for a in tok[1:4]])
        elif tok[0] == "f":
            cur["faces"].append([[int(i) - 1 for i in c.split("/")] for c in tok[1:]])
    v, vt, vn = np.array(v), np.array(vt), np.array(vn)
    return {name: {"v": v, "vt": vt, "vn": vn, "f": np.array(o["faces"])} for name, o in objs.items()}


# ---------------------------------------------------------------------------------------------------- files


def test_files_exist_and_are_small(demo_scene):
    for name in FILES:
        assert (demo_scene.root / name).is_file(), name
    assert demo_scene.room_obj.name == "demo_room.obj" and demo_scene.trajectory_json.name == "demo_trajectory.json"
    assert sum(p.stat().st_size for p in demo_scene.root.iterdir()) < 10 * 1024 * 1024


# ---------------------------------------------------------------------------------------------------- room


def test_room_shell_is_watertight_with_expected_bbox(demo_scene):
    objs = _load_obj_objects(demo_scene.room_obj)
    assert set(objs) == {"room_shell", "table", "pillar_0", "pillar_1"}
    shell = _closed(objs["room_shell"])
    assert shell.is_watertight and shell.is_winding_consistent
    np.testing.assert_allclose(shell.bounds, [ROOM_MIN, ROOM_MAX], atol=1e-6)
    np.testing.assert_allclose(shell.bounds[1] - shell.bounds[0], [6.0, 3.0, 5.0], atol=1e-6)
    assert shell.volume == pytest.approx(-90.0, abs=1e-6)     # negative: normals point into the room
    # the whole file is exactly the room (sub-volumes are inside it)
    whole = trimesh.load(str(demo_scene.room_obj), force="mesh")
    np.testing.assert_allclose(whole.bounds, [ROOM_MIN, ROOM_MAX], atol=1e-6)


def test_room_normals_point_inwards_and_agree_with_winding(demo_scene):
    parsed = _parse_obj(demo_scene.room_obj)
    centre = (np.asarray(ROOM_MIN) + np.asarray(ROOM_MAX)) / 2
    for name, o in parsed.items():
        idx = o["f"][:, :, 0]
        tri = o["v"][idx]
        geo = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm = o["vn"][o["f"][:, :, 2]].sum(axis=1)
        assert (np.einsum("ij,ij->i", geo, nrm) > 0).all(), f"{name}: winding disagrees with vn"
        if name == "room_shell":
            assert (np.einsum("ij,ij->i", tri.mean(axis=1) - centre, nrm) < 0).all()      # inward-facing
        else:
            own = o["v"][np.unique(idx)].mean(axis=0)
            assert (np.einsum("ij,ij->i", tri.mean(axis=1) - own, nrm) > 0).all(), name   # outward-facing


def test_room_subvolumes_are_closed_outward_and_inside(demo_scene):
    objs = _load_obj_objects(demo_scene.room_obj)
    lo, hi = np.asarray(ROOM_MIN), np.asarray(ROOM_MAX)
    for name in ("table", "pillar_0", "pillar_1"):
        m = _closed(objs[name])
        assert m.is_watertight and m.is_winding_consistent and m.volume > 0.1, name
        assert (m.bounds[0] > lo).all() and (m.bounds[1] < hi).all(), name     # strictly inside: no shell overlap


def test_room_uv_range_and_uniform_texel_density(demo_scene):
    parsed = _parse_obj(demo_scene.room_obj)
    for name, o in parsed.items():
        assert o["vt"].min() >= 0.0 and o["vt"].max() <= 1.0, name
        tri = o["v"][o["f"][:, :, 0]]
        world = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        uv = o["vt"][o["f"][:, :, 1]]
        e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        texels = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) * TEX_SIZE ** 2
        np.testing.assert_allclose(texels / world, PX_PER_M ** 2, rtol=0.02, err_msg=name)


def test_room_texture_has_contrast_at_several_scales(demo_scene):
    img = Image.open(demo_scene.room_texture)
    assert img.size == (TEX_SIZE, TEX_SIZE) and img.mode == "RGB"
    rgb = np.asarray(img, np.float32) / 255.0
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    assert lum.std() > 0.2
    assert (lum < 0.06).mean() > 0.02 and (lum > 0.93).mean() > 0.02      # near-black and near-white regions
    for scale, min_mean_grad, min_edge_frac in ((1, 0.008, 0.02), (4, 0.03, 0.10), (16, 0.05, 0.25)):
        n = TEX_SIZE // scale
        ds = lum.reshape(n, scale, n, scale).mean(axis=(1, 3))
        grad = np.hypot(np.diff(ds, axis=1)[:-1], np.diff(ds, axis=0)[:, :-1])
        assert grad.mean() > min_mean_grad, scale
        assert (grad > 0.05).mean() > min_edge_frac, scale


# ---------------------------------------------------------------------------------------------------- glb


def test_glb_reloads_with_expected_geometries_and_textures(demo_scene):
    scene = trimesh.load(str(demo_scene.objects_glb), force="scene")
    names = {"floor", "box", "sphere", "cylinder", "cone"}
    assert set(scene.geometry) == names and len(scene.graph.nodes_geometry) == 5
    hashes, colors, centres = set(), set(), {}
    for name, mesh in scene.geometry.items():
        assert mesh.visual.kind == "texture", name
        mat = mesh.visual.material
        assert mat.baseColorTexture is not None and min(mat.baseColorTexture.size) >= 256, name
        hashes.add(hashlib.md5(np.asarray(mat.baseColorTexture).tobytes()).hexdigest())
        colors.add(tuple(int(c) for c in mat.baseColorFactor))
        assert mesh.visual.uv.min() >= 0.0 and mesh.visual.uv.max() <= 1.0, name
        closed = _closed(mesh)
        assert closed.is_watertight and closed.volume > 0, name
    assert len(hashes) == 5 and len(colors) == 5                              # different texture and colour each
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        centres[name] = transform[:3, 3]
    prims = [centres[n] for n in ("box", "sphere", "cylinder", "cone")]
    assert min(np.linalg.norm(a - b) for i, a in enumerate(prims) for b in prims[i + 1:]) > 0.5   # distinct positions
    assert scene.bounds[0][1] == pytest.approx(-0.05, abs=1e-4) and scene.bounds[1][1] > 1.0    # rug top at y = 0


def test_scene_spec_references_room_and_glb_objects(demo_scene):
    spec = json.loads(demo_scene.scene_spec.read_text())
    objs = spec["objects"]
    assert [o["path"] for o in objs] == ["demo_room.obj", "demo_objects.glb", "demo_objects.glb"]
    assert objs[0]["texture"] == "demo_room.png"
    for o in objs:
        assert (demo_scene.root / o["path"]).is_file()
        assert len(o["position"]) == 3 and len(o["rotation_euler_deg"]) == 3 and o["scale"] is not None
    assert objs[2]["scale"] != objs[1]["scale"] and objs[2]["rotation_euler_deg"] != objs[1]["rotation_euler_deg"]


# ---------------------------------------------------------------------------------------------------- trajectory


def test_trajectory_schema_timestamps_and_rate(demo_scene):
    recs = _records(demo_scene)
    assert isinstance(recs, list) and len(recs) == 401
    assert all(tuple(r) == KEYS for r in recs)
    a = _array(recs)
    assert np.isfinite(a).all()
    dt = np.diff(a[:, 0])
    assert (dt > 0).all()                                            # monotone (milliseconds)
    np.testing.assert_allclose(dt, 10.0, atol=1e-6)                  # 100 Hz
    assert a[0, 0] == 0.0 and a[-1, 0] == pytest.approx(4000.0)
    np.testing.assert_allclose(np.linalg.norm(a[:, 4:7], axis=1), 1.0, atol=1e-5)   # forward vector is unit
    assert _records(demo_scene) == make_trajectory_records()          # file == generator output


def test_trajectory_is_a_gentle_walk_at_eye_height(demo_scene):
    a = _array(_records(demo_scene))
    p, f, dt = a[:, 1:4], a[:, 4:7], 0.01
    ground_speed = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) / dt
    assert 0.7 < ground_speed.mean() < 0.9 and ground_speed.max() < 1.0 and ground_speed.min() > 0.5
    assert 1.45 < p[:, 1].mean() < 1.55
    assert 0.005 < np.ptp(p[:, 1]) < 0.06                            # slight vertical bobbing
    yaw = np.degrees(np.unwrap(np.arctan2(f[:, 2], f[:, 0])))
    assert 10.0 < np.ptp(yaw) < 90.0                                 # some yaw change, but gentle
    assert np.abs(np.diff(yaw)).max() / dt < 90.0                    # deg/s
    assert np.abs(np.diff(p, n=2, axis=0)).max() < 1e-3              # smooth path (no jumps)


def test_trajectory_keeps_clearance_from_walls_and_objects(demo_scene):
    p = _array(_records(demo_scene))[:, 1:4]
    shell, boxes = _world_boxes(demo_scene)
    np.testing.assert_allclose(shell, [ROOM_MIN, ROOM_MAX], atol=1e-6)
    assert len(boxes) == 3 + 2 * 5                                   # table, 2 pillars, 2 GLB instances x 5 geoms
    assert (p > shell[0]).all() and (p < shell[1]).all()
    wall_clearance = np.minimum(p - shell[0], shell[1] - p)          # distance to each of the six wall planes
    assert wall_clearance.min() >= MIN_CLEARANCE
    for name, lo, hi in boxes:
        assert _point_box_distance(p, lo, hi).min() >= MIN_CLEARANCE, name
        if not name.endswith(":floor"):                              # never flies over furniture / primitives either
            assert _point_box_distance(p[:, [0, 2]], lo[[0, 2]], hi[[0, 2]]).min() >= MIN_CLEARANCE, name


def test_trajectory_loads_with_waypoint_trajectory(demo_scene):
    traj_mod = _import_or_skip("backend.trajectories")
    if not hasattr(traj_mod, "WaypointTrajectory"):
        pytest.skip("backend.trajectories.WaypointTrajectory not available yet")
    traj = traj_mod.WaypointTrajectory.from_json(str(demo_scene.trajectory_json))
    assert traj.start_us == 0 and traj.end_us == 4_000_000
    a = _array(_records(demo_scene))
    np.testing.assert_allclose(traj.pose_at(0).position, a[0, 1:4], atol=1e-6)
    np.testing.assert_allclose(traj.pose_at(1_230_000).position, a[123, 1:4], atol=1e-6)
    np.testing.assert_allclose(traj.pose_at(2_000_000).forward, a[200, 4:7], atol=1e-4)


def test_scene_spec_loads_with_load_scene(demo_scene):
    scene_mod = _import_or_skip("backend.scene")
    scene = scene_mod.load_scene(str(demo_scene.scene_spec))
    assert len(scene.objects) >= 3
    np.testing.assert_allclose(scene.bounds[0][[0, 2]], [ROOM_MIN[0], ROOM_MIN[2]], atol=1e-3)
    np.testing.assert_allclose(scene.bounds[1][[0, 2]], [ROOM_MAX[0], ROOM_MAX[2]], atol=1e-3)


# ---------------------------------------------------------------------------------------------------- determinism


def _digest(root):
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in FILES}


def test_two_runs_are_byte_identical(demo_scene, tmp_path):
    fresh = create_demo_scene(tmp_path / "fresh")
    assert fresh.root != demo_scene.root
    assert _digest(fresh.root) == _digest(demo_scene.root)               # OBJ, PNG, GLB and both JSON files


def test_rerun_is_idempotent(tmp_path):
    first = create_demo_scene(tmp_path / "d")
    digest, mtimes = _digest(first.root), {n: (first.root / n).stat().st_mtime_ns for n in FILES}
    again = create_demo_scene(first.root)                                # up-to-date: reused, nothing rewritten
    assert again == first
    assert {n: (first.root / n).stat().st_mtime_ns for n in FILES} == mtimes
    create_demo_scene(first.root, force=True)                            # forced: rewritten with identical bytes
    assert _digest(first.root) == digest


def test_cli_writes_the_demo_files(demo_scene):
    res = subprocess.run([sys.executable, "-m", "backend.samples", str(demo_scene.root)], cwd=REPO,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    for name in FILES:
        assert name in res.stdout
