"""Scene loading (no GL needed): OBJ/MTL, glTF/GLB, PLY, SceneSpec transforms, texture handling."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from backend.scene import DEFAULT_BASE_COLOR, Scene, SceneObject, load_scene, object_matrix
from backend.scene.textures import srgb_decode, srgb_encode

QUAD_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
QUAD_F = np.array([[0, 1, 2], [0, 2, 3]])
QUAD_UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)  # v = 0 at the bottom


def asym_image(w: int = 64, h: int = 32) -> np.ndarray:
    img = np.full((h, w, 3), 40, np.uint8)
    img[: h // 4, : w // 4] = (255, 255, 255)   # top-left white
    img[-h // 4:, -w // 4:] = (255, 0, 0)       # bottom-right red
    return img


def write_png(path, array) -> None:
    Image.fromarray(array).save(path)


def write_obj(path, vertices, faces, uv=None, mtl=None, usemtl=None) -> None:
    lines = [f"mtllib {mtl}"] if mtl else []
    lines += ["v %f %f %f" % tuple(v) for v in vertices]
    if uv is not None:
        lines += ["vt %f %f" % tuple(t) for t in uv]
    if usemtl:
        lines.append(f"usemtl {usemtl}")
    for f in faces:
        lines.append("f " + " ".join((f"{i + 1}/{i + 1}" if uv is not None else f"{i + 1}") for i in f))
    path.write_text("\n".join(lines) + "\n")


def test_obj_with_mtl_texture_and_untextured_material(tmp_path):
    write_png(tmp_path / "tex.png", asym_image())
    (tmp_path / "m.mtl").write_text("newmtl textured\nKd 1 1 1\nmap_Kd tex.png\nnewmtl red\nKd 1 0 0\n")
    lines = ["mtllib m.mtl"] + ["v %f %f %f" % tuple(v) for v in QUAD_V] + ["v 5 0 0", "v 6 0 0", "v 6 1 0"]
    lines += ["vt %f %f" % tuple(t) for t in QUAD_UV] + ["usemtl textured", "f 1/1 2/2 3/3", "f 1/1 3/3 4/4"]
    lines += ["usemtl red", "f 5 6 7"]
    (tmp_path / "a.obj").write_text("\n".join(lines) + "\n")

    scene = load_scene(tmp_path / "a.obj")
    assert len(scene.objects) == 2
    textured = next(o for o in scene.objects if o.texture is not None)
    plain = next(o for o in scene.objects if o.texture is None)
    assert textured.texture.shape == (32, 64, 3) and textured.texture.dtype == np.uint8
    assert np.array_equal(textured.texture, asym_image())  # stored as the plain image, row 0 = top
    assert textured.uv.shape == (len(textured.vertices), 2) and len(textured.faces) == 2
    np.testing.assert_allclose(plain.base_color, [1, 0, 0, 1])
    assert plain.uv is None and textured.vertices.dtype == np.float32 and textured.faces.dtype == np.uint32
    assert scene.num_triangles == 3


def test_obj_without_mtl_uses_default_base_color(tmp_path):
    write_obj(tmp_path / "plain.obj", QUAD_V, QUAD_F)
    scene = load_scene(str(tmp_path / "plain.obj"))
    (obj,) = scene.objects
    assert obj.texture is None and obj.uv is None and obj.colors is None
    np.testing.assert_allclose(obj.base_color, DEFAULT_BASE_COLOR)
    np.testing.assert_allclose(obj.model_matrix, np.eye(4))


def test_legacy_obj_plus_texture_override(tmp_path):
    write_obj(tmp_path / "a.obj", QUAD_V, QUAD_F, uv=QUAD_UV)  # UVs but no mtl
    write_png(tmp_path / "a.png", asym_image())
    scene = load_scene({"objects": [{"path": str(tmp_path / "a.obj"), "texture": str(tmp_path / "a.png")}]})
    assert np.array_equal(scene.objects[0].texture, asym_image())
    # a bare object entry is accepted too
    scene2 = load_scene({"path": str(tmp_path / "a.obj"), "texture": str(tmp_path / "a.png")})
    assert scene2.objects[0].texture is not None


def test_texture_override_without_uv_raises(tmp_path):
    write_obj(tmp_path / "nouv.obj", QUAD_V, QUAD_F)
    write_png(tmp_path / "t.png", asym_image())
    with pytest.raises(ValueError, match="UV"):
        load_scene({"objects": [{"path": str(tmp_path / "nouv.obj"), "texture": str(tmp_path / "t.png")}]})


def test_default_texture_only_fills_objects_without_one(tmp_path):
    write_png(tmp_path / "own.png", np.full((8, 8, 3), 200, np.uint8))
    write_png(tmp_path / "default.png", np.full((4, 4, 3), 9, np.uint8))
    (tmp_path / "m.mtl").write_text("newmtl m\nmap_Kd own.png\n")
    write_obj(tmp_path / "a.obj", QUAD_V, QUAD_F, uv=QUAD_UV, mtl="m.mtl", usemtl="m")
    write_obj(tmp_path / "b.obj", QUAD_V, QUAD_F, uv=QUAD_UV)
    scene = load_scene({
        "texture": str(tmp_path / "default.png"),
        "objects": [{"path": str(tmp_path / "a.obj")}, {"path": str(tmp_path / "b.obj")}],
    })
    assert scene.objects[0].texture.shape[0] == 8 and scene.objects[1].texture.shape[0] == 4


def test_texture_is_capped_with_lanczos(tmp_path):
    write_png(tmp_path / "big.png", np.zeros((100, 300, 3), np.uint8))
    write_obj(tmp_path / "a.obj", QUAD_V, QUAD_F, uv=QUAD_UV)
    scene = load_scene({"objects": [{"path": str(tmp_path / "a.obj"), "texture": str(tmp_path / "big.png")}]},
                       max_texture_size=64)
    assert scene.objects[0].texture.shape == (21, 64, 3)
    scene = load_scene({"objects": [{"path": str(tmp_path / "a.obj"), "texture": str(tmp_path / "big.png")}]})
    assert scene.objects[0].texture.shape == (100, 300, 3)


def test_shared_texture_file_is_loaded_once(tmp_path):
    write_png(tmp_path / "t.png", asym_image())
    write_obj(tmp_path / "a.obj", QUAD_V, QUAD_F, uv=QUAD_UV)
    entry = {"path": str(tmp_path / "a.obj"), "texture": str(tmp_path / "t.png")}
    scene = load_scene({"objects": [entry, dict(entry, position=[2, 0, 0])]})
    assert scene.objects[0].texture is scene.objects[1].texture
    assert scene.objects[0].vertices is scene.objects[1].vertices  # geometry shared, only the model matrix differs


def test_object_matrix_is_threejs_xyz_euler_order():
    # three.js 'XYZ': R = Rx @ Ry @ Rz -> the vector is rotated about Z first, then Y, then X
    m = object_matrix(rotation_euler_deg=[90, 90, 0])
    np.testing.assert_allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    m = object_matrix(rotation_euler_deg=[0, 0, 90])
    np.testing.assert_allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    m = object_matrix(position=[1, 2, 3], scale=[2, 3, 4])
    np.testing.assert_allclose(m @ [1, 1, 1, 1], [3, 5, 7, 1])


def test_spec_transform_bounds_and_diagonal(tmp_path):
    cube = trimesh.creation.box(extents=(1, 1, 1))
    cube.apply_translation([0.5, 0.5, 0.5])  # [0,1]^3
    cube.export(tmp_path / "cube.obj")
    scene = load_scene({"objects": [
        {"path": str(tmp_path / "cube.obj"), "position": [10, 0, 0], "rotation_euler_deg": [0, 0, 90], "scale": 2.0},
        {"path": str(tmp_path / "cube.obj"), "position": [0, 0, -5], "scale": [1, 2, 3]},
    ]})
    # cube 1: scale 2 -> [0,2]^3, rotate 90 deg about Z (x,y) -> (-y,x): x in [-2,0], y in [0,2]; + (10,0,0)
    # cube 2: x [0,1], y [0,2], z [-5,-2]
    np.testing.assert_allclose(scene.bounds, [[0, 0, -5], [10, 2, 2]], atol=1e-5)
    assert scene.diagonal == pytest.approx(np.linalg.norm([10, 2, 7]), rel=1e-6)
    np.testing.assert_allclose(scene.objects[0].world_vertices().min(axis=0), [8, 0, 0], atol=1e-5)


def test_to_trimesh_is_merged_world_space(tmp_path):
    write_obj(tmp_path / "a.obj", QUAD_V, QUAD_F)
    scene = load_scene({"objects": [{"path": str(tmp_path / "a.obj")},
                                    {"path": str(tmp_path / "a.obj"), "position": [0, 0, -3]}]})
    mesh = scene.to_trimesh()
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.faces.shape == (4, 3) and mesh.vertices.shape == (8, 3)
    np.testing.assert_allclose(mesh.bounds, scene.bounds)
    assert mesh.area == pytest.approx(2.0)


def test_json_spec_resolves_relative_to_the_file(tmp_path):
    sub = tmp_path / "assets"
    sub.mkdir()
    write_png(sub / "t.png", asym_image())
    write_obj(sub / "a.obj", QUAD_V, QUAD_F, uv=QUAD_UV)
    (sub / "scene.json").write_text(json.dumps({"objects": [{"path": "a.obj", "texture": "t.png", "position": [1, 2, 3]}]}))
    scene = load_scene(sub / "scene.json")
    np.testing.assert_allclose(scene.objects[0].model_matrix[:3, 3], [1, 2, 3])
    assert scene.objects[0].texture is not None


def test_glb_scene_graph_transforms_textures_and_factors(tmp_path):
    tex = asym_image()
    textured = trimesh.Trimesh(
        QUAD_V, QUAD_F, process=False,
        visual=TextureVisuals(uv=QUAD_UV, material=PBRMaterial(baseColorTexture=Image.fromarray(tex))),
    )
    tinted = trimesh.Trimesh(QUAD_V, QUAD_F, process=False,
                             visual=TextureVisuals(material=PBRMaterial(baseColorFactor=[0, 128, 255, 255])))
    scene = trimesh.Scene()
    scene.add_geometry(textured, node_name="a", geom_name="ga",
                       transform=trimesh.transformations.translation_matrix([2, 0, 0]))
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0], point=[0, 0, 0])
    rot[:3, 3] = [0, 5, 0]
    scene.add_geometry(tinted, node_name="b", geom_name="gb", transform=rot)
    scene.export(tmp_path / "s.glb")

    loaded = load_scene(tmp_path / "s.glb")
    assert len(loaded.objects) == 2
    a = next(o for o in loaded.objects if o.texture is not None)
    b = next(o for o in loaded.objects if o.texture is None)
    np.testing.assert_allclose(a.vertices.min(axis=0), [0, 0, 0], atol=1e-6)      # mesh-local vertices ...
    np.testing.assert_allclose(a.model_matrix[:3, 3], [2, 0, 0])                    # ... node transform in float64
    np.testing.assert_allclose(a.world_vertices().min(axis=0), [2, 0, 0], atol=1e-6)
    assert np.array_equal(a.texture, tex)
    np.testing.assert_allclose(a.uv, QUAD_UV, atol=1e-6)                             # bottom-left convention
    # rotated 90 deg about Y: (x, y, 0) -> (0, y, -x); translated to y + 5
    np.testing.assert_allclose(b.world_vertices().min(axis=0), [0, 5, -1], atol=1e-5)
    np.testing.assert_allclose(b.world_vertices().max(axis=0), [0, 6, 0], atol=1e-5)
    # glTF baseColorFactor is linear -> stored sRGB-encoded
    expected = srgb_encode(np.array([0, 128, 255]) / 255.0)
    np.testing.assert_allclose(b.base_color[:3], expected, atol=1e-6)
    np.testing.assert_allclose(loaded.bounds, [[0, 0, -1], [3, 6, 0]], atol=1e-5)


def test_glb_base_color_factor_is_multiplied_into_texture(tmp_path):
    white = np.full((4, 4, 3), 255, np.uint8)
    mesh = trimesh.Trimesh(
        QUAD_V, QUAD_F, process=False,
        visual=TextureVisuals(uv=QUAD_UV, material=PBRMaterial(baseColorTexture=Image.fromarray(white),
                                                               baseColorFactor=[255, 0, 0, 255])),
    )
    trimesh.Scene(mesh).export(tmp_path / "t.glb")
    (obj,) = load_scene(tmp_path / "t.glb").objects
    assert set(map(tuple, obj.texture.reshape(-1, 3))) == {(255, 0, 0)}


def test_ply_vertex_and_face_colors(tmp_path):
    colors = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]], np.uint8)
    trimesh.Trimesh(QUAD_V, QUAD_F, vertex_colors=colors, process=False).export(tmp_path / "v.ply")
    (obj,) = load_scene(tmp_path / "v.ply").objects
    assert obj.texture is None and obj.colors.shape == (4, 4)
    np.testing.assert_allclose(obj.colors, colors / 255.0, atol=1e-6)

    face_colors = np.array([[255, 0, 0, 255], [0, 0, 255, 255]], np.uint8)
    trimesh.Trimesh(QUAD_V, QUAD_F, face_colors=face_colors, process=False).export(tmp_path / "f.ply")
    (obj,) = load_scene(tmp_path / "f.ply").objects
    assert len(obj.vertices) == 6  # unwelded: one colour per triangle corner
    np.testing.assert_allclose(obj.colors[:3], np.tile([1, 0, 0, 1], (3, 1)))
    np.testing.assert_allclose(obj.colors[3:], np.tile([0, 0, 1, 1], (3, 1)))


def test_srgb_roundtrip_and_reference_values():
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(srgb_decode(srgb_encode(x)), x, atol=1e-9)
    assert float(srgb_decode(128 / 255)) == pytest.approx(0.2158, abs=1e-4)


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scene(tmp_path / "missing.obj")
    with pytest.raises(FileNotFoundError):
        load_scene({"objects": [{"path": str(tmp_path / "missing.obj")}]})
    with pytest.raises(ValueError, match="unknown"):
        load_scene({"objects": [{"path": "x.obj", "rotation": [0, 0, 0]}]})
    with pytest.raises(ValueError, match="no triangle geometry"):
        load_scene({"objects": []})
    with pytest.raises(TypeError):
        load_scene(42)


def test_scene_object_validation():
    v = np.zeros((3, 3))
    with pytest.raises(ValueError):
        SceneObject(v, np.array([[0, 1, 5]]))
    with pytest.raises(ValueError):
        SceneObject(v, np.array([[0, 1, 2]]), texture=np.zeros((2, 2, 3), np.uint8))  # texture without uv
    obj = SceneObject(v, np.array([[0, 1, 2]]))
    assert isinstance(Scene([obj]).to_trimesh(), trimesh.Trimesh)
    assert Scene([]).diagonal == 0.0


# ---- safety: confinement to a root directory, texture / geometry caps ---------------------------------------
REPO = Path(__file__).resolve().parents[1]
QUAD_OBJ = ("v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "usemtl m\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
SECRET_RGB = (7, 99, 201)


@pytest.fixture
def sandbox(tmp_path):
    """root = the upload directory; secret/ lies outside of it (root/../../secret)."""
    root = tmp_path / "uploads" / "scene1"
    secret = tmp_path / "secret"
    root.mkdir(parents=True)
    secret.mkdir()
    Image.fromarray(np.full((4, 4, 3), SECRET_RGB, np.uint8)).save(secret / "private.png")
    (secret / "evil.mtl").write_text("newmtl m\nKd 1 1 1\nmap_Kd private.png\n")
    (root / "quad.obj").write_text(QUAD_OBJ)
    Image.fromarray(np.full((4, 4, 3), 200, np.uint8)).save(root / "own.png")
    return SimpleNamespace(root=root, secret=secret)


def write_gltf(path, buffer_uri: str, image_uri=None):
    v = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0]], np.float32)
    blob = v.tobytes() + np.array([0, 1, 2], np.uint16).tobytes() + b"\0\0"
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"uri": buffer_uri, "byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 6}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}],
    }
    if image_uri:
        doc["images"], doc["textures"] = [{"uri": image_uri}], [{"source": 0}]
        doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
        doc["meshes"][0]["primitives"][0]["material"] = 0
    path.write_text(json.dumps(doc))
    return blob


def test_root_rejects_spec_paths_outside(sandbox):
    root, secret = sandbox.root, sandbox.secret
    inside = str(root / "quad.obj")
    for entry in (
        {"path": "quad.obj", "texture": "../../secret/private.png"},        # traversal
        {"path": "quad.obj", "texture": str(secret / "private.png")},        # absolute
        {"path": str(secret / "evil.mtl")},                                   # absolute object path
        {"path": "../../secret/evil.mtl"},
    ):
        with pytest.raises(ValueError, match="outside"):
            load_scene({"objects": [entry]}, root=root)
    (secret / "spec.json").write_text(json.dumps({"objects": [{"path": inside}]}))
    with pytest.raises(ValueError, match="outside"):
        load_scene(secret / "spec.json", root=root)
    with pytest.raises(ValueError, match="outside"):
        load_scene(str(secret / "evil.mtl"), root=root)
    # a symlink inside the root that points outside is outside as well
    try:
        (root / "link.png").symlink_to(secret / "private.png")
    except OSError:
        pytest.skip("symlinks not available")
    with pytest.raises(ValueError, match="outside"):
        load_scene({"objects": [{"path": "quad.obj", "texture": "link.png"}]}, root=root)


def test_root_does_not_expand_home(sandbox):
    with pytest.raises(FileNotFoundError):
        load_scene({"objects": [{"path": "~/quad.obj"}]}, root=sandbox.root)


def test_root_allows_files_inside(sandbox):
    root = sandbox.root
    (root / "sub").mkdir()
    (root / "sub" / "t.png").write_bytes((root / "own.png").read_bytes())
    (root / "sub" / "m.mtl").write_text("newmtl m\nmap_Kd t.png\n")
    (root / "sub" / "a.obj").write_text("mtllib m.mtl\n" + QUAD_OBJ)
    assert load_scene("sub/a.obj", root=root).objects[0].texture is not None      # relative to the root
    assert load_scene(root / "sub" / "a.obj", root=root).objects[0].texture is not None
    (root / "spec.json").write_text(json.dumps({"objects": [{"path": "quad.obj", "texture": "own.png"}]}))
    assert load_scene(root / "spec.json", root=root).objects[0].texture[0, 0, 0] == 200
    scene = load_scene({"objects": [{"path": str(root / "quad.obj"), "texture": str(root / "own.png")}]}, root=root)
    assert scene.objects[0].texture[0, 0, 0] == 200


@pytest.mark.parametrize("mtl_line", ["mtllib ../../secret/evil.mtl", "mtllib {secret}/evil.mtl"])
def test_root_confines_mtl_libraries(sandbox, mtl_line):
    (sandbox.root / "a.obj").write_text(mtl_line.format(secret=sandbox.secret) + "\n" + QUAD_OBJ)
    with pytest.raises(ValueError, match="outside"):
        load_scene(sandbox.root / "a.obj", root=sandbox.root)


@pytest.mark.parametrize("map_kd", ["../../secret/private.png", "{secret}/private.png"])
def test_root_confines_mtl_texture_references(sandbox, map_kd):
    (sandbox.root / "b.mtl").write_text("newmtl m\nKd 1 1 1\nmap_Kd " + map_kd.format(secret=sandbox.secret) + "\n")
    (sandbox.root / "b.obj").write_text("mtllib b.mtl\n" + QUAD_OBJ)
    with pytest.raises(ValueError, match="outside"):
        load_scene(sandbox.root / "b.obj", root=sandbox.root)


@pytest.mark.parametrize("uri", ["../../secret/geom.bin", "{secret}/geom.bin"])
def test_root_confines_gltf_buffers_and_images(sandbox, uri):
    blob = write_gltf(sandbox.root / "c.gltf", uri.format(secret=sandbox.secret))
    (sandbox.secret / "geom.bin").write_bytes(blob)
    with pytest.raises(ValueError, match="outside"):
        load_scene(sandbox.root / "c.gltf", root=sandbox.root)
    # buffer inside the root, image outside
    (sandbox.root / "geom.bin").write_bytes(blob)
    write_gltf(sandbox.root / "d.gltf", "geom.bin", image_uri=uri.replace("geom.bin", "private.png").format(secret=sandbox.secret))
    with pytest.raises(ValueError, match="outside"):
        load_scene(sandbox.root / "d.gltf", root=sandbox.root)


def test_root_allows_gltf_with_external_files_inside(sandbox):
    blob = write_gltf(sandbox.root / "e.gltf", "geom.bin", image_uri="own.png")
    (sandbox.root / "geom.bin").write_bytes(blob)
    (obj,) = load_scene(sandbox.root / "e.gltf", root=sandbox.root).objects
    assert len(obj.faces) == 1


def test_without_root_paths_are_unrestricted(sandbox):
    """Trusted local use (CLI): absolute paths, '..' and '~' keep working."""
    scene = load_scene({"objects": [{"path": str(sandbox.root / "quad.obj"),
                                     "texture": str(sandbox.root / ".." / ".." / "secret" / "private.png")}]})
    assert tuple(scene.objects[0].texture[0, 0]) == SECRET_RGB


def write_bomb_png(path, w=16000, h=16000):
    """1-bit grayscale PNG that decodes to w*h pixels but is ~30 kB on disk (streamed, no big allocation)."""
    import struct
    import zlib

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    row = b"\x00" * (1 + (w + 7) // 8)
    comp = zlib.compressobj(9)
    idat = b"".join(comp.compress(row * 1000) for _ in range(h // 1000)) + comp.compress(row * (h % 1000)) + comp.flush()
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 1, 0, 0, 0, 0))
                     + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def test_huge_texture_is_decoded_with_bounded_memory(sandbox):
    """16000x16000 px in ~30 kB: used to cost +1.7 GB peak RSS (RGB conversion before downscaling)."""
    write_bomb_png(sandbox.root / "bomb.png")
    assert (sandbox.root / "bomb.png").stat().st_size < 100_000
    code = f"""
import resource, sys
sys.path.insert(0, {str(REPO)!r})
from backend.scene import load_scene
def peak():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024
before = peak()
scene = load_scene({{"objects": [{{"path": "quad.obj", "texture": "bomb.png"}}]}}, root={str(sandbox.root)!r})
print(scene.objects[0].texture.shape[0], scene.objects[0].texture.shape[1], (peak() - before) / 1e6)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    h, w, growth_mb = out.stdout.split()
    assert max(int(h), int(w)) <= 8192
    assert float(growth_mb) < 900, f"peak RSS grew by {growth_mb} MB"


def test_texture_pixel_cap_is_checked_before_decoding(sandbox):
    write_bomb_png(sandbox.root / "bomb.png", 4000, 3000)
    with pytest.raises(ValueError, match="pixel"):
        load_scene({"objects": [{"path": "quad.obj", "texture": "bomb.png"}]}, root=sandbox.root,
                   max_texture_pixels=1_000_000)
    # the same cap applies to images decoded by trimesh itself (mtl / glTF materials)
    (sandbox.root / "m.mtl").write_text("newmtl m\nmap_Kd bomb.png\n")
    (sandbox.root / "b.obj").write_text("mtllib m.mtl\n" + QUAD_OBJ)
    with pytest.raises(ValueError, match="pixel"):
        load_scene(sandbox.root / "b.obj", root=sandbox.root, max_texture_pixels=1_000_000)
    assert load_scene(sandbox.root / "b.obj", root=sandbox.root).objects[0].texture is not None


def test_pil_global_pixel_limit_is_left_alone(sandbox):
    before = Image.MAX_IMAGE_PIXELS
    assert before is not None  # the old code set it to None for the whole process
    load_scene({"objects": [{"path": "quad.obj", "texture": "own.png"}]}, root=sandbox.root)
    assert Image.MAX_IMAGE_PIXELS == before


def test_non_finite_geometry_is_rejected_with_the_object_name(sandbox):
    (sandbox.root / "nan.obj").write_text("v nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with pytest.raises(ValueError, match=r"nan\b.*(finite|NaN)|(finite|NaN).*nan\b"):
        load_scene(sandbox.root / "nan.obj", root=sandbox.root)
    with pytest.raises(ValueError, match="finite"):
        SceneObject(np.array([[0, 0, 0], [1, 0, np.inf], [0, 1, 0]]), [[0, 1, 2]], name="broken")
    with pytest.raises(ValueError, match="broken"):
        SceneObject(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), [[0, 1, 2]], uv=np.full((3, 2), np.nan), name="broken")


def test_triangle_cap(sandbox):
    with pytest.raises(ValueError, match="triangle"):
        load_scene(sandbox.root / "quad.obj", root=sandbox.root, max_triangles=1)
    assert load_scene(sandbox.root / "quad.obj", root=sandbox.root, max_triangles=2).num_triangles == 2
    with pytest.raises(ValueError, match="triangle"):  # the cap counts all objects of the scene
        load_scene({"objects": [{"path": "quad.obj"}, {"path": "quad.obj"}]}, root=sandbox.root, max_triangles=3)


def test_georeferenced_obj_keeps_precision(tmp_path):
    """Coordinates ~1e5 (UTM-like): vertices are re-centred so float32 storage costs no precision."""
    off = np.array([123456.0, 654321.0, 10.0])
    v = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float) + off + 1e-3
    (tmp_path / "g.obj").write_text("".join("v %.6f %.6f %.6f\n" % tuple(p) for p in v) + "f 1 2 3\nf 1 3 4\n")
    (obj,) = load_scene(tmp_path / "g.obj").objects
    assert np.abs(obj.vertices).max() < 10                      # small local coordinates
    np.testing.assert_allclose(obj.world_vertices(), v, atol=1e-9)
    np.testing.assert_allclose(Scene([obj]).bounds, [v.min(axis=0), v.max(axis=0)], atol=1e-9)
