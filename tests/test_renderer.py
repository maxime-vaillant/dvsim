"""Renderer tests against analytic ground truth (pinhole geometry), no snapshots. Need a GL context (skip otherwise)."""
from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from backend.config import CameraConfig
from backend.graphical_computation.utils.renderer import Renderer, RenderOutput, auto_clip_planes
from backend.pose import Pose, focal_length_px
from backend.scene import Scene, SceneObject, load_scene
from backend.scene.textures import srgb_decode

QUAD_UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)  # v = 0 at the bottom (OBJ / trimesh)
QUAD_F = np.array([[0, 1, 2], [0, 2, 3]])
LUMA = np.array([0.2126, 0.7152, 0.0722])
REPO = Path(__file__).resolve().parents[1]


# ---- helpers ---------------------------------------------------------------------------------------------
def cfg(w: int = 320, h: int = 240, fov: float = 60.0, **kw) -> CameraConfig:
    return CameraConfig(width=w, height=h, fov_y_deg=fov, **kw)


def look(pos, front=(0, 0, -1)) -> Pose:
    return Pose.from_look(pos, front)


def quad(corners, color=None, texture=None, colors=None) -> SceneObject:
    """Quad from 4 counter-clockwise corners; ``color`` is an sRGB base colour in [0, 1]."""
    kw = {}
    if color is not None:
        kw["base_color"] = np.array([*color, 1.0], dtype=np.float32)
    if texture is not None:
        kw.update(texture=texture, uv=QUAD_UV)
    return SceneObject(np.array(corners, dtype=np.float32), QUAD_F, colors=colors, **kw)


def plane_z(z, half=30.0, **kw) -> SceneObject:
    return quad([(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)], **kw)


def fill_view(z=-1.0, **kw) -> SceneObject:
    return plane_z(z, half=10.0, **kw)


def focal(c: CameraConfig) -> float:
    return focal_length_px(c.fov_y_deg, c.height)


def project(c: CameraConfig, pose: Pose, points: np.ndarray):
    """Independent pinhole model: world points (N, 3) -> pixel (u, v), +v down."""
    cam = (np.asarray(points, float) - pose.position) @ pose.rotation()
    f = focal(c)
    return c.width / 2 + f * cam[:, 0] / -cam[:, 2], c.height / 2 - f * cam[:, 1] / -cam[:, 2]


def unproject(c: CameraConfig, pose: Pose, depth: np.ndarray) -> np.ndarray:
    """z-depth image -> world points (H, W, 3) at pixel centres."""
    cols, rows = np.meshgrid(np.arange(c.width) + 0.5, np.arange(c.height) + 0.5)
    f = focal(c)
    cam = np.stack([(cols - c.width / 2) / f * depth, -(rows - c.height / 2) / f * depth, -depth], axis=-1)
    return cam @ pose.rotation().T + pose.position


def assert_flow_matches_pinhole(c, pose, next_pose, out: RenderOutput, min_pixels=1000):
    mask = (out.depth > 0) & (out.flow_valid == 1)
    assert mask.sum() >= min_pixels
    u, v = project(c, next_pose, unproject(c, pose, out.depth)[mask])
    cols, rows = np.meshgrid(np.arange(c.width) + 0.5, np.arange(c.height) + 0.5)
    expected = np.stack([u - cols[mask], v - rows[mask]], axis=-1)
    err = np.abs(out.flow[mask] - expected).max(axis=-1)
    assert np.all(err <= 0.02 + 0.005 * np.linalg.norm(expected, axis=-1)), err.max()


def png_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def make_renderer(gl_context):
    made = []

    def make(objects, camera=None, **kw) -> Renderer:
        r = Renderer(Scene(list(objects)), camera or cfg(), gl_context, **kw)
        made.append(r)
        return r

    yield make
    for r in made:
        r.close()


def near_far_room(far_first=False):
    near = quad([(-1, -1, -2), (1, -1, -2), (1, 1, -2), (-1, 1, -2)], color=(1, 0, 0))
    far = plane_z(-5, color=(0, 1, 0))
    ground = quad([(-30, -1, -0.5), (30, -1, -0.5), (30, -1, -90), (-30, -1, -90)], color=(0.3, 0.3, 0.9))
    return [far, near, ground] if far_first else [near, far, ground]


# ---- optical flow ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("far_first", [False, True])
def test_flow_lateral_translation_near_and_far_in_one_frame(make_renderer, far_first):
    c = cfg()
    f, tx = focal(c), 0.1
    r = make_renderer(near_far_room(far_first), c)
    p0, p1 = look([0, 0, 0]), look([tx, 0, 0])
    out = r.render(p0, p1, depth=True, flow=True)

    near_px, far_px = (c.height // 2, c.width // 2), (c.height // 2, 20)  # quad centre / wall left of the quad
    assert out.depth[near_px] == pytest.approx(2.0, rel=1e-4)
    assert out.depth[far_px] == pytest.approx(5.0, rel=1e-4)
    assert out.flow[near_px][0] == pytest.approx(-f * tx / 2.0, rel=0.01)  # occlusion: near quad wins the pixel
    assert out.flow[far_px][0] == pytest.approx(-f * tx / 5.0, rel=0.01)
    assert abs(out.flow[near_px][1]) < 0.01 and abs(out.flow[far_px][1]) < 0.01
    assert_flow_matches_pinhole(c, p0, p1, out)


@pytest.mark.parametrize("move, sign", [((0.05, 0, 0), (-1, 0)), ((0, 0.05, 0), (0, 1)), ((-0.05, 0, 0), (1, 0))])
def test_flow_sign_conventions(make_renderer, move, sign):
    c = cfg()
    r = make_renderer(near_far_room(), c)
    out = r.render(look([0, 0, 0]), look(move), flow=True)
    fx, fy = out.flow[c.height // 2, c.width // 2]
    expected = focal(c) * 0.05 / 2.0
    assert fx == pytest.approx(sign[0] * expected, abs=0.02)  # camera right -> scene moves left (dx < 0)
    assert fy == pytest.approx(sign[1] * expected, abs=0.02)  # camera up -> scene moves down (dy > 0)


def test_flow_pure_yaw_matches_f_tan_theta(make_renderer):
    c = cfg(321, 241)  # odd size: a pixel centre sits exactly on the principal point
    f, theta = focal(c), np.radians(5.0)
    r = make_renderer(near_far_room(), c)
    p0 = look([0, 0, 0])
    p1 = look([0, 0, 0], (-np.sin(theta), 0, -np.cos(theta)))  # turn left
    out = r.render(p0, p1, flow=True)
    fx, fy = out.flow[c.height // 2, c.width // 2]
    assert fx == pytest.approx(f * np.tan(theta), rel=0.005)  # scene moves right when the camera turns left
    assert abs(fy) < 0.02


def test_flow_matches_pinhole_for_general_motion(make_renderer):
    c = cfg(200, 150, 70.0)
    r = make_renderer(near_far_room(), c)
    p0 = look([0.2, 0.1, 0.5], (0.1, -0.05, -1))
    p1 = look([0.3, 0.15, 0.3], (0.25, 0.02, -1))
    out = r.render(p0, p1, depth=True, flow=True)
    assert_flow_matches_pinhole(c, p0, p1, out, min_pixels=10000)


def test_flow_valid_false_for_background_and_true_on_surfaces(make_renderer):
    c = cfg()
    small = quad([(-0.5, -0.5, -3), (0.5, -0.5, -3), (0.5, 0.5, -3), (-0.5, 0.5, -3)], color=(1, 1, 1))
    r = make_renderer([small], c)
    out = r.render(look([0, 0, 0]), look([0.05, 0, 0]), depth=True, flow=True)
    assert out.flow_valid.dtype == np.uint8 and set(np.unique(out.flow_valid)) == {0, 1}
    assert np.array_equal(out.flow_valid == 1, out.depth > 0)
    assert 0 < (out.depth > 0).sum() < out.depth.size
    assert np.all(out.flow[out.depth == 0] == 0)


def test_flow_valid_false_for_points_behind_the_next_camera(make_renderer):
    c = cfg()
    r = make_renderer(near_far_room(), c)
    # the next camera moved back to z=-3.5 looking -Z: the quad at z=-2 is behind it, the wall at z=-5 is ahead
    out = r.render(look([0, 0, 0]), look([0, 0, -3.5]), depth=True, flow=True)
    near_px, far_px = (c.height // 2, c.width // 2), (c.height // 2, 20)
    assert out.depth[near_px] == pytest.approx(2.0, rel=1e-4) and out.flow_valid[near_px] == 0
    assert out.depth[far_px] == pytest.approx(5.0, rel=1e-4) and out.flow_valid[far_px] == 1
    assert np.isfinite(out.flow).all()
    # the far wall flow is a pure zoom: dx = (u - cx) * (Z / (Z - 3.5) - 1)
    u = far_px[1] + 0.5 - c.width / 2
    assert out.flow[far_px][0] == pytest.approx(u * (5.0 / 1.5 - 1.0), rel=0.01)


# ---- depth --------------------------------------------------------------------------------------------
def test_depth_of_a_ground_plane_is_analytic_z_depth(make_renderer):
    c = cfg()
    ground = quad([(-100, -1, 1), (100, -1, 1), (100, -1, -300), (-100, -1, -300)], color=(0.5, 0.5, 0.5))
    r = make_renderer([ground], c)
    out = r.render(look([0, 0, 0]), depth=True)
    f = focal(c)
    rows = np.arange(c.height)
    offset = rows + 0.5 - c.height / 2
    expected = np.where(offset > 0, f / np.where(offset > 0, offset, 1.0), 0.0)  # z-depth of the y=-1 plane
    sel = (expected >= 4.0) & (expected <= 80.0)
    assert sel.sum() > 40
    for col in (c.width // 2, 5, c.width - 6):  # same z-depth off-axis: z-depth, not Euclidean range
        np.testing.assert_allclose(out.depth[sel, col], expected[sel], rtol=0.005)
    assert np.all(out.depth[: c.height // 2] == 0)  # above the horizon: no surface
    assert out.depth.dtype == np.float32


def test_depth_precision_with_auto_planes_on_a_large_scene(make_renderer):
    c = cfg()
    ground = quad([(-500, -1, 1), (500, -1, 1), (500, -1, -1000), (-500, -1, -1000)])
    r = make_renderer([ground], c)
    assert r.far > 1000 and r.near >= 0.01
    out = r.render(look([0, 0, 0]), depth=True)
    row = c.height // 2 + 3
    assert out.depth[row, 10] == pytest.approx(focal(c) / (row + 0.5 - c.height / 2), rel=0.005)


# ---- colour, luminance ----------------------------------------------------------------------------------
@pytest.mark.parametrize("color, lum", [((1, 0, 0), 0.2126), ((0, 1, 0), 0.7152), ((0, 0, 1), 0.0722)])
def test_rgb_channel_order_and_luminance_weights(make_renderer, color, lum):
    r = make_renderer([fill_view(color=color)])
    out = r.render(look([0, 0, 0]), rgb=True)
    assert out.rgb.shape == (240, 320, 3) and out.rgb.dtype == np.uint8
    assert tuple(out.rgb[120, 160]) == tuple(int(v * 255) for v in color)  # RGB order, not BGR
    np.testing.assert_allclose(out.lum, lum, atol=1e-4)
    assert out.lum.dtype == np.float32


def test_textured_red_is_rgb_red_and_midgrey_is_linearised(make_renderer):
    red = np.zeros((4, 4, 3), np.uint8)
    red[..., 0] = 255
    out = make_renderer([fill_view(texture=red)]).render(look([0, 0, 0]), rgb=True)
    assert tuple(out.rgb[120, 160]) == (255, 0, 0)
    assert out.lum[120, 160] == pytest.approx(0.2126, abs=1e-4)

    grey = np.full((4, 4, 3), 128, np.uint8)
    out = make_renderer([fill_view(texture=grey)]).render(look([0, 0, 0]), rgb=True)
    assert abs(int(out.rgb[120, 160, 0]) - 128) <= 1
    assert out.lum[120, 160] == pytest.approx(0.2158, abs=5e-4)  # sRGB 128 -> 0.2158, not 0.502

    out = make_renderer([fill_view(color=(0.5, 0.5, 0.5))]).render(look([0, 0, 0]))
    assert out.lum[120, 160] == pytest.approx(float(srgb_decode(0.5)), abs=1e-4)


def test_vertex_colors(make_renderer):
    rgb = np.array([200, 100, 50]) / 255.0
    colors = np.tile([*rgb, 1.0], (4, 1))
    out = make_renderer([fill_view(colors=colors)]).render(look([0, 0, 0]), rgb=True)
    assert np.abs(out.rgb[120, 160].astype(int) - [200, 100, 50]).max() <= 1
    assert out.lum[120, 160] == pytest.approx(float(srgb_decode(rgb) @ LUMA), abs=1e-3)


def test_texture_is_filtered_in_linear_light(make_renderer):
    n = 512
    checker = ((np.indices((n, n)).sum(axis=0) % 2) * 255).astype(np.uint8)  # 1-texel black/white checker
    tex = np.repeat(checker[..., None], 3, axis=2)
    quad_far = quad([(-1, -1, -20), (1, -1, -20), (1, 1, -20), (-1, 1, -20)], texture=tex)
    out = make_renderer([quad_far], cfg(128, 128)).render(look([0, 0, 0]))
    patch = out.lum[60:68, 60:68]
    assert patch.mean() == pytest.approx(0.5, abs=0.05)  # sRGB-space averaging would give 0.214


# ---- stereo ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("front", [(0, 0, -1), (1, 0, 0), (0.6, 0.0, -0.8)])
def test_stereo_disparity_is_f_b_over_z(make_renderer, front):
    c = cfg(512, 512, 60.0)
    f, baseline, dist = focal(c), 0.1, 3.0
    left = look([0, 0, 0], front)
    right = left.translated_local(baseline)
    assert np.dot(right.position, left.right) == pytest.approx(baseline)
    ctr, rt, up, fw = left.forward * dist, left.right, left.up, left.forward

    def rect(centre, half):
        return [centre - half * rt - half * up, centre + half * rt - half * up,
                centre + half * rt + half * up, centre - half * rt + half * up]

    patch = quad(rect(ctr, 0.25), color=(1, 1, 1))
    backdrop = quad(rect(fw * 6.0, 40.0), color=(0, 0, 0))
    r = make_renderer([patch, backdrop], c)

    def centre_col(pose):
        lit = (r.render(pose).lum > 0.5).any(axis=0)
        cols = np.flatnonzero(lit)
        return (cols.min() + cols.max() + 1) / 2.0

    disparity = centre_col(left) - centre_col(right)
    assert disparity == pytest.approx(f * baseline / dist, abs=1.0)  # 14.8 px
    out = r.render(left, right, flow=True)  # the same fact through the flow output
    fx, fy = out.flow[256, 256]
    assert fx == pytest.approx(-f * baseline / dist, rel=0.01) and abs(fy) < 0.02


# ---- UV orientation, formats -----------------------------------------------------------------------------
def asym_texture() -> np.ndarray:
    tex = np.full((32, 64, 3), 40, np.uint8)
    tex[:8, :16] = (255, 255, 255)   # top-left white
    tex[:8, -16:] = (0, 0, 255)      # top-right blue
    tex[-8:, -16:] = (255, 0, 0)     # bottom-right red
    return tex                        # bottom-left stays dark grey


def assert_upright(rgb: np.ndarray):
    """The texture must be seen upright and unmirrored (quad spans ~9%..91% of a 128 px view)."""
    def at(fx, fy):
        return rgb[int(8.6 + 110.8 * fy), int(8.6 + 110.8 * fx)].astype(int)

    assert np.abs(at(0.1, 0.1) - [255, 255, 255]).max() <= 3   # top-left
    assert np.abs(at(0.9, 0.1) - [0, 0, 255]).max() <= 3       # top-right
    assert np.abs(at(0.9, 0.9) - [255, 0, 0]).max() <= 3       # bottom-right
    assert np.abs(at(0.1, 0.9) - [40, 40, 40]).max() <= 3      # bottom-left


def render_scene_view(gl_context, scene) -> np.ndarray:
    r = Renderer(scene, cfg(128, 128), gl_context)
    try:
        return r.render(look([0, 0, 0]), rgb=True).rgb
    finally:
        r.close()


def test_uv_orientation_obj(gl_context, tmp_path):
    Image.fromarray(asym_texture()).save(tmp_path / "t.png")
    (tmp_path / "q.mtl").write_text("newmtl m\nmap_Kd t.png\n")
    (tmp_path / "q.obj").write_text(
        "mtllib q.mtl\nv -0.5 -0.5 -1\nv 0.5 -0.5 -1\nv 0.5 0.5 -1\nv -0.5 0.5 -1\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl m\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"  # OBJ: v = 0 is the image bottom
    )
    assert_upright(render_scene_view(gl_context, load_scene(tmp_path / "q.obj")))
    # legacy usage: bare OBJ (UVs, no mtl) + separate texture image
    (tmp_path / "bare.obj").write_text((tmp_path / "q.obj").read_text().replace("mtllib q.mtl\n", "").replace("usemtl m\n", ""))
    scene = load_scene({"objects": [{"path": str(tmp_path / "bare.obj"), "texture": str(tmp_path / "t.png")}]})
    assert_upright(render_scene_view(gl_context, scene))


def test_uv_orientation_glb_roundtrip_via_trimesh(gl_context, tmp_path):
    verts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], float)
    mesh = trimesh.Trimesh(verts, QUAD_F, process=False, visual=TextureVisuals(
        uv=QUAD_UV, material=PBRMaterial(baseColorTexture=Image.fromarray(asym_texture()))))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="n", transform=trimesh.transformations.translation_matrix([0, 0, -1]))
    scene.export(tmp_path / "q.glb")
    assert_upright(render_scene_view(gl_context, load_scene(tmp_path / "q.glb")))


def test_uv_orientation_handwritten_gltf(gl_context, tmp_path):
    """glTF puts v = 0 at the image TOP; written by hand so a trimesh import/export bug cannot cancel out."""
    pos = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], "<f4")
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], "<f4")  # bottom-left vertex samples the BOTTOM of the image
    idx = np.array([0, 1, 2, 0, 2, 3], "<u2")
    blob = pos.tobytes() + uv.tobytes() + idx.tobytes()
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": [0, 0, -1]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}],
        "images": [{"uri": "data:image/png;base64," + base64.b64encode(png_bytes(asym_texture())).decode()}],
        "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode(),
                     "byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": 32},
                        {"buffer": 0, "byteOffset": 80, "byteLength": 12}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-0.5, -0.5, 0], "max": [0.5, 0.5, 0]},
                      {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
                      {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}],
    }
    (tmp_path / "q.gltf").write_text(json.dumps(doc))
    scene = load_scene(tmp_path / "q.gltf")
    np.testing.assert_allclose(scene.objects[0].uv[[0, 3]], [[0, 0], [0, 1]], atol=1e-6)  # converted to v-up
    assert_upright(render_scene_view(gl_context, scene))


def test_ply_vertex_colors_render(gl_context, tmp_path):
    colors = np.tile(np.array([[0, 255, 0, 255]], np.uint8), (4, 1))
    verts = np.array([[-5, -5, -1], [5, -5, -1], [5, 5, -1], [-5, 5, -1]], float)
    trimesh.Trimesh(verts, QUAD_F, vertex_colors=colors, process=False).export(tmp_path / "c.ply")
    rgb = render_scene_view(gl_context, load_scene(tmp_path / "c.ply"))
    assert tuple(rgb[64, 64]) == (0, 255, 0)


# ---- multi-object scenes ----------------------------------------------------------------------------------
def test_multi_object_spec_with_per_object_transforms(gl_context, tmp_path):
    (tmp_path / "unit.obj").write_text("v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nf 1 2 3\nf 1 3 4\n")
    scene = load_scene({"objects": [
        {"path": str(tmp_path / "unit.obj"), "position": [-1, 0, -3], "scale": 2.0, "base_color": [1, 0, 0]},
        {"path": str(tmp_path / "unit.obj"), "position": [1.5, 0, -6], "rotation_euler_deg": [0, 60, 0],
         "base_color": [0, 1, 0]},
    ]})
    c = cfg()
    r = Renderer(scene, c, gl_context)
    try:
        out = r.render(look([0, 0, 0]), rgb=True, depth=True)
    finally:
        r.close()
    f = focal(c)
    # object A: fronto-parallel plane z = -3, centred at pixel column 160 - f/3
    a = (c.height // 2, int(c.width / 2 - f / 3))
    assert out.depth[a] == pytest.approx(3.0, rel=1e-4) and tuple(out.rgb[a]) == (255, 0, 0)
    # object B: plane through (1.5, 0, -6) with normal Ry(60) e_z = (sin60, 0, cos60), z-depth = n.P0 / n.d
    b = (c.height // 2, int(c.width / 2 + f * 1.5 / 6))
    n, p0 = np.array([np.sin(np.radians(60)), 0, np.cos(np.radians(60))]), np.array([1.5, 0, -6.0])
    d = np.array([(b[1] + 0.5 - c.width / 2) / f, -(b[0] + 0.5 - c.height / 2) / f, -1.0])
    assert out.depth[b] == pytest.approx(float(n @ p0 / (n @ d)), rel=1e-3) and tuple(out.rgb[b]) == (0, 255, 0)
    assert out.depth[0, 0] == 0
    np.testing.assert_allclose(scene.bounds[0][2], -6.0 - 0.5 * np.sin(np.radians(60)), atol=1e-5)


def test_glb_scene_graph_renders_with_node_transforms_and_materials(gl_context, tmp_path):
    verts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], float)
    red = np.zeros((4, 4, 3), np.uint8)
    red[..., 0] = 255
    textured = trimesh.Trimesh(verts, QUAD_F, process=False, visual=TextureVisuals(
        uv=QUAD_UV, material=PBRMaterial(baseColorTexture=Image.fromarray(red))))
    tinted = trimesh.Trimesh(verts, QUAD_F, process=False,
                             visual=TextureVisuals(material=PBRMaterial(baseColorFactor=[0, 255, 0, 255])))
    tm = trimesh.Scene()
    tm.add_geometry(textured, node_name="a", geom_name="ga",
                    transform=trimesh.transformations.translation_matrix([-1, 0, -3]))
    tm.add_geometry(tinted, node_name="b", geom_name="gb",
                    transform=trimesh.transformations.translation_matrix([1, 0, -5]))
    tm.export(tmp_path / "s.glb")
    scene = load_scene(tmp_path / "s.glb")
    assert len(scene.objects) == 2
    c = cfg()
    r = Renderer(scene, c, gl_context)
    try:
        out = r.render(look([0, 0, 0]), rgb=True, depth=True)
    finally:
        r.close()
    f = focal(c)
    a = (c.height // 2, int(c.width / 2 - f / 3))
    b = (c.height // 2, int(c.width / 2 + f / 5))
    assert out.depth[a] == pytest.approx(3.0, rel=1e-4) and tuple(out.rgb[a]) == (255, 0, 0)
    assert out.depth[b] == pytest.approx(5.0, rel=1e-4) and tuple(out.rgb[b]) == (0, 255, 0)


# ---- API contract -------------------------------------------------------------------------------------------
def test_requesting_nothing_returns_only_luminance_and_reads_back_only_that(make_renderer):
    r = make_renderer(near_far_room())
    reads = []
    original = r._read
    r._read = lambda att, *a, **k: (reads.append(att), original(att, *a, **k))[1]
    p = look([0, 0, 0])

    out = r.render(p)
    assert isinstance(out, RenderOutput)
    assert out.rgb is None and out.depth is None and out.flow is None and out.flow_valid is None
    assert out.lum.shape == (240, 320) and reads == [1]  # one read-back: luminance

    reads.clear()
    r.render(p, rgb=True)
    assert reads == [1, 0]
    reads.clear()
    r.render(p, depth=True)
    assert reads == [1, 3]
    reads.clear()
    out = r.render(p, look([0.1, 0, 0]), flow=True)
    assert reads == [1, 2, 4] and out.depth is None and out.rgb is None and out.flow_valid is not None
    reads.clear()
    r.render(p, look([0.1, 0, 0]), rgb=True, depth=True)  # next_pose given but flow not requested
    assert reads == [1, 0, 3]


def test_flow_requires_next_pose(make_renderer):
    r = make_renderer(near_far_room())
    with pytest.raises(ValueError, match="next_pose"):
        r.render(look([0, 0, 0]), flow=True)


def test_outputs_are_independent_contiguous_and_writable(make_renderer):
    r = make_renderer(near_far_room())
    p0, p1 = look([0, 0, 0]), look([0.1, 0, 0])
    a = r.render(p0, p1, rgb=True, depth=True, flow=True)
    b = r.render(p1, p0, rgb=True, depth=True, flow=True)
    for name in ("lum", "rgb", "depth", "flow", "flow_valid"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.flags.c_contiguous and x.flags.writeable and x.flags.owndata
        assert not np.shares_memory(x, y)
    keep = a.lum.copy()
    a.lum[:] = 7  # scribbling on one result must not change later renders
    again = r.render(p0)
    np.testing.assert_array_equal(again.lum, keep)
    assert a.depth.shape == (240, 320) and a.flow.shape == (240, 320, 2) and a.rgb.shape == (240, 320, 3)


def test_rendering_is_stateless_between_calls(make_renderer):
    r = make_renderer(near_far_room())
    p0, p1 = look([0, 0, 0]), look([0.1, 0.02, 0.05], (0.1, 0, -1))
    first = r.render(p0, p1, depth=True, flow=True)
    r.render(p1, p0, rgb=True)
    second = r.render(p0, p1, depth=True, flow=True)
    np.testing.assert_array_equal(first.lum, second.lum)
    np.testing.assert_array_equal(first.flow, second.flow)
    np.testing.assert_array_equal(first.depth, second.depth)


def test_auto_clip_planes_and_overrides(make_renderer):
    scene = Scene(near_far_room())
    near, far = auto_clip_planes(scene)
    diag = scene.diagonal
    assert near == pytest.approx(min(max(0.01, 1e-3 * diag), 0.05)) and far >= 10 * diag - 1e-9
    _, far_path = auto_clip_planes(scene, np.array([[200.0, 0, 0]]))
    assert far_path > 200  # camera far outside the scene bounds
    r = make_renderer(scene.objects, cfg(near=0.5, far=50.0))
    assert (r.near, r.far) == (0.5, 50.0)
    with pytest.raises(ValueError):
        make_renderer(scene.objects, cfg(near=5.0, far=1.0))


def test_close_releases_resources_and_keeps_a_shared_context(gl_context):
    r = Renderer(Scene(near_far_room()), cfg(), gl_context)
    r.close()
    r.close()  # idempotent
    with pytest.raises(RuntimeError):
        r.render(look([0, 0, 0]))
    gl_context.buffer(b"\0" * 16).release()  # still usable

    owned = Renderer(Scene(near_far_room()), cfg())  # creates and owns its context
    ctx = owned.ctx
    owned.render(look([0, 0, 0]))
    owned.close()
    with pytest.raises(AttributeError):  # moderngl swaps a released context's handle for an InvalidObject
        ctx.buffer(b"\0" * 16)


def test_two_renderers_can_share_one_context(gl_context):
    a = Renderer(Scene([fill_view(color=(1, 0, 0))]), cfg(), gl_context)
    b = Renderer(Scene([fill_view(color=(0, 1, 0))]), cfg(64, 48), gl_context)
    try:
        assert tuple(a.render(look([0, 0, 0]), rgb=True).rgb[120, 160]) == (255, 0, 0)
        assert tuple(b.render(look([0, 0, 0]), rgb=True).rgb[24, 32]) == (0, 255, 0)
        assert tuple(a.render(look([0, 0, 0]), rgb=True).rgb[120, 160]) == (255, 0, 0)
    finally:
        a.close()
        b.close()


def test_empty_scene_is_rejected(gl_context):
    with pytest.raises(ValueError):
        Renderer(Scene([]), cfg(), gl_context)


# ---- GL context handling: several renderers, threads ------------------------------------------------------
def test_closing_one_renderer_does_not_blank_another(gl_context):
    """Each Renderer() owns a context; render() must make its own context current (used to return black frames)."""
    cam = cfg(32, 24, near=0.1, far=100.0)
    p = look([0, 0, 0])
    r1 = Renderer(Scene([fill_view(color=(1, 0, 0))]), cam)
    r2 = Renderer(Scene([fill_view(color=(0, 0, 1))]), cam)
    try:
        for _ in range(2):  # interleaved use
            assert tuple(r1.render(p, rgb=True).rgb[0, 0]) == (255, 0, 0)
            assert tuple(r2.render(p, rgb=True).rgb[0, 0]) == (0, 0, 255)
        r2.close()
        out = r1.render(p, rgb=True, depth=True)
        assert tuple(out.rgb[0, 0]) == (255, 0, 0) and out.depth[0, 0] == pytest.approx(1.0, rel=1e-4)
        r3 = Renderer(Scene([fill_view(color=(0, 1, 0))]), cam)  # a new one, then close the older
        r1.close()
        assert tuple(r3.render(p, rgb=True).rgb[0, 0]) == (0, 255, 0)
        r3.close()
    finally:
        r1.close()
        r2.close()


def test_shared_context_stays_current_for_the_caller(gl_context):
    r = Renderer(Scene([fill_view(color=(1, 0, 0))]), cfg(16, 16), gl_context)
    r.render(look([0, 0, 0]))
    gl_context.buffer(b"\0" * 16).release()  # the caller's own GL calls still work after render()
    r.close()
    gl_context.buffer(b"\0" * 16).release()


CROSS_THREAD_SCRIPT = """
import sys, threading
sys.path.insert(0, {repo!r})
import numpy as np
from backend.config import CameraConfig
from backend.pose import Pose
from backend.scene import Scene, SceneObject
from backend.graphical_computation.utils.renderer import Renderer

v = np.array([[-10, -10, -5], [10, -10, -5], [10, 10, -5], [-10, 10, -5]], np.float32)
def scene(color):
    return Scene([SceneObject(v, [[0, 1, 2], [0, 2, 3]], base_color=color)])
cam = CameraConfig(width=32, height=24, near=0.1, far=100.0)
p = Pose([0, 0, 0], [0, 0, 0, 1])
r1 = Renderer(scene((1, 0, 0, 1)), cam)
res = {{}}
def use():
    for name, fn in (("render", lambda: r1.render(p, rgb=True)), ("close", r1.close)):
        try:
            fn()
            res[name] = "no error"
        except RuntimeError as e:
            res[name] = "RuntimeError: " + str(e)
def own():
    r = Renderer(scene((0, 1, 0, 1)), cam)
    res["own"] = r.render(p, rgb=True).rgb[0, 0].tolist()
    r.close()
for target in (use, own):
    t = threading.Thread(target=target); t.start(); t.join()
print(res["render"]); print(res["close"]); print(res["own"])
print(r1.render(p, rgb=True).rgb[0, 0].tolist(), flush=True)   # the creating thread is unaffected
"""


def test_cross_thread_use_raises_instead_of_crashing(gl_context):
    """A moderngl context used from another thread segfaults the process (exit 139); it must be a RuntimeError."""
    out = subprocess.run([sys.executable, "-c", CROSS_THREAD_SCRIPT.format(repo=str(REPO))],
                         capture_output=True, text=True, timeout=180, cwd=str(REPO))
    assert out.returncode == 0, (out.returncode, out.stderr[-500:])
    render_msg, close_msg, own, main = out.stdout.strip().splitlines()[-4:]
    assert render_msg.startswith("RuntimeError") and "thread" in render_msg
    assert close_msg.startswith("RuntimeError") and "thread" in close_msg
    assert own == "[0, 255, 0]"  # a renderer created and used entirely inside a worker thread is fine
    assert main == "[255, 0, 0]"


# ---- clip planes ----------------------------------------------------------------------------------------------
def test_set_camera_positions_extends_far_for_distant_cameras(make_renderer):
    q = quad([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], color=(1, 1, 1))  # 2x2: diagonal 2.83
    r = make_renderer([q], cfg(128, 128))
    far_before = r.far
    p = look([0, 0, 40])  # 14 diagonals away
    assert far_before < 40
    assert (r.render(p, depth=True).depth > 0).sum() == 0  # beyond the automatic far plane
    r.set_camera_positions(np.array([[0.0, 0.0, 40.0], [0.0, 0.0, 30.0]]))
    assert r.far > 40 and r.near == pytest.approx(0.01)
    depth = r.render(p, depth=True).depth
    assert (depth > 0).sum() > 20 and depth[64, 64] == pytest.approx(40.0, rel=1e-4)  # ~5.5 px wide at 40 units
    r.set_camera_positions(np.zeros((0, 3)))  # empty path -> back to the scene-size default
    assert r.far == pytest.approx(far_before)


def test_set_camera_positions_keeps_explicit_planes(make_renderer):
    q = quad([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], color=(1, 1, 1))
    r = make_renderer([q], cfg(16, 16, near=0.5))
    r.set_camera_positions(np.array([[0.0, 0.0, 400.0]]))
    assert r.near == 0.5 and r.far > 400
    r = make_renderer([q], cfg(16, 16, near=0.5, far=60.0))
    r.set_camera_positions(np.array([[0.0, 0.0, 400.0]]))
    assert (r.near, r.far) == (0.5, 60.0)


def test_auto_near_does_not_clip_close_geometry_in_large_scenes(make_renderer):
    """diag ~ 500 gave near = 0.5, hiding anything closer than 0.5 m."""
    big = plane_z(-400.0, half=250.0, color=(0, 1, 0))
    small = quad([(-0.05, -0.05, -0.3), (0.05, -0.05, -0.3), (0.05, 0.05, -0.3), (-0.05, 0.05, -0.3)], color=(1, 0, 0))
    r = make_renderer([big, small], cfg(64, 64))
    assert r.near <= 0.05
    out = r.render(look([0, 0, 0]), depth=True)
    assert out.depth[32, 32] == pytest.approx(0.3, rel=1e-3)


# ---- large world coordinates: node transforms stay in float64 -----------------------------------------------------
def flow_error_at_offset(gl_context, scene, offset) -> float:
    c = cfg(64, 48)
    r = Renderer(scene, c, gl_context)
    try:
        p0 = look([offset, 0, 0])
        out = r.render(p0, look([offset + 0.1, 0, 0]), depth=True, flow=True)
    finally:
        r.close()
    expected = -focal(c) * 0.1 / 5.0
    assert (out.depth > 0).all() and (out.flow_valid == 1).all()
    return float(np.abs(out.flow[..., 0] - expected).max())


def test_glb_node_transform_at_1e5_keeps_flow_precision(gl_context, tmp_path):
    verts = np.array([[-20, -20, 0], [20, -20, 0], [20, 20, 0], [-20, 20, 0]], float)
    mesh = trimesh.Trimesh(verts, QUAD_F, process=False,
                           visual=TextureVisuals(material=PBRMaterial(baseColorFactor=[255, 255, 255, 255])))
    tm = trimesh.Scene()
    tm.add_geometry(mesh, node_name="n", transform=trimesh.transformations.translation_matrix([1e5, 0, -5]))
    tm.export(tmp_path / "far_away.glb")
    scene = load_scene(tmp_path / "far_away.glb")
    assert np.abs(scene.objects[0].vertices).max() <= 20 and scene.objects[0].model_matrix[0, 3] == 1e5
    assert flow_error_at_offset(gl_context, scene, 1e5) < 1e-3  # 3e-2 px when baked into float32 vertices


def test_georeferenced_obj_at_1e5_keeps_flow_precision(gl_context, tmp_path):
    v = np.array([[-20, -20, -5], [20, -20, -5], [20, 20, -5], [-20, 20, -5]], float) + [1e5, 0, 0]
    (tmp_path / "g.obj").write_text("".join("v %.6f %.6f %.6f\n" % tuple(p) for p in v) + "f 1 2 3\nf 1 3 4\n")
    assert flow_error_at_offset(gl_context, load_scene(tmp_path / "g.obj"), 1e5) < 1e-3
