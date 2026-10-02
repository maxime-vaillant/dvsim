"""Procedural demo scene: textured closed room + sub-volumes (OBJ), a multi-object GLB, a SceneSpec and a walk.

Everything is deterministic given the seed and needs only numpy / PIL / trimesh. World frame: +Y up, metres.
Files written to `out_dir`:
    demo_room.obj / demo_room.png   6 x 3 x 5 m closed room (inward normals) + table + 2 pillars (separate objects)
    demo_objects.glb                rug + box + sphere + cylinder + cone, one embedded PBR base-colour texture each
    demo_scene.json                 SceneSpec (paths relative to this file): room + the GLB placed twice
    demo_trajectory.json            legacy records (timestamp in ms, x y z, rx ry rz forward vector), 4 s @ 100 Hz
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from . import _meshes as mm
from . import _textures as tx

DEMO_SEED = 7
ROOM_MIN = (-3.0, 0.0, -2.5)
ROOM_MAX = (3.0, 3.0, 2.5)
TEX_SIZE = 2048
PX_PER_M = 160
GAP = 0.005                           # sub-volumes stay 5 mm off floor/ceiling (no coplanar faces, no shell overlap)
TABLE_CENTER_XZ = (1.5, -1.6)
TABLE_SIZE = (1.4, 0.75, 0.8)
PILLAR_CENTERS_XZ = [(-2.0, -1.7), (2.4, -0.3)]
PILLAR_WIDTH = 0.4
# (position, rotation_euler_deg, scale) of the two instances of demo_objects.glb in demo_scene.json
GLB_INSTANCES = [((-1.3, 0.01, 0.9), (0.0, 0.0, 0.0), 1.0), ((1.9, 0.01, 1.3), (0.0, 180.0, 0.0), 0.7)]
TRAJ_DURATION_S = 4.0
TRAJ_RATE_HZ = 100.0
EYE_HEIGHT = 1.5

FILES = ("demo_room.obj", "demo_room.png", "demo_objects.glb", "demo_scene.json", "demo_trajectory.json")
_STAMP = ".demo_stamp"

Rect = Tuple[int, int, int, int]      # x, y, w, h in atlas pixels (row 0 = top)


@dataclass(frozen=True)
class DemoScene:
    root: Path
    room_obj: Path
    room_texture: Path
    objects_glb: Path
    scene_spec: Path
    trajectory_json: Path


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream])


def _px(m: float) -> int:
    return int(round(m * PX_PER_M))


def _atlas_rects() -> Dict[str, Rect]:
    """Texel-density-uniform (160 px/m) atlas layout: room faces, table patches, pillar patches."""
    sx, sy, sz = (h - lo for lo, h in zip(ROOM_MIN, ROOM_MAX))
    tw, th, td = TABLE_SIZE
    r: Dict[str, Rect] = {}
    r["room-y"] = (16, 16, _px(sx), _px(sz))                        # floor
    r["room+y"] = (1008, 16, _px(sx), _px(sz))                      # ceiling
    r["room-z"] = (16, 848, _px(sx), _px(sy))
    r["room+z"] = (1008, 848, _px(sx), _px(sy))
    r["room-x"] = (16, 1360, _px(sz), _px(sy))
    r["room+x"] = (1008, 1360, _px(sz), _px(sy))
    r["table_top"] = (16, 1872, _px(tw), _px(td))
    r["table_long"] = (272, 1872, _px(tw), _px(th))
    r["table_short"] = (528, 1872, _px(td), _px(th))
    r["pillar_side"] = (1840, 1360, _px(PILLAR_WIDTH), _px(sy))
    r["pillar_cap"] = (1840, 1872, _px(PILLAR_WIDTH), _px(PILLAR_WIDTH))
    return r


def _uv_from_rect(rect: Rect, st: np.ndarray) -> np.ndarray:
    x, y, w, h = rect
    px, py = x + st[:, 0] * w, y + (1 - st[:, 1]) * h
    return np.stack([px / TEX_SIZE, 1 - py / TEX_SIZE], -1)


def _build_atlas(seed: int) -> Image.Image:
    rects = _atlas_rects()
    rng = _rng(seed, 0)
    wall_px = _px(0.9)
    styles = {
        "room-y": dict(c0=(96, 66, 40), c1=(196, 158, 108), tile_px=_px(0.5), n_lines=6, n_posters=1, n_checkers=2),
        "room+y": dict(c0=(200, 202, 210), c1=(250, 250, 246), tile_px=_px(1.0), n_lines=4, n_posters=0,
                       n_dark=3, n_light=3),
        "room-z": dict(c0=(30, 50, 90), c1=(140, 170, 210), wainscot_px=wall_px, n_posters=5),
        "room+z": dict(c0=(150, 90, 40), c1=(240, 200, 150), wainscot_px=wall_px, n_posters=4, n_checkers=2),
        "room-x": dict(c0=(40, 100, 70), c1=(170, 220, 180), wainscot_px=wall_px, n_posters=4, n_dark=2),
        "room+x": dict(c0=(90, 60, 110), c1=(210, 190, 225), wainscot_px=wall_px, n_posters=5, n_light=2),
        "table_top": dict(c0=(120, 80, 45), c1=(205, 155, 100), tile_px=32),
        "table_long": dict(c0=(120, 80, 45), c1=(205, 155, 100), tile_px=48),
        "table_short": dict(c0=(120, 80, 45), c1=(205, 155, 100), tile_px=48),
        "pillar_side": dict(c0=(70, 70, 80), c1=(200, 200, 206), tile_px=80),
        "pillar_cap": dict(c0=(70, 70, 80), c1=(200, 200, 206), tile_px=32),
    }
    small = dict(base_cell=32, n_lines=6, n_posters=0, n_checkers=0, n_dark=0, n_light=0, n_dots=30)
    atlas = np.full((TEX_SIZE, TEX_SIZE, 3), 96, np.uint8)
    pad = 6
    for name, (x, y, w, h) in rects.items():
        kw = dict(styles[name])
        if not name.startswith("room"):
            kw = {**small, **kw}
        img = np.asarray(tx.surface_texture(rng, w, h, **kw))
        atlas[y - pad:y + h + pad, x - pad:x + w + pad] = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return Image.fromarray(atlas)


def _room_meshes() -> List[mm.Mesh]:
    rects = _atlas_rects()
    room = mm.box("room_shell", ROOM_MIN, ROOM_MAX, lambda f, st: _uv_from_rect(rects["room" + f], st), inside=True)

    def table_uv(face: str, st: np.ndarray) -> np.ndarray:
        key = {"x": "table_short", "y": "table_top", "z": "table_long"}[face[1]]
        return _uv_from_rect(rects[key], st)

    def pillar_uv(face: str, st: np.ndarray) -> np.ndarray:
        return _uv_from_rect(rects["pillar_cap" if face[1] == "y" else "pillar_side"], st)

    cx, cz = TABLE_CENTER_XZ
    tw, th, td = TABLE_SIZE
    meshes = [room, mm.box("table", (cx - tw / 2, GAP, cz - td / 2), (cx + tw / 2, th, cz + td / 2), table_uv)]
    for i, (px, pz) in enumerate(PILLAR_CENTERS_XZ):
        h = PILLAR_WIDTH / 2
        lo, hi = (px - h, GAP, pz - h), (px + h, ROOM_MAX[1] - GAP, pz + h)
        meshes.append(mm.box(f"pillar_{i}", lo, hi, pillar_uv))
    return meshes


def _box_uv(face: str, st: np.ndarray) -> np.ndarray:
    return st


def _write_glb(path: Path, seed: int) -> None:
    """Rug (thin box, top at y = 0) + box, sphere, cylinder, cone standing on it; textures embedded as PNG."""
    import trimesh
    from trimesh.transformations import rotation_matrix, translation_matrix
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    rng = _rng(seed, 1)
    rug = mm.box("floor", (-1.0, -0.05, -1.0), (1.0, 0.0, 1.0), _box_uv)
    items = [   # name, mesh, transform, texture kind, texture size, base colour
        (rug, np.eye(4), "rug", 512, (255, 255, 255, 255)),
        (mm.box("box", (-0.3, -0.3, -0.3), (0.3, 0.3, 0.3), _box_uv),
         translation_matrix((-0.5, 0.3, -0.5)) @ rotation_matrix(np.radians(30), [0, 1, 0]),
         "checker", 256, (255, 200, 200, 255)),
        (mm.uv_sphere("sphere", 0.35), translation_matrix((0.5, 0.35, -0.4)), "bands", 256, (200, 220, 255, 255)),
        (mm.cylinder("cylinder", 0.2, 1.1), translation_matrix((-0.5, 0.0, 0.5)), "stripes", 256,
         (210, 255, 210, 255)),
        (mm.cone("cone", 0.3, 0.9), translation_matrix((0.5, 0.0, 0.5)), "diagonal", 256, (255, 240, 180, 255)),
    ]
    scene = trimesh.Scene()
    for mesh, transform, kind, size, color in items:
        material = PBRMaterial(name=mesh.name, baseColorTexture=tx.object_texture(kind, size, rng),
                               baseColorFactor=color, metallicFactor=0.0, roughnessFactor=1.0)
        tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_normals=mesh.normals,
                              visual=TextureVisuals(uv=mesh.uv, material=material), process=False)
        scene.add_geometry(tri, node_name=mesh.name, geom_name=mesh.name, transform=transform)
    Path(path).write_bytes(scene.export(file_type="glb"))


def make_trajectory_records(duration_s: float = TRAJ_DURATION_S, rate_hz: float = TRAJ_RATE_HZ) -> List[dict]:
    """Smooth ~0.8 m/s walk (cubic Bezier ground path at constant speed) at eye height with gentle look-around yaw,
    slight pitch and vertical bobbing; legacy records: timestamp in ms, x y z, rx ry rz = unit forward vector."""
    p = np.array([[-2.1, -0.75], [-1.0, -0.85], [0.0, -0.5], [0.9, 0.2]])
    u_dense = np.linspace(0.0, 1.0, 20001)[:, None]
    w = 1 - u_dense
    coef = np.hstack([w ** 3, 3 * u_dense * w ** 2, 3 * u_dense ** 2 * w, u_dense ** 3])
    dcoef = np.hstack([-3 * (1 - u_dense) ** 2, 3 * (1 - u_dense) ** 2 - 6 * u_dense * (1 - u_dense),
                       6 * u_dense * (1 - u_dense) - 3 * u_dense ** 2, 3 * u_dense ** 2])
    pos, tan = coef @ p, dcoef @ p
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    n = int(round(duration_s * rate_hz)) + 1
    t = np.arange(n) / rate_hz
    idx = np.interp(arc[-1] * t / duration_s, arc, np.arange(len(arc)))       # constant ground speed
    i0 = np.minimum(idx.astype(int), len(arc) - 2)
    a = (idx - i0)[:, None]
    xz = (1 - a) * pos[i0] + a * pos[i0 + 1]
    tg = (1 - a) * tan[i0] + a * tan[i0 + 1]
    heading = np.arctan2(tg[:, 1], tg[:, 0])
    yaw = heading + 0.25 + 0.30 * np.sin(2 * np.pi * t / 3.0 + 0.4)
    pitch = -0.04 + 0.06 * np.sin(2 * np.pi * t / 2.2)
    y = EYE_HEIGHT + 0.015 * np.sin(2 * np.pi * 1.6 * t) + 0.01 * np.sin(2 * np.pi * 0.4 * t)
    fwd = np.stack([np.cos(pitch) * np.cos(yaw), np.sin(pitch), np.cos(pitch) * np.sin(yaw)], -1)
    keys = ("timestamp", "x", "y", "z", "rx", "ry", "rz")
    vals = np.column_stack([t * 1000.0, xz[:, 0], y, xz[:, 1], fwd])
    return [{k: round(float(v), 6) for k, v in zip(keys, row)} for row in vals]


def _scene_spec() -> dict:
    objects: List[dict] = [{"path": "demo_room.obj", "texture": "demo_room.png", "position": [0.0, 0.0, 0.0],
                            "rotation_euler_deg": [0.0, 0.0, 0.0], "scale": 1.0}]
    for pos, rot, scale in GLB_INSTANCES:
        objects.append({"path": "demo_objects.glb", "texture": None, "position": list(pos),
                        "rotation_euler_deg": list(rot), "scale": scale})
    return {"objects": objects}


def _source_stamp(seed: int) -> str:
    h = hashlib.sha256(f"seed={seed}".encode())
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


def create_demo_scene(out_dir: Union[str, Path], seed: int = DEMO_SEED, force: bool = False) -> DemoScene:
    """Build (or reuse, when the files and the generator are unchanged) the demo scene in `out_dir`."""
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    scene = DemoScene(root, *(root / f for f in FILES))
    stamp, stamp_file = _source_stamp(seed), root / _STAMP
    if not force and all((root / f).exists() for f in FILES) and stamp_file.exists() \
            and stamp_file.read_text() == stamp:
        return scene

    _build_atlas(seed).save(scene.room_texture, format="PNG")
    mm.write_obj(scene.room_obj, _room_meshes(), "DVSim demo room: shell (inward normals), table, pillars")
    _write_glb(scene.objects_glb, seed)
    records = make_trajectory_records()
    scene.trajectory_json.write_text("[\n" + ",\n".join(json.dumps(r) for r in records) + "\n]\n")
    scene.scene_spec.write_text(json.dumps(_scene_spec(), indent=2) + "\n")
    stamp_file.write_text(stamp)
    return scene
