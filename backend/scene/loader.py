"""Scene loading on top of trimesh: .obj (+ .mtl), .glb/.gltf, .ply, or a SceneSpec (dict / .json).

SceneSpec = {"objects": [{"path", "texture", "position", "rotation_euler_deg", "scale", "base_color", "name"}],
             "texture": <default texture for objects whose file has none>}

Untrusted input (uploaded scenes): pass ``root`` and every file that gets opened is confined to that directory.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from trimesh.resolvers import FilePathResolver
from trimesh.visual import color as tm_color
from trimesh.visual.material import PBRMaterial

from backend.scene.scene import Scene, SceneObject
from backend.scene.textures import (
    DEFAULT_MAX_TEXTURE_PIXELS,
    DEFAULT_MAX_TEXTURE_SIZE,
    image_to_array,
    load_texture,
    pil_limit_lifted,
    srgb_encode,
)

DEFAULT_MAX_TRIANGLES = 20_000_000

_OBJECT_KEYS = {"path", "texture", "position", "rotation_euler_deg", "scale", "base_color", "name"}
_SPEC_KEYS = {"objects", "texture"}
_TRIMESH_DEFAULT_RGBA = (100, 100, 100, 255)  # colour of trimesh's placeholder material
_RECENTER_ABOVE = 1000.0                       # |centre| beyond this (scene units) moves into the model matrix


@dataclass(frozen=True)
class _Limits:
    max_texture_size: int
    max_texture_pixels: int
    max_triangles: int


def load_scene(
    spec: Union[str, os.PathLike, dict, Scene],
    *,
    root: Optional[Union[str, os.PathLike]] = None,
    max_texture_size: int = DEFAULT_MAX_TEXTURE_SIZE,
    max_texture_pixels: int = DEFAULT_MAX_TEXTURE_PIXELS,
    max_triangles: int = DEFAULT_MAX_TRIANGLES,
) -> Scene:
    """Load a scene from a model path, a SceneSpec dict or a ``.json`` SceneSpec file.

    Relative paths inside a JSON spec resolve against the JSON file's directory, otherwise against the CWD
    (or ``root``, see below). Textures whose longest side exceeds ``max_texture_size`` are downscaled; images over
    ``max_texture_pixels`` pixels and scenes over ``max_triangles`` triangles are refused (ValueError), as are
    non-finite vertices / UVs.

    ``root`` (untrusted input): every path that is opened (object paths, textures, the JSON spec, and the
    .mtl / texture / glTF buffer / image references that trimesh follows) must resolve inside ``root``
    (symlinks followed, no ``~`` expansion; relative paths are taken relative to ``root``), else ValueError.
    Without ``root`` paths are unrestricted (trusted local use).
    """
    if isinstance(spec, Scene):
        return spec
    limits = _Limits(max_texture_size, max_texture_pixels, max_triangles)
    root_path = _resolve_root(root)
    base: Optional[Path] = None
    if isinstance(spec, (str, os.PathLike)):
        path = _resolve(spec, None, root_path)
        if path.suffix.lower() == ".json":
            if not path.is_file():
                raise FileNotFoundError(f"scene spec not found: {path}")
            spec, base = json.loads(path.read_text()), path.parent
        else:
            spec = {"objects": [{"path": str(path)}]}
    if not isinstance(spec, dict):
        raise TypeError(f"scene spec must be a path or a dict, got {type(spec).__name__}")
    if "objects" not in spec and "path" in spec:  # a bare object entry
        spec = {"objects": [spec]}
    _check_keys(spec, _SPEC_KEYS, "scene spec")

    files: Dict[Path, List[SceneObject]] = {}
    textures: Dict[Path, np.ndarray] = {}

    def texture_from(value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        path = _resolve(value, base, root_path)
        if path not in textures:
            textures[path] = load_texture(path, limits.max_texture_size, limits.max_texture_pixels)
        return textures[path]

    default_texture = texture_from(spec["texture"]) if spec.get("texture") is not None else None
    objects: List[SceneObject] = []
    triangles = 0
    for index, entry in enumerate(spec.get("objects") or []):
        _check_keys(entry, _OBJECT_KEYS, f"objects[{index}]")
        path = _resolve(entry["path"], base, root_path)
        if path not in files:
            files[path] = _load_file(path, limits, root_path)
        model = object_matrix(entry.get("position"), entry.get("rotation_euler_deg"), entry.get("scale"))
        override = texture_from(entry["texture"]) if entry.get("texture") is not None else None
        base_color = _rgba(entry["base_color"]) if entry.get("base_color") is not None else None
        for piece in files[path]:
            triangles += len(piece.faces)
            if triangles > limits.max_triangles:
                raise ValueError(f"scene has more than {limits.max_triangles} triangles")
            changes: Dict[str, Any] = {"model_matrix": model @ piece.model_matrix}
            if override is not None or (default_texture is not None and piece.texture is None and piece.uv is not None):
                if piece.uv is None:
                    raise ValueError(f"{path.name}: cannot apply a texture, the mesh has no UV coordinates")
                changes["texture"] = override if override is not None else default_texture
            if base_color is not None:
                changes["base_color"] = base_color
            if entry.get("name"):
                changes["name"] = f"{entry['name']}/{piece.name}" if len(files[path]) > 1 else entry["name"]
            objects.append(replace(piece, **changes))
    if not objects:
        raise ValueError("scene has no triangle geometry")
    return Scene(objects)


def object_matrix(position=None, rotation_euler_deg=None, scale=None) -> np.ndarray:
    """(4,4) = T(position) @ Rx @ Ry @ Rz (degrees, three.js 'XYZ' Euler order) @ S(scale)."""
    m = np.eye(4)
    rx, ry, rz = np.radians(np.zeros(3) if rotation_euler_deg is None else np.asarray(rotation_euler_deg, float))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    s = np.ones(3) if scale is None else np.broadcast_to(np.asarray(scale, dtype=float), (3,))
    m[:3, :3] = (mx @ my @ mz) * s  # scale acts first (columns)
    if position is not None:
        m[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return m


# ---- path handling --------------------------------------------------------------------------------------------
def _check_keys(d: dict, allowed: set, where: str) -> None:
    if not isinstance(d, dict):
        raise TypeError(f"{where} must be a dict")
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"unknown {where} keys: {sorted(unknown)}")
    if where.startswith("objects[") and "path" not in d:
        raise ValueError(f"{where} needs a 'path'")


def _resolve_root(root) -> Optional[Path]:
    if root is None:
        return None
    path = Path(root).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"scene root is not a directory: {path}")
    return path


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:  # different drives
        return False


def _resolve(path, base: Optional[Path], root: Optional[Path]) -> Path:
    """Path -> absolute path. With ``root``: relative to ``base`` (or root), symlinks resolved, confined to root."""
    p = Path(path) if root is not None else Path(path).expanduser()
    if not p.is_absolute():
        anchor = base if base is not None else root
        if anchor is not None:
            p = anchor / p
    if root is None:
        return p
    real = p.resolve()
    if not _inside(real, root):
        raise ValueError(f"path {str(path)!r} is outside the scene root {root}")
    return real


class _ConfinedResolver(FilePathResolver):
    """trimesh resolver that only serves files inside ``root``.

    trimesh < 5 lets absolute names and ``..`` escape the directory of the loaded file (trimesh 5 confines them
    by itself); this class does not depend on the version. Escapes raise ValueError and are recorded in
    ``violations`` because trimesh swallows exceptions while parsing .mtl files.
    """

    def __init__(self, source, root: Path, violations: Optional[List[str]] = None):
        super().__init__(str(source))
        self._dir = Path(self.parent)
        self._root = root
        self.violations: List[str] = violations if violations is not None else []

    def _refuse(self, name, record: bool):
        if record:
            self.violations.append(str(name))
        raise ValueError(f"reference {str(name)!r} is outside the scene root {self._root}")

    def _locate(self, name, record: bool = True) -> Path:
        raw = str(name).strip()
        posix = raw.replace("\\", "/")
        if not raw or "\0" in raw or os.path.isabs(raw) or posix.startswith("/") or (len(posix) > 1 and posix[1] == ":"):
            self._refuse(name, record)
        for candidate in (posix, posix.rsplit("/", 1)[-1]):  # trimesh also falls back to the bare file name
            target = (self._dir / candidate).resolve()
            if not _inside(target, self._root):
                self._refuse(name, record)
            if target.is_file():
                return target
        raise FileNotFoundError(str(name))

    def get(self, name: str) -> bytes:
        return self._locate(name).read_bytes()

    def exists(self, name: str) -> bool:
        try:
            self._locate(name, record=False)
            return True
        except (ValueError, FileNotFoundError):
            return False

    def write(self, name, data) -> None:
        raise PermissionError("scene resolvers are read-only")

    def namespaced(self, namespace: str) -> "_ConfinedResolver":
        target = (self._dir / str(namespace)).resolve()
        if not _inside(target, self._root):
            self._refuse(namespace, True)
        return _ConfinedResolver(target, self._root, self.violations)


# ---- files -> SceneObjects ---------------------------------------------------------------------------------------
@dataclass
class _Geometry:
    vertices: np.ndarray                     # float32, relative to ``origin``
    faces: np.ndarray
    origin: np.ndarray                       # float64 (3,)
    uv: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    texture: Optional[np.ndarray] = None
    base_color: Optional[np.ndarray] = None


def _load_file(path: Path, limits: _Limits, root: Optional[Path]) -> List[SceneObject]:
    if not path.is_file():
        raise FileNotFoundError(f"scene file not found: {path}")
    resolver = _ConfinedResolver(path.parent, root) if root is not None else None

    def escaped() -> ValueError:
        return ValueError(f"{path.name}: references files outside the scene root {root}: {sorted(set(resolver.violations))}")

    try:
        with pil_limit_lifted():  # trimesh only opens (header) the images; the size cap is applied in _material
            loaded = trimesh.load(str(path), force="scene", process=False, **({"resolver": resolver} if resolver else {}))
    except Exception as exc:
        if resolver is not None and resolver.violations:
            raise escaped() from exc
        raise ValueError(f"cannot load {path}: {exc}") from exc
    if resolver is not None and resolver.violations:
        raise escaped()

    nodes = sorted(loaded.graph.nodes_geometry)
    count = 0
    for node in nodes:
        geom = loaded.geometry[loaded.graph[node][1]]
        count += len(geom.faces) if isinstance(geom, trimesh.Trimesh) else 0
    if count > limits.max_triangles:
        raise ValueError(f"{path.name}: {count} triangles, over the limit of {limits.max_triangles}")

    images: Dict[Any, np.ndarray] = {}
    geometries: Dict[str, Optional[_Geometry]] = {}
    pieces: List[SceneObject] = []
    for node in nodes:
        transform, geom_name = loaded.graph[node]
        name = path.stem if len(nodes) == 1 else f"{path.stem}/{node}"
        if geom_name not in geometries:
            try:
                geometries[geom_name] = _convert(loaded.geometry[geom_name], images, limits)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc
        geom = geometries[geom_name]
        if geom is None:
            continue
        model = np.asarray(transform, dtype=np.float64).copy()
        model[:3, 3] += model[:3, :3] @ geom.origin  # node transform @ translate(origin), kept in float64
        kwargs = {} if geom.base_color is None else {"base_color": geom.base_color}
        pieces.append(
            SceneObject(vertices=geom.vertices, faces=geom.faces, uv=geom.uv, colors=geom.colors,
                        texture=geom.texture, model_matrix=model, name=name, **kwargs)
        )
    if not pieces:
        raise ValueError(f"{path}: no triangle mesh found (point clouds and paths are not supported)")
    return pieces


def _convert(g, images: Dict[Any, np.ndarray], limits: _Limits) -> Optional[_Geometry]:
    if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0 or len(g.vertices) == 0:
        return None
    vertices, faces = np.asarray(g.vertices, dtype=np.float64), np.asarray(g.faces)
    uv = colors = texture = base_color = None
    visual = g.visual
    kind = visual.kind
    if kind == "texture":
        if getattr(visual, "uv", None) is not None and len(visual.uv) == len(vertices):
            uv = np.asarray(visual.uv, dtype=np.float32)
        texture, base_color = _material(getattr(visual, "material", None), images, limits)
        if uv is None:
            texture = None
    elif kind == "vertex":
        colors = tm_color.to_float(visual.vertex_colors).astype(np.float32)
    elif kind == "face":  # unweld so each triangle keeps its own colour
        vertices = vertices[faces].reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        colors = np.repeat(tm_color.to_float(visual.face_colors).astype(np.float32), 3, axis=0)
    # georeferenced data (UTM-like offsets): float32 vertices would quantise to millimetres, so keep the offset
    # in the float64 model matrix and store small local coordinates
    centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    origin = centre if np.abs(centre).max() > _RECENTER_ABOVE else np.zeros(3)
    return _Geometry((vertices - origin).astype(np.float32), faces, origin, uv, colors, texture, base_color)


def _material(mat, images: Dict[Any, np.ndarray], limits: _Limits) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Material -> (texture array, sRGB base colour). glTF baseColorFactor is linear and is multiplied into the texture."""
    if mat is None:
        return None, None
    pbr = isinstance(mat, PBRMaterial)
    if pbr:
        image = mat.baseColorTexture
        factor = None
        if mat.baseColorFactor is not None:
            f = np.asarray(tm_color.to_float(np.asarray(mat.baseColorFactor)), dtype=np.float64).reshape(-1)
            factor = np.append(srgb_encode(f[:3]), f[3] if f.size > 3 else 1.0).astype(np.float32)
    else:
        image = getattr(mat, "image", None)
        factor = None
        if "kd" in getattr(mat, "kwargs", {}):
            factor = tm_color.to_float(np.asarray(mat.diffuse)).astype(np.float32)
    if image is not None and not pbr and image.size[0] * image.size[1] <= 4:
        pixels = np.asarray(image.convert("RGBA")).reshape(-1, 4)
        if (pixels == pixels[0]).all():  # trimesh's 2x2 solid-colour stand-in for "no texture"
            image = None
            if factor is None and tuple(pixels[0]) != _TRIMESH_DEFAULT_RGBA:
                factor = pixels[0].astype(np.float32) / 255.0
    if image is None:
        return None, factor
    if id(image) not in images:
        images[id(image)] = image_to_array(image, limits.max_texture_size, limits.max_texture_pixels)  # size checked first
    texture = images[id(image)]
    if pbr and factor is not None and np.any(factor[:3] < 254.5 / 255):
        key = (id(image), tuple(np.round(factor[:3], 4)))
        if key not in images:
            images[key] = _tint(texture, factor[:3])
        texture = images[key]
    return texture, None


def _tint(texture: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(texture)
    for i in range(0, len(texture), 512):
        out[i:i + 512] = np.clip(texture[i:i + 512] * rgb, 0, 255).astype(np.uint8)
    return out


def _rgba(value) -> np.ndarray:
    c = np.asarray(value, dtype=np.float32).reshape(-1)
    if c.size == 3:
        c = np.append(c, np.float32(1.0))
    if c.size != 4:
        raise ValueError(f"base_color must have 3 or 4 components, got {c.size}")
    return c
