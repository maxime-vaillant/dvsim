"""Scene containers: triangles + material per drawable, in world units (assume metres), +Y up."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import trimesh

DEFAULT_BASE_COLOR = (0.8, 0.8, 0.8, 1.0)


def _default_base_color() -> np.ndarray:
    return np.array(DEFAULT_BASE_COLOR, dtype=np.float32)


@dataclass
class SceneObject:
    """One drawable: indexed triangles plus a material.

    ``vertices`` are in object space (float32, small local coordinates: glTF node transforms and far-from-origin
    offsets live in the float64 ``model_matrix``); the world position is ``model_matrix @ vertex``. ``uv`` uses the trimesh / OBJ convention (v = 0 at the *bottom* of the image,
    glTF is converted on load); ``texture`` is the plain image (row 0 = top). Colours (``colors``, ``base_color``)
    are sRGB-encoded floats in [0, 1]. Precedence: texture > vertex colours > base_color. Treat as immutable.
    """

    vertices: np.ndarray                     # (V, 3) float32
    faces: np.ndarray                        # (F, 3) uint32
    uv: Optional[np.ndarray] = None          # (V, 2) float32
    colors: Optional[np.ndarray] = None      # (V, 4) float32
    texture: Optional[np.ndarray] = None     # (h, w, 3) uint8 RGB
    base_color: np.ndarray = field(default_factory=_default_base_color)  # (4,) float32
    model_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))  # (4, 4) float64
    name: str = ""

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.uint32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3), got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3) triangles, got {self.faces.shape}")
        if len(self.faces) and int(self.faces.max()) >= len(self.vertices):
            raise ValueError("face index out of range")
        n = len(self.vertices)
        if not np.isfinite(self.vertices).all():
            raise ValueError(f"object {self.name!r}: vertices must be finite (found NaN or inf)")
        if self.uv is not None:
            self.uv = np.ascontiguousarray(self.uv, dtype=np.float32)
            if self.uv.shape != (n, 2):
                raise ValueError(f"uv must be ({n}, 2), got {self.uv.shape}")
            if not np.isfinite(self.uv).all():
                raise ValueError(f"object {self.name!r}: uv coordinates must be finite (found NaN or inf)")
        if self.colors is not None:
            self.colors = np.ascontiguousarray(self.colors, dtype=np.float32)
            if self.colors.shape != (n, 4):
                raise ValueError(f"colors must be ({n}, 4), got {self.colors.shape}")
        if self.texture is not None:
            if self.uv is None:
                raise ValueError("a texture needs uv coordinates")
            if self.texture.dtype != np.uint8 or self.texture.ndim != 3 or self.texture.shape[2] != 3:
                raise ValueError(f"texture must be (h, w, 3) uint8, got {self.texture.dtype} {self.texture.shape}")
            self.texture = np.ascontiguousarray(self.texture)
        self.base_color = np.asarray(self.base_color, dtype=np.float32).reshape(4)
        self.model_matrix = np.asarray(self.model_matrix, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(self.model_matrix).all():
            raise ValueError(f"object {self.name!r}: model_matrix must be finite")

    def world_vertices(self) -> np.ndarray:
        """(V, 3) float64 vertices after the model matrix."""
        m = self.model_matrix
        return self.vertices.astype(np.float64) @ m[:3, :3].T + m[:3, 3]


@dataclass
class Scene:
    objects: List[SceneObject] = field(default_factory=list)

    @property
    def bounds(self) -> np.ndarray:
        """(2, 3) float64 world AABB (min, max) after the model matrices."""
        if not self.objects:
            return np.zeros((2, 3))
        lo = np.min([o.world_vertices().min(axis=0) for o in self.objects if len(o.vertices)], axis=0)
        hi = np.max([o.world_vertices().max(axis=0) for o in self.objects if len(o.vertices)], axis=0)
        return np.stack([lo, hi])

    @property
    def diagonal(self) -> float:
        lo, hi = self.bounds
        return float(np.linalg.norm(hi - lo))

    @property
    def num_triangles(self) -> int:
        return int(sum(len(o.faces) for o in self.objects))

    def to_trimesh(self) -> trimesh.Trimesh:
        """All objects merged into one world-space, geometry-only mesh (for trajectory generation)."""
        if not self.objects:
            return trimesh.Trimesh()
        vertices, faces, offset = [], [], 0
        for o in self.objects:
            vertices.append(o.world_vertices())
            faces.append(o.faces.astype(np.int64) + offset)
            offset += len(o.vertices)
        return trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False)
