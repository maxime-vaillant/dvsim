"""Closed procedural meshes with UVs and analytic normals (numpy only).

UV convention is the OBJ / trimesh one: (0, 0) is the BOTTOM-left of the image (v grows upwards).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# face name -> (outward normal, right, up) for a viewer standing outside the box; right x up == normal
BOX_FACES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]] = {
    "+x": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "-x": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "+y": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    "-y": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "+z": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "-z": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
}
_ST = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)   # bottom-left, bottom-right, top-right, top-left

UVFunc = Callable[[str, np.ndarray], np.ndarray]                 # (face name, (4, 2) face-local st) -> (4, 2) uv


@dataclass
class Mesh:
    name: str
    vertices: np.ndarray    # (N, 3) float64
    uv: np.ndarray          # (N, 2) float64
    normals: np.ndarray     # (N, 3) float64
    faces: np.ndarray       # (M, 3) int64, wound CCW around `normals`

    def signed_volume(self) -> float:
        v = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def _finish(name: str, v: np.ndarray, uv: np.ndarray, vn: np.ndarray, f: np.ndarray) -> Mesh:
    """Drop degenerate triangles, wind the rest to agree with `vn`, drop unreferenced vertices."""
    f = np.asarray(f, np.int64)
    g = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    f = f[np.linalg.norm(g, axis=1) > 1e-12]
    g = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = np.einsum("ij,ij->i", g, vn[f].sum(axis=1)) < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    used, inv = np.unique(f, return_inverse=True)
    return Mesh(name, v[used], uv[used], vn[used], inv.reshape(-1, 3).astype(np.int64))


def _grid_faces(rows: int, cols: int) -> np.ndarray:
    i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (i * cols + j).ravel()
    b, c = a + 1, a + cols
    d = c + 1
    return np.concatenate([np.stack([a, c, b], -1), np.stack([b, c, d], -1)])


def _merge(name: str, parts: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Mesh:
    offs = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    v, uv, vn = (np.concatenate([p[k] for p in parts]) for k in range(3))
    f = np.concatenate([p[3] + o for p, o in zip(parts, offs)])
    return _finish(name, v, uv, vn, f)


def box(name: str, lo: Sequence[float], hi: Sequence[float], uv_of: UVFunc, inside: bool = False) -> Mesh:
    """Axis-aligned box, 4 vertices per face (flat normals). `inside=True` -> normals/winding point inwards (a room)."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    c, half = (lo + hi) / 2, (hi - lo) / 2
    sign = -1.0 if inside else 1.0
    parts = []
    for fname, (n, r, u) in BOX_FACES.items():
        n, r, u = (np.asarray(a, np.float64) for a in (n, r, u))
        hr, hu, hn = np.abs(r) @ half, np.abs(u) @ half, np.abs(n) @ half
        centre = c + n * hn
        v = np.stack([centre + (2 * s - 1) * hr * r + (2 * t - 1) * hu * u for s, t in _ST])
        st = _ST.copy()
        if inside:
            st[:, 0] = 1 - st[:, 0]     # a viewer inside sees the face mirrored w.r.t. one outside
        parts.append((v, uv_of(fname, st), np.tile(sign * n, (4, 1)), np.array([[0, 1, 2], [0, 2, 3]])))
    return _merge(name, parts)


def uv_sphere(name: str, radius: float, n_lon: int = 48, n_lat: int = 24) -> Mesh:
    """Equirectangular UVs, duplicated seam column (u = 0 and 1), pole rows collapsed."""
    th = np.linspace(0, np.pi, n_lat + 1)[:, None]
    ph = np.linspace(0, 2 * np.pi, n_lon + 1)[None, :]
    nrm = np.stack([np.sin(th) * np.cos(ph), np.cos(th) * np.ones_like(ph), np.sin(th) * np.sin(ph)], -1)
    shape = nrm.shape[:2]
    uv = np.stack([np.broadcast_to(ph / (2 * np.pi), shape), np.broadcast_to(1 - th / np.pi, shape)], -1)
    return _finish(name, (nrm * radius).reshape(-1, 3), uv.reshape(-1, 2), nrm.reshape(-1, 3),
                   _grid_faces(n_lat + 1, n_lon + 1))


def _cap(radius: float, y: float, ny: float, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(ph), np.full(n, y), radius * np.sin(ph)], -1)
    v = np.concatenate([[[0.0, y, 0.0]], ring])
    uv = np.concatenate([[[0.5, 0.5]], 0.5 + 0.5 * np.stack([np.cos(ph), np.sin(ph)], -1)])
    j = np.arange(n)
    f = np.stack([np.zeros(n, np.int64), 1 + j, 1 + (j + 1) % n], -1)
    return v, uv, np.tile([0.0, ny, 0.0], (n + 1, 1)), f


def _revolved(radius_top: float, radius_bottom: float, height: float, n: int) -> Tuple[np.ndarray, ...]:
    """Lateral surface between y = 0 (bottom ring) and y = height (top ring); rows: top, bottom."""
    ph = np.linspace(0, 2 * np.pi, n + 1)
    cs, sn = np.cos(ph), np.sin(ph)
    top = np.stack([radius_top * cs, np.full(n + 1, height), radius_top * sn], -1)
    bot = np.stack([radius_bottom * cs, np.zeros(n + 1), radius_bottom * sn], -1)
    slope = (radius_bottom - radius_top) / height     # outward normal ~ (cos, slope, sin)
    nrm = np.stack([cs, np.full(n + 1, slope), sn], -1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    u = ph / (2 * np.pi)
    uv = np.concatenate([np.stack([u, np.ones(n + 1)], -1), np.stack([u, np.zeros(n + 1)], -1)])
    return np.concatenate([top, bot]), uv, np.concatenate([nrm, nrm]), _grid_faces(2, n + 1)


def cylinder(name: str, radius: float, height: float, n: int = 48) -> Mesh:
    """Base on y = 0, top on y = height; closed with two caps."""
    parts = [_revolved(radius, radius, height, n), _cap(radius, 0.0, -1.0, n), _cap(radius, height, 1.0, n)]
    return _merge(name, parts)


def cone(name: str, radius: float, height: float, n: int = 48) -> Mesh:
    """Base (radius) on y = 0, apex on y = height; closed with a base cap."""
    return _merge(name, [_revolved(0.0, radius, height, n), _cap(radius, 0.0, -1.0, n)])


def translated(mesh: Mesh, offset: Sequence[float]) -> Mesh:
    return Mesh(mesh.name, mesh.vertices + np.asarray(offset, np.float64), mesh.uv, mesh.normals, mesh.faces)


def write_obj(path: Path, meshes: List[Mesh], comment: str) -> None:
    """Write meshes as separate `o` objects (global 1-based indices; one v/vt/vn per vertex). Deterministic bytes."""
    lines = [f"# {comment}"]
    off = 1
    for m in meshes:
        lines.append(f"o {m.name}")
        v, n = np.round(m.vertices, 6) + 0.0, np.round(m.normals, 6) + 0.0
        uv = np.round(m.uv, 6) + 0.0
        lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v]
        lines += [f"vt {a:.6f} {b:.6f}" for a, b in uv]
        lines += [f"vn {a:.6f} {b:.6f} {c:.6f}" for a, b, c in n]
        lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in (m.faces + off)]
        off += len(m.vertices)
    Path(path).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
