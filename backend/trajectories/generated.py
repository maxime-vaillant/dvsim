"""Free-space voxel grid (built once per mesh) and the base class of the random trajectory generators.

The grid stores a signed clearance field: distance (scene units) to the nearest surface voxel, positive in free space,
negative in solid / outside space. Free space = the volume enclosed by the mesh (what ``trimesh.contains`` used to
report) with nested shells alternating solid / free. All queries are vectorised and use trilinear interpolation of the
field; the field is accurate to about one voxel (``pitch``) because it measures the distance to voxel centres.
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import trimesh
from scipy import ndimage

from backend.trajectories.trajectory import WaypointTrajectory

_PAD = 2  # empty voxels around the mesh bounds, so the outside is one connected region
_RAYS = np.array([[0.5773, 0.3781, 0.7231], [-0.3419, 0.8517, 0.3960], [0.2718, -0.4142, -0.8683]])
_RAYS /= np.linalg.norm(_RAYS, axis=1, keepdims=True)


def as_rng(rng=None) -> np.random.Generator:
    """``np.random.Generator`` (returned as is), an integer seed, or None (fresh entropy)."""
    return np.random.default_rng(rng)


def _as_trimesh(obj) -> trimesh.Trimesh:
    if hasattr(obj, "to_trimesh"):  # backend.scene.Scene
        obj = obj.to_trimesh()
    if isinstance(obj, trimesh.Scene):
        obj = obj.dump(concatenate=True)
    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"expected a trimesh.Trimesh or an object with to_trimesh(), got {type(obj).__name__}")
    if len(obj.faces) == 0:
        raise ValueError("the mesh has no faces")
    return obj


def _num_voxels(extent: np.ndarray, pitch: float) -> int:
    return int(np.prod(np.floor(extent / pitch) + 1 + 2 * _PAD))


def _ragged(counts: np.ndarray):
    """(owner, local index) of every element of ``len(counts)`` ranges of the given lengths."""
    owner = np.repeat(np.arange(len(counts)), counts)
    local = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    return owner, local


def _mark(shell: np.ndarray, pts: np.ndarray, origin: np.ndarray, pitch: float) -> None:
    idx = np.rint((pts - origin) / pitch).astype(np.int64)
    idx = np.clip(idx, 0, np.array(shell.shape) - 1)
    shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True


def _rasterize(tri: np.ndarray, origin: np.ndarray, pitch: float, shape, chunk: int = 3_000_000) -> np.ndarray:
    """Boolean (26-connected) shell of the voxels hit by point samples spaced <= pitch on every triangle.

    Each triangle is sampled on the lattice spanned by the two edges meeting at its largest angle (spacing <= pitch/2
    along the edges), plus its three edges; cost is bounded by area/pitch^2 + perimeter/pitch even for slivers.
    """
    h = 0.5 * pitch
    f = np.arange(len(tri))
    edge_len = np.linalg.norm(tri[:, [2, 0, 1]] - tri[:, [1, 2, 0]], axis=2)  # edge i is opposite vertex i
    a = np.argmax(edge_len, axis=1)
    corner = tri[f, a]
    e1 = tri[f, (a + 1) % 3] - corner
    e2 = tri[f, (a + 2) % 3] - corner
    e3 = e2 - e1

    def n_of(v):
        return np.maximum(np.ceil(np.linalg.norm(v, axis=1) / h), 1).astype(np.int64)

    n1, n2, n3 = n_of(e1), n_of(e2), n_of(e3)
    cum = np.cumsum((n1 + 1) * (n2 + 1) + n1 + n2 + n3 + 3)
    shell = np.zeros(shape, dtype=bool)
    start = 0
    while start < len(tri):
        base = cum[start - 1] if start else 0
        stop = max(int(np.searchsorted(cum, base + chunk, side="right")), start + 1)
        s = slice(start, stop)
        start = stop
        c, v1, v2, v3, a1, a2, a3 = corner[s], e1[s], e2[s], e3[s], n1[s], n2[s], n3[s]
        own, loc = _ragged((a1 + 1) * (a2 + 1))
        u, v = loc // (a2 + 1)[own] / a1[own], loc % (a2 + 1)[own] / a2[own]
        keep = u + v <= 1.0 + 1e-9
        own, u, v = own[keep], u[keep], v[keep]
        _mark(shell, c[own] + u[:, None] * v1[own] + v[:, None] * v2[own], origin, pitch)
        for p0, vec, cnt in ((c, v1, a1), (c, v2, a2), (c + v1, v3, a3)):
            own, loc = _ragged(cnt + 1)
            _mark(shell, p0[own] + (loc / cnt[own])[:, None] * vec[own], origin, pitch)
    return shell


def _crossings(tri, lo, hi, origin, direction) -> int:
    """Number of triangles crossed by the ray origin + t * direction, t > 0 (slab prefilter + Moller-Trumbore)."""
    inv = 1.0 / direction
    t1, t2 = (lo - origin) * inv, (hi - origin) * inv
    near, far = np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)
    cand = np.flatnonzero((far >= near) & (far > 0))
    v0 = tri[cand, 0]
    e1, e2 = tri[cand, 1] - v0, tri[cand, 2] - v0
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = origin - v0
    u = np.einsum("ij,ij->i", tv, p) * inv_det
    q = np.cross(tv, e1)
    v = (q @ direction) * inv_det
    t = np.einsum("ij,ij->i", e2, q) * inv_det
    return int(np.count_nonzero(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)))


class FreeSpaceGrid:
    """Voxel free-space / clearance grid of a mesh, built once and queried in bulk.

    ``mesh``: a ``trimesh.Trimesh`` or anything with ``to_trimesh()`` (``backend.scene.Scene``).
    ``pitch``: voxel size in scene units; default ``min_clearance / 3``, enlarged until the grid has at most
    ``max_voxels`` voxels (an explicit pitch is used as given, up to 4 * ``max_voxels``). At roughly 80 bytes
    per voxel, the default (8e6) is about 0.6 GB worst case; raise it for large scenes on a machine with more RAM.

    Attributes: ``pitch``, ``origin`` (position of voxel (0,0,0)), ``shape``, ``field`` (float32 signed clearance),
    ``bounds`` ((2,3) mesh AABB) and ``enclosed`` (False if the mesh encloses no volume; then every non-surface voxel
    inside ``bounds`` is free and a warning was issued).
    """

    def __init__(self, mesh, pitch: Optional[float] = None, min_clearance: float = 0.3, max_voxels: int = 8_000_000):
        mesh = _as_trimesh(mesh)
        tri = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)]
        lo, hi = tri.min(axis=(0, 1)), tri.max(axis=(0, 1))
        extent = hi - lo
        if pitch is None:
            pitch = min_clearance / 3.0
            while _num_voxels(extent, pitch) > max_voxels:
                pitch *= 1.05
            if pitch > min_clearance:
                warnings.warn(f"FreeSpaceGrid: pitch {pitch:.3g} exceeds min_clearance {min_clearance:.3g} "
                              f"(scene too large for {max_voxels} voxels); clearances are coarse", stacklevel=2)
        elif not pitch > 0:
            raise ValueError(f"pitch must be positive, got {pitch}")
        elif _num_voxels(extent, pitch) > 4 * max_voxels:
            raise ValueError(f"pitch {pitch} gives {_num_voxels(extent, pitch)} voxels (> 4 * max_voxels); increase it")
        self.pitch = float(pitch)
        self.bounds = np.stack([lo, hi])
        self.origin = lo - _PAD * self.pitch
        self.shape = tuple(int(n) for n in np.floor(extent / self.pitch) + 1 + 2 * _PAD)
        self.min_clearance = float(min_clearance)
        self._build(tri)

    def _build(self, tri: np.ndarray) -> None:
        shell = _rasterize(tri, self.origin, self.pitch, self.shape)
        labels, n = ndimage.label(~shell)
        edt = ndimage.distance_transform_edt(~shell)
        free = self._enclosed_free(tri, labels, n, edt)
        self.enclosed = free is not None
        if free is None:
            warnings.warn(
                "FreeSpaceGrid: the mesh does not enclose a volume (open or leaky surface); treating every "
                "non-surface voxel inside its bounding box as free", stacklevel=3)
            free = np.zeros(self.shape, dtype=bool)
            free[_PAD:-_PAD, _PAD:-_PAD, _PAD:-_PAD] = ~shell[_PAD:-_PAD, _PAD:-_PAD, _PAD:-_PAD]
        del labels, shell
        edt *= self.pitch
        self.field = edt.astype(np.float32)
        del edt
        self.field[~free] *= -1.0

    def _enclosed_free(self, tri, labels, n, edt, max_components: int = 64, min_depth_vox: float = 1.5):
        """Boolean mask of the free voxels (background regions with an odd number of shells around them), or None."""
        outside = labels[0, 0, 0]
        idx = np.arange(1, n + 1)
        depth = np.asarray(ndimage.maximum(edt, labels, idx)) if n else np.zeros(0)
        cand = idx[(depth >= min_depth_vox) & (idx != outside)]
        if len(cand) == 0:
            return None
        cand = cand[np.argsort(-depth[cand - 1])[:max_components]]
        lo, hi = tri.min(axis=1), tri.max(axis=1)
        table = np.zeros(n + 1, dtype=bool)
        for label, pos in zip(cand, ndimage.maximum_position(edt, labels, cand)):
            point = self.origin + self.pitch * np.asarray(pos)
            table[label] = sum(_crossings(tri, lo, hi, point, d) % 2 for d in _RAYS) >= 2
        return table[labels] if table.any() else None

    # ---- queries ----------------------------------------------------------
    def clearance_at(self, points) -> np.ndarray:
        """(M,) trilinearly interpolated signed clearance (> 0 free); -inf outside the grid."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        g = (p - self.origin) / self.pitch
        top = np.array(self.shape, dtype=np.float64) - 1.0
        inside = np.all((g >= 0.0) & (g <= top), axis=1)
        g = np.clip(np.nan_to_num(g), 0.0, top)
        i0 = np.minimum(np.floor(g).astype(np.intp), np.array(self.shape) - 2)
        f = g - i0
        flat = self.field.reshape(-1)
        ny, nz = self.shape[1], self.shape[2]
        base = (i0[:, 0] * ny + i0[:, 1]) * nz + i0[:, 2]
        out = np.zeros(len(p))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    out += w * flat[base + (dx * ny + dy) * nz + dz]
        return np.where(inside, out, -np.inf)

    def is_free(self, points, clearance: float) -> np.ndarray:
        """(M,) bool: interpolated clearance >= ``clearance`` (points outside the grid are not free)."""
        return self.clearance_at(points) >= clearance

    def segments_free(self, p1, p2, clearance: float, n_samples: Optional[int] = None) -> np.ndarray:
        """(M,) bool: every sample on the segment p1[i] -> p2[i] (endpoints included) has the clearance.

        Default sampling is ``n_samples`` = enough for a spacing <= pitch (finer when clearance < pitch / 2),
        the same count for all segments.
        """
        p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 3)
        p2 = np.asarray(p2, dtype=np.float64).reshape(-1, 3)
        m = len(p1)
        if m == 0:
            return np.zeros(0, dtype=bool)
        if n_samples is None:
            step = float(np.clip(2.0 * clearance, self.pitch / 4.0, self.pitch))
            n_samples = int(np.ceil(np.linalg.norm(p2 - p1, axis=1).max() / step)) + 1
        n_samples = max(int(n_samples), 2)
        s = np.linspace(0.0, 1.0, n_samples)[None, :, None]
        ok = np.empty(m, dtype=bool)
        per = max(1, 2_000_000 // n_samples)
        for a in range(0, m, per):
            pts = p1[a:a + per, None, :] + s * (p2[a:a + per] - p1[a:a + per])[:, None, :]
            ok[a:a + per] = self.is_free(pts.reshape(-1, 3), clearance).reshape(len(pts), n_samples).all(axis=1)
        return ok

    def sample_free(self, n: int, clearance: float, rng=None, max_tries: int = 16) -> np.ndarray:
        """(n,3) uniform random free points (vectorised rejection sampling inside ``bounds``)."""
        rng = as_rng(rng)
        lo, hi = self.bounds
        got, batches, drawn, rate = 0, [], 0, 0.25
        for _ in range(max_tries):
            m = int(min(max((n - got) / max(rate, 1e-3) * 1.3, 1024), 1_000_000))
            pts = rng.uniform(lo, hi, size=(m, 3))
            ok = self.is_free(pts, clearance)
            drawn += m
            rate = max(float(ok.mean()), 1e-6)
            batches.append(pts[ok])
            got += int(ok.sum())
            if got >= n:
                return np.concatenate(batches)[:n]
        raise RuntimeError(
            f"no free space with clearance {clearance}: only {got} of {drawn} random points inside the mesh bounds were "
            f"free (mesh does not enclose a volume, or min_clearance is larger than the free space allows)")

    def __repr__(self) -> str:
        return f"FreeSpaceGrid(shape={self.shape}, pitch={self.pitch:.4g}, enclosed={self.enclosed})"


class TrajectoryGenerator:
    """Base of the random generators: holds the grid, seeds the rng, offers ``generate_many``."""

    def __init__(self, mesh_or_grid, min_clearance: float = 0.3):
        self.min_clearance = float(min_clearance)
        self.grid = mesh_or_grid if isinstance(mesh_or_grid, FreeSpaceGrid) else FreeSpaceGrid(
            mesh_or_grid, min_clearance=min_clearance)

    def generate(self, duration_us: int, *args, rng=None, initial_point=None, **kw) -> WaypointTrajectory:
        raise NotImplementedError

    def generate_many(self, n: int, duration_us: int, *args, rng=None, **kw) -> List[WaypointTrajectory]:
        """``n`` independent trajectories drawn from one rng (seed it once for reproducible batches)."""
        rng = as_rng(rng)
        return [self.generate(duration_us, *args, rng=rng, **kw) for _ in range(n)]

    def _start_point(self, initial_point) -> np.ndarray:
        p = np.asarray(initial_point, dtype=np.float64).reshape(3)
        c = float(self.grid.clearance_at(p[None])[0])
        if not c >= self.min_clearance:
            raise ValueError(f"initial_point {p.tolist()} is not in free space (clearance {c:.3g} < {self.min_clearance})")
        return p

    @staticmethod
    def _check_duration(duration_us) -> int:
        d = int(duration_us)
        if d <= 0:
            raise ValueError(f"duration_us must be positive, got {duration_us}")
        return d
