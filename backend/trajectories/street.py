"""Walks at eye height along the open ground of a scene (streets, squares, room floors).

The scene is +Y up, in metres. A top-down map of ``cell``-sized columns is built once from points sampled on the
surface: the ground of a column is its lowest sample, and the column is walkable when no sample lies between
``step_max`` and ``headroom`` above that ground. Roofs look like flat ground from above, so columns whose ground is
higher than ``terrain_step`` above the local terrain (lowest ground within ``terrain_window``) are not walkable.
Columns closer than ``min_clearance`` to a non-walkable one, or than ``edge_margin`` to where the scene ends (so the
camera does not look into the void; ignored when nothing would be left, e.g. in a small room) are dropped, and the
largest connected walkable area is kept.

A walk chains shortest routes between random points of that area (the route cost favours the middle of the street;
each new leg starts in roughly the direction the previous one ended, so there are no U-turns) until it is
``speed * duration`` long. It is then smoothed, sampled at constant speed, lifted ``eye_height`` above the smoothed
ground, and the camera looks ``look_ahead`` metres along the path, tilted by ``pitch_deg``, without roll.
"""
from __future__ import annotations

import warnings
from typing import List

import numpy as np
import trimesh
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from backend.trajectories.generated import _as_trimesh, as_rng
from backend.trajectories.trajectory import WaypointTrajectory

_NEIGHBOURS = ((0, 1), (1, 0), (1, 1), (1, -1))
_CENTRE_WEIGHT = 3.0   # route cost per cell = 1 + _CENTRE_WEIGHT / distance to the nearest obstacle (m)
_WAYPOINT_SPACING = 0.2  # m
_MIN_OBSTACLE_CELLS = 3  # smaller blocked blobs are noise


class StreetTrajectoryGenerator:
    """Random eye-height walks along the walkable ground of a +Y-up metric mesh (see the module docstring).

    ``mesh``: trimesh.Trimesh, trimesh.Scene or an object with ``to_trimesh()`` (backend.scene.Scene).
    The map is built in the constructor (deterministic); ``generate`` only plans on it.
    Attributes: ``cell`` (m, enlarged until the map has at most ``max_cells`` columns), ``origin`` ((x, z) of the
    corner of column (0, 0)), ``ground`` (ground height per column), ``walkable`` (bool map of the kept area).
    """

    def __init__(self, mesh, speed: float = 1.4, eye_height: float = 1.6, min_clearance: float = 1.0,
                 headroom: float = 2.2, step_max: float = 0.3, pitch_deg: float = -5.0, look_ahead: float = 2.0,
                 cell: float = 0.25, terrain_window: float = 20.0, terrain_step: float = 1.0, edge_margin: float = 5.0,
                 max_cells: int = 4_000_000, samples_per_cell: int = 24, max_samples: int = 20_000_000):
        for name, value in (("speed", speed), ("eye_height", eye_height), ("min_clearance", min_clearance),
                            ("headroom", headroom), ("cell", cell), ("look_ahead", look_ahead)):
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        if not (np.isfinite(edge_margin) and edge_margin >= 0):
            raise ValueError(f"edge_margin must be a non-negative number, got {edge_margin!r}")
        if not -89.0 <= pitch_deg <= 89.0:
            raise ValueError(f"pitch_deg must be in [-89, 89], got {pitch_deg!r}")
        self.speed, self.eye_height, self.min_clearance = float(speed), float(eye_height), float(min_clearance)
        self.pitch_deg, self.look_ahead = float(pitch_deg), float(look_ahead)
        mesh = _as_trimesh(mesh)
        self._build_map(mesh, float(cell), float(headroom), float(step_max), float(terrain_window),
                        float(terrain_step), float(edge_margin), int(max_cells), int(samples_per_cell),
                        int(max_samples))
        self._build_graph()

    # ---- map ------------------------------------------------------------
    def _build_map(self, mesh, cell, headroom, step_max, terrain_window, terrain_step, edge_margin, max_cells,
                   per_cell, max_samples):
        lo, hi = mesh.bounds[0, [0, 2]], mesh.bounds[1, [0, 2]]
        extent = np.maximum(hi - lo, 1e-9)
        while np.prod(np.floor(extent / cell) + 1) > max_cells:
            cell *= 1.25
        shape = tuple((np.floor(extent / cell) + 1).astype(int))
        n = int(np.clip(mesh.area / cell ** 2 * per_cell, 100_000, max_samples))
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
        ij = np.clip(np.floor((pts[:, [0, 2]] - lo) / cell).astype(int), 0, np.array(shape) - 1)
        flat = np.ravel_multi_index(ij.T, shape)

        ground = np.full(np.prod(shape), np.inf)
        np.minimum.at(ground, flat, pts[:, 1])
        ground = ground.reshape(shape)
        known = np.isfinite(ground)
        if not known.any():
            raise ValueError("the mesh has no surface to walk on")
        # Columns without samples take the nearest known ground (they stay non-walkable).
        nearest = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
        ground = ndimage.minimum_filter(ground[tuple(nearest)], size=3)
        # The walking surface is the top of the lowest layer (a ground slab has a bottom face too).
        low = pts[:, 1] - ground.ravel()[flat] <= step_max
        top = np.full(np.prod(shape), -np.inf)
        np.maximum.at(top, flat[low], pts[low, 1])
        top = top.reshape(shape)
        ground = np.where(np.isfinite(top), top, ground)

        rel = pts[:, 1] - ground.ravel()[flat]
        blocked = np.zeros(np.prod(shape), bool)
        blocked[flat[(rel > step_max) & (rel < headroom)]] = True
        blocked = blocked.reshape(shape) | ~known
        k = max(3, int(terrain_window / cell) | 1)
        terrain = ndimage.uniform_filter(ndimage.minimum_filter(ground, size=k), size=k)
        blocked |= ground > terrain + terrain_step

        # Tiny isolated specks (reconstruction noise) do not block a street; walls are long, so they stay.
        specks, n_specks = ndimage.label(blocked, structure=np.ones((3, 3)))
        sizes = np.bincount(specks.ravel(), minlength=n_specks + 1)
        blocked &= sizes[specks] >= _MIN_OBSTACLE_CELLS
        free = np.pad(~blocked, 1, constant_values=False)  # beyond the map counts as an obstacle
        dist = ndimage.distance_transform_edt(free)[1:-1, 1:-1] * cell
        ok = dist >= self.min_clearance
        # Stay away from where the scene ends (looking into the void), unless that leaves nothing (small rooms).
        inside = np.pad(known, 1, constant_values=False)
        edge = ndimage.distance_transform_edt(inside)[1:-1, 1:-1] * cell
        if (ok & (edge >= edge_margin)).any():
            ok &= edge >= edge_margin
        elif ok.any():
            warnings.warn(f"StreetTrajectoryGenerator: no walkable ground {edge_margin} m from the scene edge; "
                          "ignoring edge_margin", stacklevel=3)
        labels, count = ndimage.label(ok)
        if count == 0:
            raise ValueError(
                f"no walkable ground with min_clearance {self.min_clearance} m and headroom {headroom} m "
                "(the scene must be +Y up, in metres, with open ground)")
        ok = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1

        self.cell, self.origin, self.shape = cell, lo, shape
        self.ground = np.minimum(ground, terrain + terrain_step)
        self.walkable, self._dist, self._edge = ok, dist, edge

    @property
    def walkable_area(self) -> float:
        """Area of the kept walkable region, m²."""
        return float(self.walkable.sum()) * self.cell ** 2

    def _build_graph(self):
        ok, shape = self.walkable, self.shape
        cells = np.flatnonzero(ok)
        index = np.full(ok.size, -1)
        index[cells] = np.arange(len(cells))
        r, c = np.unravel_index(cells, shape)
        cost = 1.0 + _CENTRE_WEIGHT / self._dist.ravel()[cells]
        rows, cols, weights = [], [], []
        for dr, dc in _NEIGHBOURS:
            rr, cc = r + dr, c + dc
            inside = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            j = np.full(len(cells), -1)
            j[inside] = index[np.ravel_multi_index((rr[inside], cc[inside]), shape)]
            a = np.flatnonzero(j >= 0)
            rows.append(a)
            cols.append(j[a])
            weights.append(np.hypot(dr, dc) * 0.5 * (cost[a] + cost[j[a]]))
        n = len(cells)
        self._graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(n, n)).tocsr()
        self._cells, self._index = cells, index
        self._xz = (np.stack([r, c], axis=1) + 0.5) * self.cell + self.origin

    def ground_at(self, xz) -> np.ndarray:
        """(M,) ground height of the columns containing the (M, 2) points (x, z)."""
        ij = np.floor((np.atleast_2d(xz) - self.origin) / self.cell).astype(int)
        ij = np.clip(ij, 0, np.array(self.shape) - 1)
        return self.ground[ij[:, 0], ij[:, 1]]

    # ---- planning -------------------------------------------------------
    def generate_many(self, n: int, duration_us: int, rng=None) -> List[WaypointTrajectory]:
        rng = as_rng(rng)
        return [self.generate(duration_us, rng=rng) for _ in range(n)]

    def generate(self, duration_us: int, rng=None, initial_point=None) -> WaypointTrajectory:
        """One walk of ``duration_us`` at ``speed``; ``initial_point`` (x, y, z) starts at the nearest walkable spot."""
        duration_us = int(duration_us)
        if duration_us <= 0:
            raise ValueError(f"duration_us must be positive, got {duration_us}")
        rng = as_rng(rng)
        length = self.speed * duration_us / 1e6
        if initial_point is None:
            start = int(rng.integers(len(self._cells)))
        else:
            p = np.asarray(initial_point, dtype=np.float64).reshape(3)
            start = int(np.argmin(np.sum((self._xz - p[[0, 2]]) ** 2, axis=1)))
        # Smoothing shortens corners: plan a bit more than needed, then cut at the exact length.
        route = self._route(start, length * 1.2 + 2 * self.look_ahead + 5.0, rng)
        for sigma in (1.5, 0.75, 0.0):
            xz = self._smooth(route, sigma)
            if self._dist_at(xz).min() >= 0.5 * self.min_clearance:
                break
        return self._to_trajectory(xz, length, duration_us)

    def _route(self, start: int, length: float, rng, max_legs: int = 1000) -> np.ndarray:
        """Cell indices (into ``_cells``) of a chain of shortest routes at least ``length`` m long."""
        leg_min, leg_max = 15.0, 40.0
        limit = leg_max * (1.0 + _CENTRE_WEIGHT / self.min_clearance) / self.cell
        path, total, heading = [start], 0.0, None
        for _ in range(max_legs):
            if total >= length:
                return np.array(path)
            cur = path[-1]
            dist, pred = dijkstra(self._graph, directed=False, indices=cur, return_predecessors=True, limit=limit)
            reach = np.flatnonzero(np.isfinite(dist))
            if len(reach) < 2:
                raise ValueError("the walkable area is a single column; lower min_clearance")
            far = reach[dist[reach] >= leg_min / self.cell]
            pool = far if len(far) else reach[dist[reach] >= 0.5 * dist[reach].max()]
            # Targets deep inside the scene are likelier: legs then face the scene rather than its edge.
            weight = self._edge.ravel()[self._cells[pool]] ** 2
            cand = rng.choice(pool, size=min(32, len(pool)), replace=False, p=weight / weight.sum())
            legs = [self._trace(pred, cur, int(e)) for e in cand]
            if heading is not None:  # keep going roughly the same way
                score = np.array([self._leg_heading(leg) @ heading for leg in legs])
                best = np.flatnonzero(score >= min(score.max(), max(0.0, score.max() - 0.5)))
                leg = legs[int(rng.choice(best))]
            else:
                leg = legs[0]
            seg = np.linalg.norm(np.diff(self._xz[leg], axis=0), axis=1)
            total += float(seg.sum())
            path.extend(leg[1:].tolist())
            heading = self._leg_heading(leg[::-1]) * -1.0
        raise RuntimeError(f"could not plan a {length:.0f} m walk in {max_legs} legs")

    @staticmethod
    def _trace(pred: np.ndarray, start: int, end: int) -> np.ndarray:
        out = [end]
        while out[-1] != start:
            out.append(int(pred[out[-1]]))
        return np.array(out[::-1])

    def _leg_heading(self, leg: np.ndarray, span: float = 4.0) -> np.ndarray:
        """Unit (x, z) direction of the first ``span`` metres of a leg."""
        xz = self._xz[leg]
        d = np.linalg.norm(xz - xz[0], axis=1)
        k = int(np.searchsorted(d, span)) if d[-1] > span else len(xz) - 1
        v = xz[max(k, 1)] - xz[0]
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.zeros(2)

    def _smooth(self, route: np.ndarray, sigma_m: float) -> np.ndarray:
        xz = self._xz[route]
        if sigma_m > 0:
            sigma = sigma_m / self.cell
            smooth = ndimage.gaussian_filter1d(xz, sigma=sigma, axis=0, mode="nearest")
            fade = np.clip(1.0 - np.arange(len(xz)) / (3.0 * sigma), 0.0, 1.0)  # keep the start where it was
            xz = smooth + fade[:, None] * (xz[0] - smooth[0])
        keep = np.concatenate([[True], np.linalg.norm(np.diff(xz, axis=0), axis=1) > 1e-9])
        return xz[keep]

    def _dist_at(self, xz: np.ndarray) -> np.ndarray:
        ij = np.clip(np.floor((xz - self.origin) / self.cell).astype(int), 0, np.array(self.shape) - 1)
        return self._dist[ij[:, 0], ij[:, 1]]

    def _to_trajectory(self, xz: np.ndarray, length: float, duration_us: int) -> WaypointTrajectory:
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
        length = min(length, s[-1])
        n = max(2, int(np.ceil(length / _WAYPOINT_SPACING)) + 1)
        si = np.linspace(0.0, length, n)
        x, z = np.interp(si, s, xz[:, 0]), np.interp(si, s, xz[:, 1])
        sigma = max(1.0, 2.0 / _WAYPOINT_SPACING)  # ground smoothed over ~2 m: no jolts at kerbs
        ground = ndimage.gaussian_filter1d(self.ground_at(np.stack([x, z], 1)), sigma=sigma, mode="nearest")
        pos = np.stack([x, ground + self.eye_height, z], axis=1)

        ahead = np.minimum(si + self.look_ahead, s[-1])
        dx, dz = np.interp(ahead, s, xz[:, 0]) - x, np.interp(ahead, s, xz[:, 1]) - z
        h = np.stack([dx, dz], axis=1)
        norm = np.linalg.norm(h, axis=1)
        last = np.maximum.accumulate(np.where(norm > 1e-6, np.arange(n), -1))  # end of the walk: keep the heading
        h = h[np.maximum(last, 0)] / np.maximum(norm[np.maximum(last, 0)], 1e-12)[:, None]
        h = ndimage.gaussian_filter1d(h, sigma=sigma, axis=0, mode="nearest")
        h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
        tilt = np.tan(np.radians(self.pitch_deg))
        fronts = np.stack([h[:, 0], np.full(n, tilt), h[:, 1]], axis=1)
        times = np.rint(np.linspace(0.0, duration_us, n)).astype(np.int64)
        return WaypointTrajectory.from_waypoints(times, pos, fronts=fronts)
