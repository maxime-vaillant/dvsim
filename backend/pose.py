"""Camera pose (position + orientation) used everywhere in the simulator.

Conventions (see docs/architecture.md):
  * world frame  = the frame of the scene files, right-handed, +Y up
  * camera frame = OpenGL: +X right, +Y up, camera looks along local -Z
  * ``quat`` is (x, y, z, w) and rotates camera-frame vectors into the world frame
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_WORLD_UP = np.array([0.0, 1.0, 0.0])
_FALLBACK_UP = np.array([0.0, 0.0, -1.0])  # used when looking (almost) straight up/down


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return v / n


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w), with w >= 0."""
    t = np.trace(m)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    q = q / np.linalg.norm(q)
    return -q if q[3] < 0 else q


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the short way round
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly identical: normalized lerp
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    q = (np.sin((1 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / s
    return q / np.linalg.norm(q)


@dataclass(frozen=True)
class Pose:
    position: np.ndarray  # (3,) float64
    quat: np.ndarray      # (4,) float64, (x, y, z, w), camera -> world

    def __post_init__(self):
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.float64).reshape(3))
        q = np.asarray(self.quat, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(f"orientation quaternion must be finite and non-zero, got {q}")
        object.__setattr__(self, "quat", q / norm)

    # ---- constructors -------------------------------------------------
    @staticmethod
    def from_look(position, front, up=None) -> "Pose":
        """Camera at ``position`` looking along ``front`` (no roll relative to ``up``)."""
        f = _normalize(np.asarray(front, dtype=np.float64))
        up = _WORLD_UP if up is None else np.asarray(up, dtype=np.float64)
        if abs(float(np.dot(f, _normalize(up)))) > 0.999:
            up = _FALLBACK_UP
        x = _normalize(np.cross(f, up))   # camera right
        y = np.cross(x, f)                # camera up
        rot = np.stack([x, y, -f], axis=1)  # columns = camera axes in world coords
        return Pose(position, matrix_to_quat(rot))

    @staticmethod
    def from_array(a) -> "Pose":
        a = np.asarray(a, dtype=np.float64)
        return Pose(a[:3], a[3:7])

    # ---- accessors ----------------------------------------------------
    def rotation(self) -> np.ndarray:
        """(3,3) camera -> world rotation."""
        return quat_to_matrix(self.quat)

    @property
    def right(self) -> np.ndarray:
        return self.rotation()[:, 0]

    @property
    def up(self) -> np.ndarray:
        return self.rotation()[:, 1]

    @property
    def forward(self) -> np.ndarray:
        return -self.rotation()[:, 2]

    def matrix(self) -> np.ndarray:
        """(4,4) camera -> world."""
        m = np.eye(4)
        m[:3, :3] = self.rotation()
        m[:3, 3] = self.position
        return m

    def view_matrix(self) -> np.ndarray:
        """(4,4) world -> camera (the classic OpenGL "view" matrix), float64."""
        r = self.rotation()
        m = np.eye(4)
        m[:3, :3] = r.T
        m[:3, 3] = -r.T @ self.position
        return m

    def as_array(self) -> np.ndarray:
        """(7,) = x y z qx qy qz qw (TUM trajectory order)."""
        return np.concatenate([self.position, self.quat])

    def translated_local(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Pose":
        """Same orientation, moved along the camera's own axes (e.g. dx = baseline -> right eye)."""
        return Pose(self.position + self.rotation() @ np.array([dx, dy, dz]), self.quat)


def interpolate(p0: Pose, p1: Pose, alpha: float) -> Pose:
    """Lerp position, slerp orientation."""
    return Pose((1 - alpha) * p0.position + alpha * p1.position, slerp(p0.quat, p1.quat, alpha))


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """OpenGL perspective projection (4,4), float64."""
    f = 1.0 / np.tan(np.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def focal_length_px(fov_y_deg: float, height: int) -> float:
    return (height / 2.0) / np.tan(np.radians(fov_y_deg) / 2.0)
