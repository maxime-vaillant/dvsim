"""Single-pass MRT renderer: luminance, RGB, depth, forward optical flow and flow validity in one draw.

Conventions: see docs/architecture.md. Implementation notes:
  * one framebuffer, five colour attachments + a depth renderbuffer (real depth test, all attachments cleared to 0);
  * the projection is y-flipped so GL's first row is the image's top row: read-backs need no flip and no copy;
  * mvp matrices are composed in float64 on the CPU (camera-relative precision) and uploaded column-major float32;
  * GPU resources (buffers, VAOs, textures) are built once; ``render`` only writes uniforms, draws and reads back
    what was requested;
  * a Renderer belongs to the thread that created it (a GL context used from another thread crashes the process):
    build it in the thread that renders. Every GL entry point makes its own context current (``with self.ctx``),
    so several renderers, or a renderer next to other GL code, can coexist in one thread.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import moderngl
import numpy as np

from backend.config import CameraConfig
from backend.graphical_computation.utils.gl_context import create_gl_context
from backend.graphical_computation.utils.image_texture import ImageTexture
from backend.graphical_computation.utils.loaders import load_shader_source
from backend.pose import Pose, perspective
from backend.scene import Scene, SceneObject

_ATT_RGB, _ATT_LUM, _ATT_FLOW, _ATT_DEPTH, _ATT_VALID = range(5)
_MODE_BASE, _MODE_TEXTURE, _MODE_COLORS = range(3)
_FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])


@dataclass
class RenderOutput:
    """Every array is freshly allocated per ``render`` call (C-contiguous, writable, never reused): safe to keep."""

    lum: np.ndarray                          # (H, W) float32 linear luminance, always filled
    rgb: Optional[np.ndarray] = None         # (H, W, 3) uint8, RGB order
    depth: Optional[np.ndarray] = None       # (H, W) float32 z-depth, 0 = no surface
    flow: Optional[np.ndarray] = None        # (H, W, 2) float32 pixels (dx right, dy down); 0 where invalid
    flow_valid: Optional[np.ndarray] = None  # (H, W) uint8


@dataclass
class _Draw:
    vao: moderngl.VertexArray
    mode: int
    texture: Optional[ImageTexture]
    base_color: bytes


def auto_clip_planes(scene: Scene, camera_positions: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """(near, far) from the scene size. ``camera_positions`` (M, 3), if given, extends far to reach the
    farthest scene-bounds corner from any of them; otherwise far = 10 scene diagonals (covers cameras inside
    the scene and orbiting it). near = clamp(1e-3 * diagonal, 0.01, 0.05) so close geometry survives in large scenes."""
    diag = scene.diagonal
    near = min(max(0.01, 1e-3 * diag), 0.05)
    if camera_positions is None or len(camera_positions) == 0:
        far = 10.0 * diag
    else:
        lo, hi = scene.bounds
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pos = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
        far = 1.05 * float(np.max(np.linalg.norm(pos[:, None, :] - corners[None, :, :], axis=2)))
    return near, max(far, 100.0 * near)


class Renderer:
    """Renders ``scene`` from camera poses. ``context=None`` creates (and owns) a standalone GL context.

    ``camera.near`` / ``camera.far`` of None are derived from the scene (see ``auto_clip_planes``);
    ``camera_positions`` (optional) or ``set_camera_positions`` refines them for a known camera path.
    Not thread-safe: use it from the thread that created it.
    """

    def __init__(self, scene: Scene, camera: CameraConfig, context: Optional[moderngl.Context] = None,
                 *, camera_positions: Optional[np.ndarray] = None):
        if not scene.objects:
            raise ValueError("cannot render an empty scene")
        self.scene = scene
        self.camera = camera
        self.width, self.height = int(camera.width), int(camera.height)
        near, far = auto_clip_planes(scene, camera_positions)
        self.near = float(camera.near) if camera.near is not None else near
        self.far = float(camera.far) if camera.far is not None else far
        if not 0.0 < self.near < self.far:
            raise ValueError(f"invalid clip planes near={self.near}, far={self.far}")

        self._owns_context = context is None
        self.ctx = create_gl_context() if context is None else context
        self._thread = threading.get_ident()
        self._released = False
        self._buffers: List[moderngl.Buffer] = []
        self._vaos: List[moderngl.VertexArray] = []
        self._textures: List[ImageTexture] = []
        try:
            with self.ctx:
                self._init_program()
                self._init_framebuffer()
                self._init_draws(scene)
        except BaseException:
            self._released = True
            if self._owns_context:
                self.ctx.release()
            raise
        self._update_projection()

    def _update_projection(self) -> None:
        self._proj = _FLIP_Y @ perspective(self.camera.fov_y_deg, self.width / self.height, self.near, self.far)

    def set_camera_positions(self, positions: Optional[np.ndarray]) -> None:
        """Tell the renderer where the camera will be (``(M, 3)`` world positions of the path) so the automatic
        clip planes cover it. Only planes left as None in ``CameraConfig`` change; it is cheap (projection only).
        An empty array restores the scene-size default."""
        near, far = auto_clip_planes(self.scene, positions)
        near = float(self.camera.near) if self.camera.near is not None else near
        far = float(self.camera.far) if self.camera.far is not None else far
        if not 0.0 < near < far:
            raise ValueError(f"invalid clip planes near={near}, far={far}")
        self.near, self.far = near, far
        self._update_projection()

    def _check_thread(self) -> None:
        if self._released:
            raise RuntimeError("renderer is closed")
        if threading.get_ident() != self._thread:
            raise RuntimeError(
                "Renderer used from a different thread than the one that created it (an OpenGL context must not "
                "cross threads): create the Renderer / Simulator in the thread that renders"
            )

    # ---- setup ----------------------------------------------------------
    def _init_program(self) -> None:
        self.prog = self.ctx.program(
            vertex_shader=load_shader_source("scene.vert"), fragment_shader=load_shader_source("scene.frag")
        )
        self.prog["tex"].value = 0
        self._blank = ImageTexture(self.ctx, np.zeros((1, 1, 3), np.uint8))  # keeps the sampler valid for untextured draws
        self._textures.append(self._blank)
        self.prog["screen_size"].value = (float(self.width), float(self.height))
        self._u_cur, self._u_next = self.prog["cur_mvp"], self.prog["next_mvp"]
        self._u_mode, self._u_base = self.prog["color_mode"], self.prog["base_color"]

    def _init_framebuffer(self) -> None:
        size = (self.width, self.height)
        self._targets = [
            self.ctx.texture(size, 4, dtype="f1"),   # rgb (sRGB-encoded RGBA8)
            self.ctx.texture(size, 1, dtype="f4"),   # linear luminance
            self.ctx.texture(size, 2, dtype="f4"),   # flow
            self.ctx.texture(size, 1, dtype="f4"),   # z-depth
            self.ctx.texture(size, 1, dtype="f1"),   # flow valid (R8, 0 / 255)
        ]
        self._depth_rb = self.ctx.depth_renderbuffer(size)
        self.fbo = self.ctx.framebuffer(color_attachments=self._targets, depth_attachment=self._depth_rb)

    def _init_draws(self, scene: Scene) -> None:
        self._draws: List[_Draw] = []
        vaos: Dict[tuple, moderngl.VertexArray] = {}
        gpu_textures: Dict[int, ImageTexture] = {}
        for obj in scene.objects:
            if obj.texture is not None:
                mode = _MODE_TEXTURE
            elif obj.colors is not None:
                mode = _MODE_COLORS
            else:
                mode = _MODE_BASE
            key = (id(obj.vertices), id(obj.faces), id(obj.uv), id(obj.colors), mode)
            if key not in vaos:
                vaos[key] = self._make_vao(obj, mode)
            texture = None
            if mode == _MODE_TEXTURE:
                if id(obj.texture) not in gpu_textures:
                    gpu_textures[id(obj.texture)] = ImageTexture(self.ctx, obj.texture)
                    self._textures.append(gpu_textures[id(obj.texture)])
                texture = gpu_textures[id(obj.texture)]
            self._draws.append(_Draw(vaos[key], mode, texture, obj.base_color.astype("f4").tobytes()))
        self._models = np.stack([o.model_matrix for o in scene.objects])  # (N, 4, 4) float64

    def _make_vao(self, obj: SceneObject, mode: int) -> moderngl.VertexArray:
        if mode == _MODE_TEXTURE:
            data, fmt, names = np.hstack([obj.vertices, obj.uv]), "3f 2f", ("in_vertex", "in_uv")
        elif mode == _MODE_COLORS:
            data, fmt, names = np.hstack([obj.vertices, obj.colors]), "3f 4f", ("in_vertex", "in_color")
        else:
            data, fmt, names = obj.vertices, "3f", ("in_vertex",)
        vbo = self.ctx.buffer(np.ascontiguousarray(data, dtype=np.float32))
        ibo = self.ctx.buffer(obj.faces)
        vao = self.ctx.vertex_array(self.prog, [(vbo, fmt, *names)], index_buffer=ibo, index_element_size=4)
        self._buffers += [vbo, ibo]
        self._vaos.append(vao)
        return vao

    # ---- rendering --------------------------------------------------------
    def render(self, pose: Pose, next_pose: Optional[Pose] = None, *,
               rgb: bool = False, depth: bool = False, flow: bool = False) -> RenderOutput:
        """Render one view. Luminance is always returned; ``rgb`` / ``depth`` / ``flow`` (which also fills
        ``flow_valid``) are read back only when requested. ``flow`` is the forward flow ``pose -> next_pose``.
        Stereo: call again with ``pose.translated_local(baseline)``."""
        self._check_thread()
        if flow and next_pose is None:
            raise ValueError("flow=True requires next_pose")

        cur_vp = self._proj @ pose.view_matrix()
        cur = _column_major(cur_vp @ self._models)
        nxt = _column_major((self._proj @ next_pose.view_matrix()) @ self._models) if flow else None

        with self.ctx:  # make our context current (other renderers / contexts may exist), restore the previous one
            self.fbo.use()
            self.ctx.enable_only(moderngl.DEPTH_TEST)
            self.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            self._blank.use(0)
            last_mode = -1
            for i, d in enumerate(self._draws):
                self._u_cur.write(cur[i])
                if nxt is not None:
                    self._u_next.write(nxt[i])
                if d.mode != last_mode:
                    self._u_mode.value = d.mode
                    last_mode = d.mode
                if d.mode == _MODE_BASE:
                    self._u_base.write(d.base_color)
                elif d.texture is not None:
                    d.texture.use(0)
                d.vao.render()

            out = RenderOutput(lum=self._read(_ATT_LUM, 1, "f4", np.float32))
            if rgb:
                out.rgb = self._read(_ATT_RGB, 3, "f1", np.uint8)
            if depth:
                out.depth = self._read(_ATT_DEPTH, 1, "f4", np.float32)
            if flow:
                out.flow = self._read(_ATT_FLOW, 2, "f4", np.float32)
                valid = self._read(_ATT_VALID, 1, "f1", np.uint8)
                np.minimum(valid, 1, out=valid)  # R8 holds 0 / 255
                out.flow_valid = valid
        return out

    def _read(self, attachment: int, components: int, gl_dtype: str, dtype) -> np.ndarray:
        shape = (self.height, self.width) if components == 1 else (self.height, self.width, components)
        arr = np.empty(shape, dtype=dtype)
        self.fbo.read_into(arr, components=components, attachment=attachment, dtype=gl_dtype)
        return arr

    # ---- lifetime ---------------------------------------------------------
    def close(self) -> None:
        """Release all GL objects (and the context if this renderer created it). Idempotent; same thread as
        the creating one."""
        if self._released:
            return
        self._check_thread()
        self._released = True
        with self.ctx:
            for obj in [*self._vaos, *self._buffers, *self._textures, self.fbo, *self._targets, self._depth_rb, self.prog]:
                obj.release()
        if self._owns_context:
            self.ctx.release()

    def __enter__(self) -> "Renderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _column_major(m: np.ndarray) -> np.ndarray:
    """(N, 4, 4) float64 row-major matrices -> (N, 4, 4) float32 whose memory is column-major (GL layout)."""
    return np.ascontiguousarray(np.swapaxes(m, -1, -2), dtype=np.float32)
