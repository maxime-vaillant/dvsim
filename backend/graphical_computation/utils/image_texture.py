from __future__ import annotations

import moderngl
import numpy as np

GL_SRGB8 = 0x8C41


class ImageTexture:
    """GPU texture from an (h, w, 3) uint8 sRGB image.

    Stored as SRGB8 so the hardware decodes to linear light before bilinear / mipmap / anisotropic filtering.
    Row 0 of the array is the image top; shaders sample with ``t = 1 - v``.
    """

    def __init__(self, context: moderngl.Context, array: np.ndarray, anisotropy: float = 16.0):
        h, w = array.shape[:2]
        self.texture = context.texture((w, h), 3, np.ascontiguousarray(array), internal_format=GL_SRGB8)
        self.texture.build_mipmaps()
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.texture.anisotropy = anisotropy
        self.texture.repeat_x = self.texture.repeat_y = True

    def use(self, location: int = 0) -> None:
        self.texture.use(location)

    def release(self) -> None:
        self.texture.release()
