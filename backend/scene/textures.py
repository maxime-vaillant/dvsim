"""Texture image helpers (CPU side): safe decode with size caps, sRGB conversions."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

DEFAULT_MAX_TEXTURE_SIZE = 8192                   # longest side kept (LANCZOS downscale above)
DEFAULT_MAX_TEXTURE_PIXELS = 16384 * 16384        # refuse to decode anything larger (decompression bombs)

_PIL_LIMIT_LOCK = threading.Lock()


@contextmanager
def pil_limit_lifted():
    """Temporarily lift PIL's decompression-bomb check around ``Image.open`` (header parsing only).

    Callers enforce their own explicit pixel cap from ``image.size`` *before* decoding (see ``image_to_array``),
    which reports a clear error and allows sizes PIL's default would reject; a process-wide
    ``Image.MAX_IMAGE_PIXELS = None`` would silently disable the protection for everything else in the process.
    """
    with _PIL_LIMIT_LOCK:
        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            yield
        finally:
            Image.MAX_IMAGE_PIXELS = previous


def image_to_array(image: Image.Image, max_size: int = DEFAULT_MAX_TEXTURE_SIZE,
                   max_pixels: int = DEFAULT_MAX_TEXTURE_PIXELS) -> np.ndarray:
    """PIL image -> (h, w, 3) uint8 RGB, C-contiguous; the longest side is downscaled to ``max_size``.

    The pixel count is checked from the header before anything is decoded (ValueError above ``max_pixels``), and
    the image is shrunk (JPEG draft mode, integer box reduce, or nearest for 1-bit / palette images) before it
    is widened to RGB, so peak memory stays near the decoded size of the source instead of several times it.
    """
    w, h = image.size
    if w * h > max_pixels:
        raise ValueError(f"texture is {w}x{h} = {w * h} pixels, over the {max_pixels}-pixel limit")
    scale = min(1.0, max_size / max(w, h))
    target = (max(1, round(w * scale)), max(1, round(h * scale)))
    if scale < 1.0:
        image.draft("RGB", target)  # JPEG only: decode at 1/2, 1/4, 1/8 size; no-op for other formats
    if image.mode not in ("1", "P", "L", "LA", "RGB", "RGBA"):
        image = image.convert("RGB")
    if scale < 1.0:
        if image.mode in ("1", "P"):  # PIL resizes these with NEAREST only, and widening first would triple memory
            image = image.resize(target, Image.NEAREST)
        else:
            k = int(min(image.size[0] / target[0], image.size[1] / target[1]))
            if k >= 2:
                image = image.reduce(k)
            if image.size != target:
                image = image.resize(target, Image.LANCZOS)
    return _rgb_array(image)


def _rgb_array(image: Image.Image) -> np.ndarray:
    """(h, w, 3) uint8, converted block by block (a whole-image tobytes / RGB copy would double the peak)."""
    w, h = image.size
    out = np.empty((h, w, 3), dtype=np.uint8)
    step = max(1, (1 << 22) // w)
    for i in range(0, h, step):
        block = image.crop((0, i, w, min(h, i + step)))
        out[i:i + step] = np.asarray(block if block.mode == "RGB" else block.convert("RGB"))
    return out


def load_texture(path: Union[str, Path], max_size: int = DEFAULT_MAX_TEXTURE_SIZE,
                 max_pixels: int = DEFAULT_MAX_TEXTURE_PIXELS) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"texture file not found: {path}")
    with pil_limit_lifted():
        image = Image.open(path)  # parses the header only
    with image:
        return image_to_array(image, max_size, max_pixels)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Linear [0,1] -> sRGB-encoded [0,1] (IEC 61966-2-1)."""
    c = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """sRGB-encoded [0,1] -> linear [0,1]."""
    c = np.clip(np.asarray(encoded, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))
