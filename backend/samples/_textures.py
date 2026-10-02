"""Procedural textures for the demo scene (numpy + PIL only, fully deterministic given the rng)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

Color = Tuple[int, int, int]

NEAR_BLACK: Color = (12, 12, 14)
NEAR_WHITE: Color = (247, 246, 240)
POSTER_COLORS = [(196, 40, 44), (32, 92, 200), (238, 190, 28), (36, 158, 92), (168, 62, 178), (240, 120, 30)]


def _int(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, max(hi, lo + 1)))


def value_noise(rng: np.random.Generator, h: int, w: int, cell: int) -> np.ndarray:
    """(h, w) float32 smooth value noise in [0, 1] with lattice spacing `cell` pixels."""
    gh, gw = h // cell + 2, w // cell + 2
    g = rng.random((gh, gw), dtype=np.float32)
    ys = np.arange(h, dtype=np.float32) / cell
    xs = np.arange(w, dtype=np.float32) / cell
    y0, x0 = ys.astype(np.int64), xs.astype(np.int64)
    ty, tx = ys - y0, xs - x0
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    top = g[y0][:, x0] + (g[y0][:, x0 + 1] - g[y0][:, x0]) * tx[None, :]
    bot = g[y0 + 1][:, x0] + (g[y0 + 1][:, x0 + 1] - g[y0 + 1][:, x0]) * tx[None, :]
    return top + (bot - top) * ty[:, None]


def fbm(rng: np.random.Generator, h: int, w: int, base_cell: int, octaves: int,
        persistence: float = 0.55) -> np.ndarray:
    """Multi-octave noise normalised to [0, 1] (lattice halves every octave: base_cell, base_cell/2, ...)."""
    acc = np.zeros((h, w), np.float32)
    amp, cell = 1.0, base_cell
    for _ in range(octaves):
        acc += amp * value_noise(rng, h, w, max(2, int(cell)))
        amp *= persistence
        cell //= 2
    acc -= acc.min()
    return acc / max(float(acc.max()), 1e-6)


def _noise_colors(rng: np.random.Generator, h: int, w: int, c0: Color, c1: Color, base_cell: int) -> np.ndarray:
    """(h, w, 3) float32 in [0, 255]: two-colour gradient driven by multi-octave noise + a slow brightness drift."""
    n = fbm(rng, h, w, base_cell, 6)
    n = np.clip((n - 0.5) * 1.7 + 0.5, 0.0, 1.0)
    m = fbm(rng, h, w, base_cell * 2, 2)
    col = np.asarray(c0, np.float32) * (1 - n[..., None]) + np.asarray(c1, np.float32) * n[..., None]
    return np.clip(col * (0.78 + 0.44 * m[..., None]), 0, 255)


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _poster(d: ImageDraw.ImageDraw, rng: np.random.Generator, w: int, h: int) -> None:
    pw, ph = _int(rng, 70, 200), _int(rng, 90, 240)
    x, y = _int(rng, 10, w - pw - 10), _int(rng, 10, h - ph - 10)
    col = POSTER_COLORS[_int(rng, 0, len(POSTER_COLORS))]
    d.rectangle([x, y, x + pw, y + ph], fill=col)
    d.rectangle([x + 6, y + 6, x + pw - 6, y + ph - 6], outline=NEAR_WHITE, width=3)
    d.rectangle([x + pw // 4, y + ph // 4, x + 3 * pw // 4, y + 3 * ph // 4], fill=NEAR_BLACK)
    d.ellipse([x + pw // 3, y + ph // 3, x + 2 * pw // 3, y + 2 * ph // 3], fill=NEAR_WHITE)


def _checker(d: ImageDraw.ImageDraw, rng: np.random.Generator, w: int, h: int) -> None:
    nx, ny, c = _int(rng, 5, 10), _int(rng, 4, 8), _int(rng, 20, 40)
    x0, y0 = _int(rng, 5, w - nx * c - 5), _int(rng, 5, h - ny * c - 5)
    for i in range(nx):
        for j in range(ny):
            fill = NEAR_BLACK if (i + j) % 2 else NEAR_WHITE
            d.rectangle([x0 + i * c, y0 + j * c, x0 + (i + 1) * c - 1, y0 + (j + 1) * c - 1], fill=fill)


def _patch(d: ImageDraw.ImageDraw, rng: np.random.Generator, w: int, h: int, fill: Color, edge: Color) -> None:
    pw, ph = _int(rng, 100, 260), _int(rng, 60, 160)
    x, y = _int(rng, 5, w - pw - 5), _int(rng, 5, h - ph - 5)
    d.rectangle([x, y, x + pw, y + ph], fill=fill)
    d.rectangle([x + 5, y + 5, x + pw - 5, y + ph - 5], outline=edge, width=2)


def _line(d: ImageDraw.ImageDraw, rng: np.random.Generator, w: int, h: int, accent: Color) -> None:
    kind = _int(rng, 0, 3)
    x0, y0 = _int(rng, 0, w), _int(rng, 0, h)
    length = _int(rng, 100, 520)
    dx, dy = ((length, 0), (0, length), (length, length))[kind]
    if kind == 2 and _int(rng, 0, 2):
        dy = -dy
    col = (NEAR_BLACK, NEAR_WHITE, accent)[_int(rng, 0, 3)]
    d.line([(x0, y0), (x0 + dx, y0 + dy)], fill=col, width=_int(rng, 2, 6))


def surface_texture(rng: np.random.Generator, w: int, h: int, c0: Color, c1: Color, *, base_cell: int = 160,
                    tile_px: int = 0, wainscot_px: int = 0, n_lines: int = 12, n_posters: int = 4,
                    n_checkers: int = 1, n_dark: int = 1, n_light: int = 1, n_dots: int = 160) -> Image.Image:
    """Room-surface texture: noise gradients at 6 scales + tile grid / wainscot + lines, posters, checkers, dark
    and light patches, dots (edges and corners at every scale, both near-black and near-white areas)."""
    arr = _noise_colors(rng, h, w, c0, c1, base_cell)
    if wainscot_px:
        arr[h - wainscot_px:] *= 0.55
    img = _to_image(arr)
    d = ImageDraw.Draw(img)
    accent = POSTER_COLORS[_int(rng, 0, len(POSTER_COLORS))]
    dark = tuple(int(0.3 * (a + b) / 2) for a, b in zip(c0, c1))
    if tile_px:
        for x in range(0, w, tile_px):
            d.line([(x, 0), (x, h)], fill=dark, width=3)
        for y in range(0, h, tile_px):
            d.line([(0, y), (w, y)], fill=dark, width=3)
    if wainscot_px:
        d.line([(0, h - wainscot_px), (w, h - wainscot_px)], fill=NEAR_WHITE, width=4)
    for _ in range(n_lines):
        _line(d, rng, w, h, accent)
    for _ in range(n_checkers):
        _checker(d, rng, w, h)
    for _ in range(n_posters):
        _poster(d, rng, w, h)
    for _ in range(n_dark):
        _patch(d, rng, w, h, NEAR_BLACK, (90, 90, 96))
    for _ in range(n_light):
        _patch(d, rng, w, h, NEAR_WHITE, (170, 170, 176))
    for _ in range(n_dots):
        r, x, y = _int(rng, 2, 5), _int(rng, 5, w - 5), _int(rng, 5, h - 5)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(NEAR_BLACK, NEAR_WHITE)[_int(rng, 0, 2)])
    return img


def object_texture(kind: str, size: int, rng: np.random.Generator) -> Image.Image:
    """Small texture for one of the demo primitives: rug | checker | bands | stripes | diagonal."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    n = fbm(rng, size, size, size // 4, 5)
    if kind == "checker":
        cells = 8
        chk = (((xx * cells // size) + (yy * cells // size)) % 2).astype(np.float32)
        a, b = np.array([235, 235, 225], np.float32), np.array([30, 40, 90], np.float32)
        col = a * chk[..., None] + b * (1 - chk[..., None])
        col *= (0.7 + 0.5 * n[..., None])
    elif kind == "bands":
        v = np.sin(yy / size * 2 * np.pi * 5 + 3.0 * n) > 0
        col = np.where(v[..., None], np.array([220, 90, 40], np.float32), np.array([245, 225, 170], np.float32))
        col = col * (0.75 + 0.4 * n[..., None])
        col[(xx.astype(int) % (size // 4)) < 3] = NEAR_BLACK
    elif kind == "stripes":
        widths = rng.integers(size // 32, size // 8, size).cumsum()
        idx = np.searchsorted(widths, np.arange(size))
        pal = np.array([[40, 160, 120], [240, 240, 235], [20, 30, 60], [230, 170, 40]], np.float32)
        row = pal[idx % len(pal)]
        col = np.broadcast_to(row[None, :, :], (size, size, 3)) * (0.75 + 0.4 * n[..., None])
        col = np.array(col)
        col[(yy.astype(int) % (size // 4)) < 4] = NEAR_BLACK
    elif kind == "diagonal":
        period = size // 8
        diag = (((xx + yy) // period) % 2).astype(np.float32)
        a, b = np.array([200, 60, 160], np.float32), np.array([250, 240, 200], np.float32)
        col = a * diag[..., None] + b * (1 - diag[..., None])
        col *= (0.75 + 0.4 * n[..., None])
    elif kind == "rug":
        col = _noise_colors(rng, size, size, (120, 30, 40), (230, 200, 120), size // 4)
        img = _to_image(col)
        d = ImageDraw.Draw(img)
        for i, c in enumerate([NEAR_WHITE, (30, 50, 120), NEAR_BLACK, (240, 180, 40)]):
            m = 12 + i * size // 12
            d.rectangle([m, m, size - m, size - m], outline=c, width=6)
        c = size // 2
        d.polygon([(c, c - size // 6), (c + size // 6, c), (c, c + size // 6), (c - size // 6, c)], fill=(20, 60, 110))
        return img
    else:
        raise ValueError(f"unknown texture kind {kind!r}")
    img = _to_image(col)
    d = ImageDraw.Draw(img)
    for _ in range(40):
        r, x, y = _int(rng, 2, size // 32 + 3), _int(rng, 0, size), _int(rng, 0, size)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(NEAR_BLACK, NEAR_WHITE)[_int(rng, 0, 2)])
    return img
