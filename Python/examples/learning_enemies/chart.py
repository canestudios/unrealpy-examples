"""The learning curve as a picture: numpy only (no plotting library in the game), encoded as a PNG with the
standard library's zlib, ready for ImportBufferAsTexture2D."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]

BACKGROUND = (14, 18, 26)
GRID = (40, 48, 62)
CURVE = (255, 170, 40)
LOSS = (90, 200, 255)


def polyline(
    image: Image, points: Sequence[tuple[float, float]], colour: tuple[int, int, int], width: int = 2
) -> None:
    """Draw connected segments: every half pixel along them, a (width x width) square (vectorised)."""
    if not points:
        return
    radius = max(0, width // 2)
    xy = np.asarray(points, dtype=np.float64)
    if len(xy) == 1:
        samples = xy
    else:
        starts, ends = xy[:-1], xy[1:]
        counts = np.maximum(1, (np.abs(ends - starts).max(axis=1) * 2).astype(int)) + 1
        index = np.repeat(np.arange(len(starts)), counts)
        first = np.cumsum(counts) - counts  # where each segment's samples start
        t = (np.arange(int(counts.sum())) - first[index]) / (counts[index] - 1)
        samples = starts[index] + (ends[index] - starts[index]) * t[:, None]
    xs, ys = np.rint(samples[:, 0]).astype(int), np.rint(samples[:, 1]).astype(int)
    height, width_px = image.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = xs + dx, ys + dy
            inside = (x >= 0) & (x < width_px) & (y >= 0) & (y < height)
            image[y[inside], x[inside]] = colour


def learning_chart(
    curve: Sequence[float], losses: Sequence[float], *, width: int = 640, height: int = 320, margin: int = 16
) -> Image:
    """RGB picture: the dodge rate per episode (0..1, orange) over a grid at 25/50/75 %, and the training
    loss (blue, scaled to its own maximum) underneath."""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    left, right, top, bottom = margin, width - margin, margin, height - margin
    for level in (0.25, 0.5, 0.75):
        y = bottom - (bottom - top) * level
        image[round(y), left:right] = GRID
    image[top:bottom, left] = GRID
    image[bottom, left:right] = GRID

    def series(values: Sequence[float], scale: float) -> list[tuple[float, float]]:
        if not values:
            return []
        span = max(1, len(values) - 1)
        return [
            (left + (right - left) * i / span, bottom - (bottom - top) * min(max(v / scale, 0.0), 1.0))
            for i, v in enumerate(values)
        ]

    if losses:
        polyline(image, series(_thin(losses, right - left), max(max(losses), 1e-6)), LOSS, 1)
    if curve:
        polyline(image, series(_thin(curve, right - left), 1.0), CURVE, 3)
    return image


def _thin(values: Sequence[float], count: int) -> list[float]:
    """At most `count` values, evenly picked (a long run still draws in constant time)."""
    if len(values) <= count:
        return list(values)
    step = (len(values) - 1) / (count - 1)
    return [values[round(i * step)] for i in range(count)]


def encode_png(image: Image) -> bytes:
    """An 8-bit RGB PNG (filter 0 rows, zlib level 1: flat charts compress well anyway)."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("encode_png needs an H x W x 3 uint8 image")
    height, width = image.shape[:2]
    rows = np.concatenate([np.zeros((height, 1), np.uint8), image.reshape(height, width * 3)], axis=1)

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 1))
        + chunk(b"IEND", b"")
    )
