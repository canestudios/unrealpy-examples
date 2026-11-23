"""Pure parts of the AI texture example, shared by the editor and the model process: no Unreal, no torch.

Asset naming, the request/response wire format (one JSON object per line), pixel conversion and PNG encoding,
and the procedural stand-in model the example runs with before any weights are downloaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

TEXTURE_PREFIX = "T_AI_"
INSTANCE_PREFIX = "MI_AI_"
MAX_STEM = 40  # Unreal asset names may be longer, but the Content Browser tile shows ~40 characters
MIN_SIZE, MAX_SIZE = 256, 1024
MAX_STEPS = 8  # the LCM adapter is distilled for 2-8 steps
# Appended to the user's words unless they already describe a texture; SDXL-family models otherwise tend to
# draw a scene (horizon, perspective) instead of a flat, evenly lit surface.
TEXTURE_SUFFIX = "seamless tileable texture, top-down orthographic view, flat even lighting, highly detailed"


# -- naming ------------------------------------------------------------------------------------------------


def slug(prompt: str, limit: int = MAX_STEM) -> str:
    """ASCII words of the prompt joined by underscores, cut at a word boundary within `limit` characters."""
    ascii_text = unicodedata.normalize("NFKD", prompt).encode("ascii", "ignore").decode("ascii")
    words = [str(word) for word in re.findall(r"[a-z0-9]+", ascii_text.lower())]
    stem = ""
    for word in words:
        candidate = f"{stem}_{word}" if stem else word
        if len(candidate) > limit:
            break
        stem = candidate
    if not stem and words:
        stem = words[0][:limit]  # a single word longer than the limit
    return stem or "texture"


def asset_name(prompt: str, exists: Callable[[str], bool], prefix: str = TEXTURE_PREFIX) -> str:
    """A new asset name for the prompt: T_AI_mossy_cobblestone, then _2, _3... while `exists(name)`."""
    base = prefix + slug(prompt)
    name, number = base, 2
    while exists(name):
        name = f"{base}_{number}"
        number += 1
    return name


def instance_name(texture_name: str) -> str:
    """The material instance that shows a generated texture: T_AI_x -> MI_AI_x."""
    stem = texture_name.removeprefix(TEXTURE_PREFIX)
    return INSTANCE_PREFIX + stem


def texture_prompt(prompt: str) -> str:
    """The prompt sent to the model: the user's words, plus the texture framing unless already there."""
    text = " ".join(prompt.split())
    if not text:
        raise ValueError("the prompt is empty")
    if "texture" in text.lower():
        return text
    return f"{text}, {TEXTURE_SUFFIX}"


# -- wire format -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """One generation. `out` is the PNG the model process writes; seed -1 picks a random one."""

    id: int
    prompt: str
    out: str
    size: int = 1024
    steps: int = 4
    seed: int = -1
    seamless: bool = True

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("the prompt is empty")
        if not self.out:
            raise ValueError("no output path")
        if not MIN_SIZE <= self.size <= MAX_SIZE or self.size % 64:
            raise ValueError(f"size must be a multiple of 64 in {MIN_SIZE}..{MAX_SIZE}, not {self.size}")
        if not 1 <= self.steps <= MAX_STEPS:
            raise ValueError(f"steps must be in 1..{MAX_STEPS}, not {self.steps}")
        if self.seed < -1 or self.seed >= 2**32:
            raise ValueError(f"seed must be -1 (random) or in 0..2**32-1, not {self.seed}")

    def to_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False) + "\n"

    @classmethod
    def from_line(cls, line: str) -> Request:
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError("a request is a JSON object")
        fields = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**fields)


def response(request_id: int, **fields: Any) -> str:
    """A response line of the model process: {"id": ..., "ok": ..., ...}."""
    return json.dumps({"id": request_id, **fields}, ensure_ascii=False) + "\n"


def event(name: str, **fields: Any) -> str:
    """A progress line of the model process that answers no request (loading, loaded, exiting)."""
    return json.dumps({"event": name, **fields}, ensure_ascii=False) + "\n"


# -- pixels ------------------------------------------------------------------------------------------------


def to_rgba8(image: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """An H x W x 4 uint8 array from what image models return: H x W gray, H x W x 1/3/4 channels, floats in
    [0, 1] (clipped) or uint8. NaN pixels are an error: SDXL's VAE overflows in fp16 and decodes black."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4) or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"expected an H x W, H x W x 1, 3 or 4 image, not shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("the image has NaN or infinite pixels (fp16 VAE overflow?)")
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError(f"expected float or uint8 pixels, not {array.dtype}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] == 3:
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
        array = np.concatenate([array, alpha], axis=2)
    return np.ascontiguousarray(array)


def encode_png(rgba: npt.NDArray[np.uint8]) -> bytes:
    """An 8-bit RGBA PNG of an H x W x 4 uint8 array (no Pillow needed, so the editor side can write one)."""
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected H x W x 4 uint8 pixels, not {rgba.dtype} {rgba.shape}")
    height, width = rgba.shape[:2]

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = np.zeros((height, width * 4 + 1), dtype=np.uint8)  # filter byte 0 (None) before every row
    rows[:, 1:] = rgba.reshape(height, width * 4)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8 bits, RGBA, deflate, no interlace
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + chunk(b"IEND", b"")
    )


# -- stand-in model ----------------------------------------------------------------------------------------


def procedural_texture(prompt: str, size: int, seed: int) -> npt.NDArray[np.float32]:
    """A seamless H x W x 3 image in [0, 1] from the prompt and seed: sums of sine waves with whole periods
    across the image (so it tiles), coloured by a hash of the prompt. Lets the whole editor flow run before
    torch and the weights are installed (`--fake`)."""
    digest = hashlib.sha256(f"{prompt}\0{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    axis = np.arange(size, dtype=np.float32) * (2.0 * math.pi / size)
    u, v = np.meshgrid(axis, axis, indexing="xy")
    field = np.zeros((size, size), dtype=np.float32)
    for octave in range(5):
        for _ in range(3):
            fx, fy = rng.integers(-(2 << octave), (2 << octave) + 1, size=2)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            field += np.sin(fx * u + fy * v + phase).astype(np.float32) / (octave + 1)
    field = (field - field.min()) / max(float(np.ptp(field)), 1e-6)
    dark = np.frombuffer(digest[8:11], dtype=np.uint8).astype(np.float32) / 255.0 * 0.5
    light = 0.5 + np.frombuffer(digest[11:14], dtype=np.uint8).astype(np.float32) / 255.0 * 0.5
    return dark + field[:, :, None] * (light - dark)
