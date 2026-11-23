"""The model process of the AI texture example.

    python -m examples.ai_texture.worker serve [--fake] [--models DIR] [--idle SECONDS]
    python -m examples.ai_texture.worker fetch --models DIR
    python -m examples.ai_texture.worker check

`serve` reads one JSON request per line on stdin and answers one JSON line per request on stdout, after
writing the PNG the request names; the model loads on the first request and stays on the GPU until stdin
closes (the editor exited) or nothing arrived for --idle seconds, which frees its VRAM and RAM. Library
output goes to stderr. It runs in its own Python environment (env/pyproject.toml), never in the editor.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TextIO

import numpy as np
import numpy.typing as npt

from examples.ai_texture import model
from examples.ai_texture.core import Request, encode_png, event, response, to_rgba8


class Generator(Protocol):
    device: str
    offloaded: bool

    def generate(self, request: Request) -> tuple[npt.NDArray[np.float32], int]: ...


def write_png(path: Path, image: npt.ArrayLike) -> tuple[int, int]:
    """Write the image atomically (the editor never imports half a file); returns width, height."""
    rgba = to_rgba8(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(encode_png(rgba))
    partial.replace(path)
    return int(rgba.shape[1]), int(rgba.shape[0])


def serve(
    make_generator: Callable[[], Generator],
    lines: queue.Queue[str | None],
    send: Callable[[str], None],
    idle: float | None = None,
) -> str:
    """Answer requests until `lines` yields None (stdin closed) or stays empty for `idle` seconds. Returns why
    it stopped. A request that fails gets {"ok": false, "error": ...}; the process keeps serving."""
    generator: Generator | None = None
    while True:
        try:
            line = lines.get(timeout=idle)
        except queue.Empty:
            return "idle"
        if line is None:
            return "closed"
        if not line.strip():
            continue
        try:
            request = Request.from_line(line)
        except (ValueError, TypeError) as exc:
            send(response(-1, ok=False, error=f"bad request: {exc}"))
            continue
        started = time.perf_counter()
        try:
            if generator is None:
                send(event("loading"))
                generator = make_generator()
                send(event("loaded", device=generator.device, offloaded=generator.offloaded,
                           seconds=round(time.perf_counter() - started, 2)))  # fmt: skip
                started = time.perf_counter()
            image, seed = generator.generate(request)
            width, height = write_png(Path(request.out), image)
        except Exception as exc:  # reported to the editor; the next request may well succeed
            traceback.print_exc(file=sys.stderr)
            send(response(request.id, ok=False, error=f"{type(exc).__name__}: {exc}"))
            continue
        seconds = round(time.perf_counter() - started, 2)
        send(response(request.id, ok=True, path=request.out, seed=seed, width=width, height=height,
                      seconds=seconds))  # fmt: skip


def _read_lines(stream: TextIO, lines: queue.Queue[str | None]) -> None:
    for line in stream:
        lines.put(line)
    lines.put(None)


def _serve_main(args: argparse.Namespace) -> int:
    protocol = sys.stdout
    sys.stdout = sys.stderr  # print() in libraries must not corrupt the protocol

    def send(text: str) -> None:
        protocol.write(text)
        protocol.flush()

    def make() -> Generator:
        if args.fake:
            return model.ProceduralGenerator(args.delay)
        return model.VegaGenerator(Path(args.models), lambda text: print(text, file=sys.stderr), args.offload)

    lines: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=_read_lines, args=(sys.stdin, lines), daemon=True).start()
    reason = serve(make, lines, send, args.idle if args.idle > 0 else None)
    send(event("exiting", reason=reason))
    return 0


def _check_main(_args: argparse.Namespace) -> int:
    """Report what the environment has (torch build, CUDA, free VRAM) without loading the model."""
    try:
        import torch
    except ImportError:
        print("torch: not installed (set up env/pyproject.toml; see README.md)")
        return 1
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA: not available")
        return 1
    free, total = torch.cuda.mem_get_info()
    print(f"GPU {torch.cuda.get_device_name(0)}: {free / 1024**3:.1f} of {total / 1024**3:.1f} GB free")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.ai_texture.worker")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve", help="answer generation requests on stdin/stdout")
    serve_parser.add_argument("--fake", action="store_true", help="procedural stand-in, no torch needed")
    serve_parser.add_argument("--delay", type=float, default=0.0, help="--fake: seconds per image")
    serve_parser.add_argument("--models", default=os.environ.get("AI_TEXTURE_MODELS", ""))
    serve_parser.add_argument(
        "--offload",
        choices=model.OFFLOAD_MODES,
        default=os.environ.get("AI_TEXTURE_OFFLOAD", "auto"),
        help="what waits in RAM: auto (by free VRAM), none, encoders (the text encoders), model (all)",
    )
    serve_parser.add_argument(
        "--idle", type=float, default=600.0, help="exit after this long idle (0: never)"
    )
    fetch_parser = commands.add_parser("fetch", help="download the model files (3.7 GB)")
    fetch_parser.add_argument("--models", required=True)
    commands.add_parser("check", help="report torch, CUDA and free VRAM")
    args = parser.parse_args(argv)
    if args.command == "serve":
        if not args.fake and not args.models:
            parser.error("serve needs --models (or AI_TEXTURE_MODELS) unless --fake")
        return _serve_main(args)
    if args.command == "fetch":
        model.fetch(Path(args.models))
        return 0
    return _check_main(args)


if __name__ == "__main__":
    sys.exit(main())
