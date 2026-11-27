"""Shared by the runtime examples, outside the engine: frame-time statistics and the clip encoder.

No Unreal here: `PerfMeter` is fed by the game and summarised anywhere; `encode_clip` and `encode_still` run
ffmpeg (from the imageio-ffmpeg wheel, fetched by uv on demand) on the PNG frames a recording wrote.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

IMAGEIO_FFMPEG = "imageio-ffmpeg==0.6.0"
# Lumen (software tracing) needs mesh distance fields, a startup setting: this engine argument turns them on.


def percentile(values: Sequence[float], fraction: float) -> float:
    """The nearest-rank percentile (fraction in [0, 1]) of a non-empty sequence."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class PerfMeter:
    """Frame times and the cost of named Python systems, in milliseconds."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self.frames: list[float] = []
        self.costs: dict[str, list[float]] = defaultdict(list)
        self._last: float | None = None

    def frame(self) -> None:
        """Call once per engine frame: records the time since the previous call."""
        now = self._clock()
        if self._last is not None:
            self.frames.append((now - self._last) * 1000.0)
        self._last = now

    def add(self, name: str, milliseconds: float) -> None:
        self.costs[name].append(milliseconds)

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        started = self._clock()
        try:
            yield
        finally:
            self.add(name, (self._clock() - started) * 1000.0)

    def reset(self) -> None:
        self.frames.clear()
        self.costs.clear()
        self._last = None

    def summary(self) -> dict[str, Any]:
        """fps and frame-time percentiles; per system: calls, mean and max ms, and mean ms per frame."""
        out: dict[str, Any] = {"frames": len(self.frames)}
        if self.frames:
            mean = sum(self.frames) / len(self.frames)
            out.update(
                fps=round(1000.0 / mean, 1),
                frame_ms_mean=round(mean, 2),
                frame_ms_p95=round(percentile(self.frames, 0.95), 2),
                frame_ms_max=round(max(self.frames), 2),
            )
        systems = {}
        for name, samples in self.costs.items():
            systems[name] = {
                "calls": len(samples),
                "mean_ms": round(sum(samples) / len(samples), 3),
                "max_ms": round(max(samples), 3),
                "ms_per_frame": round(sum(samples) / max(1, len(self.frames)), 3),
            }
        out["systems"] = systems
        return out


# -- running Unreal (outside the engine) ---------------------------------------------------------------------


def unrealpy(project: Path) -> list[str]:
    """The `unrealpy` command: installed in the environment, else the plugin's own CLI run through uvx."""
    local = Path(sys.executable).with_name("unrealpy.exe" if sys.platform == "win32" else "unrealpy")
    found = str(local) if local.is_file() else shutil.which("unrealpy")
    if found is not None:
        return [found]
    plugin = project / "Plugins" / "UnrealPy"
    uvx = shutil.which("uvx")
    if uvx:
        for tools_dir in (plugin / "Content" / "Python" / "Tools", plugin / "Tools"):
            cli = tools_dir / "unrealpy_cli"
            if cli.is_dir():
                return [uvx, "--from", str(cli), "unrealpy"]
    raise SystemExit("Install the UnrealPy CLI from your plugin or add it to PATH (see README.md).")


def run_unrealpy(arguments: list[str], project: Path) -> dict[str, Any]:
    """One `unrealpy run`; its JSON report, or exit with the log's errors. Before it: the mannequins are
    copied from the engine's template if the project lacks them."""
    from examples import mannequins

    mannequins.install(project.resolve(), mannequins.engine_root())
    completed = subprocess.run(
        [*unrealpy(project), "run", *arguments],
        cwd=project,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    try:
        report = json.loads(completed.stdout[completed.stdout.index("{") :])
    except ValueError:
        raise SystemExit(f"unrealpy run failed:\n{completed.stdout}\n{completed.stderr}") from None
    result = report.get("result") or {}
    if completed.returncode != 0 or not result.get("ok"):
        errors = (report.get("log_summary") or {}).get("errors") or []
        raise SystemExit(
            f"unrealpy run failed (log {report.get('log')}):\n{result.get('error')}\n" + "\n".join(errors)
        )
    return {"value": result.get("value"), "seconds": report.get("seconds")}


# -- encoding (outside the engine) ---------------------------------------------------------------------------


def find_ffmpeg() -> str:
    """ffmpeg from the imageio-ffmpeg wheel (uv fetches it into its cache; nothing enters the project)."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv not found on PATH")
    completed = subprocess.run(
        [
            uv,
            "run",
            "--no-project",
            "--with",
            IMAGEIO_FFMPEG,
            "python",
            "-c",
            "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stdout.strip().splitlines()[-1]


def clip_command(
    ffmpeg: str, frames: Path, fps: int, out: Path, *, crf: int = 22, banner: Path | None = None
) -> list[str]:
    """H.264 MP4 from frames/frame_00000.png... (yuv420p: plays everywhere), with `banner` (a PNG with alpha)
    over the top if given. yuv420p needs even sizes: an odd frame (a Play In Editor viewport can be one) loses
    its last row or column. Unreal pads its exported PNGs after the image end, which ffmpeg reports but
    decodes fine: only fatal errors are shown."""
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "fatal",
        "-framerate",
        str(fps),
        "-i",
        str(frames / "frame_%05d.png"),
    ]
    even = "crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0"
    if banner is not None:
        command += ["-i", str(banner), "-filter_complex", f"[0:v]{even}[frames];[frames][1:v]overlay=0:0"]
    else:
        command += ["-vf", even]
    return [
        *command,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]


def banner(text: str, out: Path, width: int = 1920, height: int = 84) -> Path:
    """A one-line caption across the top of a clip: white bold text on a dark translucent band (PNG)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width / 100, height / 100), dpi=100)
    figure.patch.set_facecolor((0.02, 0.03, 0.05, 0.62))
    FigureCanvasAgg(figure)
    figure.text(
        0.5, 0.5, text, color="white", fontsize=height * 0.38, fontweight="bold", ha="center", va="center"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, format="png", facecolor=figure.get_facecolor())
    return out


def still_command(ffmpeg: str, source: Path, out: Path, *, width: int = 1280, quality: int = 4) -> list[str]:
    """A compressed JPEG of one frame, `width` pixels wide (for docs/images)."""
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "fatal",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        str(quality),
        str(out),
    ]


def encode_clip(
    frames: Path, fps: int, out: Path, ffmpeg: str | None = None, banner: Path | None = None
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(clip_command(ffmpeg or find_ffmpeg(), frames, fps, out, banner=banner), check=True)
    return out


def encode_still(source: Path, out: Path, ffmpeg: str | None = None, width: int = 1280) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(still_command(ffmpeg or find_ffmpeg(), source, out, width=width), check=True)
    return out
