"""The whole example in one command, from the project folder:

    uv run python -m examples.fluid.demo [--packaged [--packaged-dir DIR]] [--clip SECONDS]

1. builds the water's materials in a headless editor if the project lacks them (fluid_assets.py);
2. runs the game (Play In Editor in a headless editor, or the packaged game with --packaged): Warp compiles
   the kernels (or loads them from Saved/WarpCache), the clip is shot frame by frame at 1920x1080 while
   the dam breaks, then the frame rate is measured for a few seconds of real time (game.py);
3. ffmpeg (from the imageio-ffmpeg wheel, fetched by uv) makes the MP4: a 3 s card with one of the kernels'
   Python source, then the clip with a one-line banner; and a JPEG of the last frame.

Prints one JSON summary. No Unreal and no Warp in this process.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from examples import showreel  # Content/Python/unrealpy is on the project environment's path

HERE = Path(__file__).resolve().parent
CARD_SECONDS = 3
CARD_KERNEL = "lambdas"  # the kernel shown on the card


def banner_text(particles: int) -> str:
    return f"NVIDIA Warp: {particles:,}-particle fluid, Python GPU kernels inside the game"


def kernel_source(name: str = CARD_KERNEL) -> str:
    """The source of one @wp.kernel in sim.py, decorator included."""
    text = (HERE / "sim.py").read_text(encoding="utf-8")
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = min(d.lineno for d in node.decorator_list) if node.decorator_list else node.lineno
            lines = text.splitlines()[start - 1 : node.end_lineno]
            return "\n".join(line.split("  #")[0].rstrip() for line in lines)  # no comments
    raise KeyError(name)


def card(out: Path, width: int = 1920, height: int = 1080) -> Path:
    """The title card: the kernel's Python source on a dark background, with a caption (PNG)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#0d1117")
    FigureCanvasAgg(figure)
    figure.text(
        0.06,
        0.92,
        "Python, compiled to a CUDA kernel by NVIDIA Warp, called by the game every frame",
        color="#7ee787",
        fontsize=26,
        fontweight="bold",
        va="top",
    )
    figure.text(0.06, 0.84, kernel_source(), color="#e6edf3", fontsize=17, family="monospace", va="top")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, format="png", facecolor=figure.get_facecolor())
    return out


def clip_command(ffmpeg: str, frames: Path, fps: int, title: Path, banner: Path, out: Path) -> list[str]:
    """The card for CARD_SECONDS, then the frames with the banner over the top, as one H.264 MP4."""
    graph = (
        "[0:v]scale=1920:1080,format=yuv420p,setsar=1[card];"
        "[1:v]scale=1920:1080[frames];[frames][2:v]overlay=0:0,format=yuv420p,setsar=1[clip];"
        "[card][clip]concat=n=2:v=1:a=0[out]"
    )
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "fatal",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-t",
        str(CARD_SECONDS),
        "-i",
        str(title),
        "-framerate",
        str(fps),
        "-i",
        str(frames / "frame_%05d.png"),
        "-i",
        str(banner),
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]


def ensure_assets(project: Path) -> None:
    folder = project / "Content" / "Fluid"
    if all((folder / f"{name}.uasset").is_file() for name in ("M_FluidParticle", "M_FluidSurface")):
        return
    completed = subprocess.run(
        [*showreel.unrealpy(project), "run", "--editor", "Content/Python/editor/fluid_assets.py:main"],
        cwd=project,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise SystemExit(f"building the fluid material failed:\n{completed.stdout[-3000:]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--packaged", action="store_true", help="run the packaged game instead of PIE")
    parser.add_argument("--packaged-dir", help="the archive of `unrealpy package --archive-dir`")
    parser.add_argument("--clip", type=float, default=16.0, help="seconds of clip (0: none)")
    parser.add_argument("--measure", type=float, default=4.0, help="seconds of real-time frame rate")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, help="where the MP4 and JPEG go (default: Saved/Fluid)")
    options = parser.parse_args(argv)
    project = options.project.resolve()
    extra = ["--", "--clip", str(options.clip), "--measure", str(options.measure)]
    if options.packaged:
        where = ["--packaged-dir", options.packaged_dir] if options.packaged_dir else []
        target = ["--packaged", *where, "--render", "--timeout", "2400", "examples.fluid.game:demo"]
    else:
        ensure_assets(project)
        target = ["--editor", "--render", "--timeout", "2400", "examples.fluid.game:pie"]
    made = showreel.run_unrealpy([*target, *extra], project)
    value = made["value"]
    out = options.out or project / "Saved" / "Fluid"
    name = "fluid_packaged" if options.packaged else "fluid_pie"
    ffmpeg = showreel.find_ffmpeg()
    summary: dict[str, Any] = {
        "where": value["where"],
        "unreal_seconds": made["seconds"],
        "perf": value["perf"],
    }
    if value.get("recording"):
        title = card(out / "card.png")
        banner = showreel.banner(banner_text(value["particles"]), out / "banner.png")
        clip = out / f"{name}.mp4"
        command = clip_command(ffmpeg, Path(value["recording"]["folder"]), 30, title, banner, clip)
        subprocess.run(command, check=True)
        summary["clip"] = str(clip)
        summary["clip_frames"] = value["recording"]["frames"]
    summary["screenshot"] = str(showreel.encode_still(Path(value["still"]), out / f"{name}.jpg", ffmpeg))
    summary["still_png"] = value["still"]
    summary["status"] = {
        key: value.get(key)
        for key in (
            "particles",
            "sim_seconds",
            "warp_setup_seconds",
            "fallen",
            "sim_ms_mean",
            "frame_ms_mean",
        )
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
