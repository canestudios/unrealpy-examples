"""The whole example in one command, from the project folder:

    uv run python -m examples.security_camera.demo [--packaged] [--clip SECONDS]

1. a headless editor (UnrealEditor-Cmd, rendering offscreen, no window) starts Play In Editor with
   SecurityMode, lets the mannequins walk their rounds, measures the frame time and the
   vision's cost in real time, then shoots the clip frame by frame through a camera at 1920x1080 (HighResShot)
   (game.py:pie); with --packaged the packaged game does the same (game.py:demo, no editor);
2. ffmpeg (from the imageio-ffmpeg wheel, fetched by uv) turns the frames into an MP4 and the last frame
   into a compressed JPEG.

Prints one JSON summary. No Unreal and no OpenCV in this process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from examples import showreel  # Content/Python/unrealpy is on the project environment's path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--packaged", action="store_true", help="run the packaged game instead of PIE")
    parser.add_argument(
        "--packaged-dir", help="the archive of `unrealpy package --archive-dir` (default: Saved/Packaged)"
    )
    parser.add_argument("--clip", type=float, default=12.0, help="seconds of clip (0: none)")
    parser.add_argument(
        "--measure", type=float, default=0.0, help="seconds of frame-time measurement (0: none)"
    )
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, help="where the MP4 and JPEG go (default: Saved/SecurityCamera)")
    options = parser.parse_args(argv)
    extra = ["--", "--clip", str(options.clip), "--measure", str(options.measure)]
    if options.packaged:
        where = ["--packaged-dir", options.packaged_dir] if options.packaged_dir else []
        made = showreel.run_unrealpy(
            [
                "--packaged",
                *where,
                "--render",
                "--timeout",
                "2400",
                "examples.security_camera.game:demo",
                *extra,
            ],
            options.project,
        )
    else:
        made = showreel.run_unrealpy(
            ["--editor", "--render", "--timeout", "2400", "examples.security_camera.game:pie", *extra],
            options.project,
        )
    value = made["value"]
    out = options.out or options.project / "Saved" / "SecurityCamera"
    name = "security_camera_packaged" if options.packaged else "security_camera_pie"
    ffmpeg = showreel.find_ffmpeg()
    summary: dict[str, Any] = {
        "where": value["where"],
        "unreal_seconds": made["seconds"],
        "perf": value["perf"],
    }
    if value.get("recording"):
        clip = showreel.encode_clip(Path(value["recording"]["folder"]), 30, out / f"{name}.mp4", ffmpeg)
        summary["clip"] = str(clip)
        summary["clip_frames"] = value["recording"]["frames"]
    summary["screenshot"] = str(showreel.encode_still(Path(value["still"]), out / f"{name}.jpg", ffmpeg))
    summary["still_png"] = value["still"]
    summary["status"] = {key: value.get(key) for key in ("vision_frames", "tracks", "alarm", "target")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
