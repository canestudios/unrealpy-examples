"""The whole example in one command, from the project folder:

    uv run python -m examples.live_analytics.demo [--packaged] [--clip SECONDS]

1. a headless editor (UnrealEditor-Cmd, rendering offscreen, no window) starts Play In Editor with
   AnalyticsMode, lets the first playtesters get going, then shoots the clip frame by frame through a
   moving camera at 1920x1080 while the analytics build up (game.py:pie); with --packaged the packaged
   game does the same (game.py:demo, no editor);
2. ffmpeg (from the imageio-ffmpeg wheel, fetched by uv) turns the frames into an MP4 and the last frame
   into a compressed JPEG.

Prints one JSON summary. No Unreal, pandas or matplotlib in this process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from examples import showreel  # Content/Python/gameplay is on the project environment's path

BANNER = "Live playtest analytics: pandas + matplotlib + NumPy inside the game"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--packaged", action="store_true", help="run the packaged game instead of PIE")
    parser.add_argument(
        "--packaged-dir", help="the archive of `unrealpy package --archive-dir` (default: Saved/Packaged)"
    )
    parser.add_argument("--clip", type=float, default=15.0, help="seconds of clip (0: none)")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, help="where the MP4 and JPEG go (default: Saved/LiveAnalytics)")
    options = parser.parse_args(argv)
    extra = ["--", "--clip", str(options.clip)]
    if options.packaged:
        where = ["--packaged-dir", options.packaged_dir] if options.packaged_dir else []
        made = showreel.run_unrealpy(
            [
                "--packaged",
                *where,
                "--render",
                "--timeout",
                "2400",
                "examples.live_analytics.game:demo",
                *extra,
            ],
            options.project,
        )
    else:
        made = showreel.run_unrealpy(
            ["--editor", "--render", "--timeout", "2400", "examples.live_analytics.game:pie", *extra],
            options.project,
        )
    value = made["value"]
    out = options.out or options.project / "Saved" / "LiveAnalytics"
    name = "live_analytics_packaged" if options.packaged else "live_analytics_pie"
    ffmpeg = showreel.find_ffmpeg()
    summary: dict[str, Any] = {
        "where": value["where"],
        "unreal_seconds": made["seconds"],
    }
    if value.get("recording"):
        title = showreel.banner(BANNER, out / "banner.png")
        clip = showreel.encode_clip(
            Path(value["recording"]["folder"]), 30, out / f"{name}.mp4", ffmpeg, title
        )
        summary["clip"] = str(clip)
        summary["clip_frames"] = value["recording"]["frames"]
    summary["screenshot"] = str(showreel.encode_still(Path(value["still"]), out / f"{name}.jpg", ffmpeg))
    summary["still_png"] = value["still"]
    summary["status"] = {
        key: value.get(key)
        for key in (
            "game_seconds",
            "runs",
            "deaths",
            "deaths_on_bridge",
            "finished",
            "samples",
            "renders",
            "render_ms_mean",
            "renderer",
        )
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
