"""The whole example in one command, from the project folder:

    uv run python Python/examples/ai_texture/demo.py "mossy cobblestone" [--fake] [--seed N]

1. a headless editor (UnrealEditor-Cmd, no window) asks the model process for the texture, imports it as
   /Game/AI/Textures/T_AI_<prompt>, makes /Game/AI/Materials/MI_AI_<prompt> and puts it on the shapes of the
   demo level /Game/AI/AITexture_Demo (editor.py:main);
2. the game, rendered offscreen, opens that level, looks through its camera and saves the viewport as a PNG
   with ue.agent.capture_viewport (editor.py:capture).

The two steps run one after the other (never two Unreal processes at once). Prints one JSON summary.
No Unreal and no torch in this process: it only runs `unrealpy run` twice.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

EDITOR = "Python/examples/ai_texture/editor.py"
DEMO_MAP = "/Game/AI/AITexture_Demo"  # editor.DEMO_MAP (editor.py imports ue, so it is not imported here)


def unrealpy() -> str:
    """The project's unrealpy command (next to this interpreter in .venv, else on PATH)."""
    local = Path(sys.executable).with_name("unrealpy.exe" if sys.platform == "win32" else "unrealpy")
    found = str(local) if local.is_file() else shutil.which("unrealpy")
    if found is None:
        raise SystemExit("unrealpy not found: run this with `uv run python ...` from the project folder")
    return found


def run(arguments: list[str], project: Path) -> dict[str, Any]:
    """One `unrealpy run`; its JSON report, or exit with the log's errors."""
    completed = subprocess.run(
        [unrealpy(), "run", *arguments], cwd=project, capture_output=True, text=True, encoding="utf-8"
    )
    try:
        report = json.loads(completed.stdout)
    except ValueError:
        raise SystemExit(
            f"unrealpy run {' '.join(arguments)} failed:\n{completed.stdout}\n{completed.stderr}"
        ) from None
    result = report.get("result") or {}
    if completed.returncode != 0 or not result.get("ok"):
        errors = (report.get("log_summary") or {}).get("errors") or []
        raise SystemExit(
            f"unrealpy run {arguments[-1]} failed (log {report.get('log')}):\n" + "\n".join(errors)
        )
    return {"value": result.get("value"), "seconds": report.get("seconds")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("prompt")
    parser.add_argument("--fake", action="store_true", help="procedural stand-in instead of the model")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    options = parser.parse_args(argv)
    generate = [options.prompt, "--seed", str(options.seed)] + (["--fake"] if options.fake else [])
    made = run(["--editor", "--timeout", "900", f"{EDITOR}:main", "--", *generate], options.project)
    shot = run(["--render", "--map", DEMO_MAP, "--timeout", "300", f"{EDITOR}:capture"], options.project)
    summary = {
        **made["value"],
        "editor_seconds": made["seconds"],
        "capture": shot["value"]["path"],
        "capture_seconds": shot["seconds"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
