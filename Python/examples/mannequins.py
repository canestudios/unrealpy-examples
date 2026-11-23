"""Epic's UE5 mannequins (Manny and Quinn) for the runtime examples, from the engine's own template content.

The repository ships no Unreal assets. The first run copies the mannequin pack (skeletal meshes, materials,
textures, the unarmed animations, the locomotion blend space and anim blueprint, the control rigs) from

    <engine>/Templates/TemplateResources/High/Characters/Content/Mannequins

to the project's Content/Characters/Mannequins, the same package paths the third-person template uses, so
every reference inside the pack stays valid. It is what adding the template's feature pack does, as a
script: reproducible, and skipped when the files are already there. Config/DefaultGame.ini cooks the
folder, so a packaged game has it too.

    uv run python -m examples.mannequins [--engine <engine root>] [--project .]

No Unreal here: it runs before the engine starts (the asset registry then finds the files).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PACK = Path("Templates") / "TemplateResources" / "High" / "Characters" / "Content" / "Mannequins"
DESTINATION = Path("Content") / "Characters" / "Mannequins"
# What the examples load, by package path (the rest of the pack comes along for its references).
MESH_MANNY = "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple.SKM_Manny_Simple"
MESH_QUINN = "/Game/Characters/Mannequins/Meshes/SKM_Quinn_Simple.SKM_Quinn_Simple"
ANIM_CLASS = "/Game/Characters/Mannequins/Anims/Unarmed/ABP_Unarmed.ABP_Unarmed_C"


def engine_root(explicit: str | None = None) -> Path:
    """The engine to copy from: --engine, else UE_ENGINE_ROOT, else the engine whose Python runs this
    (the project's environment is made from the engine's interpreter)."""
    if explicit is not None and explicit != "":
        return Path(explicit)
    configured = os.environ.get("UE_ENGINE_ROOT", "")
    if configured != "":
        return Path(configured)
    home = Path(sys.base_prefix).resolve()  # <engine>/Engine/Binaries/ThirdParty/Python3/Win64
    return home.parents[4]


def plan(source: Path, destination: Path) -> list[tuple[Path, Path]]:
    """The files to copy: every file of the pack missing at the destination or of a different size."""
    copies = []
    for file in sorted(source.rglob("*")):
        if not file.is_file():
            continue
        target = destination / file.relative_to(source)
        if not target.is_file() or target.stat().st_size != file.stat().st_size:
            copies.append((file, target))
    return copies


def install(project: Path, engine: Path) -> dict[str, object]:
    """Copy what is missing; returns what was done."""
    source = engine / PACK
    if not source.is_dir():
        raise FileNotFoundError(f"no mannequin pack at {source} (is this a UE 5 engine with its templates?)")
    destination = project / DESTINATION
    copies = plan(source, destination)
    for file, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    return {
        "source": str(source),
        "destination": str(destination),
        "copied": len(copies),
        "megabytes": round(sum(file.stat().st_size for file, _ in copies) / 2**20, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.mannequins")
    parser.add_argument("--engine", help="engine root (default: UE_ENGINE_ROOT, else this Python's engine)")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    options = parser.parse_args(argv)
    print(json.dumps(install(options.project.resolve(), engine_root(options.engine)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
