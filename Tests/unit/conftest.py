"""The unit tests run outside Unreal, on UnrealPy's engine-less fake of the `ue` module.

They need the UnrealPy plugin (for the fake), in this project's Plugins/UnrealPy or where UNREALPY_PLUGIN
points, and the type database the editor writes at its first start (Intermediate/PythonStubs/ue-types.json;
`unrealpy stubs` makes it too), or where UNREALPY_TYPES points.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PLUGIN = Path(os.environ.get("UNREALPY_PLUGIN", PROJECT / "Plugins" / "UnrealPy")).resolve()
if not (PLUGIN / "Content" / "Python" / "unrealpy_testing").is_dir():
    raise RuntimeError(f"UnrealPy plugin not found at {PLUGIN}: copy it into Plugins/ (Plugins/README.md)")
sys.path.insert(0, str(PLUGIN / "Content" / "Python"))
os.environ.setdefault("UNREALPY_TYPES", str(PROJECT / "Intermediate" / "PythonStubs" / "ue-types.json"))
pytest_plugins = ["unrealpy_testing.pytest_plugin"]
