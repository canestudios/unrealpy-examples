"""The tests run outside Unreal, on UnrealPy's engine-less fake of the `ue` module.

They need the UnrealPy plugin (for the fake) and a type database made by `unrealpy stubs` in a project that
has the plugin. Point at them with UNREALPY_PLUGIN (the plugin folder) and, if the database is not in that
project's Intermediate/PythonStubs/ue-types.json, UNREALPY_TYPES.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _plugin() -> Path:
    named = os.environ.get("UNREALPY_PLUGIN")
    if named:
        return Path(named).resolve()
    return (Path(__file__).resolve().parents[2] / "cogitator" / "Plugins" / "UnrealPy").resolve()


PLUGIN = _plugin()
if not (PLUGIN / "Content" / "Python" / "unrealpy_testing").is_dir():
    raise RuntimeError(f"UnrealPy plugin not found at {PLUGIN}: set UNREALPY_PLUGIN to its folder")
sys.path.insert(0, str(PLUGIN / "Content" / "Python"))
os.environ.setdefault(
    "UNREALPY_TYPES", str(PLUGIN.parents[1] / "Intermediate" / "PythonStubs" / "ue-types.json")
)
pytest_plugins = ["unrealpy_testing.pytest_plugin"]
