"""Every example writes its clips and pictures under Unreal's saved directory, which a packaged game can write
to wherever it is installed (the project folder may be read-only)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import ue
from examples import stage


@pytest.mark.parametrize(
    ("example", "folder"),
    [
        ("fluid", "Fluid"),
        ("learning_enemies", "LearningEnemies"),
        ("live_analytics", "LiveAnalytics"),
        ("security_camera", "SecurityCamera"),
    ],
)
def test_output_uses_unreal_saved_dir(
    example: str, folder: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stage, "shape", lambda _name: None)
    game = importlib.import_module(f"examples.{example}.game")

    saved = tmp_path / "user" / "Saved"
    monkeypatch.setattr(ue, "saved_dir", lambda: str(saved))
    monkeypatch.setattr(ue, "project_dir", lambda: str(tmp_path / "install"))
    assert game._out_dir() == saved.resolve() / folder
