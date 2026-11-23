"""The mannequin installer shared by the runtime examples: copies the engine template's mannequin pack."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples import mannequins


def test_mannequin_install_copies_only_what_is_missing(tmp_path: Path) -> None:
    engine, project = tmp_path / "engine", tmp_path / "project"
    pack = engine / mannequins.PACK
    (pack / "Meshes").mkdir(parents=True)
    (pack / "Meshes" / "SKM_Manny_Simple.uasset").write_bytes(b"mesh")
    (pack / "Anims").mkdir()
    (pack / "Anims" / "MM_Idle.uasset").write_bytes(b"idle")
    first = mannequins.install(project, engine)
    assert first["copied"] == 2
    assert (project / mannequins.DESTINATION / "Meshes" / "SKM_Manny_Simple.uasset").read_bytes() == b"mesh"
    assert mannequins.install(project, engine)["copied"] == 0
    (pack / "Anims" / "MM_Idle.uasset").write_bytes(b"idle, longer")
    assert mannequins.plan(pack, project / mannequins.DESTINATION) == [
        (pack / "Anims" / "MM_Idle.uasset", project / mannequins.DESTINATION / "Anims" / "MM_Idle.uasset")
    ]
    with pytest.raises(FileNotFoundError):
        mannequins.install(project, tmp_path / "nowhere")


def test_engine_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert mannequins.engine_root(str(tmp_path)) == tmp_path
    monkeypatch.setenv("UE_ENGINE_ROOT", str(tmp_path / "env"))
    assert mannequins.engine_root() == tmp_path / "env"
