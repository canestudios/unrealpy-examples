"""AI texture example: the parts that need neither torch nor the engine (naming, wire format, pixels, PNG,
the model process with its procedural stand-in, the editor's import task on the fake `ue`)."""

from __future__ import annotations

import json
import os
import queue
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.ai_texture import client as worker_client
from examples.ai_texture import core, model, worker


def _decode_png(data: bytes) -> np.ndarray:
    """RGBA pixels of a PNG written by core.encode_png, checking every chunk's CRC."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position, idat, size = 8, b"", (0, 0)
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        body = data[position + 8 : position + 8 + length]
        (crc,) = struct.unpack(">I", data[position + 8 + length : position + 12 + length])
        assert crc == zlib.crc32(kind + body)
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            assert (depth, colour) == (8, 6)
            size = (height, width)
        elif kind == b"IDAT":
            idat += body
        position += 12 + length
    rows = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(size[0], size[1] * 4 + 1)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].reshape(size[0], size[1], 4)


# -- naming --------------------------------------------------------------------------------------------------


def test_asset_name_is_ascii_words_and_unique() -> None:
    taken = {"T_AI_mossy_cobblestone", "T_AI_mossy_cobblestone_2"}
    assert core.asset_name("Mossy  cobblestone!", taken.__contains__) == "T_AI_mossy_cobblestone_3"
    assert core.asset_name("Pietra sbozzata, città", lambda _: False) == "T_AI_pietra_sbozzata_citta"
    assert core.asset_name("!!! ???", lambda _: False) == "T_AI_texture"


def test_asset_name_cuts_long_prompts_at_a_word_boundary() -> None:
    stem = core.slug("ancient " * 20)
    assert len(stem) <= core.MAX_STEM and not stem.endswith("_") and set(stem.split("_")) == {"ancient"}
    assert core.slug("x" * 100) == "x" * core.MAX_STEM


def test_instance_name_and_texture_prompt() -> None:
    assert core.instance_name("T_AI_lava_rock_2") == "MI_AI_lava_rock_2"
    assert core.texture_prompt("  lava   rock ") == f"lava rock, {core.TEXTURE_SUFFIX}"
    assert core.texture_prompt("Brick texture, red") == "Brick texture, red"
    with pytest.raises(ValueError, match="empty"):
        core.texture_prompt("   ")


# -- wire format ---------------------------------------------------------------------------------------------


def test_request_round_trips_and_validates() -> None:
    request = core.Request(id=3, prompt="città", out="textures/x.png", size=512, steps=2, seed=7, seamless=False)
    assert core.Request.from_line(request.to_line()) == request
    assert core.Request.from_line('{"id": 1, "prompt": "a", "out": "o.png", "extra": 1}').size == 1024
    for bad in ({"size": 1000}, {"size": 2048}, {"steps": 0}, {"steps": 9}, {"seed": -2}, {"prompt": " "}):
        with pytest.raises(ValueError):
            fields: dict[str, Any] = {"id": 1, "prompt": "a", "out": "o.png", **bad}
            core.Request(**fields)
    with pytest.raises(ValueError):
        core.Request.from_line("[1, 2]")


# -- pixels and PNG ------------------------------------------------------------------------------------------


def test_to_rgba8_converts_model_outputs() -> None:
    floats = np.array([[[0.0, 0.5, 1.2], [-1.0, 0.25, 1.0]]], dtype=np.float32)  # 1 x 2 x 3
    rgba = core.to_rgba8(floats)
    assert rgba.dtype == np.uint8 and rgba.shape == (1, 2, 4)
    assert rgba.tolist() == [[[0, 128, 255, 255], [0, 64, 255, 255]]]
    gray = core.to_rgba8(np.full((2, 2), 10, dtype=np.uint8))
    assert gray[0, 0].tolist() == [10, 10, 10, 255]
    four = np.arange(16, dtype=np.uint8).reshape(2, 2, 4)
    assert np.array_equal(core.to_rgba8(four), four)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.full((2, 2, 3), np.nan, dtype=np.float32), "NaN"),
        (np.zeros((2, 2, 2), dtype=np.float32), "image"),
        (np.zeros((0, 2, 3), dtype=np.float32), "image"),
        (np.zeros((2, 2, 3), dtype=np.int32), "uint8"),
    ],
)
def test_to_rgba8_rejects_unusable_images(image: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        core.to_rgba8(image)


def test_encode_png_round_trips() -> None:
    rgba = np.random.default_rng(1).integers(0, 256, size=(5, 7, 4), dtype=np.uint8)
    assert np.array_equal(_decode_png(core.encode_png(rgba)), rgba)
    with pytest.raises(ValueError):
        core.encode_png(rgba[:, :, :3].copy())


def test_procedural_texture_is_deterministic_and_seamless() -> None:
    image = core.procedural_texture("lava", 128, seed=5)
    assert image.shape == (128, 128, 3) and image.min() >= 0.0 and image.max() <= 1.0
    assert np.array_equal(image, core.procedural_texture("lava", 128, seed=5))
    assert not np.array_equal(image, core.procedural_texture("lava", 128, seed=6))
    # Across the wrap seam the image changes no more than between any two neighbouring columns or rows.
    inner_x = np.abs(np.diff(image, axis=1)).max()
    inner_y = np.abs(np.diff(image, axis=0)).max()
    assert np.abs(image[:, 0] - image[:, -1]).max() <= inner_x * 1.01
    assert np.abs(image[0] - image[-1]).max() <= inner_y * 1.01


# -- model process -------------------------------------------------------------------------------------------


def _serve(
    lines: list[str | None], make: Any = None, idle: float | None = None
) -> tuple[list[dict[str, Any]], str]:
    feed: queue.Queue[str | None] = queue.Queue()
    for line in lines:
        feed.put(line)
    sent: list[dict[str, Any]] = []
    reason = worker.serve(
        make or model.ProceduralGenerator, feed, lambda text: sent.append(json.loads(text)), idle
    )
    return sent, reason


def test_serve_writes_pngs_and_answers_each_request(tmp_path: Path) -> None:
    first = core.Request(id=1, prompt="moss", out=str(tmp_path / "a" / "T_AI_moss.png"), size=256, seed=3)
    second = core.Request(id=2, prompt="sand", out=str(tmp_path / "T_AI_sand.png"), size=256)
    sent, reason = _serve([first.to_line(), "\n", "not json\n", second.to_line(), None])
    assert reason == "closed"
    assert [message.get("event") for message in sent[:2]] == ["loading", "loaded"]
    answers = {message["id"]: message for message in sent if "id" in message}
    assert answers[-1]["ok"] is False and "bad request" in answers[-1]["error"]
    assert (
        answers[1]["ok"]
        and answers[1]["seed"] == 3
        and (answers[1]["width"], answers[1]["height"]) == (256, 256)
    )
    assert answers[2]["ok"] and 0 <= answers[2]["seed"] < 2**32
    png = _decode_png(Path(first.out).read_bytes())
    expected = core.to_rgba8(core.procedural_texture("moss", 256, 3))
    assert np.array_equal(png, expected)
    assert not list(tmp_path.rglob("*.partial"))


def test_serve_reports_failures_and_keeps_going(tmp_path: Path) -> None:
    class Flaky:
        device, offloaded = "test", False
        calls = 0

        def generate(self, request: core.Request) -> tuple[np.ndarray, int]:
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("out of video memory")
            return np.zeros((64, 64, 3), dtype=np.float32), 0

    requests = [core.Request(id=n, prompt="x", out=str(tmp_path / f"{n}.png"), size=256) for n in (1, 2)]
    sent, _ = _serve([r.to_line() for r in requests] + [None], make=Flaky)
    answers = {message["id"]: message for message in sent if "id" in message}
    assert answers[1] == {"id": 1, "ok": False, "error": "RuntimeError: out of video memory"}
    assert answers[2]["ok"] and (tmp_path / "2.png").is_file()


def test_serve_stops_when_idle_without_loading_the_model() -> None:
    def never() -> worker.Generator:
        raise AssertionError("the model loads on the first request only")

    sent, reason = _serve([], make=never, idle=0.05)
    assert (sent, reason) == ([], "idle")


def test_client_drives_the_real_process_with_the_stand_in(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]  # the repository: Python/examples, like a project
    paths = worker_client.Paths(project)
    command = worker_client.worker_command(paths, fake=True, idle=30)
    command[0] = sys.executable  # the interpreter running the tests has numpy, like the project's .venv
    client = worker_client.WorkerClient(
        command, worker_client.worker_environment(paths), tmp_path / "worker.log", project
    )
    request = core.Request(id=9, prompt="lava", out=str(tmp_path / "T_AI_lava.png"), size=256, seed=1)
    client.submit(request)
    messages: list[dict[str, Any]] = []
    deadline = time.monotonic() + 60
    try:
        while client.pending and time.monotonic() < deadline:
            messages += client.poll()
            time.sleep(0.02)
    finally:
        client.stop()
    answer = next(message for message in messages if message.get("id") == 9)
    assert answer["ok"], (answer, (tmp_path / "worker.log").read_text(encoding="utf-8"))
    assert Path(answer["path"]).is_file() and not client.running


def test_client_answers_lost_requests_when_the_process_dies(tmp_path: Path) -> None:
    client = worker_client.WorkerClient(
        [sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(3)"],
        os.environ,
        tmp_path / "log",
        tmp_path,
    )
    client.submit(core.Request(id=4, prompt="x", out=str(tmp_path / "x.png")))
    messages: list[dict[str, Any]] = []
    deadline = time.monotonic() + 30
    while client.pending and time.monotonic() < deadline:
        messages += client.poll()
        time.sleep(0.02)
    client.stop()
    lost = next(message for message in messages if message.get("id") == 4)
    assert lost["ok"] is False and "exited (3)" in lost["error"]


def test_worker_environment_replaces_the_editors_python_settings(tmp_path: Path) -> None:
    paths = worker_client.Paths(tmp_path)
    env = worker_client.worker_environment(paths, {"PYTHONHOME": "ue", "PythonPath": "ue", "PATH": "p"})
    assert env["PATH"] == "p" and "PYTHONHOME" not in env
    assert env["PYTHONPATH"] == str(tmp_path / "Python") and env["HF_HUB_OFFLINE"] == "1"
    assert Path(env["HF_HOME"]).is_relative_to(tmp_path / "Intermediate")
    command = worker_client.worker_command(paths, fake=False)
    assert command[0] == str(paths.venv_python) and command[-2:] == ["--models", str(paths.models)]
    assert "not set up" in str(worker_client.setup_problem(paths, fake=False))


def test_model_files_cover_the_generator_folders() -> None:
    assert set(model.MODEL_FILES) == {"vega", "vega_rt", "vae_fp16_fix"}
    patterns = [pattern for _, files in model.MODEL_FILES.values() for pattern in files]
    assert not [pattern for pattern in patterns if pattern.endswith(".safetensors") and "fp16" not in pattern
                and "lora" not in pattern and "diffusion_pytorch_model" not in pattern]  # fmt: skip


def test_offload_plan_picks_the_least_offload_that_fits() -> None:
    gb = 1024**3
    assert model.offload_plan("auto", 8 * gb) == "none"
    assert model.offload_plan("auto", 6 * gb) == "encoders"
    assert model.offload_plan("auto", 4 * gb) == "model"
    assert model.offload_plan("model", 8 * gb) == "model"


def test_offload_mode_is_validated_before_torch_loads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="offload must be one of"):
        model.VegaGenerator(tmp_path, print, offload="sometimes")
    with pytest.raises(SystemExit):
        worker.main(["serve", "--fake", "--offload", "sometimes"])


# -- one-command demo ----------------------------------------------------------------------------------------


def test_demo_runs_the_editor_then_the_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.ai_texture import demo

    calls: list[list[str]] = []

    class Done:
        returncode = 0

        def __init__(self, value: Any) -> None:
            self.stdout = json.dumps({"result": {"ok": True, "value": value}, "seconds": 1.5})
            self.stderr = ""

    def fake_run(command: list[str], **_kwargs: Any) -> Done:
        calls.append(command[1:])
        if command[-1].endswith(":capture"):
            return Done({"path": "shot.png", "width": 1600, "height": 900})
        return Done({"texture": "/Game/AI/Textures/T_AI_moss"})

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    monkeypatch.setattr(demo, "unrealpy", lambda: "unrealpy")
    assert demo.main(["moss", "--fake", "--project", str(tmp_path)]) == 0
    editor_call, capture_call = calls
    assert editor_call[:2] == ["run", "--editor"] and editor_call[-4:] == ["moss", "--seed", "-1", "--fake"]
    assert capture_call[:4] == ["run", "--render", "--map", demo.DEMO_MAP]


def test_demo_reports_a_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.ai_texture import demo

    class Failed:
        returncode = 1
        stdout = json.dumps({"result": {"ok": False}, "log": "x.log", "log_summary": {"errors": ["boom"]}})
        stderr = ""

    monkeypatch.setattr(demo.subprocess, "run", lambda *_args, **_kwargs: Failed())
    monkeypatch.setattr(demo, "unrealpy", lambda: "unrealpy")
    with pytest.raises(SystemExit, match="boom"):
        demo.main(["moss"])


def test_demo_map_matches_the_editor() -> None:
    from examples.ai_texture import demo, editor

    assert demo.DEMO_MAP == editor.DEMO_MAP


# -- editor side on the fake `ue` ----------------------------------------------------------------------------


def test_import_task_targets_the_generated_texture(tmp_path: Path) -> None:
    from examples.ai_texture import editor

    source = tmp_path / "T_AI_moss.png"
    task = editor.import_task(source, "T_AI_moss")
    assert (task.filename, task.destination_path, task.destination_name) == (
        str(source),
        "/Game/AI/Textures",
        "T_AI_moss",
    )
    assert task.automated and task.save and task.replace_existing


def test_import_texture_returns_the_imported_texture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ue
    from examples.ai_texture import editor

    class Tools:
        def __init__(self) -> None:
            self.tasks: list[Any] = []

        def import_asset_tasks(self, tasks: list[Any]) -> None:
            self.tasks.extend(tasks)

    with pytest.raises(FileNotFoundError):
        editor.import_texture(Tools(), tmp_path / "missing.png", "T_AI_x")  # ty: ignore[invalid-argument-type]
    source = tmp_path / "T_AI_x.png"
    source.write_bytes(core.encode_png(core.to_rgba8(np.zeros((4, 4, 3), np.float32))))
    imported: list[Any] = []
    monkeypatch.setattr(ue.AssetImportTask, "get_objects", lambda _task: imported)
    tools = Tools()
    with pytest.raises(RuntimeError, match="no Texture2D"):
        editor.import_texture(tools, source, "T_AI_x")  # ty: ignore[invalid-argument-type]
    texture = ue.new_object(ue.Texture2D)
    imported += [ue.new_object(ue.Material), texture]
    assert editor.import_texture(tools, source, "T_AI_x") is texture  # ty: ignore[invalid-argument-type]
    assert [task.destination_name for task in tools.tasks] == ["T_AI_x", "T_AI_x"]


def test_dialog_settings_class_holds_every_dialog_value() -> None:
    import ue
    from examples.ai_texture import editor

    settings: Any = ue.new_object(editor.settings_class())
    assert editor.settings_class() is editor.settings_class()
    for name, value in editor._State.last.items():
        setattr(settings, name, value)
        assert getattr(settings, name) == value
