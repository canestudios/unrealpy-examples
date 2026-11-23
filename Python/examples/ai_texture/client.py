"""The editor's side of the model process: start it, send requests, collect answers. No Unreal, no torch.

The model runs in a separate process with its own Python environment (Intermediate/AITexture/venv, from
env/pyproject.toml), not in the editor's interpreter:
- torch never enters the project environment, so it can neither ship in a packaged game nor be removed by
  the exact `uv sync --group dev` the editor runs whenever pyproject.toml changes;
- inference never holds the editor's GIL or game thread, and a CUDA crash or out-of-memory cannot take the
  editor down;
- the process exits after a while idle, which gives the VRAM and RAM back.
A reader thread turns the process's stdout lines into dicts; the game thread drains them with poll().
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from examples.ai_texture.core import Request
from examples.ai_texture.model import missing


@dataclass(frozen=True)
class Paths:
    """Where the example keeps its files, all under the project (never C:, never staged in a package)."""

    project: Path

    @property
    def root(self) -> Path:
        return self.project / "Intermediate" / "AITexture"

    @property
    def venv_python(self) -> Path:
        return self.root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    @property
    def project_python(self) -> Path:
        """The project's development environment: has numpy, enough for the procedural stand-in."""
        return self.project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def hf_home(self) -> Path:
        return self.root / "hf-home"

    @property
    def log(self) -> Path:
        return self.root / "worker.log"

    @property
    def images(self) -> Path:
        """The generated PNGs, kept as the textures' import sources (Reimport works)."""
        return self.project / "Saved" / "AITexture"


def worker_command(paths: Paths, *, fake: bool, delay: float = 0.0, idle: float = 600.0) -> list[str]:
    python = paths.project_python if fake else paths.venv_python
    command = [str(python), "-m", "examples.ai_texture.worker", "serve", "--idle", str(idle)]
    if fake:
        command += ["--fake", "--delay", str(delay)]
    else:
        command += ["--models", str(paths.models)]
    return command


def worker_environment(paths: Paths, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The editor's environment minus its Python settings (Unreal's embedded interpreter sets some), plus
    the example's package on the path and a Hugging Face cache under the project that never downloads: only
    `worker fetch` does."""
    source = os.environ if base is None else base
    skip = {"PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "VIRTUAL_ENV", "PYTHONEXECUTABLE"}
    env = {key: value for key, value in source.items() if key.upper() not in skip}
    env.update(
        PYTHONPATH=str(paths.project / "Python"),
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        HF_HOME=str(paths.hf_home),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    return env


def setup_problem(paths: Paths, *, fake: bool) -> str | None:
    """Why the model process cannot start yet, or None."""
    if fake:
        return None if paths.project_python.is_file() else f"no project environment at {paths.project_python}"
    if not paths.venv_python.is_file():
        return (
            f"the model environment is not set up ({paths.venv_python} is missing). "
            "See Python/examples/ai_texture/README.md, or use the procedural stand-in (fake)."
        )
    absent = missing(paths.models)
    if absent:
        return f"the model files are not downloaded ({', '.join(absent)} missing in {paths.models})"
    return None


class WorkerClient:
    """One model process, started on the first submit and again whenever it exited (idle or crashed)."""

    def __init__(self, command: list[str], env: Mapping[str, str], log_path: Path, cwd: Path) -> None:
        self.command = command
        self.env = dict(env)
        self.log_path = log_path
        self.cwd = cwd
        self._process: subprocess.Popen[str] | None = None
        self._log: IO[bytes] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.pending: dict[int, Request] = {}

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.running:
            return
        self._close_log()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")  # a file, not a pipe: a chatty library can never block on it
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            cwd=self.cwd,
            env=self.env,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console window pops up
        )
        self._process = process
        threading.Thread(target=self._read, args=(process,), daemon=True, name="ai-texture-reader").start()

    def _read(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                message = {"event": "stray", "text": line.rstrip()}
            if not isinstance(message, dict):
                message = {"event": "stray", "text": line.rstrip()}
            self._messages.put(message)
        self._messages.put({"event": "stopped", "code": process.wait()})

    def submit(self, request: Request) -> None:
        self.start()
        assert self._process is not None and self._process.stdin is not None
        self.pending[request.id] = request
        try:
            self._process.stdin.write(request.to_line())
            self._process.stdin.flush()
        except OSError as exc:  # the process died between start() and now
            self._messages.put({"id": request.id, "ok": False, "error": f"model process gone: {exc}"})

    def poll(self) -> list[dict[str, Any]]:
        """Everything the process said since the last poll. When it stopped, each request it never answered
        gets an error answer, so no job waits forever."""
        out: list[dict[str, Any]] = []
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                return out
            request_id = message.get("id")
            if isinstance(request_id, int):
                self.pending.pop(request_id, None)
            out.append(message)
            if message.get("event") == "stopped":
                for lost in sorted(self.pending):
                    out.append({
                        "id": lost,
                        "ok": False,
                        "error": f"the model process exited ({message.get('code')}); see {self.log_path}",
                    })  # fmt: skip
                self.pending.clear()

    def stop(self, timeout: float = 10.0) -> None:
        """Close stdin (the process finishes the current image and exits); kill it only if it hangs."""
        process = self._process
        if process is not None and process.poll() is None:
            if process.stdin is not None:
                with contextlib.suppress(OSError):
                    process.stdin.close()
            try:
                process.wait(timeout)
            except subprocess.TimeoutExpired:
                process.kill()  # the process this client started
                process.wait()
        self._close_log()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
