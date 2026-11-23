"""The editor side of the AI texture example: menu entry, prompt dialog, texture and material assets.

    Tools > UnrealPy > AI Texture from Prompt...
        installed in a running editor by
        uv run unrealpy editor exec "import examples.ai_texture.editor as e; e.install_menu()"
    uv run unrealpy run Python/examples/ai_texture/editor.py:main -- "mossy cobblestone" [--fake]
        headless: generates one texture and its material instance, then exits

The model runs in its own process (client.py); this module only talks to it from the game thread, on ticks,
so the editor stays responsive while the GPU works. Each result becomes /Game/AI/Textures/T_AI_<prompt>
(Texture2D) and /Game/AI/Materials/MI_AI_<prompt> (an instance of /Game/AI/M_AITexture), assigned to every
material slot of the actors selected when the request was made, in one undoable transaction.
"""

from __future__ import annotations

import argparse
import atexit
import itertools
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import ue
from examples.ai_texture import client as worker_client
from examples.ai_texture.core import Request, asset_name, instance_name
from ue import agent

CONTENT = "/Game/AI"
TEXTURES = CONTENT + "/Textures"
INSTANCES = CONTENT + "/Materials"
MATERIAL_NAME = "M_AITexture"
MATERIAL = f"{CONTENT}/{MATERIAL_NAME}"
DEFAULT_TEXTURE = "/Engine/EngineResources/DefaultTexture.DefaultTexture"
MENU = "LevelEditor.MainMenu.Tools"
SECTION = "UnrealPy"
ENTRY = "UnrealPyAITexture"
_COMMAND = "import examples.ai_texture.editor as ai_texture; ai_texture.open_dialog()"


@dataclass
class Job:
    request: Request
    texture: str  # asset name, reserved when the request was made
    tiling: float
    targets: list[str] = field(default_factory=list)  # actor path names
    result: dict[str, Any] | None = None
    error: str | None = None  # set when the answer or an asset step failed


class _State:
    client: worker_client.WorkerClient | None = None
    fake: bool | None = None
    jobs: dict[int, Job] = {}
    ids = itertools.count(1)
    pumping = False
    # The dialog's last values (the settings object itself is a UObject the engine may collect).
    last: dict[str, Any] = {
        "prompt": "mossy cobblestone path",
        "size": 1024,
        "steps": 4,
        "seed": -1,
        "seamless": True,
        "tiling": 2.0,
        "apply_to_selection": True,
        "fake": False,
    }


# -- menu and dialog ---------------------------------------------------------------------------------------


def install_menu(_args: list[str] | None = None) -> dict[str, str]:
    """Add Tools > UnrealPy > AI Texture from Prompt... (the editor's ToolMenus, through Epic's `unreal`)."""
    import unreal  # ty: ignore[import-not-found]

    menus = unreal.ToolMenus.get()
    menu = menus.extend_menu(MENU)
    menu.add_section(SECTION, SECTION)
    entry = unreal.ToolMenuEntry(name=ENTRY, type=unreal.MultiBlockType.MENU_ENTRY)
    entry.set_label("AI Texture from Prompt...")
    entry.set_tool_tip(
        "Generate a seamless texture with an open-weight model on the GPU (Segmind-Vega + VegaRT), import "
        "it, and put it on the selected actors."
    )
    entry.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "", _COMMAND)
    menu.add_menu_entry(SECTION, entry)
    menus.refresh_all_widgets()
    return {"menu": f"{MENU}.{SECTION}.{ENTRY}"}


_settings_class: type[ue.Object] | None = None


def settings_class() -> type[ue.Object]:
    """The dialog's settings, a script class (its properties are what the details view shows). Defined on
    first use, so importing this module defines no Unreal type."""
    global _settings_class
    if _settings_class is None:

        @ue.uclass
        class AITextureSettings(ue.Object):
            prompt: str = ue.uproperty("", category="Texture")
            size: int = ue.uproperty(1024, category="Texture", meta={"ClampMin": "256", "ClampMax": "1024"})
            steps: int = ue.uproperty(4, category="Texture", meta={"ClampMin": "1", "ClampMax": "8"})
            seed: int = ue.uproperty(-1, category="Texture", meta={"ToolTip": "-1: a new random seed"})
            seamless: bool = ue.uproperty(True, category="Texture")
            tiling: float = ue.uproperty(2.0, category="Material", meta={"ClampMin": "0.01"})
            apply_to_selection: bool = ue.uproperty(True, category="Material")
            fake: bool = ue.uproperty(
                False, category="Model", meta={"ToolTip": "Procedural stand-in: no model, no GPU"}
            )

        _settings_class = AITextureSettings
    return _settings_class


def open_dialog() -> int | None:
    """Ask for the prompt and settings (a modal details view), then generate. Returns the request id."""
    settings: Any = ue.new_object(settings_class())
    for name, value in _State.last.items():
        setattr(settings, name, value)
    options = ue.EditorDialogLibraryObjectDetailsViewOptions(
        show_object_name=False, allow_search=False, min_width=520, min_height=320
    )
    if not ue.EditorDialogLibrary.show_object_details_view("AI Texture from Prompt", settings, options):
        return None
    _State.last = {name: getattr(settings, name) for name in _State.last}
    values = dict(_State.last)
    try:
        return generate(
            values.pop("prompt"),
            size=round(values.pop("size") / 64) * 64,
            **values,
        )
    except (ValueError, RuntimeError) as exc:
        ue.EditorDialogLibrary.show_message("AI Texture", str(exc), ue.EAppMsgType.Ok)
        return None


# -- requests ----------------------------------------------------------------------------------------------


def _paths() -> worker_client.Paths:
    return worker_client.Paths(Path(ue.project_dir()).resolve())


def _client(fake: bool) -> worker_client.WorkerClient:
    """The model process client; a switch between the model and the stand-in restarts the process."""
    if _State.client is not None and _State.fake != fake:
        _State.client.stop()
        _State.client = None
    if _State.client is None:
        paths = _paths()
        problem = worker_client.setup_problem(paths, fake=fake)
        if problem is not None:
            raise RuntimeError(problem)
        _State.client = worker_client.WorkerClient(
            worker_client.worker_command(paths, fake=fake, delay=1.0 if fake else 0.0),
            worker_client.worker_environment(paths),
            paths.log,
            paths.project,
        )
        _State.fake = fake
        atexit.register(_State.client.stop)
    return _State.client


def _asset_exists(name: str) -> bool:
    return ue.EditorAssetLibrary.does_asset_exist(f"{TEXTURES}/{name}") or any(
        job.texture == name for job in _State.jobs.values()
    )


def selected_actors() -> list[ue.Actor]:
    subsystem = ue.EditorSubsystemBlueprintLibrary.get_editor_subsystem(ue.EditorActorSubsystem)
    if not isinstance(subsystem, ue.EditorActorSubsystem):
        return []
    return [actor for actor in subsystem.get_selected_level_actors() if actor is not None]


def generate(
    prompt: str,
    *,
    size: int = 1024,
    steps: int = 4,
    seed: int = -1,
    seamless: bool = True,
    tiling: float = 2.0,
    apply_to_selection: bool = True,
    fake: bool = False,
) -> int:
    """Queue one generation and return its id at once; the assets appear a few seconds later."""
    paths = _paths()
    name = asset_name(prompt, _asset_exists)
    request = Request(
        id=next(_State.ids),
        prompt=prompt,
        out=str(paths.images / f"{name}.png"),
        size=size,
        steps=steps,
        seed=seed,
        seamless=seamless,
    )
    targets = [actor.get_path_name() for actor in selected_actors()] if apply_to_selection else []
    _client(fake).submit(request)
    _State.jobs[request.id] = Job(request, name, tiling, targets)
    ue.log(f"AI Texture #{request.id}: '{prompt}' -> {TEXTURES}/{name} ({len(targets)} actors selected)")
    _ensure_pump()
    return request.id


def _ensure_pump() -> None:
    if not _State.pumping:
        _State.pumping = True
        ue.call_on_next_tick(_pump)


def _pump() -> None:
    """Every tick while jobs are pending: turn the model process's answers into assets."""
    client = _State.client
    for message in client.poll() if client is not None else []:
        _handle(message)
    if _State.client is not None and _State.client.pending:
        ue.call_on_next_tick(_pump)
    else:
        _State.pumping = False


def _handle(message: dict[str, Any]) -> None:
    kind = message.get("event")
    if kind == "loading":
        ue.log("AI Texture: loading the model on the GPU (the first time takes a while)...")
    elif kind == "loaded":
        ue.log(f"AI Texture: model ready on {message.get('device')} in {message.get('seconds')} s")
    elif kind == "stray":
        ue.warning(f"AI Texture model process: {message.get('text')}")
    elif kind in ("exiting", "stopped"):
        ue.log(f"AI Texture: model process {kind} ({message.get('reason', message.get('code'))})")
    request_id = message.get("id")
    if not isinstance(request_id, int):
        return
    job = _State.jobs.pop(request_id, None)
    if job is None:
        ue.warning(f"AI Texture: answer to unknown request {request_id}: {message}")
        return
    job.result = message
    if not message.get("ok"):
        job.error = str(message.get("error"))
        ue.error(f"AI Texture #{request_id} failed: {job.error}")
        return
    try:
        finish(job)
    except Exception as exc:  # an asset step failed: say so, keep the editor and the pump going
        job.error = f"{type(exc).__name__}: {exc}"
        ue.error(
            f"AI Texture #{request_id}: the image is at {message.get('path')}, but {job.error}\n"
            + traceback.format_exc()
        )


# -- assets ------------------------------------------------------------------------------------------------


def finish(job: Job) -> dict[str, Any]:
    """Import the PNG, make the material instance and assign it to the job's actors (one undo step)."""
    tools = ue.AssetToolsHelpers.get_asset_tools()
    assert tools is not None
    assert job.result is not None
    texture = import_texture(tools, Path(job.result["path"]), job.texture)
    instance = make_instance(tools, texture, instance_name(job.texture), job.tiling)
    actors = [actor for path in job.targets if isinstance(actor := ue.find_object(path), ue.Actor)]
    with agent.transaction(f"AI Texture: {job.request.prompt}"):
        slots = assign(actors, instance)
    ue.EditorAssetLibrary.sync_browser_to_objects([instance.get_path_name()])
    seconds = job.result.get("seconds")
    ue.log(
        f"AI Texture #{job.request.id}: {texture.get_path_name()} ({job.result.get('width')} px, seed "
        f"{job.result.get('seed')}, {seconds} s on the GPU), {slots} material slots on {len(actors)} actors"
    )
    return {"texture": texture.get_path_name(), "instance": instance.get_path_name(), "slots": slots}


def import_task(source: Path, name: str, destination: str = TEXTURES) -> ue.AssetImportTask:
    task = ue.new_object(ue.AssetImportTask)
    task.filename = str(source)
    task.destination_path = destination
    task.destination_name = name
    task.automated = True  # no import dialog
    task.save = True
    task.replace_existing = True
    return task


def import_texture(tools: ue.AssetTools, source: Path, name: str) -> ue.Texture2D:
    """The generated PNG as /Game/AI/Textures/<name>: sRGB colour, mipmapped, wrapping (it tiles)."""
    if not source.is_file():
        raise FileNotFoundError(f"no generated image at {source}")
    task = import_task(source, name)
    tools.import_asset_tasks([task])
    textures = [obj for obj in task.get_objects() if isinstance(obj, ue.Texture2D)]
    if not textures:
        raise RuntimeError(f"importing {source} made no Texture2D")
    return textures[0]


def ensure_material(tools: ue.AssetTools) -> ue.Material:
    """/Game/AI/M_AITexture: Albedo texture parameter sampled at TexCoord * Tiling into Base Color, and a
    Roughness parameter. Made once; every generated texture is an instance of it."""
    existing = ue.load_object(MATERIAL)
    if isinstance(existing, ue.Material):
        return existing
    material = tools.create_asset(MATERIAL_NAME, CONTENT, ue.Material, ue.new_object(ue.MaterialFactoryNew))
    assert isinstance(material, ue.Material)
    edit = ue.MaterialEditingLibrary

    def node(kind: type[ue.MaterialExpression], x: int, y: int) -> Any:
        return edit.create_material_expression(material, kind, x, y)

    coordinates = node(ue.MaterialExpressionTextureCoordinate, -900, 0)
    tiling = node(ue.MaterialExpressionScalarParameter, -900, 120)
    tiling.parameter_name = "Tiling"
    tiling.default_value = 1.0
    uv = node(ue.MaterialExpressionMultiply, -650, 40)
    assert edit.connect_material_expressions(coordinates, "", uv, "A")
    assert edit.connect_material_expressions(tiling, "", uv, "B")
    albedo = node(ue.MaterialExpressionTextureSampleParameter2D, -420, 0)
    albedo.parameter_name = "Albedo"
    albedo.texture = cast(ue.Texture, ue.load_object(DEFAULT_TEXTURE))
    assert edit.connect_material_expressions(uv, "", albedo, "UVs")
    roughness = node(ue.MaterialExpressionScalarParameter, -420, 260)
    roughness.parameter_name = "Roughness"
    roughness.default_value = 0.85
    assert edit.connect_material_property(albedo, "RGB", ue.EMaterialProperty.MP_BaseColor)
    assert edit.connect_material_property(roughness, "", ue.EMaterialProperty.MP_Roughness)
    edit.recompile_material(material)
    assert ue.EditorAssetLibrary.save_loaded_asset(material, False)
    return material


def make_instance(
    tools: ue.AssetTools, texture: ue.Texture2D, name: str, tiling: float
) -> ue.MaterialInstanceConstant:
    """/Game/AI/Materials/<name>, an instance of M_AITexture showing the texture (replaced if it exists)."""
    parent = ensure_material(tools)
    path = f"{INSTANCES}/{name}"
    instance = ue.load_object(path) if ue.EditorAssetLibrary.does_asset_exist(path) else None
    if not isinstance(instance, ue.MaterialInstanceConstant):
        instance = tools.create_asset(
            name, INSTANCES, ue.MaterialInstanceConstant, ue.new_object(ue.MaterialInstanceConstantFactoryNew)
        )
    assert isinstance(instance, ue.MaterialInstanceConstant)
    edit = ue.MaterialEditingLibrary
    edit.set_material_instance_parent(instance, parent)
    # These setters return False even when they succeed (the engine never sets the result): read back.
    edit.set_material_instance_texture_parameter_value(instance, "Albedo", texture)
    edit.set_material_instance_scalar_parameter_value(instance, "Tiling", tiling)
    shown = edit.get_material_instance_texture_parameter_value(instance, "Albedo")
    if shown is None or shown.get_path_name() != texture.get_path_name():
        raise RuntimeError(f"{path}: the Albedo parameter did not take {texture.get_path_name()}")
    edit.update_material_instance(instance)
    assert ue.EditorAssetLibrary.save_loaded_asset(instance, False)
    return instance


def assign(actors: list[ue.Actor], material: ue.MaterialInterface) -> int:
    """Put the material in every slot of the actors' mesh components; returns the number of slots."""
    slots = 0
    for actor in actors:
        for component in actor.get_components_by_class(ue.MeshComponent):
            if not isinstance(component, ue.MeshComponent):
                continue
            ue.KismetSystemLibrary.transact_object(component)  # undo restores the old materials
            for index in range(component.get_num_materials()):
                component.set_material(index, material)
                slots += 1
    return slots


# -- headless entry point ----------------------------------------------------------------------------------


async def main(args: list[str]) -> dict[str, Any]:
    """`unrealpy run Python/examples/ai_texture/editor.py:main -- PROMPT [--fake] [--size N] [--seed N]`:
    one texture and its material instance, waiting for the model (the first run loads it)."""
    parser = argparse.ArgumentParser(prog="editor.py:main")
    parser.add_argument("prompt")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-seamless", dest="seamless", action="store_false")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-level", dest="level", action="store_false", help=f"skip the demo level ({DEMO_MAP})"
    )
    options = parser.parse_args(args)
    request_id = generate(
        options.prompt,
        size=options.size,
        steps=options.steps,
        seed=options.seed,
        seamless=options.seamless,
        apply_to_selection=False,
        fake=options.fake,
    )
    job = _State.jobs[request_id]
    deadline = time.monotonic() + options.timeout
    while request_id in _State.jobs:
        if time.monotonic() > deadline:
            raise TimeoutError(f"no image after {options.timeout} s; see {_paths().log}")
        await agent.frames(1)
    if _State.client is not None:
        _State.client.stop()
    result = job.result or {}
    if job.error is not None:
        raise RuntimeError(f"AI Texture failed: {job.error}")
    instance = f"{INSTANCES}/{instance_name(job.texture)}"
    summary: dict[str, Any] = {
        "texture": f"{TEXTURES}/{job.texture}",
        "instance": instance,
        "image": result.get("path"),
        "seed": result.get("seed"),
        "gpu_seconds": result.get("seconds"),
    }
    if options.level:
        material = ue.load_object(instance)
        assert isinstance(material, ue.MaterialInstanceConstant), instance
        summary["level"] = build_demo_level(material)
    return summary


# -- demo level and capture --------------------------------------------------------------------------------

DEMO_MAP = f"{CONTENT}/AITexture_Demo"
DEMO_TAG = "AITextureDemo"
CAMERA_TAG = "AITextureDemoCamera"
# Basic shapes are 100 units across with the pivot at the centre: scaled 1.5, z 75 rests them on the floor.
DEMO_SHAPES = (("Cube", -230.0), ("Sphere", 0.0), ("Cylinder", 230.0))
DEMO_CAMERA = ((-520.0, 0.0, 380.0), (-28.0, 0.0, 0.0), 62.0)  # location, rotation (pitch, yaw, roll), FOV


def _subsystem(cls: type[Any]) -> Any:
    subsystem = ue.EditorSubsystemBlueprintLibrary.get_editor_subsystem(cls)
    assert isinstance(subsystem, cls), cls
    return subsystem


def build_demo_level(material: ue.MaterialInterface) -> dict[str, Any]:
    """/Game/AI/AITexture_Demo: a cube, a sphere and a cylinder wearing `material` on a grey floor, sun, sky
    and a camera, saved (replacing the previous demo's actors). Opens in the editor like any level."""
    level_editor = _subsystem(ue.LevelEditorSubsystem)
    actors = _subsystem(ue.EditorActorSubsystem)
    if ue.EditorAssetLibrary.does_asset_exist(DEMO_MAP):
        assert level_editor.load_level(DEMO_MAP), f"cannot open {DEMO_MAP}"
        for old in actors.get_all_level_actors():
            if old is not None and old.actor_has_tag(DEMO_TAG):
                actors.destroy_actor(old)
    else:
        assert level_editor.new_level(DEMO_MAP), f"cannot create {DEMO_MAP}"

    def spawn(cls: type[Any], label: str, location: ue.Vector, rotation: ue.Rotator | None = None) -> Any:
        actor = actors.spawn_actor_from_class(cls, location, rotation or ue.Rotator())
        assert isinstance(actor, cls), f"cannot spawn {cls.__name__}"
        actor.set_actor_label(label)
        actor.tags = [DEMO_TAG]
        return actor

    def shape(
        mesh: str, label: str, location: ue.Vector, scale: ue.Vector, skin: ue.MaterialInterface
    ) -> None:
        actor = spawn(ue.StaticMeshActor, label, location)
        component = actor.static_mesh_component
        component.set_static_mesh(ue.load_object(f"/Engine/BasicShapes/{mesh}.{mesh}"))
        component.set_material(0, skin)
        actor.set_actor_scale3d(scale)

    grey = cast(
        ue.MaterialInterface, ue.load_object("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")
    )
    shape("Plane", "Floor", ue.Vector(0.0, 0.0, 0.0), ue.Vector(50.0, 50.0, 1.0), grey)
    for mesh, y in DEMO_SHAPES:
        shape(mesh, f"AI {mesh}", ue.Vector(0.0, y, 75.0), ue.Vector(1.5, 1.5, 1.5), material)
    sun = spawn(ue.DirectionalLight, "Sun", ue.Vector(0.0, 0.0, 600.0), ue.Rotator(-38.0, 35.0, 0.0))
    sun.light_component.set_intensity(4.0)
    sky = spawn(ue.SkyLight, "Sky Light", ue.Vector(0.0, 0.0, 400.0))
    sky.light_component.set_mobility(ue.EComponentMobility.Movable)
    sky.light_component.real_time_capture = True
    spawn(ue.SkyAtmosphere, "Sky Atmosphere", ue.Vector(0.0, 0.0, 0.0))
    location, rotation, fov = DEMO_CAMERA
    camera = spawn(ue.CameraActor, "Demo Camera", ue.Vector(*location), ue.Rotator(*rotation))
    camera.tags = [DEMO_TAG, CAMERA_TAG]
    camera.camera_component.set_field_of_view(fov)
    spawn(ue.PlayerStart, "Player Start", ue.Vector(-900.0, 0.0, 100.0))  # out of the camera's view
    world = ue.get_world()
    assert world is not None
    settings = world.get_world_settings()
    assert settings is not None
    settings.default_game_mode = ue.GameModeBase  # whatever the project's default is
    assert level_editor.save_current_level(), f"cannot save {DEMO_MAP}"
    return {"map": DEMO_MAP, "shapes": [mesh for mesh, _ in DEMO_SHAPES]}


async def capture(args: list[str]) -> dict[str, Any]:
    """`unrealpy run --render --map /Game/AI/AITexture_Demo Python/examples/ai_texture/editor.py:capture`:
    look through the demo level's camera and save the game viewport as a PNG (ue.agent.capture_viewport)."""
    parser = argparse.ArgumentParser(prog="editor.py:capture")
    parser.add_argument("--out", help="a new .png (default: Saved/AITexture/Captures/<map>-<time>.png)")
    parser.add_argument("--resolution", default="1600x900w")
    parser.add_argument("--settle", type=int, default=90, help="frames for exposure and the sky capture")
    options = parser.parse_args(args)
    ue.KismetSystemLibrary.execute_console_command(f"r.SetRes {options.resolution}")
    cameras = [actor for actor in agent.actors(ue.CameraActor) if actor.actor_has_tag(CAMERA_TAG)]
    if not cameras:
        raise RuntimeError(f"no demo camera: start on {DEMO_MAP} (--map), made by editor.py:main")
    controller = ue.GameplayStatics.get_player_controller(0)
    assert controller is not None
    controller.set_view_target_with_blend(cameras[0], 0.0)
    await agent.frames(options.settle)
    out = (
        Path(options.out)
        if options.out
        else (
            Path(ue.project_dir())
            / "Saved"
            / "AITexture"
            / "Captures"
            / f"AITexture_Demo-{time.strftime('%Y%m%d-%H%M%S')}.png"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    return dict(await agent.capture_viewport(out.resolve(), show_ui=False))
