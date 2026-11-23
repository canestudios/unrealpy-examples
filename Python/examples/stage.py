"""Shared by the runtime examples, inside the game: greybox props built from the engine's basic shapes,
in-world screens fed with pictures made in Python, a 1920x1080 film camera, and the launchers that start a
Python game mode in Play In Editor or in a standalone/packaged game.

Everything here is runtime code: no editor API, no project assets (only /Engine content).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar, cast

import ue
from ue import agent

SHAPES = "/Engine/BasicShapes"
BASIC_MATERIAL = f"{SHAPES}/BasicShapeMaterial.BasicShapeMaterial"
# Engine materials with useful parameters, so no project asset is needed:
# - the mesh-merge "flatten" material: base colour, metallic, roughness, emissive colour and scale;
# - the in-world widget material: shows a texture parameter ("SlateUI") unlit.
SURFACE_MATERIAL = "/Engine/EngineMaterials/FlattenMaterial.FlattenMaterial"
SCREEN_MATERIAL = "/Engine/EngineMaterials/Widget3DPassThrough_Opaque.Widget3DPassThrough_Opaque"

Colour = tuple[float, float, float]


def look_at(source: Iterable[float], target: Iterable[float]) -> tuple[float, float, float]:
    """The (pitch, yaw, roll) in degrees that points Unreal's forward axis from source at target."""
    dx, dy, dz = (float(t) - float(s) for s, t in zip(source, target, strict=True))
    return (math.degrees(math.atan2(dz, math.hypot(dx, dy))), math.degrees(math.atan2(dy, dx)), 0.0)


def turn_towards(current: float, target: float, max_step: float) -> float:
    """Move an angle in degrees towards another by at most max_step, the short way round."""
    delta = (target - current + 180.0) % 360.0 - 180.0
    return current + max(-max_step, min(max_step, delta))


def shape(name: str) -> ue.StaticMesh:
    mesh = ue.load_object(f"{SHAPES}/{name}.{name}")
    assert isinstance(mesh, ue.StaticMesh), name
    return mesh


_MISSING: set[str] = set()


def _instance(component: ue.PrimitiveComponent, path: str, element: int = 0) -> ue.MaterialInstanceDynamic:
    """A dynamic instance of the material at `path` (the basic shape material if it is not available, e.g.
    not cooked) on one element of the component."""
    source = ue.load_object(path)
    if not isinstance(source, ue.MaterialInterface):
        if path not in _MISSING:
            _MISSING.add(path)
            ue.warning(f"stage: {path} is not available (not cooked?); using the basic shape material")
        source = ue.load_object(BASIC_MATERIAL)
    material = component.create_dynamic_material_instance(element, cast(ue.MaterialInterface, source))
    assert material is not None
    return material


def recolour(
    material: ue.MaterialInstanceDynamic, colour: Colour, glow: Colour | None = None, strength: float = 1.0
) -> None:
    """Change a surface's base colour, and its glow (emissive colour x strength; None: none)."""
    linear = ue.LinearColor(*colour, 1.0)
    material.set_vector_parameter_value("BaseColorConst", linear)
    material.set_vector_parameter_value("Color", linear)  # the basic shape material's fallback
    emissive = glow if glow is not None else (0.0, 0.0, 0.0)
    material.set_vector_parameter_value("EmissiveColorConst", ue.LinearColor(*emissive, 1.0))
    material.set_scalar_parameter_value("EmissiveScale", strength if glow is not None else 0.0)


def surface(
    component: ue.PrimitiveComponent,
    colour: Colour,
    *,
    roughness: float = 0.6,
    metallic: float = 0.0,
    glow: Colour | None = None,
    strength: float = 1.0,
    element: int = 0,
) -> ue.MaterialInstanceDynamic:
    """A physically based surface: base colour (linear), roughness, metallic, optional emissive glow."""
    material = _instance(component, SURFACE_MATERIAL, element)
    recolour(material, colour, glow, strength)
    material.set_scalar_parameter_value("RoughnessConst", roughness)
    material.set_scalar_parameter_value("Roughness", roughness)
    material.set_scalar_parameter_value("MetallicConst", metallic)
    material.set_scalar_parameter_value("SpecularConst", 0.5)
    return material


def paint(component: ue.PrimitiveComponent, colour: Colour) -> ue.MaterialInstanceDynamic:
    """A plain, fairly rough surface in the given linear colour."""
    return surface(component, colour)


def glowing(
    component: ue.PrimitiveComponent, colour: Colour, strength: float = 10.0
) -> ue.MaterialInstanceDynamic:
    """An emissive surface (lamps, lasers, indicator lights): bright enough to bloom and light with Lumen."""
    return surface(component, (0.02, 0.02, 0.02), roughness=0.3, glow=colour, strength=strength)


def grid(
    component: ue.PrimitiveComponent,
    colour: Colour,
    lines: Colour,
    *,
    size: float = 100.0,
    roughness: float = 0.55,
    major: int = 5,
) -> ue.MaterialInstanceDynamic:
    """A floor with a grid on it: the floor in `colour`, thin lines every `size` cm on its top face, every
    `major`-th line brighter. Lines are geometry (thin boxes), so it needs no grid material to be cooked."""
    material = surface(component, colour, roughness=roughness)
    owner = component.get_owner()
    assert owner is not None
    origin, extent = owner.get_actor_bounds(False, False)
    top = float(origin[2] + extent[2]) + 0.15
    x0, x1 = float(origin[0] - extent[0]), float(origin[0] + extent[0])
    y0, y1 = float(origin[1] - extent[1]), float(origin[1] + extent[1])
    bright = tuple(min(1.0, value * 1.8) for value in lines)
    for axis, (start, end), (low, high) in ((0, (x0, x1), (y0, y1)), (1, (y0, y1), (x0, x1))):
        count = int((end - start) // size)
        for index in range(1, count):
            at = start + index * size
            width = 0.04 if index % major else 0.07
            tint = cast(Colour, bright) if index % major == 0 else lines
            middle, length = (low + high) / 2.0, (high - low) / 100.0
            location = (at, middle, top) if axis == 0 else (middle, at, top)
            scale = (width, length, 0.003) if axis == 0 else (length, width, 0.003)
            prop("Cube", location, scale, tint, roughness=0.4, collision=False, shadow=False)
    return material


def prop(
    mesh: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    colour: Colour | None = None,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    collision: bool = True,
    roughness: float = 0.6,
    metallic: float = 0.0,
    glow: Colour | None = None,
    strength: float = 1.0,
    shadow: bool = True,
) -> ue.StaticMeshActor:
    """A movable StaticMeshActor of /Engine/BasicShapes/<mesh> (a 100-unit shape, pivot at its centre)."""
    actor = ue.spawn_actor(ue.StaticMeshActor, ue.Vector(*location), ue.Rotator(*rotation))
    component = actor.static_mesh_component
    component.set_mobility(ue.EComponentMobility.Movable)
    component.set_static_mesh(shape(mesh))
    actor.set_actor_scale3d(ue.Vector(*scale))
    if not collision:
        actor.set_actor_enable_collision(False)
    if not shadow:
        component.set_cast_shadow(False)
    if colour is not None:
        surface(component, colour, roughness=roughness, metallic=metallic, glow=glow, strength=strength)
    return actor


def lighting(
    sun_pitch: float = -40.0,
    sun_yaw: float = 30.0,
    sun: float = 3.0,
    sky: float = 1.0,
    fog: float = 0.0,
    *,
    sun_colour: Colour = (1.0, 0.92, 0.8),
    sky_colour: Colour = (0.75, 0.85, 1.0),
    fog_colour: Colour = (0.12, 0.17, 0.28),
) -> dict[str, ue.Actor]:
    """Sun (atmosphere sun light), sky atmosphere, sky light with real-time capture, and exponential height
    fog (volumetric when `fog` > 0: light beams become visible)."""
    sun_actor = ue.spawn_actor(
        ue.DirectionalLight, ue.Vector(0.0, 0.0, 800.0), ue.Rotator(sun_pitch, sun_yaw, 0.0)
    )
    light = cast(ue.DirectionalLightComponent, sun_actor.light_component)
    light.set_mobility(ue.EComponentMobility.Movable)
    sun_actor.set_actor_rotation(ue.Rotator(sun_pitch, sun_yaw, 0.0), False)
    light.set_intensity(sun)
    light.set_light_color(ue.LinearColor(*sun_colour, 1.0), True)
    light.set_atmosphere_sun_light(True)
    light.set_cast_shadows(True)
    light.set_volumetric_scattering_intensity(2.0)
    actors: dict[str, ue.Actor] = {"sun": sun_actor}
    fog_actor = ue.spawn_actor(ue.ExponentialHeightFog, ue.Vector(0.0, 0.0, -50.0))
    component = fog_actor.component
    component.set_fog_density(max(fog, 0.002))
    component.set_fog_height_falloff(0.3)
    component.set_fog_inscattering_color(ue.LinearColor(*fog_colour, 1.0))
    component.set_start_distance(400.0)
    component.set_fog_max_opacity(0.6)
    component.set_volumetric_fog(fog > 0.0)
    component.set_volumetric_fog_extinction_scale(2.0)
    actors["fog"] = fog_actor
    sky_actor = ue.spawn_actor(ue.SkyLight, ue.Vector(0.0, 0.0, 600.0))
    sky_light = sky_actor.light_component
    sky_light.set_mobility(ue.EComponentMobility.Movable)
    sky_light.set_real_time_capture(True)
    sky_light.set_intensity(sky)
    sky_light.set_light_color(ue.LinearColor(*sky_colour, 1.0))
    atmosphere = ue.spawn_actor(ue.SkyAtmosphere, ue.Vector(0.0, 0.0, 0.0))
    return {**actors, "sky": sky_actor, "atmosphere": atmosphere}


def accent(
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    colour: Colour,
    lumens: float,
    *,
    cone: float = 35.0,
    radius: float = 2500.0,
    rect: bool = False,
    shadows: bool = True,
) -> ue.Light:
    """A practical light aimed at `target`: a spot light, or a rect light (a soft panel) with `rect`."""
    rotation = ue.Rotator(*look_at(location, target))
    light: ue.RectLightComponent | ue.SpotLightComponent
    if rect:
        actor: ue.Light = ue.spawn_actor(ue.RectLight, ue.Vector(*location), rotation)
        light = actor.rect_light_component
    else:
        spot = ue.spawn_actor(ue.SpotLight, ue.Vector(*location), rotation)
        light = spot.spot_light_component
        light.set_outer_cone_angle(cone)
        light.set_inner_cone_angle(cone * 0.3)
        actor = spot
    light.set_mobility(ue.EComponentMobility.Movable)
    actor.set_actor_rotation(rotation, False)  # a light spawns static: its spawn rotation does not stick
    light.set_intensity_units(ue.ELightUnits.Lumens)
    light.set_intensity(lumens)
    light.set_light_color(ue.LinearColor(*colour, 1.0), True)
    light.set_attenuation_radius(radius)
    light.set_cast_shadows(shadows)
    light.set_volumetric_scattering_intensity(0.0)
    return actor


def lumen_available() -> bool:
    """Lumen needs mesh distance fields (LUMEN_ARG at startup); without them: screen-space GI."""
    return ue.KismetSystemLibrary.get_console_variable_int_value("r.GenerateMeshDistanceFields") == 1


def look(*, exposure_bias: float = 0.0, bloom: float = 0.8, vignette: float = 0.3) -> ue.PostProcessVolume:
    """The renderer and an unbound post-process volume for every example: Lumen GI and reflections (when
    distance fields are on), virtual shadows, volumetric fog; auto-exposure clamped to one brightness (no
    pumping), no motion blur, no film grain, no chromatic aberration, a light vignette."""
    lumen = lumen_available()
    commands = [
        "r.VolumetricFog 1",
        "r.SkyLight.RealTimeReflectionCapture 1",
        "r.Shadow.Virtual.Enable 1",
        "r.MotionBlurQuality 0",
        "r.AntiAliasingMethod 4",  # TSR
        "r.DynamicRes.OperationMode 0",
        "r.ScreenPercentage 100",
    ]
    if lumen:
        commands += [
            "r.DynamicGlobalIlluminationMethod 1",
            "r.ReflectionMethod 1",
            "r.Lumen.TraceMeshSDFs 1",
            "r.Lumen.ScreenProbeGather.ScreenTraces 1",
            "r.Lumen.Reflections.ScreenTraces 1",
            "r.SSGI.Enable 0",
        ]
    else:
        commands += ["r.SSGI.Enable 1"]
    for command in commands:
        ue.KismetSystemLibrary.execute_console_command(command)
    volume = ue.spawn_actor(ue.PostProcessVolume, ue.Vector(0.0, 0.0, 0.0))
    volume.unbound = True
    settings = volume.settings
    values: dict[str, Any] = {
        "auto_exposure_method": ue.EAutoExposureMethod.AEM_Histogram,
        "auto_exposure_bias": exposure_bias,
        "auto_exposure_min_brightness": 1.0,  # min = max: a fixed exposure, it never pumps
        "auto_exposure_max_brightness": 1.0,
        "bloom_intensity": bloom,
        "bloom_threshold": 1.0,
        "vignette_intensity": vignette,
        "film_grain_intensity": 0.0,
        "scene_fringe_intensity": 0.0,
        "motion_blur_amount": 0.0,
        "ambient_occlusion_intensity": 0.6,
        "color_contrast": ue.Vector4(1.0, 1.0, 1.0, 1.1),
        "color_saturation": ue.Vector4(1.0, 1.0, 1.0, 1.12),
        "dynamic_global_illumination_method": (
            ue.EDynamicGlobalIlluminationMethod.Lumen
            if lumen
            else ue.EDynamicGlobalIlluminationMethod.ScreenSpace
        ),
        "reflection_method": ue.EReflectionMethod.Lumen if lumen else ue.EReflectionMethod.ScreenSpace,
    }
    for key, value in values.items():
        setattr(settings, key, value)
        if hasattr(settings, "override_" + key):
            setattr(settings, "override_" + key, True)
    volume.settings = settings
    return volume


class Screen:
    """A plane that shows pictures made in Python (image file bytes: PNG, JPEG or BMP), unlit.

    Upright and facing `yaw` by default; `flat` lays it on the floor facing up (a decal-like overlay: use the
    translucent widget material so a picture's alpha shows the floor through)."""

    def __init__(
        self,
        location: tuple[float, float, float],
        yaw: float,
        width: float,
        height: float,
        brightness: float = 1.0,
        *,
        material: str = SCREEN_MATERIAL,
        flat: bool = False,
    ) -> None:
        # The plane lies in XY facing +Z; rolled upright its X stays horizontal, its Y turns vertical and it
        # faces along Y, so a quarter turn more makes it face `yaw`.
        rotation = (0.0, yaw, 0.0) if flat else (0.0, yaw - 90.0, 90.0)
        self.actor = prop(
            "Plane",
            location,
            (width / 100.0, height / 100.0, 1.0),
            None,
            rotation,
            collision=False,
            shadow=False,
        )
        component = self.actor.static_mesh_component
        dynamic = component.create_dynamic_material_instance(
            0, cast(ue.MaterialInterface, ue.load_object(material))
        )
        assert dynamic is not None
        dynamic.set_vector_parameter_value(
            "TintColorAndOpacity", ue.LinearColor(brightness, brightness, brightness, 1.0)
        )
        self.material = dynamic
        self.texture: ue.Texture2D | None = None

    def use(self, texture: ue.Texture2D) -> None:
        """Show a texture made earlier (texture_from), without decoding anything."""
        self.material.set_texture_parameter_value("SlateUI", texture)
        self.texture = texture

    def face(self, yaw: float) -> None:
        """Turn an upright screen to face `yaw` (a billboard towards a moving camera)."""
        self.actor.set_actor_rotation(ue.Rotator(0.0, yaw - 90.0, 90.0), False)

    def show(self, image_file: bytes) -> None:
        """Decode an image file (BMP/PNG/JPEG bytes) into a new transient texture and put it on the screen."""
        texture = ue.KismetRenderingLibrary.import_buffer_as_texture2d(
            image_file, world_context_object=ue.get_world()
        )
        if texture is None:
            raise RuntimeError("ImportBufferAsTexture2D could not decode the picture")
        self.material.set_texture_parameter_value("SlateUI", texture)
        self.texture = texture  # the previous one is garbage once nothing references it


def texture_from(image_file: bytes) -> ue.Texture2D:
    """Decode an image file (PNG, JPEG, BMP bytes) into a transient Texture2D."""
    texture = ue.KismetRenderingLibrary.import_buffer_as_texture2d(
        image_file, world_context_object=ue.get_world()
    )
    if texture is None:
        raise RuntimeError("ImportBufferAsTexture2D could not decode the picture")
    return texture


def _mix(
    p: tuple[float, float, float], q: tuple[float, float, float], s: float
) -> tuple[float, float, float]:
    return (p[0] + (q[0] - p[0]) * s, p[1] + (q[1] - p[1]) * s, p[2] + (q[2] - p[2]) * s)


class CameraPath:
    """Camera keyframes (time, location, look-at target, field of view), eased between with smoothstep:
    holds, pushes and pulls with no cut and no jerk."""

    def __init__(
        self, keys: list[tuple[float, tuple[float, float, float], tuple[float, float, float], float]]
    ) -> None:
        self.keys = sorted(keys, key=lambda key: key[0])

    def at(self, t: float) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        keys = self.keys
        if t <= keys[0][0]:
            return keys[0][1], keys[0][2], keys[0][3]
        for (t0, l0, a0, f0), (t1, l1, a1, f1) in zip(keys, keys[1:], strict=False):
            if t <= t1:
                u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                s = u * u * (3.0 - 2.0 * u)

                return _mix(l0, l1, s), _mix(a0, a1, s), f0 + (f1 - f0) * s
        return keys[-1][1], keys[-1][2], keys[-1][3]


class Film:
    """The showreel camera: the player's view through a CameraActor, shot with HighResShot at 1920x1080 (the
    full renderer: Lumen, virtual shadows, volumetric fog, post-process; a SceneCapture2D lacks some of it),
    whatever the size of the game's window or editor viewport.

    A recording steps the game by exactly one clip frame between shots, and holds it while a shot is
    written."""

    def __init__(
        self,
        location: tuple[float, float, float],
        rotation: tuple[float, float, float],
        *,
        fov: float = 60.0,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        self.width, self.height = width, height
        self.actor = ue.spawn_actor(ue.CameraActor, ue.Vector(*location), ue.Rotator(*rotation))
        self.actor.camera_component.set_field_of_view(fov)
        self.actor.camera_component.set_constraint_aspect_ratio(False)
        self.aim()

    def place(
        self, location: tuple[float, float, float], target: tuple[float, float, float], fov: float
    ) -> None:
        """Move the camera (a keyframe of a CameraPath, for instance)."""
        self.actor.set_actor_location_and_rotation(
            ue.Vector(*location), ue.Rotator(*look_at(location, target)), False, True
        )
        self.actor.camera_component.set_field_of_view(fov)

    def aim(self) -> None:
        """Look through this camera (call again if something else took the view)."""
        controller = ue.GameplayStatics.get_player_controller(0)
        if controller is not None:
            controller.auto_manage_active_camera_target = False
            controller.set_view_target_with_blend(self.actor, 0.0)

    async def warm_up(self, frames: int = 30) -> None:
        """Let exposure, temporal AA and Lumen settle on this view before the first shot."""
        self.aim()
        await agent.frames(frames)

    def request(self, path: Path) -> Path:
        """Ask for a shot of the next rendered frame; the PNG is written a little later, asynchronously."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        ue.KismetSystemLibrary.execute_console_command(
            f'HighResShot {self.width}x{self.height} filename="{path.as_posix()}"'
        )
        return path

    async def shoot(self, path: Path, timeout: float = 30.0) -> Path:
        """One shot, waiting until the file is complete. In Play In Editor the shot is the editor viewport
        at its own size (agent.capture_viewport): there HighResShot stops the play world from ticking."""
        if ue.is_editor():
            path.unlink(missing_ok=True)
            await agent.capture_viewport(path, show_ui=False, timeout=timeout)
            return path
        self.request(path)
        await wait_for_files([path], timeout)
        return path


async def wait_for_files(paths: list[Path], timeout: float = 120.0) -> None:
    """Wait (frame by frame) until every file exists and has stopped growing."""
    deadline = time.monotonic() + timeout
    sizes: dict[Path, int] = {}
    pending = list(paths)
    while pending:
        await agent.frames(1)
        still = []
        for path in pending:
            size = path.stat().st_size if path.is_file() else -1
            if size <= 0 or sizes.get(path) != size:
                sizes[path] = size
                still.append(path)
        pending = still
        if pending and time.monotonic() > deadline:
            raise TimeoutError(f"{len(pending)} screenshots not written, e.g. {pending[0]}")


def manual_exposure(component: ue.SceneCaptureComponent2D, bias: float) -> None:
    """Fixed exposure (no eye adaptation): the same scene always renders the same brightness."""
    settings = component.post_process_settings
    settings.override_auto_exposure_method = True
    settings.auto_exposure_method = ue.EAutoExposureMethod.AEM_Manual
    settings.override_auto_exposure_bias = True
    settings.auto_exposure_bias = bias
    settings.override_motion_blur_amount = True
    settings.motion_blur_amount = 0.0
    component.post_process_settings = settings
    component.post_process_blend_weight = 1.0


async def record(
    film: Film,
    folder: Path,
    seconds: float,
    fps: int = 30,
    on_step: Callable[[float], None] | None = None,
    *,
    game_speed: float = 1.0,
) -> dict[str, Any]:
    """Shoot `seconds` of clip as PNG frames (frame_00000.png...), stepping the game by exactly
    game_speed/fps per frame, so the clip plays at game_speed times real speed however long each shot takes.
    `on_step(dt)` runs before each shot (an autopilot)."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("frame_*.png"):
        old.unlink()
    count = round(seconds * fps)
    started = time.perf_counter()
    # Each clip frame: one engine frame of exactly one clip step, then the shot, written while the game is
    # paused (it renders, nothing moves), so every shot is one clip step after the last however long the
    # 1080p PNG takes. Paused, not slowed down: frames of a few microseconds made physics unstable (a
    # ragdoll's joint solver turned its corrections into huge velocities and flung the limbs apart).
    step = game_speed / fps
    world = ue.get_world()
    agent.set_fixed_timestep(step)
    try:
        for index in range(count):
            await agent.frames(1)
            if on_step is not None:
                on_step(step)
            ue.GameplayStatics.set_game_paused(True, world_context_object=world)
            try:
                await film.shoot(folder / f"frame_{index:05d}.png")
            finally:
                ue.GameplayStatics.set_game_paused(False, world_context_object=world)
    finally:
        agent.set_fixed_timestep(None)
    return {
        "frames": count,
        "fps": fps,
        "folder": str(folder),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


# -- launching a Python game mode ----------------------------------------------------------------------------


Mode = TypeVar("Mode", bound=ue.GameModeBase)


def max_fps(cap: float) -> None:
    """Cap the frame rate (t.MaxFPS; 0 lifts the cap)."""
    ue.KismetSystemLibrary.execute_console_command(f"t.MaxFPS {cap:g}")


def class_path(cls: type[ue.Object]) -> str:
    return f"/Script/UnrealPyTypes.{cls.__name__}"


async def start_in_editor(mode: type[Mode], timeout: float = 60.0) -> Mode:
    """Play In Editor on the open level with `mode` as its game mode; returns the running game mode."""
    world = ue.get_world()
    assert world is not None
    settings = world.get_world_settings()
    assert settings is not None
    settings.default_game_mode = mode  # the editor world's override; Play In Editor copies the world
    agent.begin_play("standalone", 1)
    return await _wait_for_mode(mode, timeout)


async def start_in_game(mode: type[Mode], timeout: float = 60.0) -> Mode:
    """In a standalone or packaged game: reopen the current level with `mode` as its game mode."""
    world = ue.get_world()
    assert world is not None
    level = world.get_path_name().split(".", 1)[0]
    ue.GameplayStatics.open_level(level, True, f"game={class_path(mode)}", world_context_object=world)
    return await _wait_for_mode(mode, timeout)


async def _wait_for_mode(mode: type[Mode], timeout: float) -> Mode:
    deadline = time.monotonic() + timeout
    while True:
        found = ue.GameplayStatics.get_game_mode()
        if isinstance(found, mode):
            await agent.frames(2)
            return found
        if time.monotonic() > deadline:
            raise TimeoutError(f"{mode.__name__} did not start within {timeout} s")
        await agent.frames(1)


def stop_in_editor() -> None:
    if ue.is_editor() and agent.play_state()["in_play"]:
        agent.end_play()
