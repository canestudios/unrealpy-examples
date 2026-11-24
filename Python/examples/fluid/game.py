"""A GPU fluid simulated by Python kernels, inside the game: NVIDIA Warp, from gameplay code written with
UnrealPy.

150,000 particles of water (sim.py: position-based fluids, @wp.kernel functions on the GPU, neighbours from
a wp.HashGrid) are held behind a gate at one end of a concrete tank. The gate lifts, the dam breaks over
pillars, a ramp and Epic's mannequins standing in the tank, and a wave paddle at the far end keeps sending
waves back. Every frame the game hands the kernels the mannequins' capsules and reads back how the water
flows around each one: the water shoves them, and a strong enough wave knocks them over (a ragdoll).

The particles go to the screen without leaving the GPU's memory more than once a frame: a kernel packs them
(position in cm, speed), the game copies them into a float texture (FloatTextureLibrary: one memory copy
from a NumPy array), and one instanced quad per particle reads its texel in the vertex shader and turns
into a little sphere of water (M_FluidParticle, see assets.py), white foam where it is fast.

    uv run python Python/examples/fluid/demo.py                       # Play In Editor, clip + screenshot
    unrealpy run --packaged examples.fluid.game:demo                  # packaged build (--group fluid)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import ue
from examples import stage
from examples.characters import DemoMannequin
from ue import agent

if TYPE_CHECKING:
    from examples.fluid import sim

# The project imports every module under Python/examples at startup (ClassPackages) to define its classes:
# Warp waits until a FluidMode actually starts.

TEXTURE_WIDTH = 128  # the spray texture: 128 x 128 = sim.SPRAY drops
CM = 100.0  # the simulation is in metres
MATERIAL = "/Game/Fluid/M_FluidParticle.M_FluidParticle"
SURFACE_MATERIAL = "/Game/Fluid/M_FluidSurface.M_FluidSurface"
PILLARS = ((0.5, 1.8, 0.35, 3.0), (0.5, -1.8, 0.35, 3.0), (-0.9, 0.0, 0.3, 3.0))  # x, y, radius, height (m)
RAMPS = (((3.6, 0.0, 0.0), (1.2, 1.3, 0.5), -15.0),)  # centre, half extents (m), pitch about y (degrees)
STANDING = (  # the mannequins (m): in the dam's way, between the pillars, by the ramp
    (-1.6, -2.4),
    (-1.9, 2.3),
    (-0.3, -0.9),
    (0.6, 0.3),
    (1.3, -1.7),
    (2.1, 2.3),
    (2.3, -0.4),
)
CAPSULE_RADIUS, CAPSULE_HALF = 0.32, 0.62  # metres: around the mannequin's body, feet to head
SHOVE, KNOCK_OVER = 1.0, 3.4  # m/s of water flow around a mannequin: it is pushed / it falls
WALL_HEIGHT = 150.0
TINTS = ((0.85, 0.35, 0.08), (0.78, 0.8, 0.84), (0.12, 0.45, 0.9), (0.9, 0.75, 0.1), (0.3, 0.3, 0.32))
CAMERA_PATH = (  # seconds, location, target, fov: from the north (the dam on the left, the paddle on the
    # right): the dam, down among the mannequins, the paddle's waves, wide
    (0.0, (-1500.0, 1350.0, 1000.0), (-300.0, 0.0, 40.0), 55.0),
    (2.5, (-1000.0, 1200.0, 800.0), (-100.0, 0.0, 40.0), 55.0),
    (6.0, (-200.0, 1150.0, 650.0), (150.0, 0.0, 60.0), 60.0),
    (10.0, (900.0, 1100.0, 700.0), (300.0, 0.0, 60.0), 60.0),
    (13.0, (1300.0, 500.0, 900.0), (0.0, 0.0, 0.0), 62.0),
    (16.0, (300.0, 1800.0, 1700.0), (0.0, 0.0, 0.0), 58.0),
)


@ue.uclass
class FluidParticles(ue.Actor):
    """All the particles: one instanced quad each, placed by the material from the positions texture."""

    instances = ue.component(ue.InstancedStaticMeshComponent)


@ue.uclass
class Wader(DemoMannequin):
    """A mannequin in the tank: stands (or strolls), gets shoved by the water, falls over in a big wave."""

    def fall(self, velocity: tuple[float, float, float]) -> None:
        self.character_movement.disable_movement()
        self.capsule_component.set_collision_enabled(ue.ECollisionEnabled.NoCollision)
        mesh = self.mesh
        mesh.set_collision_profile_name("Ragdoll")
        mesh.set_all_bodies_simulate_physics(True)
        mesh.set_all_physics_linear_velocity(ue.Vector(*velocity), False)


@ue.uclass
class FluidMode(ue.GameModeBase):
    default_pawn_class = ue.SpectatorPawn

    def begin_play(self) -> None:
        # Warp builds its kernels from their Python source: a packaged game must ship sim.py as source
        # (Config/DefaultUnrealPy.ini: +StageSource=examples/fluid/sim.py; see README.md).
        from examples.fluid import sim

        sim.init(Path(ue.saved_dir()).resolve() / "WarpCache")
        self.sim = sim
        self.tank = sim.Tank(pillars=list(PILLARS), boxes=list(RAMPS))
        started = time.perf_counter()
        self.fluid = sim.Fluid(self.tank, iterations=3)
        self.fluid.step(1.0 / 120.0)  # compile (or load from the cache) every kernel now, not mid-clip
        self.setup_seconds = time.perf_counter() - started
        self.count = self.fluid.count
        built = build_tank(self.tank)
        self.gate: ue.StaticMeshActor = built["gate"]
        self.paddle: ue.StaticMeshActor = built["paddle"]
        self.height = math.ceil(sim.SPRAY / TEXTURE_WIDTH)
        self.texture = ue.FloatTextureLibrary.create_float_texture(TEXTURE_WIDTH, self.height)
        self.texels = np.zeros((TEXTURE_WIDTH * self.height, 4), dtype=np.float32)
        self.texels[:, 2] = -10000.0  # the texture's spare texels: far below the tank
        self.particles = spawn_particles(sim.SPRAY, self.texture, self.height)
        rows, columns = self.fluid.grid_size
        self.surface_texture = ue.FloatTextureLibrary.create_float_texture(columns, rows)
        self.surface = spawn_surface(self.tank, self.fluid.cell, rows, columns, self.surface_texture)
        self.waders: list[Wader] = []
        for index, (x, y) in enumerate(STANDING):
            wader = ue.spawn_actor(
                Wader, ue.Vector(x * CM, y * CM, 100.0), ue.Rotator(0.0, 180.0 - 25.0 * index, 0.0)
            )
            self.waders.append(wader)
        self.fallen: set[int] = set()
        self.dressed = False
        self.running = not stage.scripted()  # a showcase starts it when its camera is ready
        self.frame_ms: list[float] = []
        self.sim_ms: list[float] = []
        self.last = time.perf_counter()
        self._upload(self.fluid.texel_data(), self.fluid.surface_data())
        ue.log(f"FluidMode: {self.count} particles, Warp ready in {self.setup_seconds:.1f} s")

    def tick(self, delta_seconds: float) -> None:
        now = time.perf_counter()
        self.frame_ms.append((now - self.last) * 1000.0)
        self.last = now
        if not self.dressed:
            for index, wader in enumerate(self.waders):
                wader.dress(quinn=index % 2 == 1, tint=TINTS[index % len(TINTS)])
            self.dressed = True
        if not self.running:
            return
        started = time.perf_counter()
        # Last frame's results (copied to the host while the GPU went on), then this frame's step: the game
        # thread never waits for the kernels it has just launched.
        results = self.fluid.collect()
        self._upload(results.particles, results.surface)
        flow = results.flow
        self._shove(flow)
        self.fluid.set_capsules([self._capsule(wader) for wader in self.waders])
        self.fluid.step(min(delta_seconds, 1.0 / 20.0))
        self.fluid.publish()
        self.sim_ms.append((time.perf_counter() - started) * 1000.0)
        self._machines()

    def _capsule(self, wader: Wader) -> sim.Capsule:
        if id(wader) in self.fallen:  # a ragdoll: its pelvis, lying in the water
            at = wader.mesh.get_socket_location("pelvis")
            centre = (float(at[0]) / CM, float(at[1]) / CM, float(at[2]) / CM)
            return self.sim.Capsule(centre, centre, 0.25)
        at = wader.get_actor_location()
        x, y, z = float(at[0]) / CM, float(at[1]) / CM, float(at[2]) / CM
        return self.sim.Capsule((x, y, z - CAPSULE_HALF), (x, y, z + CAPSULE_HALF), CAPSULE_RADIUS)

    def _shove(self, flow: list[tuple[tuple[float, float, float], float]]) -> None:
        """The water's flow around each mannequin pushes it; a strong wave knocks it over."""
        for wader, ((vx, vy, _), touching) in zip(self.waders, flow, strict=False):
            if id(wader) in self.fallen or touching < 15.0:
                continue
            speed = math.hypot(vx, vy)
            if speed > KNOCK_OVER and touching > 90.0:
                self.fallen.add(id(wader))
                wader.fall((vx * CM * 0.5, vy * CM * 0.5, 120.0))
            elif speed > SHOVE:
                wader.launch_character(ue.Vector(vx * CM * 0.9, vy * CM * 0.9, 0.0), True, False)

    def _upload(self, particles: np.ndarray, surface: np.ndarray) -> None:
        self.texels[: len(particles)] = particles
        ue.FloatTextureLibrary.write_float_texture(self.texture, self.texels.reshape(-1).view(np.int32))
        ue.FloatTextureLibrary.write_float_texture(self.surface_texture, surface.reshape(-1).view(np.int32))

    def _machines(self) -> None:
        """The gate lifts when the dam opens; the paddle follows the simulation's stroke."""
        t = self.fluid.time
        lift = min(1.0, max(0.0, (t - self.tank.gate_opens + 0.15) / 0.35))
        gate_x = (self.tank.lo[0] + self.tank.dam[0]) * CM
        self.gate.set_actor_location(ue.Vector(gate_x + 4.0, 0.0, 90.0 + lift * 320.0), False, True)
        self.gate.set_actor_hidden_in_game(lift >= 1.0)
        self.paddle.set_actor_location(ue.Vector(self.tank.paddle(t) * CM + 10.0, 0.0, 110.0), False, True)

    def status(self) -> dict[str, Any]:
        frames = self.frame_ms[60:] or self.frame_ms
        return {
            "particles": self.count,
            "sim_seconds": round(self.fluid.time, 2),
            "warp_setup_seconds": round(self.setup_seconds, 2),
            "fallen": len(self.fallen),
            "sim_ms_mean": round(float(np.mean(self.sim_ms)) if self.sim_ms else 0.0, 2),
            "frame_ms_mean": round(float(np.mean(frames)) if frames else 0.0, 2),
        }


def spawn_particles(count: int, texture: ue.Texture2D, height: int) -> FluidParticles:
    """One quad instance per particle; custom data 0 is its texel's index. The instances sit at the origin:
    the material moves them, and huge bounds keep them from being culled where they were placed."""
    actor = ue.spawn_actor(FluidParticles, ue.Vector(0.0, 0.0, 0.0))
    ism = actor.instances
    ism.set_static_mesh(stage.shape("Plane"))
    base = ue.load_object(MATERIAL)
    assert isinstance(base, ue.MaterialInterface), MATERIAL
    material = ism.create_dynamic_material_instance(0, base)
    assert material is not None
    material.set_texture_parameter_value("Positions", texture)
    material.set_vector_parameter_value(
        "TextureSize", ue.LinearColor(float(TEXTURE_WIDTH), float(height), 0.0, 0.0)
    )
    ism.set_collision_enabled(ue.ECollisionEnabled.NoCollision)
    ism.set_cast_shadow(False)
    ism.affect_distance_field_lighting = False
    ism.set_bounds_scale(3000.0)  # the quads are 1 cm at the origin: bounds of 15 m cover the tank
    ism.set_num_custom_data_floats(1)
    tiny = ue.Transform(ue.Quat(0.0, 0.0, 0.0, 1.0), ue.Vector(0.0, 0.0, 0.0), ue.Vector(0.01, 0.01, 0.01))
    ism.add_instances([tiny] * count, False, False, False)
    for index in range(count):
        ism.set_custom_data_value(index, 0, float(index), index == count - 1)
    return actor


@ue.uclass
class FluidSurface(ue.Actor):
    """The water's surface: a grid of small quads, one per cell of the simulation's height field."""

    instances = ue.component(ue.InstancedStaticMeshComponent)


def spawn_surface(
    tank: sim.Tank, cell: float, rows: int, columns: int, texture: ue.Texture2D
) -> FluidSurface:
    """One quad per cell; the material lifts each corner to the height field and shades the water."""
    actor = ue.spawn_actor(FluidSurface, ue.Vector(0.0, 0.0, 0.0))
    ism = actor.instances
    ism.set_static_mesh(stage.shape("Plane"))
    base = ue.load_object(SURFACE_MATERIAL)
    assert isinstance(base, ue.MaterialInterface), SURFACE_MATERIAL
    material = ism.create_dynamic_material_instance(0, base)
    assert material is not None
    x0, y0 = tank.lo[0] * CM, tank.lo[1] * CM
    material.set_texture_parameter_value("Surface", texture)
    material.set_vector_parameter_value("Grid", ue.LinearColor(x0, y0, cell * CM, 0.0))
    material.set_vector_parameter_value("GridSize", ue.LinearColor(float(columns), float(rows), 0.0, 0.0))
    ism.set_collision_enabled(ue.ECollisionEnabled.NoCollision)
    ism.set_cast_shadow(False)  # moving every frame, its shadow would be redrawn every frame
    ism.affect_distance_field_lighting = False
    ism.set_bounds_scale(40.0)  # waves rise up to 2 m above the tiles' resting height
    size = cell * CM / 100.0  # the plane is 100 cm across
    tiles = [
        ue.Transform(
            ue.Quat(0.0, 0.0, 0.0, 1.0),
            ue.Vector(x0 + (i + 0.5) * cell * CM, y0 + (j + 0.5) * cell * CM, 0.5),
            ue.Vector(size, size, 1.0),
        )
        for j in range(rows)
        for i in range(columns)
    ]
    ism.add_instances(tiles, False, False, False)
    return actor


def build_tank(tank: sim.Tank) -> dict[str, Any]:
    """The tank (concrete floor and walls), the pillars and the ramp the simulation collides with, the gate,
    the paddle, and the lighting."""
    stage.lighting(
        sun_pitch=-38.0,
        sun_yaw=-120.0,
        sun=4.0,
        sky=1.3,
        fog=0.004,
        sun_colour=(1.0, 0.86, 0.7),
        sky_colour=(0.6, 0.75, 1.0),
        fog_colour=(0.25, 0.32, 0.45),
    )
    stage.look(exposure_bias=-1.1, bloom=0.6)
    (x0, y0, _), (x1, y1, _) = tank.lo, tank.hi
    x0, x1, y0, y1 = x0 * CM, x1 * CM, y0 * CM, y1 * CM
    stage.prop("Cube", (0.0, 0.0, -45.0), (200.0, 200.0, 0.2), (0.12, 0.12, 0.11), roughness=0.9)
    floor = stage.prop("Cube", (0.0, 0.0, -20.0), ((x1 - x0) / 100.0 + 0.8, (y1 - y0) / 100.0 + 0.8, 0.4))
    stage.grid(floor.static_mesh_component, (0.3, 0.3, 0.29), (0.2, 0.2, 0.19), size=100.0, roughness=0.8)
    concrete = (0.22, 0.215, 0.2)
    for sign in (-1.0, 1.0):
        stage.prop(
            "Cube",
            (0.0, sign * (y1 + 20.0), WALL_HEIGHT / 2.0),
            ((x1 - x0) / 100.0 + 0.8, 0.4, WALL_HEIGHT / 100.0),
            concrete,
            roughness=0.85,
        )
        stage.prop(
            "Cube",
            (sign * (x1 + 20.0 + (60.0 if sign > 0 else 0.0)), 0.0, WALL_HEIGHT / 2.0),
            (0.4, (y1 - y0) / 100.0 + 0.8, WALL_HEIGHT / 100.0),
            concrete,
            roughness=0.85,
        )
    for x, y, radius, height in tank.pillars:
        stage.prop(
            "Cylinder",
            (x * CM, y * CM, height * CM / 2.0),
            (radius * 2.0, radius * 2.0, height),
            (0.36, 0.35, 0.33),
            roughness=0.7,
        )
    for (cx, cy, cz), (hx, hy, hz), pitch in tank.boxes:
        stage.prop(
            "Cube",
            (cx * CM, cy * CM, cz * CM),
            (hx * 2.0, hy * 2.0, hz * 2.0),
            (0.55, 0.3, 0.12),
            (-pitch, 0.0, 0.0),
            roughness=0.5,
            metallic=0.2,
        )
    gate = stage.prop(
        "Cube",
        ((tank.lo[0] + tank.dam[0]) * CM + 4.0, 0.0, 90.0),
        (0.08, (y1 - y0) / 100.0, 1.8),
        (0.08, 0.09, 0.1),
        roughness=0.3,
        metallic=1.0,
        collision=False,
    )
    paddle = stage.prop(
        "Cube",
        (x1 + 10.0, 0.0, 110.0),
        (0.2, (y1 - y0) / 100.0 - 0.1, 2.2),
        (0.95, 0.65, 0.05),
        roughness=0.35,
        metallic=0.6,
        collision=False,
    )
    for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        stage.accent(
            (x * 1500.0, y * 1100.0, 900.0),
            (x * 300.0, y * 200.0, 0.0),
            (1.0, 0.95, 0.88),
            9000.0,
            cone=45.0,
            radius=4500.0,
            shadows=False,
        )
    return {"gate": gate, "paddle": paddle}


# -- scripted runs -------------------------------------------------------------------------------------------


async def showcase(mode: FluidMode, *, measure: float, clip: float, out: Path) -> dict[str, Any]:
    """Shoot the clip (the camera follows the dam break, the mannequins, the waves), then measure the frame
    rate for a few seconds of real time while the paddle keeps the water moving."""
    path = stage.CameraPath(list(CAMERA_PATH))
    location, target, fov = path.at(0.0)
    film = stage.Film(location, stage.look_at(location, target), fov=fov)
    await film.warm_up()
    recording: dict[str, Any] = {}
    clock = [0.0]

    def move(dt: float) -> None:
        clock[0] += dt
        location, target, fov = path.at(clock[0])
        film.place(location, target, fov)

    mode.running = True
    if clip > 0:
        move(0.0)
        recording = await stage.record(film, out / "frames", clip, 30, move)
    still = await film.shoot(out / f"still-{time.strftime('%Y%m%d-%H%M%S')}.png")
    perf: dict[str, Any] = {}
    if measure > 0:
        await agent.frames(10)
        frames, ticks, started = 0, len(mode.sim_ms), time.perf_counter()
        while time.perf_counter() - started < measure:
            await agent.frames(1)
            frames += 1
        perf = {
            "fps": round(frames / (time.perf_counter() - started), 1),
            "sim_ms": round(float(np.mean(mode.sim_ms[ticks:] or [0.0])), 2),
        }
    return {"perf": perf, "recording": recording, "still": str(still), **mode.status()}


def _out_dir() -> Path:
    return Path(ue.project_dir()).resolve() / "Saved" / "Fluid"


def _options(args: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="fluid")
    parser.add_argument(
        "--measure", type=float, default=4.0, help="seconds of real-time frame rate before the clip"
    )
    parser.add_argument("--clip", type=float, default=16.0)
    return vars(parser.parse_args(args))


async def pie(args: list[str]) -> dict[str, Any]:
    """`unrealpy run --editor --render examples.fluid.game:pie`: Play In Editor."""
    options = _options(args)
    mode = await stage.start_in_editor(FluidMode)
    try:
        result = await showcase(mode, out=_out_dir() / "pie", **options)
    finally:
        stage.stop_in_editor()
    return {"where": "play in editor", **result}


async def demo(args: list[str]) -> dict[str, Any]:
    """`unrealpy run --render examples.fluid.game:demo` (add --packaged for the packaged game)."""
    options = _options(args)
    mode = await stage.start_in_game(FluidMode)
    return {"where": "game", **await showcase(mode, out=_out_dir() / "game", **options)}


def play(_args: list[str]) -> None:
    """Interactive: watch the water (fly around with WASD and the mouse)."""
    world = ue.get_world()
    assert world is not None
    level = world.get_path_name().split(".", 1)[0]
    ue.GameplayStatics.open_level(
        level, True, f"game={stage.class_path(FluidMode)}", world_context_object=world
    )
