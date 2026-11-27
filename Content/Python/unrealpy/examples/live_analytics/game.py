"""Live playtest analytics: pandas, NumPy and matplotlib inside the game, in gameplay code written with
UnrealPy.

A small greybox level (a start room, two rooms joined by halls, a narrow bridge over a pit with a swinging
blade, a longer detour with a spike trap, an exit) is played by autopiloted mannequin playtesters, who come
in waves and try to reach the exit, most over the bridge, some by the detour. The blade knocks some off the
bridge, the spikes catch a few in the detour; whoever dies or finishes tries again. A Python game mode
collects telemetry while they play (positions five times a second, every start, death and finish) and shows
it live, in the world:

- on the floor, where the playtesters walk (a NumPy histogram2d, a Gaussian blur, a colour map) and where
  they die (red), plus a red cross on every death;
- a screen with two matplotlib charts, redrawn twice a second: deaths per area (the bridge's bar towers)
  and the time the runs that reach the exit take, per route.

The summaries (pandas) and the drawing run in a separate Python process (matplotlib would hold the game's
GIL for a quarter of a second per chart); the game thread only appends rows and swaps textures.

    uv run python -m examples.live_analytics.demo        # Play In Editor, clip + screenshot
    unrealpy run --packaged examples.live_analytics.game:demo   # packaged build (--group analytics)

Only /Engine content and the template mannequins (see examples/mannequins.py) are used.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

import ue
from examples import stage
from examples.characters import DemoMannequin
from examples.live_analytics import level
from ue import agent

if TYPE_CHECKING:
    from examples.live_analytics import playtest, render

# ClassPackages discovers the example classes at startup, but
# pandas and matplotlib wait until an AnalyticsMode actually starts (level is plain Python).

SAMPLE_HZ = 5.0
DISPLAY_SECONDS = 0.5
SEED = 1
TESTER_TINT = (0.78, 0.8, 0.84)
DEATH_GLOW, EXIT_GLOW = (1.0, 0.02, 0.01), (0.1, 1.0, 0.25)
SCREEN_SIZE = (1400.0, 788.0)
SCREEN_AT = (0.0, -2100.0, SCREEN_SIZE[1] / 2.0 + 80.0)  # south of the level, facing it
# The cameras look from the north, so the level reads left to right: start, bridge, exit.
WIDE = ((-300.0, 2700.0, 2300.0), (0.0, -1130.0, 0.0), 62.0)  # camera: location, target, fov
HAZARD = ((-700.0, 1250.0, 800.0), (0.0, 0.0, 80.0), 45.0)  # the bridge and the blade
SCREEN = ((-300.0, -500.0, 650.0), (0.0, -2100.0, 600.0), 58.0)  # the charts fill the frame
FILM_AT, FILM_LOOKS_AT = WIDE[0], WIDE[1]
LABELS = (  # caption, colour, picture width, location, size in cm, the camera it faces
    (
        "Where playtesters walk (NumPy heatmap)",
        "#5ef0f0",
        3000,
        (-1450.0, 0.0, 330.0),
        (1300.0, 87.0),
        WIDE[0],
    ),
    ("Where they die", "#ff5a4f", 1400, (0.0, 150.0, 380.0), (650.0, 93.0), HAZARD[0]),
    (
        "pandas + matplotlib, rendered in-game",
        "#ffffff",
        2800,
        (0.0, -2100.0, 940.0),
        (1400.0, 100.0),
        SCREEN[0],
    ),
)
TRANSLUCENT_SCREEN = "/Engine/EngineMaterials/Widget3DPassThrough_Translucent.Widget3DPassThrough_Translucent"
WALL_HEIGHT = 140.0
FALLEN_Z = -700.0  # a body this far down the pit is gone


def make_renderer() -> Executor:
    """A one-process pool on the engine's own interpreter (pythonw: no console window), so matplotlib never
    holds the game's GIL; a thread if a process cannot be started."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    home = Path(sys.base_prefix).resolve()
    python = home / ("pythonw.exe" if os.name == "nt" else "bin/python3")
    if not python.is_file():
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="analytics-render")
    try:
        context = multiprocessing.get_context("spawn")
        context.set_executable(str(python))
        return ProcessPoolExecutor(max_workers=1, mp_context=context)
    except (OSError, ValueError) as exc:
        ue.warning(f"analytics: no render process ({exc}); rendering in a thread")
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="analytics-render")


@ue.uclass
class PlaytestTester(DemoMannequin):
    """A playtester: a mannequin on autopilot, running after the waypoint its run gives it. Testers pass
    through each other (no queues on the bridge), and die as a ragdoll."""

    @override
    def begin_play(self) -> None:
        super().begin_play()
        self.capsule_component.set_collision_response_to_channel(
            ue.ECollisionChannel.ECC_Pawn, ue.ECollisionResponse.ECR_Ignore
        )

    def die(self, velocity: tuple[float, float, float]) -> None:
        """Go limp (the mannequin's physics asset) and fly off with `velocity`."""
        self.character_movement.disable_movement()
        self.capsule_component.set_collision_enabled(ue.ECollisionEnabled.NoCollision)
        mesh = self.mesh
        mesh.set_collision_profile_name("Ragdoll")
        mesh.set_all_bodies_simulate_physics(True)
        mesh.set_all_physics_linear_velocity(ue.Vector(*velocity), False)


@ue.uclass
class AnalyticsMode(ue.GameModeBase):
    default_pawn_class = ue.SpectatorPawn

    def begin_play(self) -> None:
        from examples.live_analytics import playtest, render, telemetry

        self.render_module = render
        self.session: playtest.Session = playtest.Session(seed=SEED)
        self.telemetry = telemetry.Telemetry()
        self.bounds = telemetry.LEVEL_BOUNDS
        built = build_level()
        self.screen: stage.Screen = built["screen"]
        self.heat: stage.Screen = built["heat"]
        self.labels: list[stage.Screen] = built["labels"]
        self.blade: tuple[ue.StaticMeshActor, ue.StaticMeshActor] = built["blade"]
        self.spikes: list[ue.StaticMeshActor] = built["spikes"]
        self.markers: list[tuple[ue.StaticMeshActor, ...]] = built["markers"]
        self.next_marker = 0
        self.rings: list[tuple[ue.StaticMeshActor, ue.MaterialInstanceDynamic]] = built["rings"]
        self.ring_life = [0.0] * len(self.rings)
        self.ring_span = [1.0] * len(self.rings)
        self.testers: dict[int, PlaytestTester] = {}  # run id -> its mannequin
        self.undressed: list[PlaytestTester] = []
        self.bodies: list[tuple[PlaytestTester, float]] = []  # ragdolls, with the time they died
        self.running = False  # the showcase starts the session once the captions are ready
        self.recording = False
        self.sample_timer = 0.0
        self.display_timer = 0.0
        self.renderer = make_renderer()
        self.job: Future[render.Pictures] | None = None
        self.render_seconds: list[float] = []
        texts = [(text, colour, width) for text, colour, width, _, _, _ in LABELS]
        self.captions: Future[list[bytes]] | None = self.renderer.submit(render.captions, texts)
        # Empty displays to start from (a clear floor, charts at zero), not the material's white.
        self.blank: Future[render.Pictures] | None = self.renderer.submit(
            render.render, self.telemetry.snapshot(), 0.0, self.bounds
        )
        self._hazards()
        ue.log("AnalyticsMode: level built, waiting for the playtesters")

    @property
    def ready(self) -> bool:
        """The captions are on (the render process has started and drawn them)."""
        if self.captions is not None and self.captions.done():
            for screen, picture in zip(self.labels, self.captions.result(), strict=False):
                screen.show(picture)
            self.captions = None
        if self.blank is not None and self.blank.done():
            pictures = self.blank.result()
            self.screen.show(pictures.dashboard)
            self.heat.show(pictures.heat)
            self.blank = None
        return self.captions is None and self.blank is None

    def end_play(self, end_play_reason: Any) -> None:
        self.renderer.shutdown(wait=False, cancel_futures=True)

    def tick(self, delta_seconds: float) -> None:
        for tester in self.undressed:
            tester.dress(tint=TESTER_TINT)
        self.undressed.clear()
        if not self.running and not stage.scripted() and self.ready:
            self.running = True  # played by hand: start once the captions are on
        if not self.running:
            return
        self._play(delta_seconds)
        self._sample(delta_seconds)
        self._display(delta_seconds)
        self._animate(delta_seconds)

    def _play(self, dt: float) -> None:
        session = self.session
        for run_id, run in session.runs.items():
            tester = self.testers.get(run_id)
            if tester is None:
                continue
            tester.steer_to(run.goal, session.pace(run), arrive=10.0)
            here = tester.get_actor_location()
            run.position = (float(here[0]), float(here[1]))
        for event in session.step(dt):
            x, y = event.position
            self.telemetry.event(
                event.time, event.kind, event.run, event.tester, event.route, x, y, event.cause
            )
            if event.kind == "spawn":
                tester = ue.spawn_actor(PlaytestTester, ue.Vector(x, y, 100.0), ue.Rotator(0.0, 0.0, 0.0))
                self.testers[event.run] = tester
                self.undressed.append(tester)
                continue
            tester = self.testers.pop(event.run, None)
            if event.kind == "finish":
                self._ring((x, y), EXIT_GLOW, 0.6)
                if tester is not None:
                    tester.destroy_actor()
                continue
            self._ring((x, y), DEATH_GLOW, 0.7)  # a death: a red cross where it happened
            self._mark((x, y))
            self.display_timer = 0.0  # the deaths chart ticks up now
            if tester is not None:
                knock = (0.0, event.push * 1100.0, 450.0) if event.cause == "blade" else (0.0, 0.0, 150.0)
                tester.die(knock)
                self.bodies.append((tester, session.clock))
        self._hazards()
        for body, died in list(self.bodies):  # the ones the blade threw into the pit
            if body.get_actor_location()[2] < FALLEN_Z or session.clock - died > 30.0:
                body.destroy_actor()
                self.bodies.remove((body, died))

    def _hazards(self) -> None:
        """The blade swings on its pivot; the spikes come up and go down."""
        angle = self.session.blade_angle()
        side = 1.0 if angle >= 0.0 else -1.0
        rotation = ue.Rotator(-(90.0 - abs(angle)), 90.0 * side, 0.0)
        px, py, pz = level.BLADE_PIVOT
        down = (0.0, math.sin(math.radians(angle)), -math.cos(math.radians(angle)))
        arm, head = self.blade
        for actor, reach in ((arm, level.BLADE_LENGTH / 2.0), (head, level.BLADE_LENGTH)):
            at = ue.Vector(px + down[0] * reach, py + down[1] * reach, pz + down[2] * reach)
            actor.set_actor_location_and_rotation(at, rotation, False, True)
        height = 30.0 if self.session.spikes_up() else -60.0
        for spike in self.spikes:
            location = spike.get_actor_location()
            spike.set_actor_location(ue.Vector(location[0], location[1], height), False, True)

    def _sample(self, dt: float) -> None:
        self.sample_timer -= dt
        if self.sample_timer > 0.0:
            return
        self.sample_timer += 1.0 / SAMPLE_HZ
        clock = self.session.clock
        for run in self.session.runs.values():
            self.telemetry.position(clock, run.id, run.tester, run.route, *run.position)

    def _display(self, dt: float) -> None:
        self.display_timer -= dt
        if self.job is None and self.display_timer <= 0.0:
            self.display_timer = DISPLAY_SECONDS
            self.job = self.renderer.submit(
                self.render_module.render, self.telemetry.snapshot(), self.session.clock, self.bounds
            )
        elif self.job is not None and self.job.done():
            job, self.job = self.job, None
            try:
                pictures = job.result()
            except Exception as exc:  # a broken render must not stop the game
                ue.error(f"analytics render failed: {exc}")
                return
            self.render_seconds.append(pictures.seconds)
            self.screen.show(pictures.dashboard)
            self.heat.show(pictures.heat)

    def _animate(self, dt: float) -> None:
        """Death and finish rings grow and fade."""
        for index, (ring, material) in enumerate(self.rings):
            if self.ring_life[index] <= 0.0:
                continue
            self.ring_life[index] = max(0.0, self.ring_life[index] - dt)
            age = 1.0 - self.ring_life[index] / self.ring_span[index]
            ring.set_actor_scale3d(ue.Vector(0.8 + 2.5 * age, 0.8 + 2.5 * age, 0.02))
            material.set_scalar_parameter_value("EmissiveScale", 40.0 * (1.0 - age))
            if self.ring_life[index] == 0.0:
                ring.set_actor_hidden_in_game(True)

    def _ring(self, position: tuple[float, float], glow: tuple[float, float, float], life: float) -> None:
        index = min(range(len(self.rings)), key=lambda i: self.ring_life[i])
        ring, material = self.rings[index]
        ring.set_actor_location(ue.Vector(position[0], position[1], 6.0), False, True)
        ring.set_actor_hidden_in_game(False)
        stage.recolour(material, (0.0, 0.0, 0.0), glow, 40.0)
        self.ring_life[index] = life
        self.ring_span[index] = life

    def _mark(self, position: tuple[float, float]) -> None:
        """A red cross on the floor where a playtester died (the oldest one moves when all are used)."""
        pair = self.markers[self.next_marker % len(self.markers)]
        self.next_marker += 1
        for bar, yaw in zip(pair, (45.0, -45.0), strict=True):
            bar.set_actor_location_and_rotation(
                ue.Vector(position[0], position[1], 3.0), ue.Rotator(0.0, yaw, 0.0), False, True
            )
            bar.set_actor_hidden_in_game(False)

    def status(self) -> dict[str, Any]:
        events = self.telemetry.events
        return {
            "game_seconds": round(self.session.clock, 1),
            "runs": sum(1 for e in events if e[1] == "spawn"),
            "deaths": sum(1 for e in events if e[1] == "death"),
            "deaths_on_bridge": sum(1 for e in events if e[1] == "death" and e[7] == "blade"),
            "finished": sum(1 for e in events if e[1] == "finish"),
            "samples": len(self.telemetry.positions),
            "renders": len(self.render_seconds),
            "render_ms_mean": round(1000 * sum(self.render_seconds) / max(1, len(self.render_seconds)), 1),
            "renderer": type(self.renderer).__name__,
        }


def build_level() -> dict[str, Any]:
    # Late afternoon: a warm sun, a blue sky, floodlights; dark floors so the heatmap glows on them.
    stage.lighting(
        sun_pitch=-32.0,
        sun_yaw=-125.0,
        sun=4.5,
        sky=1.2,
        fog=0.008,
        sun_colour=(1.0, 0.78, 0.55),
        sky_colour=(0.55, 0.7, 1.0),
        fog_colour=(0.08, 0.11, 0.2),
    )
    stage.look(exposure_bias=-0.5, bloom=0.9)
    stage.prop("Cube", (0.0, 0.0, -1000.0), (120.0, 120.0, 0.2), (0.02, 0.025, 0.035), roughness=0.95)
    # The level: platforms over the void (floor top at z = 0), low walls with doorways.
    for piece in level.PIECES:
        cx, cy = piece.centre
        scale = ((piece.x1 - piece.x0) / 100.0, (piece.y1 - piece.y0) / 100.0, 0.4)
        if piece.walls:
            slab = stage.prop("Cube", (cx, cy, -20.0), scale)
            stage.grid(
                slab.static_mesh_component,
                (0.035, 0.038, 0.045),
                (0.07, 0.078, 0.095),
                size=100.0,
                roughness=0.9,
            )
        else:  # the bridge: bare metal planks
            stage.prop(
                "Cube",
                (cx, cy, -10.0),
                (scale[0], scale[1], 0.2),
                (0.09, 0.08, 0.07),
                roughness=0.45,
                metallic=0.8,
            )
    for (x0, y0), (x1, y1) in level.walls():
        length = math.hypot(x1 - x0, y1 - y0) + 20.0
        tall = WALL_HEIGHT / 100.0
        scale = (length / 100.0, 0.2, tall) if abs(y1 - y0) < 1.0 else (0.2, length / 100.0, tall)
        stage.prop(
            "Cube",
            ((x0 + x1) / 2.0, (y0 + y1) / 2.0, WALL_HEIGHT / 2.0),
            scale,
            (0.16, 0.17, 0.2),
            roughness=0.5,
        )
    # Hazard stripes where the blade sweeps the bridge, the exit pad.
    for x in (-level.BLADE_REACH, level.BLADE_REACH):
        stage.prop(
            "Cube",
            (x, 0.0, 1.0),
            (0.12, 1.4, 0.02),
            (1.0, 0.75, 0.05),
            glow=(1.0, 0.6, 0.02),
            strength=4.0,
            collision=False,
            shadow=False,
        )
    pad = level.EXIT_PAD
    stage.prop(
        "Cube",
        (*pad.centre, 1.0),
        ((pad.x1 - pad.x0) / 100.0, (pad.y1 - pad.y0) / 100.0, 0.02),
        (0.05, 0.6, 0.15),
        glow=EXIT_GLOW,
        strength=3.0,
        collision=False,
        shadow=False,
    )
    blade = _build_blade()
    spikes = _build_spikes()
    # The floor overlay: a translucent plane just above the floors, showing the latest picture.
    x_min, x_max, y_min, y_max = level.BOUNDS
    heat = stage.Screen(
        ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, 1.5),
        0.0,
        x_max - x_min,
        y_max - y_min,
        brightness=2.5,
        material=TRANSLUCENT_SCREEN,
        flat=True,
    )
    screen = _build_screen()
    markers = []
    for _ in range(40):
        pair = tuple(
            stage.prop(
                "Cube",
                (0.0, 0.0, -500.0),
                (0.9, 0.16, 0.01),
                (0.4, 0.0, 0.0),
                glow=DEATH_GLOW,
                strength=20.0,
                collision=False,
                shadow=False,
            )
            for _ in range(2)
        )
        for bar in pair:
            bar.set_actor_hidden_in_game(True)
        markers.append(pair)
    rings = []
    for _ in range(10):
        ring = stage.prop(
            "Cylinder",
            (0.0, 0.0, -500.0),
            (1.0, 1.0, 0.02),
            (0.0, 0.0, 0.0),
            glow=(1.0, 1.0, 1.0),
            strength=0.0,
            collision=False,
            shadow=False,
        )
        ring.set_actor_hidden_in_game(True)
        rings.append((ring, cast(ue.MaterialInstanceDynamic, ring.static_mesh_component.get_material(0))))
    # Floodlights from the four corners.
    for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        at = (x * 2300.0, y * 1600.0 - 300.0, 1100.0)
        stage.accent(at, (x * 700.0, -300.0, 0.0), (1.0, 0.93, 0.82), 9000.0, cone=40.0, radius=5000.0)
    # Captions over the level, the hazard and the screen (filled in when the render process has drawn them).
    labels = []
    for _, _, _, location, (width, height), camera in LABELS:
        yaw = stage.look_at(location, camera)[1]
        labels.append(stage.Screen(location, yaw, width, height, brightness=1.6, material=TRANSLUCENT_SCREEN))
    return {
        "screen": screen,
        "heat": heat,
        "labels": labels,
        "blade": blade,
        "spikes": spikes,
        "markers": markers,
        "rings": rings,
    }


def _build_blade() -> tuple[ue.StaticMeshActor, ue.StaticMeshActor]:
    """A short pendulum over the middle of the bridge, hung from a low cantilever gantry: two posts on the
    bridge's far (south) side, clear of the head's sweep and behind the action from the cameras, with an arm
    and a wide, thin head swinging across."""
    steel = (0.35, 0.36, 0.4)
    px, _, pz = level.BLADE_PIVOT
    top, bottom, post_y = pz + 20.0, -300.0, -110.0
    for x in (-150.0, 150.0):
        stage.prop(
            "Cube",
            (px + x, post_y, (top + bottom) / 2.0),
            (0.16, 0.16, (top - bottom) / 100.0),
            steel,
            roughness=0.35,
            metallic=1.0,
        )
        stage.prop(
            "Cube",
            (px + x, post_y / 2.0 + 10.0, top),
            (0.16, (20.0 - post_y) / 100.0, 0.16),
            steel,
            roughness=0.35,
            metallic=1.0,
        )
    stage.prop("Cube", (px, 0.0, top), (3.16, 0.2, 0.2), steel, roughness=0.35, metallic=1.0)
    arm = stage.prop(
        "Cube",
        (px, 0.0, pz),
        (level.BLADE_LENGTH / 100.0, 0.1, 0.1),
        steel,
        roughness=0.3,
        metallic=1.0,
        collision=False,
    )
    head = stage.prop(
        "Cube",
        (px, 0.0, pz),
        (0.55, 2.0 * level.BLADE_REACH / 100.0, 0.08),
        (0.7, 0.05, 0.03),
        roughness=0.25,
        metallic=1.0,
        glow=(1.0, 0.05, 0.02),
        strength=2.0,
        collision=False,
    )
    return arm, head


def _build_spikes() -> list[ue.StaticMeshActor]:
    """The detour's trap: a dark red plate and a grid of cones that come up through it."""
    trap = level.SPIKES
    cx, cy = trap.centre
    stage.prop(
        "Cube",
        (cx, cy, 0.5),
        ((trap.x1 - trap.x0) / 100.0, (trap.y1 - trap.y0) / 100.0, 0.01),
        (0.25, 0.02, 0.02),
        roughness=0.4,
        collision=False,
        shadow=False,
    )
    spikes = []
    for i in range(5):
        for j in range(3):
            x = trap.x0 + (i + 0.5) * (trap.x1 - trap.x0) / 5.0
            y = trap.y0 + (j + 0.5) * (trap.y1 - trap.y0) / 3.0
            spike = stage.prop(
                "Cone",
                (x, y, -60.0),
                (0.22, 0.22, 0.6),
                (0.55, 0.55, 0.6),
                roughness=0.3,
                metallic=1.0,
                collision=False,
            )
            spikes.append(spike)
    return spikes


def _build_screen() -> stage.Screen:
    """The screen for the charts, on legs behind the level, facing it."""
    sx, sy, sz = SCREEN_AT
    width, height = SCREEN_SIZE
    frame = (0.03, 0.03, 0.035)
    stage.prop(
        "Cube",
        (sx, sy - 14.0, sz),
        (width / 100.0 + 0.4, 0.2, height / 100.0 + 0.4),
        frame,
        roughness=0.25,
        metallic=1.0,
    )
    leg = (sz - height / 2.0) / 2.0
    for side in (-1.0, 1.0):
        stage.prop(
            "Cube",
            (sx + side * width * 0.38, sy - 16.0, leg),
            (0.2, 0.2, 2.0 * leg / 100.0),
            frame,
            roughness=0.25,
            metallic=1.0,
        )
    return stage.Screen(SCREEN_AT, 90.0, width, height, brightness=1.4)


# -- scripted runs -------------------------------------------------------------------------------------------


async def _drive(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        await agent.frames(1)


CAMERA_PATH = (  # 15 s, no cuts: the level, the hazard, the charts
    (0.0, *WIDE),
    (4.0, *WIDE),
    (6.0, *HAZARD),
    (9.5, *HAZARD),
    (11.5, *SCREEN),
    (15.0, *SCREEN),
)


async def showcase(mode: AnalyticsMode, *, warmup: float, clip: float, out: Path) -> dict[str, Any]:
    """Wait for the captions, start the session, let the first wave get going and shoot the clip while the
    camera moves: the heatmap, the death crosses and the charts visibly build up."""
    film = stage.Film(WIDE[0], stage.look_at(WIDE[0], WIDE[1]), fov=WIDE[2])
    path = stage.CameraPath([(t, location, target, fov) for t, location, target, fov in CAMERA_PATH])
    deadline = time.monotonic() + 60.0
    while not mode.ready and time.monotonic() < deadline:
        await agent.frames(1)
    await film.warm_up()
    mode.running = True
    await _drive(warmup)
    recording: dict[str, Any] = {}
    clock = [0.0]

    def move(dt: float) -> None:
        clock[0] += dt
        location, target, fov = path.at(clock[0])
        film.place(location, target, fov)

    if clip > 0:
        mode.recording = True
        move(0.0)
        recording = await stage.record(film, out / "frames", clip, 30, move)
        mode.recording = False
    still = await film.shoot(out / f"still-{time.strftime('%Y%m%d-%H%M%S')}.png")
    return {"recording": recording, "still": str(still), **mode.status()}


def _out_dir() -> Path:
    return Path(ue.saved_dir()).resolve() / "LiveAnalytics"


def _options(args: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="live_analytics")
    parser.add_argument("--warmup", type=float, default=3.0, help="seconds of play before the clip")
    parser.add_argument("--clip", type=float, default=15.0)
    return vars(parser.parse_args(args))


async def pie(args: list[str]) -> dict[str, Any]:
    """`unrealpy run --editor --render examples.live_analytics.game:pie`: Play In Editor."""
    options = _options(args)
    mode = await stage.start_in_editor(AnalyticsMode)
    try:
        result = await showcase(mode, out=_out_dir() / "pie", **options)
    finally:
        stage.stop_in_editor()
    return {"where": "play in editor", **result}


async def demo(args: list[str]) -> dict[str, Any]:
    """`unrealpy run --render examples.live_analytics.game:demo` (add --packaged for the packaged game)."""
    options = _options(args)
    mode = await stage.start_in_game(AnalyticsMode)
    return {"where": "game", **await showcase(mode, out=_out_dir() / "game", **options)}


def play(_args: list[str]) -> None:
    """Interactive: watch the playtesters and the analytics update (fly around with WASD and the mouse)."""
    world_ = ue.get_world()
    assert world_ is not None
    level_name = world_.get_path_name().split(".", 1)[0]
    ue.GameplayStatics.open_level(
        level_name, True, f"game={stage.class_path(AnalyticsMode)}", world_context_object=world_
    )
