"""Security camera with vision: OpenCV running inside the game, in gameplay code written with UnrealPy.

A camera on a post (a SceneCaptureComponent2D rendering into a 256x144 render target) watches a room. Eight
times a second the game reads its pixels and OpenCV finds what moves (MOG2 background subtraction, contours,
a tracker that keeps ids). The gameplay reacts: a spotlight turret swings onto the tracked target, an alarm
beacon turns red, and a monitor on the wall shows the camera feed with OpenCV's boxes drawn on it (the
annotated frame goes back into a texture).

    uv run python Python/examples/security_camera/demo.py          # Play In Editor, clip + screenshot
    unrealpy run --render examples.security_camera.game:demo       # standalone game
    unrealpy run --packaged examples.security_camera.game:demo     # packaged build (--group vision)
    UnrealEditor.exe <YourProject>.uproject -game -windowed -UnrealPyRun=examples.security_camera.game:play
        -UnrealPyKeepAlive                                         # walk in front of the camera yourself

Only /Engine content is used; the scene is built when the game mode starts.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

import ue
from examples import showreel, stage
from examples.characters import WALK_SPEED, DemoMannequin
from ue import agent

if TYPE_CHECKING:
    from examples.security_camera import vision

# The project imports every module under Python/examples at startup (ClassPackages) to define its classes:
# OpenCV waits until a SecurityMode actually starts.

# The room: x in [-ROOM_X, ROOM_X], y in [-ROOM_Y, ROOM_Y], floor at z = 0 (centimetres).
ROOM_X, ROOM_Y, WALL = 800.0, 600.0, 170.0
CAMERA_AT = (740.0, 560.0, 430.0)
CAMERA_LOOKS_AT = (-180.0, -150.0, 0.0)
FEED_SIZE = (256, 144)
FEED_FOV = 90.0
VISION_HZ = 8.0
READBACK_DELAY = 2  # frames between the capture and its read-back
TURRET_AT = (560.0, -500.0)
TURRET_HEIGHT = 390.0
BEACON_AT = (790.0, 360.0, 260.0)
SCREEN_AT = (812.0, -40.0, 360.0)
SCREEN_SIZE = (560.0, 315.0)
FILM_AT = (-1330.0, -420.0, 760.0)
FILM_LOOKS_AT = (260.0, 20.0, 130.0)

PERF = showreel.PerfMeter()


def _mesh(name: str) -> ue.StaticMesh:
    return stage.shape(name)


MOVABLE = ue.EComponentMobility.Movable


# -- actors --------------------------------------------------------------------------------------------------


@ue.uclass
class SecurityWalker(DemoMannequin):
    """A mannequin that walks a loop of waypoints, pausing now and then (idle), at the speed its walk
    animation was made for (no foot sliding); the anim blueprint blends idle, walk and jog."""

    @override
    def begin_play(self) -> None:
        super().begin_play()
        self.route: list[tuple[float, float, float]] = []  # (x, y, pause seconds)
        self.leg = 0
        self.wait = 0.0
        self.pace = WALK_SPEED

    def walk(self, route: list[tuple[float, float, float]], pace: float = WALK_SPEED) -> None:
        self.route, self.pace = route, pace

    def tick(self, delta_seconds: float) -> None:
        if not self.route:
            return
        if self.wait > 0.0:
            self.wait -= delta_seconds
            return
        x, y, pause = self.route[self.leg]
        if self.steer_to((x, y), self.pace, arrive=50.0):
            self.leg = (self.leg + 1) % len(self.route)
            self.wait = pause


@ue.uclass
class SpotlightTurret(ue.Actor):
    """A searchlight on a pole: sweeps while the room is clear, swings onto the camera's target."""

    root = ue.component(ue.SceneComponent, mobility=MOVABLE)
    pole = ue.component(
        ue.StaticMeshComponent,
        static_mesh=_mesh("Cylinder"),
        mobility=MOVABLE,
        relative_location=ue.Vector(0.0, 0.0, TURRET_HEIGHT / 2.0),
        relative_scale3d=ue.Vector(0.3, 0.3, TURRET_HEIGHT / 100.0),
    )
    head = ue.component(
        ue.SceneComponent, mobility=MOVABLE, relative_location=ue.Vector(0.0, 0.0, TURRET_HEIGHT + 30.0)
    )
    lamp = ue.component(
        ue.StaticMeshComponent,
        parent="head",
        static_mesh=_mesh("Cylinder"),
        mobility=MOVABLE,
        relative_rotation=ue.Rotator(-90.0, 0.0, 0.0),
        relative_scale3d=ue.Vector(0.55, 0.55, 0.8),
    )
    lens = ue.component(
        ue.StaticMeshComponent,
        parent="head",
        static_mesh=_mesh("Cylinder"),
        mobility=MOVABLE,
        relative_location=ue.Vector(41.0, 0.0, 0.0),
        relative_rotation=ue.Rotator(-90.0, 0.0, 0.0),
        relative_scale3d=ue.Vector(0.45, 0.45, 0.02),
    )
    beam = ue.component(
        ue.SpotLightComponent,
        parent="head",
        mobility=MOVABLE,
        relative_location=ue.Vector(60.0, 0.0, 0.0),
        intensity=25000.0,
        attenuation_radius=4000.0,
        inner_cone_angle=7.0,
        outer_cone_angle=13.0,
    )
    turn_rate: float = ue.uproperty(110.0)  # degrees per second

    def begin_play(self) -> None:
        stage.surface(self.pole, (0.05, 0.05, 0.06), roughness=0.35, metallic=1.0)
        stage.surface(self.lamp, (0.85, 0.62, 0.08), roughness=0.3, metallic=1.0)
        stage.glowing(self.lens, (1.0, 0.95, 0.85), 40.0)
        for part in (self.lamp, self.lens):
            part.set_cast_shadow(False)  # the light sits just in front of the lamp's glass
        self.beam.set_volumetric_scattering_intensity(4.0)
        self.pitch, self.yaw = -30.0, 135.0
        self.target: tuple[float, float, float] | None = None
        self.clock = 0.0

    def tick(self, delta_seconds: float) -> None:
        self.clock += delta_seconds
        origin = self.head.get_world_location()
        if self.target is not None:
            pitch, yaw, _ = stage.look_at(origin, self.target)
        else:  # idle sweep across the room
            pitch, yaw = -32.0, 140.0 + 35.0 * math.sin(self.clock * 0.8)
        step = self.turn_rate * delta_seconds
        self.pitch = stage.turn_towards(self.pitch, pitch, step)
        self.yaw = stage.turn_towards(self.yaw, yaw, step)
        self.head.set_world_rotation(ue.Rotator(self.pitch, self.yaw, 0.0), False, True)


@ue.uclass
class AlarmBeacon(ue.Actor):
    root = ue.component(ue.SceneComponent, mobility=MOVABLE)
    bulb = ue.component(
        ue.StaticMeshComponent,
        static_mesh=_mesh("Sphere"),
        mobility=MOVABLE,
        relative_scale3d=ue.Vector(0.45, 0.45, 0.45),
    )
    glow = ue.component(
        ue.PointLightComponent,
        mobility=MOVABLE,
        relative_location=ue.Vector(-40.0, 0.0, 0.0),
        intensity=4000.0,
        attenuation_radius=900.0,
    )

    def begin_play(self) -> None:
        self.bulb.set_collision_enabled(ue.ECollisionEnabled.NoCollision)
        self.paint = stage.glowing(self.bulb, (0.1, 1.0, 0.2), 20.0)
        self.armed = False
        self.clock = 0.0

    def tick(self, delta_seconds: float) -> None:
        self.clock += delta_seconds
        if self.armed:
            on = int(self.clock * 4.0) % 2 == 0
            colour = (1.0, 0.02, 0.01) if on else (0.25, 0.0, 0.0)
            self.glow.set_intensity(30000.0 if on else 2000.0)
        else:
            colour = (0.1, 1.0, 0.2)
            self.glow.set_intensity(3000.0)
        stage.recolour(self.paint, (0.02, 0.02, 0.02), colour, 40.0)
        self.glow.set_light_color(ue.LinearColor(*colour, 1.0), True)


@dataclass
class _Camera:
    """The security camera: the capture the game reads, its picture's file, the screen it feeds."""

    capture: ue.SceneCaptureComponent2D
    target: ue.TextureRenderTarget2D
    feed: Path
    screen: stage.Screen
    pipeline: vision.Pipeline
    worker: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="security-vision")
    )
    job: Future[vision.Sighting] | None = None
    last: vision.Sighting | None = None
    step: int = 0
    wait: int = 0
    timer: float = 0.0


@ue.uclass
class SecurityMode(ue.GameModeBase):
    """Builds the room, runs the camera's vision and drives the turret, beacon and monitor.

    The vision runs as a little pipeline spread over frames, so no single frame pays for all of it:
    capture (the render target renders with this frame) -> read it back next frame and hand it to the
    worker thread (OpenCV) -> when the worker answers, aim, sound the alarm and put the picture on the
    monitor."""

    default_pawn_class = ue.SpectatorPawn  # the people are the walkers; the player watches

    def begin_play(self) -> None:
        import cv2

        self.imread = cv2.imread
        self.clock = 0.0
        built = build_room()
        self.turret: SpotlightTurret = built["turret"]
        self.beacon: AlarmBeacon = built["beacon"]
        self.walkers: list[SecurityWalker] = built["walkers"]
        self.lasers: list[ue.MaterialInstanceDynamic] = built["lasers"]
        self.camera = _Camera(**built["camera"])
        ue.log("SecurityMode: room built, camera watching")

    def end_play(self, end_play_reason: Any) -> None:
        self.camera.worker.shutdown(wait=False, cancel_futures=True)

    def tick(self, delta_seconds: float) -> None:
        self.clock += delta_seconds
        PERF.frame()
        with PERF.measure("vision (game thread)"):
            self._vision_step(delta_seconds)
        armed = self.beacon.armed
        strength = (60.0 if int(self.clock * 6.0) % 2 == 0 else 25.0) if armed else 8.0
        for laser in self.lasers:
            laser.set_scalar_parameter_value("EmissiveScale", strength)

    def _vision_step(self, delta_seconds: float) -> None:
        camera = self.camera
        camera.timer += delta_seconds
        if camera.step == 0 and camera.timer >= 1.0 / VISION_HZ:
            camera.timer = 0.0
            camera.capture.capture_scene()
            camera.step, camera.wait = 1, READBACK_DELAY
        elif camera.step == 1 and camera.wait > 0:
            camera.wait -= 1  # let the GPU finish the capture: the read-back then waits less
        elif camera.step == 1:
            with PERF.measure("readback"):
                ue.KismetRenderingLibrary.export_render_target(
                    camera.target, str(camera.feed.parent), camera.feed.name, world_context_object=self
                )
                frame = cast("vision.Image | None", self.imread(str(camera.feed)))
            if frame is None:
                camera.step = 0
                return
            camera.job = camera.worker.submit(self._timed_process, frame, self.clock)
            camera.step = 2
        elif camera.step == 2 and camera.job is not None and camera.job.done():
            sighting = camera.job.result()
            camera.job, camera.last, camera.step = None, sighting, 0
            self.turret.target = sighting.target
            self.beacon.armed = sighting.alarm
            with PERF.measure("monitor"):
                camera.screen.show(sighting.picture)

    def _timed_process(self, frame: vision.Image, clock: float) -> vision.Sighting:
        """In the worker thread."""
        with PERF.measure("opencv (worker thread)"):
            return self.camera.pipeline.process(frame, clock)

    def status(self) -> dict[str, Any]:
        last = self.camera.last
        return {
            "vision_frames": self.camera.pipeline.frames,
            "tracks": list(last.tracks) if last else [],
            "alarm": bool(last and last.alarm),
            "target": last.target if last else None,
        }


def build_room() -> dict[str, Any]:
    """The room, its props, the walkers, the camera and its screen, the turret and the beacon."""
    from examples.security_camera import vision

    # Early evening: a low warm sun raking across the room, a cool sky, a little volumetric haze.
    stage.lighting(
        sun_pitch=-16.0,
        sun_yaw=-55.0,
        sun=3.5,
        sky=1.2,
        fog=0.02,
        sun_colour=(1.0, 0.68, 0.42),
        sky_colour=(0.55, 0.7, 1.0),
        fog_colour=(0.08, 0.12, 0.22),
    )
    stage.look(exposure_bias=-0.6, bloom=0.9)
    ground = stage.prop("Cube", (0.0, 0.0, -14.0), (60.0, 60.0, 0.2), (0.012, 0.013, 0.016), roughness=0.95)
    floor = stage.prop("Cube", (0.0, 0.0, -10.0), (ROOM_X / 50.0, ROOM_Y / 50.0, 0.2))
    stage.grid(
        floor.static_mesh_component, (0.035, 0.04, 0.05), (0.07, 0.085, 0.11), size=100.0, roughness=0.85
    )
    del ground
    wall = (0.12, 0.118, 0.115)  # concrete: the background, cool and calm next to the warm walkers
    for x in (-ROOM_X, ROOM_X):  # the near wall (x < 0) is low so the film camera sees in
        height = 60.0 if x < 0 else WALL
        stage.prop("Cube", (x, 0.0, height / 2.0), (0.3, ROOM_Y / 50.0, height / 100.0), wall, roughness=0.9)
    for y in (-ROOM_Y, ROOM_Y):
        stage.prop("Cube", (0.0, y, WALL / 2.0), (ROOM_X / 50.0, 0.3, WALL / 100.0), wall, roughness=0.9)
    # A cool light strip along the top of the high walls.
    strip = (0.15, 0.55, 1.0)
    stage.prop(
        "Cube",
        (ROOM_X - 18.0, 0.0, WALL - 8.0),
        (0.06, ROOM_Y / 50.0, 0.05),
        strip,
        glow=strip,
        strength=2.5,
        collision=False,
        shadow=False,
    )
    for y in (-ROOM_Y + 18.0, ROOM_Y - 18.0):
        stage.prop(
            "Cube",
            (0.0, y, WALL - 8.0),
            (ROOM_X / 50.0, 0.06, 0.05),
            strip,
            glow=strip,
            strength=2.5,
            collision=False,
            shadow=False,
        )
    crate = (0.2, 0.09, 0.03)
    for x, y, size in ((250.0, 330.0, 1.1), (360.0, 360.0, 0.8), (-350.0, -330.0, 1.2), (60.0, -420.0, 0.9)):
        stage.prop("Cube", (x, y, size * 50.0), (size, size, size), crate, roughness=0.75)
    stage.prop(
        "Cylinder", (-120.0, 200.0, 60.0), (0.7, 0.7, 1.2), (0.08, 0.2, 0.55), roughness=0.3, metallic=1.0
    )
    # Laser tripwires across the room at knee height: dim while clear, flashing during an alarm.
    lasers = []
    for y in (-300.0, 0.0, 300.0):
        beam = stage.prop(
            "Cylinder",
            (0.0, y, 45.0),
            (0.025, 0.025, ROOM_X / 50.0),
            (1.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            glow=(1.0, 0.03, 0.02),
            strength=8.0,
            collision=False,
            shadow=False,
        )
        lasers.append(beam)
        for x in (-ROOM_X + 25.0, ROOM_X - 25.0):  # emitter posts
            stage.prop(
                "Cube", (x, y, 35.0), (0.18, 0.18, 0.7), (0.05, 0.05, 0.06), roughness=0.3, metallic=1.0
            )
    # Accent lights: two warm panels inside the walls, a cool rim from behind.
    stage.accent(
        (-ROOM_X + 60.0, -ROOM_Y + 60.0, 300.0), (0.0, 0.0, 0.0), (1.0, 0.75, 0.5), 1500.0, rect=True
    )
    stage.accent((ROOM_X - 60.0, -ROOM_Y + 60.0, 300.0), (0.0, 0.0, 0.0), (1.0, 0.7, 0.45), 1500.0, rect=True)
    stage.accent(
        (ROOM_X + 400.0, 0.0, 900.0), (0.0, 0.0, 0.0), (0.4, 0.6, 1.0), 3000.0, cone=40.0, radius=4000.0
    )

    # The people: two mannequins on their rounds (walking) and an intruder (Quinn, jogging), routes that
    # keep clear of the crates; warm metal paints against the cool room.
    walkers = []
    routes = (
        (
            [(-450.0, -100.0, 1.5), (450.0, -150.0, 0.0), (500.0, 230.0, 2.0), (-300.0, 300.0, 0.0)],
            (0.8, 0.82, 0.86),
            False,
            WALK_SPEED,
            0.0,
        ),
        (
            [(300.0, 100.0, 0.5), (-500.0, 120.0, 2.5), (-180.0, -300.0, 0.0), (420.0, -280.0, 1.0)],
            (1.0, 0.62, 0.2),
            False,
            WALK_SPEED,
            2.0,
        ),
        (
            [
                (-560.0, 420.0, 0.0),
                (560.0, 470.0, 0.0),
                (600.0, -60.0, 0.0),
                (-20.0, 40.0, 0.0),
                (-560.0, -200.0, 0.0),
                (-650.0, 200.0, 0.0),
            ],
            (1.0, 0.18, 0.4),
            True,
            450.0,
            4.0,
        ),
    )
    for route, tint, quinn, pace, delay in routes:
        start = route[-1]
        walker = ue.spawn_actor(SecurityWalker, ue.Vector(start[0], start[1], 100.0))
        walker.dress(quinn=quinn, tint=tint)
        walker.walk(route, pace)
        walker.wait = delay
        walkers.append(walker)

    # The security camera: a housing on a bracket, and the SceneCapture2D the game reads.
    rotation = stage.look_at(CAMERA_AT, CAMERA_LOOKS_AT)
    stage.prop(
        "Cylinder",
        (CAMERA_AT[0], CAMERA_AT[1], CAMERA_AT[2] / 2.0),
        (0.2, 0.2, CAMERA_AT[2] / 100.0),
        (0.05, 0.05, 0.06),
        roughness=0.35,
        metallic=1.0,
    )
    housing = stage.prop(
        "Cube", CAMERA_AT, (0.6, 0.28, 0.28), (0.8, 0.8, 0.78), rotation, collision=False, roughness=0.35
    )
    led = stage.prop(
        "Sphere",
        (CAMERA_AT[0], CAMERA_AT[1], CAMERA_AT[2] + 18.0),
        (0.06, 0.06, 0.06),
        (1.0, 0.0, 0.0),
        glow=(1.0, 0.02, 0.02),
        strength=60.0,
        collision=False,
        shadow=False,
    )
    rendering = ue.KismetRenderingLibrary
    target = rendering.create_render_target2d(
        FEED_SIZE[0], FEED_SIZE[1], ue.ETextureRenderTargetFormat.RTF_RGBA8, world_context_object=housing
    )
    camera = ue.spawn_actor(ue.SceneCapture2D, ue.Vector(*CAMERA_AT), ue.Rotator(*rotation))
    capture = camera.capture_component2d
    capture.texture_target = target
    capture.capture_source = ue.ESceneCaptureSource.SCS_FinalColorLDR
    capture.fov_angle = FEED_FOV
    capture.capture_every_frame = False
    capture.capture_on_movement = False
    stage.manual_exposure(capture, 2.5)
    # The camera sees what is in the room, not the lights the game itself moves (a sweeping spotlight
    # would look like motion), nor the monitor (which would film itself).
    capture.show_flag_settings = [
        ue.EngineShowFlagsSetting(show_flag_name="SpotLights", enabled=False),
        ue.EngineShowFlagsSetting(show_flag_name="PointLights", enabled=False),
    ]
    for own in (housing, led):
        capture.hide_actor_components(own, True)

    turret = ue.spawn_actor(SpotlightTurret, ue.Vector(TURRET_AT[0], TURRET_AT[1], 0.0))
    beacon = ue.spawn_actor(AlarmBeacon, ue.Vector(*BEACON_AT))
    stage.prop(
        "Cube",
        (BEACON_AT[0] + 10.0, BEACON_AT[1], (BEACON_AT[2] + WALL) / 2.0 - 20.0),
        (0.2, 0.2, (BEACON_AT[2] - WALL) / 100.0 + 0.2),
        (0.05, 0.05, 0.06),
        roughness=0.35,
        metallic=1.0,
    )
    # The monitor: a dark bezel on two legs behind the back wall, the screen in front of it.
    sx, sy, sz = SCREEN_AT
    width, height = SCREEN_SIZE
    bezel = stage.prop(
        "Cube",
        (sx + 14.0, sy, sz),
        (0.2, width / 100.0 + 0.3, height / 100.0 + 0.3),
        (0.03, 0.03, 0.035),
        roughness=0.25,
        metallic=1.0,
    )
    legs = [
        stage.prop(
            "Cube",
            (sx + 16.0, sy + side * width * 0.35, (sz - height / 2.0) / 2.0),
            (0.15, 0.15, (sz - height / 2.0) / 100.0),
            (0.03, 0.03, 0.035),
            roughness=0.25,
            metallic=1.0,
        )
        for side in (-1.0, 1.0)
    ]
    screen = stage.Screen(SCREEN_AT, 180.0, width, height, brightness=2.0)
    laser_materials = [
        cast(ue.MaterialInstanceDynamic, beam.static_mesh_component.get_material(0)) for beam in lasers
    ]
    for hidden in (screen.actor, bezel, *legs, beacon, turret, *lasers):
        capture.hide_actor_components(hidden, True)

    feed = Path(ue.project_dir()).resolve() / "Saved" / "SecurityCamera" / "feed.png"
    feed.parent.mkdir(parents=True, exist_ok=True)
    model = vision.CameraModel(CAMERA_AT, rotation, FEED_FOV, *FEED_SIZE)
    return {
        "turret": turret,
        "beacon": beacon,
        "walkers": walkers,
        "lasers": laser_materials,
        "camera": {
            "capture": capture,
            "target": target,
            "feed": feed,
            "screen": screen,
            "pipeline": vision.Pipeline(model, (-ROOM_X, ROOM_X, -ROOM_Y, ROOM_Y), aim_height=10.0),
        },
    }


# -- scripted runs -------------------------------------------------------------------------------------------


async def _drive(seconds: float) -> None:
    """Let the game run in real time for `seconds`."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        await agent.frames(1)


async def showcase(
    mode: SecurityMode, *, warmup: float, measure: float, clip: float, out: Path
) -> dict[str, Any]:
    """Warm up, measure the frame time and the vision's cost in real time, then shoot the clip."""
    film = stage.Film(FILM_AT, stage.look_at(FILM_AT, FILM_LOOKS_AT), fov=62.0)  # measured through it too
    await _drive(warmup)
    perf = {}
    for cap in (0, 60) if measure > 0 else ():  # optional: uncapped, then t.MaxFPS 60
        stage.max_fps(cap)
        PERF.reset()
        await _drive(measure)
        perf[f"max_fps_{cap}" if cap else "uncapped"] = PERF.summary()
    stage.max_fps(0)
    status = mode.status()
    await film.warm_up()
    recording: dict[str, Any] = {}
    if clip > 0:  # each frame of the clip is one fixed 1/30 s step of the game
        recording = await stage.record(film, out / "frames", clip, 30)
    still = await film.shoot(out / f"still-{time.strftime('%Y%m%d-%H%M%S')}.png")
    return {"perf": perf, "recording": recording, "still": str(still), **status}


def _out_dir() -> Path:
    return Path(ue.project_dir()).resolve() / "Saved" / "SecurityCamera"


def _options(args: list[str]) -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(prog="security_camera")
    parser.add_argument("--warmup", type=float, default=4.0)
    parser.add_argument("--measure", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=12.0)
    return vars(parser.parse_args(args))


async def pie(args: list[str]) -> dict[str, Any]:
    """`unrealpy run --editor --render Python/examples/security_camera/game.py:pie`: Play In Editor."""
    options = _options(args)
    mode = await stage.start_in_editor(SecurityMode)
    try:
        result = await showcase(mode, out=_out_dir() / "pie", **options)
    finally:
        stage.stop_in_editor()
    return {"where": "play in editor", **result}


async def demo(args: list[str]) -> dict[str, Any]:
    """`unrealpy run --render examples.security_camera.game:demo` (add --packaged for the packaged game)."""
    options = _options(args)
    mode = await stage.start_in_game(SecurityMode)
    return {"where": "game", **await showcase(mode, out=_out_dir() / "game", **options)}


def play(_args: list[str]) -> None:
    """Interactive: the room with SecurityMode; fly in front of the camera with WASD and the mouse."""
    world = ue.get_world()
    assert world is not None
    level = world.get_path_name().split(".", 1)[0]
    ue.GameplayStatics.open_level(
        level, True, f"game={stage.class_path(SecurityMode)}", world_context_object=world
    )
