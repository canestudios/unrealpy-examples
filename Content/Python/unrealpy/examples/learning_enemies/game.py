"""Enemies that learn while you play: PyTorch training online, inside the game, in gameplay code written with
UnrealPy.

Five drones circle the hero, who throws a ball at one of them every 0.45 s. A small MLP (11 inputs, two
hidden layers of 64, 9 moves) drives all the drones from the first second; it starts knowing nothing, so
the drones just hover and get hit. Ten times a second a scripted expert labels the situation each drone is
in with the move it would make, and four times a second the policy takes a few gradient steps on those
labels (DAgger: imitation learning on the policy's own experience). In well under a minute the drones
start stepping out of the balls' way. A board behind the arena plots the dodge rate per episode (one
episode per ball), with the episode count and the training loss.

    uv run python -m examples.learning_enemies.demo       # Play In Editor, clip + screenshot
    unrealpy run --render examples.learning_enemies.game:demo    # standalone game
    unrealpy run --packaged examples.learning_enemies.game:demo  # packaged build (--group learning)

The hero is an autopilot. Only /Engine content is used; the scene is built when the game mode starts.
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ue
from examples import showreel, stage
from examples.characters import WALK_SPEED, DemoMannequin
from ue import agent

if TYPE_CHECKING:
    from examples.learning_enemies import arena as rules
    from examples.learning_enemies import brain

# ClassPackages discovers the example classes at startup, but
# torch (seconds to import) waits until a LearningMode actually starts.

ARENA = 750.0  # brain.ARENA
WALL = ARENA + 110.0
FLY_HEIGHT = 75.0
BOARD_AT = (WALL + 260.0, 0.0, 330.0)
BOARD_SIZE = (900.0, 420.0)
FILM_AT = (-1850.0, -950.0, 1150.0)
FILM_LOOKS_AT = (320.0, 40.0, 260.0)
BOARD_HZ = 2.0
DRONE_COLOUR = (0.75, 0.04, 0.03)
HIT_GLOW = (1.0, 0.9, 0.8)

PERF = showreel.PerfMeter()
MOVABLE = ue.EComponentMobility.Movable


@ue.uclass
class LearningDrone(ue.Actor):
    root = ue.component(ue.SceneComponent, mobility=MOVABLE)
    body = ue.component(
        ue.StaticMeshComponent,
        static_mesh=stage.shape("Sphere"),
        mobility=MOVABLE,
        relative_scale3d=ue.Vector(0.95, 0.95, 0.75),
    )
    fin = ue.component(
        ue.StaticMeshComponent,
        static_mesh=stage.shape("Cone"),
        mobility=MOVABLE,
        relative_location=ue.Vector(0.0, 0.0, 45.0),
        relative_scale3d=ue.Vector(0.5, 0.5, 0.45),
    )
    eye = ue.component(
        ue.StaticMeshComponent,
        static_mesh=stage.shape("Sphere"),
        mobility=MOVABLE,
        relative_location=ue.Vector(40.0, 0.0, 8.0),
        relative_scale3d=ue.Vector(0.25, 0.25, 0.25),
    )

    def begin_play(self) -> None:
        for part in (self.body, self.fin, self.eye):
            part.set_collision_enabled(ue.ECollisionEnabled.NoCollision)
        self.skin = stage.surface(self.body, DRONE_COLOUR, roughness=0.22, metallic=0.4)
        stage.surface(self.fin, (0.04, 0.04, 0.05), roughness=0.3, metallic=1.0)
        stage.glowing(self.eye, (1.0, 0.75, 0.1), 30.0)
        self.flashing = False

    def show(self, state: rules.Drone, hero: brain.Vec) -> None:
        """Place it where the rules say, facing its heading, banked and tilted (smooth: see arena.steer)."""
        x, y = state.position
        self.set_actor_location(ue.Vector(x, y, FLY_HEIGHT), False, True)
        yaw = math.degrees(state.heading)
        self.set_actor_rotation(ue.Rotator(state.tilt, yaw, state.bank), True)
        flashing = state.flash > 0.0
        if flashing != self.flashing:
            self.flashing = flashing
            stage.recolour(self.skin, DRONE_COLOUR, HIT_GLOW if flashing else None, 25.0)


@ue.uclass
class LearningHero(DemoMannequin):
    """The thrower: a mannequin walking its circle round the middle (at its walk animation's own speed)."""


@ue.uclass
class LearningMode(ue.GameModeBase):
    """Builds the arena, steps the rules, mirrors them onto actors, keeps the board up to date."""

    default_pawn_class = ue.SpectatorPawn  # the hero is an autopilot; the player watches

    def begin_play(self) -> None:
        import torch

        from examples.learning_enemies import arena as rules
        from examples.learning_enemies import chart

        torch.set_num_threads(2)  # leave the game's own threads alone
        self.chart = chart
        self.world_rules = rules.Arena(measure=PERF.measure, background=True, hero_external=True)
        self.rules = rules
        built = build_arena(len(self.world_rules.drones))
        self.drones: list[LearningDrone] = built["drones"]
        self.hero: LearningHero = built["hero"]
        self.hero.dress(tint=(0.86, 0.88, 0.92))  # silver, like the drones' nemesis should be
        self.balls: list[ue.StaticMeshActor] = built["balls"]
        self.board: stage.Screen = built["board"]
        self.title: ue.TextRenderActor = built["title"]
        self.stats: ue.TextRenderActor = built["stats"]
        self.speed_note = ""
        self.board_timer = 0.0
        self.painter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="learning-board")
        self.picture: Future[bytes] | None = None
        ue.log("LearningMode: arena built, drones untrained")

    def end_play(self, end_play_reason: Any) -> None:
        self.world_rules.stop()
        self.painter.shutdown(wait=False, cancel_futures=True)

    def tick(self, delta_seconds: float) -> None:
        PERF.frame()
        with PERF.measure("rules + learning (python)"):
            state = self.world_rules
            # The hero is a character: it walks after a point running ahead on its circle, and the rules
            # throw from where it really is.
            self.hero.steer_to(self.rules.Arena.hero_path(state.clock + 0.5), WALK_SPEED, arrive=5.0)
            here = self.hero.get_actor_location()
            state.hero = (float(here[0]), float(here[1]))
            state.step(delta_seconds)
        with PERF.measure("actors"):
            self._mirror()
        self.board_timer -= delta_seconds
        if self.board_timer <= 0.0 and self.picture is None:
            self.board_timer = 1.0 / BOARD_HZ
            with PERF.measure("board (text)"):
                self.request_board()
        elif self.picture is not None and self.picture.done():
            with PERF.measure("board (texture)"):
                self.board.show(self.picture.result())
            self.picture = None

    def _mirror(self) -> None:
        state = self.world_rules
        for actor, drone in zip(self.drones, state.drones, strict=True):
            actor.show(drone, state.hero)
        for index, ball in enumerate(self.balls):
            if index < len(state.flights):
                x, y = state.flights[index].position
                ball.set_actor_location(ue.Vector(x, y, FLY_HEIGHT), False, True)
                ball.set_actor_hidden_in_game(False)
            else:
                ball.set_actor_hidden_in_game(True)

    def request_board(self) -> None:
        """Update the text now; draw and encode the chart in a worker thread (the next ticks collect it)."""
        state = self.world_rules
        learner = state.learner
        assert learner is not None
        curve, losses = list(state.score.curve), list(learner.losses)
        self.picture = self.painter.submit(self._paint, curve, losses)
        summary = state.summary()
        loss = "-" if summary["loss"] is None else f"{summary['loss']:.3f}"
        self.stats.text_render.set_text(
            f"EPISODE {summary['episodes']}   DODGED {round(100 * float(summary['dodge_rate']))}%   "
            f"HITS {summary['hits']}   LOSS {loss}{self.speed_note}"
        )

    def _paint(self, curve: list[float], losses: list[float]) -> bytes:
        with PERF.measure("chart (worker thread)"):
            return self.chart.encode_png(self.chart.learning_chart(curve, losses))

    def status(self) -> dict[str, Any]:
        return self.world_rules.summary()


def text(
    location: tuple[float, float, float], size: float, colour: tuple[int, int, int], words: str
) -> ue.TextRenderActor:
    actor = ue.spawn_actor(ue.TextRenderActor, ue.Vector(*location), ue.Rotator(0.0, 180.0, 0.0))
    component = actor.text_render
    component.set_horizontal_alignment(ue.EHorizTextAligment.EHTA_Center)
    component.set_world_size(size)
    component.set_text_render_color(ue.Color(r=colour[0], g=colour[1], b=colour[2], a=255))
    component.set_text(words)
    return actor


def build_arena(drones: int) -> dict[str, Any]:
    # Dusk in a training hall: a low sun behind the board, a deep blue sky, floodlights on the arena.
    stage.lighting(
        sun_pitch=-14.0,
        sun_yaw=170.0,
        sun=3.0,
        sky=1.0,
        fog=0.012,
        sun_colour=(1.0, 0.6, 0.35),
        sky_colour=(0.45, 0.6, 1.0),
        fog_colour=(0.06, 0.09, 0.18),
    )
    stage.look(exposure_bias=-0.8, bloom=1.0)
    stage.prop("Cube", (0.0, 0.0, -14.0), (80.0, 80.0, 0.2), (0.012, 0.013, 0.016), roughness=0.95)
    floor = stage.prop("Cube", (0.0, 0.0, -10.0), (WALL / 50.0, WALL / 50.0, 0.2))
    stage.grid(floor.static_mesh_component, (0.04, 0.045, 0.06), (0.08, 0.1, 0.13), size=100.0, roughness=0.8)
    # The arena: a dark disc over a slightly larger glowing one, so only a cyan ring of light shows.
    stage.prop(
        "Cylinder",
        (0.0, 0.0, 1.0),
        (ARENA / 50.0 + 0.5, ARENA / 50.0 + 0.5, 0.04),
        (0.0, 0.3, 0.5),
        glow=(0.05, 0.6, 1.0),
        strength=15.0,
        collision=False,
        shadow=False,
    )
    stage.prop(
        "Cylinder",
        (0.0, 0.0, 3.5),
        (ARENA / 50.0, ARENA / 50.0, 0.04),
        (0.07, 0.08, 0.1),
        roughness=0.3,
        metallic=0.6,
        collision=False,
    )
    rim = (0.2, 0.21, 0.24)
    for sign in (-1.0, 1.0):
        stage.prop(
            "Cube", (sign * WALL, 0.0, 30.0), (0.3, WALL / 50.0, 0.6), rim, roughness=0.35, metallic=1.0
        )
        stage.prop(
            "Cube", (0.0, sign * WALL, 30.0), (WALL / 50.0, 0.3, 0.6), rim, roughness=0.35, metallic=1.0
        )
    # Floodlights on four masts, aimed at the middle.
    for x, y in ((-WALL, -WALL), (-WALL, WALL), (WALL, -WALL), (WALL, WALL)):
        stage.prop(
            "Cylinder", (x, y, 400.0), (0.25, 0.25, 8.0), (0.05, 0.05, 0.06), roughness=0.35, metallic=1.0
        )
        stage.prop(
            "Cube",
            (x, y, 810.0),
            (0.6, 0.6, 0.25),
            (0.9, 0.9, 0.85),
            glow=(1.0, 0.92, 0.8),
            strength=20.0,
            collision=False,
            shadow=False,
        )
        stage.accent(
            (x, y, 780.0), (x * 0.15, y * 0.15, 0.0), (1.0, 0.9, 0.78), 3500.0, cone=32.0, radius=3500.0
        )
    hero = ue.spawn_actor(LearningHero, ue.Vector(230.0, 0.0, 100.0))
    actors = [ue.spawn_actor(LearningDrone, ue.Vector(0.0, 0.0, FLY_HEIGHT)) for _ in range(drones)]
    balls = [
        stage.prop(
            "Sphere",
            (0.0, 0.0, -500.0),
            (0.3, 0.3, 0.3),
            (1.0, 0.5, 0.05),
            glow=(1.0, 0.45, 0.05),
            strength=30.0,
            collision=False,
            shadow=False,
        )
        for _ in range(16)
    ]
    for ball in balls:
        ball.set_actor_hidden_in_game(True)
    bx, by, bz = BOARD_AT
    width, height = BOARD_SIZE
    stage.prop(
        "Cube",
        (bx + 14.0, by, bz),
        (0.2, width / 100.0 + 0.4, height / 100.0 + 0.4),
        (0.03, 0.03, 0.035),
        roughness=0.25,
        metallic=1.0,
    )
    for side in (-1.0, 1.0):
        stage.prop(
            "Cube",
            (bx + 16.0, by + side * width * 0.4, (bz - height / 2.0) / 2.0),
            (0.18, 0.18, (bz - height / 2.0) / 100.0),
            (0.03, 0.03, 0.035),
            roughness=0.25,
            metallic=1.0,
        )
    # A lamp in front of the board for its text (the sun lights the arena from behind the board).
    lamp = ue.spawn_actor(ue.PointLight, ue.Vector(bx - 420.0, by, bz + 260.0))
    lamp.point_light_component.set_mobility(MOVABLE)
    lamp.point_light_component.set_intensity(25000.0)
    lamp.point_light_component.set_attenuation_radius(1500.0)
    board = stage.Screen(BOARD_AT, 180.0, width, height, brightness=1.6)
    text(
        (bx - 5.0, by, bz - height / 2.0 - 60.0),
        34.0,
        (200, 205, 215),
        "ORANGE: BALLS DODGED (LAST 25)     BLUE: TRAINING LOSS",
    )
    title = text((bx - 5.0, by, bz + height / 2.0 + 105.0), 64.0, (255, 190, 60), "DRONES LEARNING TO DODGE")
    stats = text((bx - 5.0, by, bz + height / 2.0 + 35.0), 42.0, (230, 235, 245), "EPISODE 0")
    return {"drones": actors, "hero": hero, "balls": balls, "board": board, "title": title, "stats": stats}


# -- scripted runs -------------------------------------------------------------------------------------------


async def showcase(
    mode: LearningMode, *, clip: float, speed: float, measure: float, out: Path
) -> dict[str, Any]:
    """Shoot the clip from the very start (untrained drones), `speed` times faster than real time, then
    measure the frame time in real time, then a last still."""
    film = stage.Film(FILM_AT, stage.look_at(FILM_AT, FILM_LOOKS_AT), fov=60.0)
    await film.warm_up()
    recording: dict[str, Any] = {}
    if clip > 0:
        mode.speed_note = f"   x{speed:g}" if speed != 1.0 else ""
        # Each frame waits for the training it asked for: a frame of the clip takes longer than its game
        # time (the 1080p shot), and the drones must learn per game second, as they do in real time.
        recording = await stage.record(
            film, out / "frames", clip, 30, lambda _dt: mode.world_rules.catch_up(), game_speed=speed
        )
        mode.speed_note = ""
    after_clip = mode.status()
    perf = {}
    for cap in (0, 60) if measure > 0 else ():  # optional: uncapped, then t.MaxFPS 60
        stage.max_fps(cap)
        PERF.reset()
        end = time.perf_counter() + measure
        while time.perf_counter() < end:
            await agent.frames(1)
        perf[f"max_fps_{cap}" if cap else "uncapped"] = PERF.summary()
    stage.max_fps(0)
    mode.request_board()
    await agent.frames(5)
    still = await film.shoot(out / f"still-{time.strftime('%Y%m%d-%H%M%S')}.png")
    return {
        "perf": perf,
        "recording": recording,
        "after_clip": after_clip,
        "final": mode.status(),
        "still": str(still),
        "curve": [round(v, 2) for v in mode.world_rules.score.curve[::10]],
    }


def _out_dir() -> Path:
    return Path(ue.saved_dir()).resolve() / "LearningEnemies"


def _options(args: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="learning_enemies")
    parser.add_argument("--clip", type=float, default=15.0, help="seconds of clip (0: none)")
    parser.add_argument("--speed", type=float, default=3.0, help="game seconds per clip second")
    parser.add_argument("--measure", type=float, default=0.0, help="seconds of real-time measurement")
    return vars(parser.parse_args(args))


async def pie(args: list[str]) -> dict[str, Any]:
    """`unrealpy run --editor --render examples.learning_enemies.game:pie`: Play In Editor."""
    options = _options(args)
    mode = await stage.start_in_editor(LearningMode)
    try:
        result = await showcase(mode, out=_out_dir() / "pie", **options)
    finally:
        stage.stop_in_editor()
    return {"where": "play in editor", **result}


async def demo(args: list[str]) -> dict[str, Any]:
    """`unrealpy run --render examples.learning_enemies.game:demo` (add --packaged for the packaged game)."""
    options = _options(args)
    mode = await stage.start_in_game(LearningMode)
    return {"where": "game", **await showcase(mode, out=_out_dir() / "game", **options)}


def play(_args: list[str]) -> None:
    """Interactive: watch the drones learn (fly around with WASD and the mouse)."""
    world = ue.get_world()
    assert world is not None
    level = world.get_path_name().split(".", 1)[0]
    ue.GameplayStatics.open_level(
        level, True, f"game={stage.class_path(LearningMode)}", world_context_object=world
    )
