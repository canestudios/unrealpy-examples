"""The game's rules without the engine: the hero walks a loop and throws balls at the drones, the balls fly
and hit, the drones move as the policy says, and the learner trains on the expert's labels. The Unreal side
(game.py) steps this world every tick and mirrors it onto actors; the unit tests step it alone.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from examples.learning_enemies import brain
from examples.learning_enemies.brain import Vec

THROW_INTERVAL = 0.45  # seconds between throws
BALL_SPEED = 1150.0  # cm/s
BALL_RANGE = 1900.0  # cm a ball flies before it counts as dodged
DECIDE_HZ = 10.0
TRAIN_HZ = 4.0
TRAIN_STEPS = 4
KNOCKBACK = 140.0  # cm
# Smooth motion: the policy picks a target velocity ten times a second; every step the drone's velocity
# moves towards it at a bounded acceleration, its heading turns at a bounded rate, and it banks into turns
# and tilts into accelerations, as a hovering drone would.
ACCELERATION = 2600.0  # cm/s^2
TURN_RATE = math.radians(420.0)  # rad/s
MAX_BANK = 28.0  # degrees
MAX_TILT = 14.0  # degrees
GRAVITY = 981.0  # cm/s^2
HERO_SPEED = 300.0  # cm/s: the mannequin's walk (its animation's own speed)
HERO_LOOP = 230.0  # cm: radius of the hero's walk around the middle


@dataclass
class Drone:
    position: Vec
    strafe: float  # +1 or -1: which way it circles the hero
    target: Vec = (0.0, 0.0)  # the velocity the brain asked for
    velocity: Vec = (0.0, 0.0)  # the velocity it has (moving towards `target`)
    moved: Vec = (0.0, 0.0)  # what it actually did last step (a wall stops it)
    heading: float = 0.0  # radians, the way it faces
    bank: float = 0.0  # degrees, rolled into a turn
    tilt: float = 0.0  # degrees, pitched into an acceleration
    action: int = 0
    flash: float = 0.0  # seconds left of the "hit" flash
    hits: int = 0


@dataclass
class Flight:
    position: Vec
    velocity: Vec
    target: int
    travelled: float = 0.0


@dataclass
class Event:
    kind: str  # "throw", "hit", "dodge"
    drone: int
    position: Vec


@dataclass
class Arena:
    drones_count: int = 5
    seed: int = 0
    learner: brain.Learner | None = None
    measure: Callable[[str], AbstractContextManager[object]] = field(default=lambda _name: nullcontext())
    background: bool = False  # train in a worker thread (the game) or inline (the tests)
    hero: Vec = (0.0, 0.0)
    clock: float = 0.0
    hero_external: bool = False  # the game moves the hero (a character) and writes `hero` itself

    def __post_init__(self) -> None:
        self.random = random.Random(self.seed)
        if self.learner is None:
            self.learner = brain.Learner(seed=self.seed)
        self.trainer = brain.BackgroundTrainer(self.learner, self.measure) if self.background else None
        self.score = brain.Scoreboard()
        self.drones: list[Drone] = []
        for index in range(self.drones_count):
            angle = 2.0 * math.pi * index / self.drones_count
            self.drones.append(
                Drone((600.0 * math.cos(angle), 600.0 * math.sin(angle)), 1.0 if index % 2 else -1.0)
            )
        self.flights: list[Flight] = []
        self._throw_timer = 0.5
        self._decide_timer = 0.0
        self._train_timer = 0.0
        self._next_target = 0
        self.decide_seconds = 0.0
        self.train_seconds = 0.0

    # -- one step ------------------------------------------------------------------------------------------

    def step(self, dt: float) -> list[Event]:
        self.clock += dt
        events: list[Event] = []
        if not self.hero_external:
            self.hero = self.hero_path(self.clock)
        self._throw_timer -= dt
        if self._throw_timer <= 0.0:
            self._throw_timer += THROW_INTERVAL
            events.append(self._throw())
        self._decide_timer -= dt
        if self._decide_timer <= 0.0:
            self._decide_timer += 1.0 / DECIDE_HZ
            with self.measure("decide (policy + expert)"):
                started = time.perf_counter()
                self.decide()
                self.decide_seconds += time.perf_counter() - started
        self._train_timer -= dt
        if self._train_timer <= 0.0:
            self._train_timer += 1.0 / TRAIN_HZ
            if self.trainer is not None:
                self.trainer.request(TRAIN_STEPS)
            else:
                with self.measure("train (4 SGD steps)"):
                    started = time.perf_counter()
                    assert self.learner is not None
                    self.learner.train(TRAIN_STEPS)
                    self.train_seconds += time.perf_counter() - started
        self._move_drones(dt)
        events.extend(self._fly(dt))
        return events

    @staticmethod
    def hero_path(t: float) -> Vec:
        """Where the hero walks at time t: a circle round the middle at walking speed."""
        angle = t * HERO_SPEED / HERO_LOOP
        return (HERO_LOOP * math.cos(angle), HERO_LOOP * math.sin(angle))

    def _throw(self) -> Event:
        index = self._next_target
        self._next_target = (self._next_target + 1) % len(self.drones)
        drone = self.drones[index]
        # Lead the target: aim where it will be if it keeps its velocity (so drifting is not dodging).
        dx, dy = drone.position[0] - self.hero[0], drone.position[1] - self.hero[1]
        flight_time = math.hypot(dx, dy) / BALL_SPEED
        aim = (
            drone.position[0] + drone.moved[0] * flight_time,
            drone.position[1] + drone.moved[1] * flight_time,
        )
        ax, ay = aim[0] - self.hero[0], aim[1] - self.hero[1]
        length = math.hypot(ax, ay) or 1.0
        spread = self.random.uniform(-0.04, 0.04)
        cos_s, sin_s = math.cos(spread), math.sin(spread)
        direction = (ax / length * cos_s - ay / length * sin_s, ax / length * sin_s + ay / length * cos_s)
        self.flights.append(Flight(self.hero, (direction[0] * BALL_SPEED, direction[1] * BALL_SPEED), index))
        return Event("throw", index, self.hero)

    def balls(self) -> list[brain.Ball]:
        return [brain.Ball(f.position, f.velocity) for f in self.flights]

    def decide(self) -> None:
        """Every drone: observe, let the policy pick its move, ask the expert for the right one, remember."""
        assert self.learner is not None
        balls = self.balls()
        observations = [brain.observe(d.position, self.hero, balls) for d in self.drones]
        features = np.stack([f for f, _ in observations])
        chosen = self.learner.act(features)
        for drone, (feature, frame), action in zip(self.drones, observations, chosen, strict=True):
            label = brain.expert(feature, drone.strafe)
            self.learner.remember(feature, label, int(action))
            drone.action = int(action)
            local = brain.ACTIONS[drone.action]
            world = frame.to_world(local)
            drone.target = (world[0] * brain.DRONE_SPEED, world[1] * brain.DRONE_SPEED)

    def _move_drones(self, dt: float) -> None:
        limit = brain.ARENA
        for drone in self.drones:
            drone.flash = max(0.0, drone.flash - dt)
            steer(drone, dt, self.hero)
            x = min(max(drone.position[0] + drone.velocity[0] * dt, -limit), limit)
            y = min(max(drone.position[1] + drone.velocity[1] * dt, -limit), limit)
            if dt > 0.0:  # what it really did: a drone pushing into a wall stands still
                drone.moved = ((x - drone.position[0]) / dt, (y - drone.position[1]) / dt)
            drone.position = (x, y)
        self._separate()

    def _separate(self, gap: float = 2.0 * brain.DRONE_RADIUS + 30.0) -> None:
        """Drones are solid: two closer than `gap` push each other apart (half the overlap each)."""
        limit = brain.ARENA
        for i, first in enumerate(self.drones):
            for second in self.drones[i + 1 :]:
                dx, dy = second.position[0] - first.position[0], second.position[1] - first.position[1]
                distance = math.hypot(dx, dy)
                if distance >= gap:
                    continue
                if distance < 1e-6:
                    dx, dy, distance = 1.0, 0.0, 1.0
                push = (gap - distance) / 2.0
                ux, uy = dx / distance * push, dy / distance * push
                first.position = (
                    min(max(first.position[0] - ux, -limit), limit),
                    min(max(first.position[1] - uy, -limit), limit),
                )
                second.position = (
                    min(max(second.position[0] + ux, -limit), limit),
                    min(max(second.position[1] + uy, -limit), limit),
                )

    def _fly(self, dt: float) -> list[Event]:
        events: list[Event] = []
        remaining: list[Flight] = []
        for flight in self.flights:
            start = flight.position
            end = (start[0] + flight.velocity[0] * dt, start[1] + flight.velocity[1] * dt)
            struck = next(
                (
                    i
                    for i, d in enumerate(self.drones)
                    if brain.segment_hits(start, end, d.position, brain.HIT_RADIUS)
                ),
                None,
            )
            if struck is not None:
                drone = self.drones[struck]
                drone.hits += 1
                drone.flash = 0.35
                speed = math.hypot(*flight.velocity) or 1.0
                push = (flight.velocity[0] / speed * KNOCKBACK, flight.velocity[1] / speed * KNOCKBACK)
                limit = brain.ARENA
                drone.position = (
                    min(max(drone.position[0] + push[0], -limit), limit),
                    min(max(drone.position[1] + push[1], -limit), limit),
                )
                self.score.record(False)
                events.append(Event("hit", struck, end))
                continue
            flight.position = end
            flight.travelled += BALL_SPEED * dt
            if flight.travelled >= BALL_RANGE:
                self.score.record(True)
                events.append(Event("dodge", flight.target, end))
                continue
            remaining.append(flight)
        self.flights = remaining
        return events

    def stop(self) -> None:
        if self.trainer is not None:
            self.trainer.stop()

    def catch_up(self) -> None:
        """Wait for the background training to finish what was asked so far (recordings only)."""
        if self.trainer is not None:
            self.trainer.wait_idle()

    def summary(self) -> dict[str, Any]:
        assert self.learner is not None
        agreement = self.learner.agreement
        return {
            "game_seconds": round(self.clock, 1),
            "episodes": self.score.episodes,
            "hits": self.score.hits,
            "dodge_rate": round(self.score.rate(), 3),
            "samples": self.learner.samples,
            "sgd_steps": self.learner.steps,
            "loss": round(self.learner.losses[-1], 4) if self.learner.losses else None,
            "agrees_with_expert": round(sum(agreement) / len(agreement), 3) if agreement else 0.0,
        }


def steer(drone: Drone, dt: float, hero: Vec) -> None:
    """One step of smooth motion: velocity towards the target at a bounded acceleration; heading towards the
    direction of travel (or the hero, when hovering) at a bounded turn rate; bank and tilt from the lateral
    and forward accelerations, eased so they never snap."""
    if dt <= 0.0:
        return
    vx, vy = drone.velocity
    dvx, dvy = drone.target[0] - vx, drone.target[1] - vy
    change = math.hypot(dvx, dvy)
    limit = ACCELERATION * dt
    if change > limit:
        dvx, dvy = dvx / change * limit, dvy / change * limit
    drone.velocity = (vx + dvx, vy + dvy)
    speed = math.hypot(*drone.velocity)
    if speed > 60.0:
        wanted = math.atan2(drone.velocity[1], drone.velocity[0])
    else:
        wanted = math.atan2(hero[1] - drone.position[1], hero[0] - drone.position[0])
    turn = (wanted - drone.heading + math.pi) % (2.0 * math.pi) - math.pi
    turn = max(-TURN_RATE * dt, min(TURN_RATE * dt, turn))
    drone.heading = (drone.heading + turn + math.pi) % (2.0 * math.pi) - math.pi
    # Accelerations in the drone's frame: forward (tilt) and to its left (bank).
    ax, ay = dvx / dt, dvy / dt
    forward = (math.cos(drone.heading), math.sin(drone.heading))
    along = ax * forward[0] + ay * forward[1]
    across = -ax * forward[1] + ay * forward[0] + speed * turn / dt
    bank = max(-MAX_BANK, min(MAX_BANK, math.degrees(math.atan2(across, GRAVITY))))
    tilt = max(-MAX_TILT, min(MAX_TILT, -math.degrees(math.atan2(along, GRAVITY))))
    ease = min(1.0, dt * 8.0)
    drone.bank += (bank - drone.bank) * ease
    drone.tilt += (tilt - drone.tilt) * ease
