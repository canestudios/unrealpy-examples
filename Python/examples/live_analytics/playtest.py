"""The playtest session, without the engine: autopiloted testers run the level from the start room to the
exit, each run taking the bridge or the detour, and the hazards kill some of them. The Unreal side (game.py)
moves a mannequin after each run's goal and writes back where it really is; the rules decide the routes,
the deaths, the finishes and when each tester tries again. The unit tests step it alone.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from examples.live_analytics import level

Vec = tuple[float, float]

BRIDGE_SHARE = 0.7  # most testers take the short way, over the bridge
SPEEDS = 420.0, 600.0  # cm/s: a brisk walk to a jog (the mannequin's own animation speeds)
BRIDGE_SPEED = 330.0  # everyone slows down on the narrow bridge
WAVES = ((0.0, 4), (1.2, 3), (2.4, 3))  # (time, testers): ten testers come in three waves
RETRY = 0.8  # seconds before a tester who died or finished starts another run
ARRIVE = 60.0  # a waypoint counts as reached within this distance


@dataclass
class Run:
    id: int
    tester: str
    route: str
    waypoints: list[Vec]
    speed: float
    started: float
    position: Vec
    index: int = 0
    state: str = "running"  # "running", "dead", "finished"

    @property
    def goal(self) -> Vec:
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]


@dataclass
class Event:
    time: float
    kind: str  # "spawn", "death", "finish"
    run: int
    tester: str
    route: str
    position: Vec
    cause: str = ""  # a death's hazard: "blade" or "spikes"
    push: float = 0.0  # the blade's direction along y (-1 or 1) when it hit


@dataclass
class Session:
    seed: int = 0
    clock: float = 0.0
    runs: dict[int, Run] = field(default_factory=dict)  # the runs under way

    def __post_init__(self) -> None:
        self.random = random.Random(self.seed)
        self.pending: list[tuple[float, str]] = []  # (when, tester) runs to start
        number = 0
        for when, count in WAVES:
            for _ in range(count):
                number += 1
                self.pending.append((when, f"P{number}"))
        self.next_id = 0

    def pace(self, run: Run) -> float:
        """How fast a run moves now: its own speed, slower on the bridge."""
        x, y = run.position
        on_bridge = abs(x) < abs(level.ROUTES["bridge"][4][0]) and abs(y) < 120.0
        return min(run.speed, BRIDGE_SPEED) if on_bridge else run.speed

    def blade_angle(self, clock: float | None = None) -> float:
        """The blade's angle from straight down, degrees (positive swings towards +y)."""
        t = self.clock if clock is None else clock
        return level.BLADE_SWING * math.sin(2.0 * math.pi * t / level.BLADE_PERIOD)

    def blade_direction(self) -> float:
        """Which way along y the blade is moving: -1 or 1."""
        return 1.0 if math.cos(2.0 * math.pi * self.clock / level.BLADE_PERIOD) >= 0.0 else -1.0

    def spikes_up(self, clock: float | None = None) -> bool:
        t = self.clock if clock is None else clock
        return t % level.SPIKES_PERIOD < level.SPIKES_UP

    def step(self, dt: float) -> list[Event]:
        """Advance by dt (run positions are whatever the game wrote last); returns the events."""
        self.clock += dt
        events: list[Event] = []
        due = [entry for entry in self.pending if entry[0] <= self.clock]
        self.pending = [entry for entry in self.pending if entry[0] > self.clock]
        for _, tester in due:
            events.append(self._start(tester))
        for run in list(self.runs.values()):
            event = self._advance(run) or self._hazards(run)
            if event is None:
                continue
            events.append(event)
            del self.runs[run.id]
            self.pending.append((self.clock + RETRY, run.tester))
        return events

    def _start(self, tester: str) -> Event:
        route = "bridge" if self.random.random() < BRIDGE_SHARE else "detour"
        start = level.START
        position = (
            self.random.uniform(start.x0 + 100.0, start.x0 + 350.0),
            self.random.uniform(start.y0 + 90.0, start.y1 - 90.0),
        )
        waypoints = []
        for x, y in level.ROUTES[route]:  # a little spread inside the rooms, none on the bridge or in halls
            jitter = 60.0 if level.area_of(x, y) in ("Room A", "Room B") and abs(x) > 500.0 else 0.0
            waypoints.append((x, y + self.random.uniform(-jitter, jitter)))
        run = Run(self.next_id, tester, route, waypoints, self.random.uniform(*SPEEDS), self.clock, position)
        self.next_id += 1
        self.runs[run.id] = run
        return Event(self.clock, "spawn", run.id, tester, route, position)

    def _advance(self, run: Run) -> Event | None:
        if _distance(run.position, run.goal) > ARRIVE:
            return None
        run.index += 1
        if run.index < len(run.waypoints):
            return None
        run.state = "finished"
        return Event(self.clock, "finish", run.id, run.tester, run.route, run.position)

    def _hazards(self, run: Run) -> Event | None:
        x, y = run.position
        on_bridge = abs(y) <= 90.0 and abs(x) <= level.BLADE_REACH
        if on_bridge and abs(self.blade_angle()) < level.BLADE_HIT_ANGLE:
            run.state = "dead"
            push = self.blade_direction()
            return Event(self.clock, "death", run.id, run.tester, run.route, run.position, "blade", push)
        if level.SPIKES.contains(x, y) and self.spikes_up():
            run.state = "dead"
            return Event(self.clock, "death", run.id, run.tester, run.route, run.position, "spikes")
        return None


def _distance(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
