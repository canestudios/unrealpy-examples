"""The enemies' brain: what a drone perceives, what a scripted expert would do, and a small PyTorch MLP that
learns to do the same while the game runs. No Unreal here: the game feeds positions in and gets moves out,
and the unit tests drive it alike.

Learning is DAgger (dataset aggregation, imitation learning): the policy always drives the drones, from the
first second, with untrained weights; the expert only labels each situation the drones actually meet with
the move it would make, and the policy trains on those labels a few times a second. Supervised, so it is
robust and fast; on-policy, so it learns to recover from its own mistakes.
"""

from __future__ import annotations

import math
import random
import threading
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from typing_extensions import override

Vec = tuple[float, float]
Array = npt.NDArray[np.float32]

ARENA = 750.0  # the drones stay in [-ARENA, ARENA]^2 (centimetres)
DRONE_RADIUS = 55.0
BALL_RADIUS = 18.0
HIT_RADIUS = DRONE_RADIUS + BALL_RADIUS
DRONE_SPEED = 430.0  # cm/s
DANGER_TIME = 1.2  # seconds: a ball arriving later than this is not a threat yet
DANGER_MISS = 150.0  # centimetres: a ball passing farther than this is not a threat
KEEP_NEAR, KEEP_FAR = 520.0, 760.0  # the distance band the drones keep from the hero

# The moves, in the drone's own frame: +x points at the hero, +y to its left. Index 0 stays put.
ACTIONS: tuple[Vec, ...] = ((0.0, 0.0),) + tuple(
    (math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 45)
)
FEATURES = 11


@dataclass(frozen=True)
class Frame:
    """The drone's frame: origin at the drone, x towards the hero."""

    origin: Vec
    x_axis: Vec

    @property
    def y_axis(self) -> Vec:
        return (-self.x_axis[1], self.x_axis[0])

    def to_local(self, vector: Vec) -> Vec:
        return (
            vector[0] * self.x_axis[0] + vector[1] * self.x_axis[1],
            vector[0] * self.y_axis[0] + vector[1] * self.y_axis[1],
        )

    def to_world(self, vector: Vec) -> Vec:
        return (
            vector[0] * self.x_axis[0] + vector[1] * self.y_axis[0],
            vector[0] * self.x_axis[1] + vector[1] * self.y_axis[1],
        )


def frame_for(drone: Vec, hero: Vec) -> Frame:
    dx, dy = hero[0] - drone[0], hero[1] - drone[1]
    length = math.hypot(dx, dy)
    return Frame(drone, (1.0, 0.0) if length < 1e-6 else (dx / length, dy / length))


@dataclass(frozen=True)
class Ball:
    position: Vec
    velocity: Vec


def closest_approach(relative: Vec, velocity: Vec) -> tuple[float, Vec]:
    """For a ball at `relative` (from the drone) moving at `velocity`: the time of closest approach (may be
    negative: it is moving away) and the ball's offset from the drone at that time."""
    speed2 = velocity[0] ** 2 + velocity[1] ** 2
    if speed2 < 1e-9:
        return 0.0, relative
    t = -(relative[0] * velocity[0] + relative[1] * velocity[1]) / speed2
    return t, (relative[0] + velocity[0] * t, relative[1] + velocity[1] * t)


def threat(drone: Vec, balls: Sequence[Ball]) -> tuple[Ball, float, Vec] | None:
    """The most urgent ball heading at the drone (soonest arrival among those that would pass close)."""
    best: tuple[Ball, float, Vec] | None = None
    for ball in balls:
        relative = (ball.position[0] - drone[0], ball.position[1] - drone[1])
        t, miss = closest_approach(relative, ball.velocity)
        if 0.0 < t < DANGER_TIME and math.hypot(*miss) < DANGER_MISS and (best is None or t < best[1]):
            best = (ball, t, miss)
    return best


def observe(drone: Vec, hero: Vec, balls: Sequence[Ball]) -> tuple[Array, Frame]:
    """The policy's input (FEATURES floats, roughly in [-1, 1]) and the frame its moves are in:
    distance to the hero; the threatening ball (flag, position, velocity, miss offset, time to closest
    approach); where the drone stands in the arena."""
    frame = frame_for(drone, hero)
    features = np.zeros(FEATURES, dtype=np.float32)
    features[0] = math.hypot(hero[0] - drone[0], hero[1] - drone[1]) / 1000.0
    danger = threat(drone, balls)
    if danger is not None:
        ball, t, miss = danger
        position = frame.to_local((ball.position[0] - drone[0], ball.position[1] - drone[1]))
        velocity = frame.to_local(ball.velocity)
        local_miss = frame.to_local(miss)
        features[1] = 1.0
        features[2:4] = (position[0] / 1000.0, position[1] / 1000.0)
        features[4:6] = (velocity[0] / 1500.0, velocity[1] / 1500.0)
        features[6:8] = (local_miss[0] / DANGER_MISS, local_miss[1] / DANGER_MISS)
        features[8] = t / DANGER_TIME
    place = frame.to_local((drone[0] / ARENA, drone[1] / ARENA))
    features[9:11] = place
    return features, frame


def nearest_action(direction: Vec) -> int:
    """The move (1..8) closest in angle to a direction in the drone's frame; 0 for no direction."""
    if math.hypot(*direction) < 1e-6:
        return 0
    return max(
        range(1, len(ACTIONS)), key=lambda i: ACTIONS[i][0] * direction[0] + ACTIONS[i][1] * direction[1]
    )


def expert(features: Array, strafe: float = 1.0) -> int:
    """The scripted expert, from the same features the policy sees: step out of a ball's way (away from the
    point where it would pass); otherwise keep in the distance band around the hero, strafing inside it;
    and never walk out of the arena."""
    if features[1] > 0.5:
        miss = (float(features[6]), float(features[7]))
        if math.hypot(*miss) < 0.05:  # dead centre: step sideways, across the ball's path
            velocity = (float(features[4]), float(features[5]))
            away = (-velocity[1], velocity[0])
        else:
            away = (-miss[0], -miss[1])
        return nearest_action(_inside(away, features))
    distance = float(features[0]) * 1000.0
    if distance > KEEP_FAR:
        return nearest_action(_inside((1.0, 0.3 * strafe), features))
    if distance < KEEP_NEAR:
        return nearest_action(_inside((-1.0, 0.3 * strafe), features))
    return nearest_action(_inside((0.0, strafe), features))


def _inside(direction: Vec, features: Array) -> Vec:
    """Bend a move back into the arena when the drone is near a wall."""
    place = (float(features[9]), float(features[10]))  # position / ARENA, in the drone's frame
    out = place[0] * direction[0] + place[1] * direction[1]
    if math.hypot(*place) > 0.85 and out > 0.0:
        return (direction[0] - place[0] * 1.5, direction[1] - place[1] * 1.5)
    return direction


def segment_hits(start: Vec, end: Vec, centre: Vec, radius: float) -> bool:
    """Whether a ball moving from start to end in one frame passes within radius of centre (so a fast ball
    never tunnels through a drone between two frames)."""
    sx, sy = end[0] - start[0], end[1] - start[1]
    length2 = sx * sx + sy * sy
    t = (
        0.0
        if length2 < 1e-9
        else max(0.0, min(1.0, ((centre[0] - start[0]) * sx + (centre[1] - start[1]) * sy) / length2))
    )
    px, py = start[0] + sx * t, start[1] + sy * t
    return math.hypot(centre[0] - px, centre[1] - py) <= radius


# -- the policy ----------------------------------------------------------------------------------------------


class Policy(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, len(ACTIONS)),
        )
        # A newborn drone knows nothing: its output layer starts at zero with a nudge towards "stay", so
        # before any training every drone just hovers where it is (and gets hit).
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[0] = 1.0

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(features))


class Learner:
    """DAgger: remember (situation, expert move) pairs, train the policy on them, let the policy act.

    Situations with a ball in the air are rarer than plain chasing but matter most, so they have their own
    memory and make up half of every training batch.

    Two copies of the network: `policy` acts (on the game thread), a twin trains, and after each training
    call the twin's weights are copied into `policy` under a lock. So training can run in a worker thread
    (`BackgroundTrainer`) while the game keeps acting, and an action never sees half-updated weights."""

    def __init__(self, *, seed: int = 0, capacity: int = 20000, batch: int = 128, lr: float = 3e-4) -> None:
        torch.manual_seed(seed)
        self.random = random.Random(seed)
        self.policy = Policy()
        self._twin = Policy()
        self._twin.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self._twin.parameters(), lr=lr)
        self.batch = batch
        self.calm: deque[tuple[Array, int]] = deque(maxlen=capacity)
        self.danger: deque[tuple[Array, int]] = deque(maxlen=capacity)
        self.steps = 0
        self.losses: list[float] = []
        self.agreement: deque[bool] = deque(maxlen=400)  # did the policy pick the expert's move?
        self._lock = threading.Lock()

    @property
    def samples(self) -> int:
        return len(self.calm) + len(self.danger)

    def act(self, features: Array) -> np.ndarray:
        """The policy's moves for a batch of observations (N x FEATURES): the most likely move each."""
        with self._lock, torch.inference_mode():
            logits = self.policy(torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)))
        return logits.argmax(dim=1).numpy()

    def remember(self, features: Array, label: int, chosen: int | None = None) -> None:
        with self._lock:
            (self.danger if features[1] > 0.5 else self.calm).append((features.copy(), int(label)))
            if chosen is not None:
                self.agreement.append(int(chosen) == int(label))

    def train(self, steps: int = 4) -> float | None:
        """A few gradient steps of cross-entropy on remembered labels; the mean loss, or None (no data)."""
        if self.samples < 32:
            return None
        total = 0.0
        for _ in range(steps):
            with self._lock:
                half = self.batch // 2
                picks = self._sample(self.danger, half) + self._sample(self.calm, self.batch - half)
                if len(picks) < self.batch:
                    picks += self._sample(self.danger or self.calm, self.batch - len(picks))
            x = torch.from_numpy(np.stack([f for f, _ in picks]))
            y = torch.tensor([label for _, label in picks], dtype=torch.long)
            loss = nn.functional.cross_entropy(self._twin(x), y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
            self.steps += 1
        with self._lock:
            self.policy.load_state_dict(self._twin.state_dict())
        mean = total / steps
        self.losses.append(mean)
        return mean

    def _sample(self, memory: deque[tuple[Array, int]], count: int) -> list[tuple[Array, int]]:
        if not memory or count <= 0:
            return []
        return [memory[self.random.randrange(len(memory))] for _ in range(count)]


class BackgroundTrainer:
    """Runs Learner.train in a worker thread: the game asks for training (`request`) at its own rate and
    never waits for it. Requests that pile up while a step runs are merged (up to 16 steps per call; the
    rest waits for the next call, nothing is dropped)."""

    def __init__(
        self, learner: Learner, measure: Callable[[str], AbstractContextManager[object]] | None = None
    ) -> None:
        self.learner = learner
        self._measure = measure
        self._pending = 0
        self._busy = False
        self._wake = threading.Condition()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="learning-enemies-trainer", daemon=True)
        self._thread.start()

    def request(self, steps: int) -> None:
        with self._wake:
            self._pending += steps
            self._wake.notify()

    def _run(self) -> None:
        while True:
            with self._wake:
                while self._pending == 0 and not self._stopped:
                    self._wake.wait()
                if self._stopped:
                    return
                steps = min(self._pending, 16)
                self._pending -= steps
                self._busy = True
            try:
                if self._measure is not None:
                    with self._measure("train (worker thread)"):
                        self.learner.train(steps)
                else:
                    self.learner.train(steps)
            finally:
                with self._wake:
                    self._busy = False
                    self._wake.notify_all()

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until every requested step has run (or `timeout` passes): for recordings, where the game
        steps in fixed increments and must not outrun its own training."""
        with self._wake:
            return self._wake.wait_for(
                lambda: self._stopped or (self._pending == 0 and not self._busy), timeout
            )

    def stop(self) -> None:
        with self._wake:
            self._stopped = True
            self._wake.notify()
        self._thread.join(timeout=5.0)


# -- the scoreboard ------------------------------------------------------------------------------------------


class Scoreboard:
    """Episodes (one per ball thrown at a drone) and the learning curve: the share of balls dodged over the
    last `window` episodes."""

    def __init__(self, window: int = 25) -> None:
        self.window = window
        self.outcomes: list[bool] = []  # True: dodged
        self.curve: list[float] = []

    def record(self, dodged: bool) -> None:
        self.outcomes.append(dodged)
        recent = self.outcomes[-self.window :]
        self.curve.append(sum(recent) / len(recent))

    @property
    def episodes(self) -> int:
        return len(self.outcomes)

    @property
    def hits(self) -> int:
        return self.outcomes.count(False)

    def rate(self) -> float:
        return self.curve[-1] if self.curve else 0.0
