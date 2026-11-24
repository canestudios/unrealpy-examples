"""Learning enemies example: the parts that need no engine (perception, the scripted expert, the policy and
its DAgger training, the arena's rules, the chart)."""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np
import pytest
import torch

from examples.learning_enemies import arena, brain, chart

# -- perception and the expert -------------------------------------------------------------------------------


def test_frame_round_trip_and_axes() -> None:
    frame = brain.frame_for((100.0, 100.0), (100.0, 400.0))  # the hero is straight up +y
    assert frame.x_axis == pytest.approx((0.0, 1.0)) and frame.y_axis == pytest.approx((-1.0, 0.0))
    assert frame.to_local((0.0, 5.0)) == pytest.approx((5.0, 0.0))
    for vector in ((3.0, -2.0), (-7.5, 0.25)):
        assert frame.to_world(frame.to_local(vector)) == pytest.approx(vector)
    assert brain.frame_for((0.0, 0.0), (0.0, 0.0)).x_axis == (1.0, 0.0)


def test_closest_approach_and_threat() -> None:
    t, miss = brain.closest_approach((-500.0, 40.0), (1000.0, 0.0))
    assert t == pytest.approx(0.5) and miss == pytest.approx((0.0, 40.0))
    t, _ = brain.closest_approach((500.0, 0.0), (1000.0, 0.0))
    assert t < 0  # flying away
    assert brain.closest_approach((10.0, 0.0), (0.0, 0.0)) == (0.0, (10.0, 0.0))
    drone = (0.0, 0.0)
    near = brain.Ball((-400.0, 30.0), (1000.0, 0.0))
    far = brain.Ball((-900.0, 0.0), (1000.0, 0.0))
    wide = brain.Ball((-300.0, 400.0), (1000.0, 0.0))
    found = brain.threat(drone, [far, wide, near])
    assert found is not None and found[0] is near
    assert brain.threat(drone, [wide]) is None
    assert brain.threat(drone, [brain.Ball((-2000.0, 0.0), (1000.0, 0.0))]) is None  # arrives too late


def test_observe_shapes_and_flags() -> None:
    calm, _ = brain.observe((0.0, 0.0), (500.0, 0.0), [])
    assert calm.shape == (brain.FEATURES,) and calm.dtype == np.float32
    assert calm[0] == pytest.approx(0.5) and calm[1] == 0.0 and not calm[2:9].any()
    danger, frame = brain.observe((0.0, 0.0), (500.0, 0.0), [brain.Ball((400.0, 20.0), (-1100.0, 0.0))])
    assert danger[1] == 1.0
    assert danger[2] == pytest.approx(0.4) and danger[4] == pytest.approx(-1100.0 / 1500.0)
    assert danger[7] == pytest.approx(20.0 / brain.DANGER_MISS)  # passes 20 cm to the drone's left
    assert frame.x_axis == (1.0, 0.0)


def test_expert_dodges_away_from_the_ball() -> None:
    features, _ = brain.observe((0.0, 0.0), (500.0, 0.0), [brain.Ball((400.0, 20.0), (-1100.0, 0.0))])
    move = brain.ACTIONS[brain.expert(features)]
    assert move[1] < 0.0  # the ball passes on the left: step right
    centred, _ = brain.observe((0.0, 0.0), (500.0, 0.0), [brain.Ball((400.0, 0.0), (-1100.0, 0.0))])
    sideways = brain.ACTIONS[brain.expert(centred)]
    assert abs(sideways[1]) > 0.5  # dead centre: across the ball's path


def test_expert_keeps_the_distance_band_and_the_arena() -> None:
    far, _ = brain.observe((0.0, 0.0), (1000.0, 0.0), [])
    assert brain.ACTIONS[brain.expert(far)][0] > 0.5  # closes in
    near, _ = brain.observe((0.0, 0.0), (300.0, 0.0), [])
    assert brain.ACTIONS[brain.expert(near)][0] < -0.5  # backs off
    band, _ = brain.observe((0.0, 0.0), (600.0, 0.0), [])
    assert brain.ACTIONS[brain.expert(band, strafe=1.0)] == pytest.approx((0.0, 1.0), abs=1e-9)
    assert brain.ACTIONS[brain.expert(band, strafe=-1.0)] == pytest.approx((0.0, -1.0), abs=1e-9)
    # At the +y wall, backing off from a hero at -y would leave the arena: the move bends back inside.
    wall, frame = brain.observe((0.0, brain.ARENA), (0.0, brain.ARENA - 300.0), [])
    world = frame.to_world(brain.ACTIONS[brain.expert(wall)])
    assert world[1] <= 1e-9


def test_nearest_action_and_segment_hits() -> None:
    assert brain.nearest_action((0.0, 0.0)) == 0
    assert brain.ACTIONS[brain.nearest_action((1.0, 0.1))] == pytest.approx((1.0, 0.0))
    assert brain.ACTIONS[brain.nearest_action((-1.0, -1.0))] == pytest.approx(
        (-math.sqrt(0.5), -math.sqrt(0.5))
    )
    assert brain.segment_hits((-100.0, 50.0), (100.0, 50.0), (0.0, 0.0), 60.0)  # passes through in one step
    assert not brain.segment_hits((-100.0, 80.0), (100.0, 80.0), (0.0, 0.0), 60.0)
    assert brain.segment_hits((0.0, 0.0), (0.0, 0.0), (10.0, 0.0), 20.0)


# -- the policy ----------------------------------------------------------------------------------------------


def test_untrained_policy_hovers() -> None:
    learner = brain.Learner(seed=1)
    features = np.random.default_rng(0).normal(size=(16, brain.FEATURES)).astype(np.float32)
    assert (learner.act(features) == 0).all()


def test_learner_imitates_the_expert() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    learner = brain.Learner(seed=0, lr=3e-3)
    assert learner.train() is None  # nothing to learn from yet
    situations = []
    for _ in range(600):
        drone = (float(rng.uniform(-600, 600)), float(rng.uniform(-600, 600)))
        hero = (float(rng.uniform(-200, 200)), float(rng.uniform(-200, 200)))
        balls = []
        if rng.random() < 0.5:
            origin = (hero[0], hero[1])
            aim = (drone[0] - origin[0] + rng.normal(0, 60), drone[1] - origin[1] + rng.normal(0, 60))
            length = math.hypot(*aim) or 1.0
            balls = [brain.Ball(origin, (aim[0] / length * 1150.0, aim[1] / length * 1150.0))]
        features, _ = brain.observe(drone, hero, balls)
        situations.append(features)
        learner.remember(features, brain.expert(features))
    first = learner.train(1)
    for _ in range(150):
        last = learner.train(4)
    assert first is not None and last is not None and last < first * 0.5
    chosen = learner.act(np.stack(situations))
    labels = np.array([brain.expert(f) for f in situations])
    assert (chosen == labels).mean() > 0.8
    assert learner.steps == 601 and len(learner.losses) == 151


def test_learner_tracks_agreement_and_memories() -> None:
    learner = brain.Learner()
    calm, _ = brain.observe((0.0, 0.0), (600.0, 0.0), [])
    danger, _ = brain.observe((0.0, 0.0), (500.0, 0.0), [brain.Ball((400.0, 20.0), (-1100.0, 0.0))])
    learner.remember(calm, 3, chosen=3)
    learner.remember(danger, 7, chosen=0)
    assert len(learner.calm) == 1 and len(learner.danger) == 1 and learner.samples == 2
    assert list(learner.agreement) == [True, False]


def test_background_trainer_trains_off_thread_and_publishes_weights() -> None:
    learner = brain.Learner(seed=2, lr=3e-3)
    calm, _ = brain.observe((0.0, 0.0), (900.0, 0.0), [])
    for _ in range(200):
        learner.remember(calm, 2)
    names: list[str] = []

    class Probe:
        def __init__(self, name: str) -> None:
            names.append(name)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    trainer = brain.BackgroundTrainer(learner, Probe)
    try:
        for _ in range(60):
            trainer.request(4)
        assert trainer.wait_idle(timeout=60.0)
    finally:
        trainer.stop()
    assert learner.act(calm[None, :])[0] == 2  # the acting copy received the trained weights
    assert learner.steps == 240 and names and set(names) == {"train (worker thread)"}


def test_scoreboard_curve() -> None:
    score = brain.Scoreboard(window=4)
    for dodged in (False, False, True, True, True, True):
        score.record(dodged)
    assert score.episodes == 6 and score.hits == 2
    assert score.curve == pytest.approx([0.0, 0.0, 1 / 3, 0.5, 0.75, 1.0])
    assert score.rate() == 1.0 and brain.Scoreboard().rate() == 0.0


# -- the arena -----------------------------------------------------------------------------------------------


def test_still_drones_always_get_hit() -> None:
    world = arena.Arena(seed=3)

    def hover() -> None:
        for drone in world.drones:
            drone.target = (0.0, 0.0)

    world.decide = hover  # type: ignore[method-assign]
    events = []
    while world.clock < 10.0:
        events += world.step(1.0 / 60.0)
    kinds = {event.kind for event in events}
    assert "throw" in kinds and "hit" in kinds and "dodge" not in kinds
    assert world.score.episodes >= 18 and world.score.hits == world.score.episodes
    assert all(abs(d.position[0]) <= brain.ARENA and abs(d.position[1]) <= brain.ARENA for d in world.drones)


def test_the_drones_learn_to_dodge_within_a_minute() -> None:
    world = arena.Arena(seed=0)
    while world.clock < 8.0:
        world.step(1.0 / 30.0)
    early = world.score.outcomes[:]
    while world.clock < 60.0:
        world.step(1.0 / 30.0)
    late = world.score.outcomes[-20:]
    assert sum(early) / len(early) < 0.3  # untrained: hit most of the time
    assert sum(late) / len(late) > 0.8  # a minute later: dodging most balls
    summary = world.summary()
    assert (
        summary["sgd_steps"] > 0 and summary["samples"] > 1000 and summary["episodes"] == world.score.episodes
    )


def test_steer_is_smooth() -> None:
    drone = arena.Drone((0.0, 0.0), 1.0)
    drone.target = (430.0, 0.0)
    dt = 1.0 / 60.0
    previous = drone.velocity
    for _ in range(30):
        arena.steer(drone, dt, (500.0, 0.0))
        change = math.hypot(drone.velocity[0] - previous[0], drone.velocity[1] - previous[1])
        assert change <= arena.ACCELERATION * dt + 1e-9  # never a jump in velocity
        previous = drone.velocity
    assert drone.velocity == pytest.approx((430.0, 0.0))
    assert -arena.MAX_TILT <= drone.tilt <= 0.0  # it leaned into the acceleration
    drone.target = (0.0, 430.0)  # a hard left turn
    headings = []
    for _ in range(40):
        arena.steer(drone, dt, (500.0, 0.0))
        headings.append(drone.heading)
        assert abs(drone.bank) <= arena.MAX_BANK
    steps = [abs(b - a) for a, b in zip(headings, headings[1:], strict=False)]
    assert max(steps) <= arena.TURN_RATE * dt + 1e-9  # turns at a bounded rate
    assert drone.heading == pytest.approx(math.pi / 2, abs=0.05)
    arena.steer(drone, 0.0, (0.0, 0.0))  # a zero step changes nothing


def test_the_hero_walks_its_circle_at_walking_speed() -> None:
    a, b = arena.Arena.hero_path(0.0), arena.Arena.hero_path(0.1)
    assert math.hypot(*a) == pytest.approx(arena.HERO_LOOP)
    assert math.hypot(b[0] - a[0], b[1] - a[1]) / 0.1 == pytest.approx(arena.HERO_SPEED, rel=0.01)


def test_drones_never_overlap() -> None:
    world = arena.Arena(seed=4)
    for drone in world.drones:
        drone.position = (10.0, 10.0)  # all in one spot
    world._separate()
    for _ in range(5):
        world._separate()
    positions = [d.position for d in world.drones]
    for i, a in enumerate(positions):
        for b in positions[i + 1 :]:
            assert math.hypot(a[0] - b[0], a[1] - b[1]) > 2.0 * brain.DRONE_RADIUS


def test_arena_measures_its_systems() -> None:
    seen: list[str] = []

    class Probe:
        def __init__(self, name: str) -> None:
            seen.append(name)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    world = arena.Arena(seed=1, measure=Probe)
    for _ in range(40):
        world.step(1.0 / 30.0)
    assert "decide (policy + expert)" in seen and "train (4 SGD steps)" in seen


# -- the chart -----------------------------------------------------------------------------------------------


def _png_pixels(data: bytes) -> np.ndarray:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position, idat, size = 8, b"", (0, 0)
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        body = data[position + 8 : position + 8 + length]
        (crc,) = struct.unpack(">I", data[position + 8 + length : position + 12 + length])
        assert crc == zlib.crc32(kind + body)
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            assert (depth, colour) == (8, 2)
            size = (height, width)
        elif kind == b"IDAT":
            idat += body
        position += 12 + length
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(size[0], size[1] * 3 + 1)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].reshape(size[0], size[1], 3)


def test_chart_draws_the_curve_and_encodes_png() -> None:
    picture = chart.learning_chart([0.0, 0.5, 1.0], [2.0, 1.0], width=200, height=100, margin=10)
    assert picture.shape == (100, 200, 3)
    orange = (picture == chart.CURVE).all(axis=2)
    assert orange[88:92, 8:14].any()  # starts at 0 on the left
    assert orange[8:12, 186:192].any()  # ends at 1 on the right
    assert (picture == chart.LOSS).all(axis=2).any()
    decoded = _png_pixels(chart.encode_png(picture))
    assert (decoded == picture).all()
    empty = chart.learning_chart([], [])
    assert (empty == chart.BACKGROUND).all(axis=2).mean() > 0.9
    with pytest.raises(ValueError):
        chart.encode_png(np.zeros((4, 4), np.uint8))


def test_chart_thins_long_runs() -> None:
    assert chart._thin(list(range(10)), 20) == list(range(10))
    thinned = chart._thin(list(range(1000)), 5)
    assert thinned[0] == 0 and thinned[-1] == 999 and len(thinned) == 5
