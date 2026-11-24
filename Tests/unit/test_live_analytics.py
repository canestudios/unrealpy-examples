"""Live playtest analytics example: the parts that need no engine (the level, the playtest rules, the
telemetry summaries with pandas and NumPy, the matplotlib pictures, the renderer)."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from matplotlib import image as mpimg

from examples.live_analytics import charts, level, playtest, render, telemetry

BOUNDS = telemetry.LEVEL_BOUNDS
CELLS = (39, 17)


def _play(seconds: float, seed: int = 1) -> tuple[playtest.Session, telemetry.Telemetry]:
    """The session with ideal movement (each run walks straight at its waypoint), and the telemetry kept."""
    session, log = playtest.Session(seed=seed), telemetry.Telemetry()
    dt, sample = 1.0 / 30.0, 0.0
    while session.clock < seconds:
        for run in session.runs.values():
            dx, dy = run.goal[0] - run.position[0], run.goal[1] - run.position[1]
            distance = math.hypot(dx, dy)
            step = min(distance, session.pace(run) * dt)
            if distance > 1e-9:
                run.position = (
                    run.position[0] + dx / distance * step,
                    run.position[1] + dy / distance * step,
                )
        for event in session.step(dt):
            log.event(
                event.time, event.kind, event.run, event.tester, event.route, *event.position, event.cause
            )
        sample -= dt
        if sample <= 0.0:
            sample += 0.2
            for run in session.runs.values():
                log.position(session.clock, run.id, run.tester, run.route, *run.position)
    return session, log


# -- the level -----------------------------------------------------------------------------------------------


def test_routes_stay_on_the_level_and_end_at_the_exit() -> None:
    for route, waypoints in level.ROUTES.items():
        assert level.EXIT_PAD.contains(*waypoints[-1]), route
        for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:], strict=False):
            for u in np.linspace(0.0, 1.0, 25):  # every leg runs over floor, never over the void
                assert level.area_of(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u) != "", (route, x0, y0)

    def areas(route: str) -> set[str]:
        points = level.ROUTES[route]
        return {
            level.area_of(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)
            for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False)
            for u in np.linspace(0.0, 1.0, 25)
        }

    assert level.HAZARD_AREA in areas("bridge") and level.HAZARD_AREA not in areas("detour")
    assert "Detour" in areas("detour") and "Detour" not in areas("bridge")


def test_walls_leave_doorways_and_the_bridge_open() -> None:
    walls = level.walls()
    assert walls

    def blocked(x: float, y: float) -> bool:
        return any(
            min(a[0], b[0]) - 1 <= x <= max(a[0], b[0]) + 1
            and min(a[1], b[1]) - 1 <= y <= max(a[1], b[1]) + 1
            for a, b in walls
        )

    assert not blocked(-1000.0, 0.0) and not blocked(-400.0, 0.0) and not blocked(400.0, 0.0)  # doorways
    assert not blocked(-710.0, -450.0) and not blocked(0.0, 70.0)  # into the detour; the bridge's edge
    assert blocked(-400.0, 300.0) and blocked(-1900.0, 0.0)  # Room A's pit side; the start room's back


# -- the playtest rules --------------------------------------------------------------------------------------


def test_playtesters_come_in_waves_and_try_again() -> None:
    session, log = _play(20.0)
    spawns = [event for event in log.events if event[1] == "spawn"]
    assert [event[0] for event in spawns[:4]] == pytest.approx([1.0 / 30.0] * 4, abs=1e-9)
    assert len({event[3] for event in spawns}) == 10 and len(spawns) > 10  # ten testers, more runs
    kinds = [event[1] for event in log.events]
    assert kinds.count("death") > 0 and kinds.count("finish") > 0
    routes = [event[4] for event in spawns]
    assert routes.count("bridge") > routes.count("detour") > 0
    assert len(session.runs) <= 10


def test_the_blade_kills_on_the_bridge_and_the_spikes_in_the_detour() -> None:
    session = playtest.Session(seed=0)
    session.step(0.01)
    run = next(iter(session.runs.values()))
    run.position = (0.0, 0.0)
    session.clock = 0.0  # the blade hangs straight down at t = 0
    event = session._hazards(run)
    assert event is not None and event.cause == "blade" and event.push in (-1.0, 1.0)
    session.clock = level.BLADE_PERIOD / 4.0  # swung all the way out
    assert session._hazards(run) is None
    run.position = level.SPIKES.centre
    session.clock = level.SPIKES_PERIOD + 0.05
    event = session._hazards(run)
    assert event is not None and event.cause == "spikes"
    session.clock = level.SPIKES_PERIOD / 2.0
    assert session._hazards(run) is None


def test_the_bridge_is_where_they_die() -> None:
    _, log = _play(20.0)
    per_area = telemetry.deaths_per_area(log.snapshot())
    assert list(per_area.index) == list(level.AREAS)
    assert per_area[level.HAZARD_AREA] == per_area.max() > 0


# -- telemetry -----------------------------------------------------------------------------------------------


def test_deaths_per_area_finish_times_and_totals() -> None:
    log = telemetry.Telemetry()
    log.event(0.0, "spawn", 0, "P1", "bridge", -1700.0, 0.0)
    log.event(0.0, "spawn", 1, "P2", "detour", -1700.0, 100.0)
    log.event(1.0, "spawn", 2, "P3", "bridge", -1700.0, 0.0)
    log.event(3.0, "death", 0, "P1", "bridge", 10.0, 5.0, "blade")
    log.event(9.5, "finish", 1, "P2", "detour", 1700.0, 0.0)
    log.event(7.0, "finish", 2, "P3", "bridge", 1700.0, 0.0)
    snapshot = log.snapshot()
    per_area = telemetry.deaths_per_area(snapshot)
    assert per_area["Bridge"] == 1 and per_area.sum() == 1
    times = telemetry.finish_times(snapshot).sort_values("seconds")
    assert times["route"].tolist() == ["bridge", "detour"] and times["seconds"].tolist() == [6.0, 9.5]
    assert telemetry.totals(snapshot) == {"runs": 3, "deaths": 1, "finished": 2}
    assert telemetry.route_share(snapshot) == pytest.approx({"bridge": 2 / 3, "detour": 1 / 3})
    empty = telemetry.Telemetry().snapshot()
    assert telemetry.deaths_per_area(empty).sum() == 0 and telemetry.finish_times(empty).empty
    assert telemetry.totals(empty) == {"runs": 0, "deaths": 0, "finished": 0}


def test_heatmaps_put_heat_where_they_walk_and_die() -> None:
    log = telemetry.Telemetry()
    for _ in range(50):
        log.position(0.0, 0, "P1", "bridge", -1800.0, 300.0)  # the start room's back corner
    log.event(1.0, "death", 0, "P1", "bridge", 0.0, 0.0, "blade")
    snapshot = log.snapshot()
    walk = telemetry.traffic(snapshot, BOUNDS, CELLS)
    assert walk.shape == (17, 39) and walk.max() == pytest.approx(1.0)
    row, column = np.unravel_index(int(np.argmax(walk)), walk.shape)
    assert column == 1 and row == 15  # x near the left, y near the top (rows grow with y)
    death = telemetry.deaths(snapshot, BOUNDS, CELLS)
    row, column = np.unravel_index(int(np.argmax(death)), death.shape)
    assert column == 19 and death[15, 1] == pytest.approx(0.0, abs=1e-6)
    young = telemetry.traffic(snapshot, BOUNDS, CELLS, saturate=1000.0)
    assert 0.0 < young.max() < 0.1  # few samples: barely visible
    assert not telemetry.traffic(telemetry.Telemetry().snapshot(), BOUNDS, CELLS).any()


def test_gaussian_blur_keeps_the_total_and_spreads() -> None:
    image = np.zeros((9, 9))
    image[4, 4] = 1.0
    blurred = telemetry.gaussian_blur(image, 1.0)
    assert blurred.sum() == pytest.approx(1.0) and blurred[4, 4] < 1.0 and blurred[4, 5] > 0
    assert (telemetry.gaussian_blur(image, 0.0) == image).all()


# -- pictures ------------------------------------------------------------------------------------------------


def test_dashboard_and_heat_picture() -> None:
    _, log = _play(15.0)
    snapshot = log.snapshot()
    assert charts.dashboard(snapshot, 15.0, (640, 360))[:3] == b"\xff\xd8\xff"
    walk = telemetry.traffic(snapshot, BOUNDS, CELLS)
    death = telemetry.deaths(snapshot, BOUNDS, CELLS)
    png = charts.heat_picture(walk, death)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pixels = mpimg.imread(io.BytesIO(png))
    assert pixels.shape == (*walk.shape, 4)
    alpha = pixels[..., 3]
    assert alpha.max() > 0.5 and alpha.min() == 0.0  # clear where nobody went
    hottest = np.unravel_index(int(np.argmax(death)), death.shape)
    red, green, blue, _ = pixels[hottest]
    assert red > 0.8 and red > green + 0.4 and red > blue + 0.4  # deaths are red
    assert charts.dashboard(telemetry.Telemetry().snapshot(), 0.0, (320, 180))[:3] == b"\xff\xd8\xff"


def test_render_bundles_everything() -> None:
    _, log = _play(10.0)
    pictures = render.render(log.snapshot(), 10.0, BOUNDS)
    assert pictures.dashboard[:3] == b"\xff\xd8\xff" and pictures.heat[:4] == b"\x89PNG"
    assert pictures.seconds > 0


def test_captions_are_transparent_pngs() -> None:
    (png,) = render.captions([("Where they die", "#ff5a4f", 800)])
    pixels = mpimg.imread(io.BytesIO(png))
    assert pixels.shape == (200, 800, 4) and pixels[0, 0, 3] == 0.0 and pixels[100, 400, 3] > 0.5


def test_camera_path_eases_between_keys() -> None:
    from examples import stage

    path = stage.CameraPath([(0.0, (0, 0, 0), (1, 0, 0), 50.0), (2.0, (10, 0, 0), (1, 1, 0), 40.0)])
    assert path.at(-1.0) == ((0, 0, 0), (1, 0, 0), 50.0)
    location, target, fov = path.at(1.0)
    assert location == pytest.approx((5.0, 0.0, 0.0)) and fov == pytest.approx(45.0)
    early, _, _ = path.at(0.2)
    assert early[0] < 1.0  # smoothstep: slow start
    assert path.at(5.0) == ((10, 0, 0), (1, 1, 0), 40.0)
