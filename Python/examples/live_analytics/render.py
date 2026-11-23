"""The analytics renderer: everything the displays need from one telemetry snapshot, in one call, so it can
run in another process (matplotlib is pure Python for much of its work and would hold the game's GIL for a
quarter of a second per chart if it ran in a thread of the game).
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.live_analytics import charts, telemetry

# The blurred counts at which the heatmaps reach full colour: they start faint and fill in.
WALK_SATURATION, DEATH_SATURATION = 2.0, 0.6
CELLS = (117, 52)  # about 33 cm a cell over the level


@dataclass(frozen=True)
class Pictures:
    dashboard: bytes  # JPEG, the in-world screen
    heat: bytes  # PNG with alpha, the floor overlay
    seconds: float  # time spent rendering


def render(snapshot: telemetry.Snapshot, now: float, bounds: telemetry.Bounds) -> Pictures:
    import time

    started = time.perf_counter()
    dashboard = charts.dashboard(snapshot, now)
    walk = telemetry.traffic(snapshot, bounds, CELLS, saturate=WALK_SATURATION)
    death = telemetry.deaths(snapshot, bounds, CELLS, saturate=DEATH_SATURATION)
    return Pictures(dashboard, charts.heat_picture(walk, death), time.perf_counter() - started)


def captions(texts: list[tuple[str, str, int]]) -> list[bytes]:
    """Caption pictures for (text, colour, width in pixels) triples, 200 pixels high: the world's labels."""
    return [charts.label_picture(text, colour, (width, 200)) for text, colour, width in texts]
