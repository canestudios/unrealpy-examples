"""The playtest level, without the engine: a small greybox layout of rooms and corridors (floor top at z = 0),
the two routes through it and the areas the analytics count deaths in.

    START -- hall -- ROOM A ==== BRIDGE (over a pit, a swinging blade) ==== ROOM B -- hall -- EXIT
                        |                                                  |
                        +------------ DETOUR (longer, a spike trap) -------+

game.py builds it (platforms over a dark void, low walls with gaps where two pieces meet) and playtest.py
walks the testers along the routes.
"""

from __future__ import annotations

from dataclasses import dataclass

Vec = tuple[float, float]


@dataclass(frozen=True)
class Rect:
    area: str  # the analytics' name for the part of the level it belongs to
    x0: float
    x1: float
    y0: float
    y1: float
    walls: bool = True  # the bridge has none: what the blade hits falls into the pit

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return self.x0 - margin <= x <= self.x1 + margin and self.y0 - margin <= y <= self.y1 + margin

    @property
    def centre(self) -> Vec:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


PIECES = (
    Rect("Start", -1900.0, -1300.0, -350.0, 350.0),
    Rect("Start", -1300.0, -1000.0, -110.0, 110.0),  # the first hall
    Rect("Room A", -1000.0, -400.0, -450.0, 450.0),
    Rect("Bridge", -400.0, 400.0, -70.0, 70.0, walls=False),
    Rect("Detour", -800.0, -620.0, -1000.0, -450.0),
    Rect("Detour", -800.0, 800.0, -1180.0, -1000.0),
    Rect("Detour", 620.0, 800.0, -1000.0, -450.0),
    Rect("Room B", 400.0, 1000.0, -450.0, 450.0),
    Rect("Exit", 1000.0, 1300.0, -110.0, 110.0),  # the last hall
    Rect("Exit", 1300.0, 1900.0, -350.0, 350.0),
)
AREAS = ("Start", "Room A", "Bridge", "Detour", "Room B", "Exit")  # the order of the deaths chart
HAZARD_AREA = "Bridge"
PIT = Rect("Pit", -400.0, 400.0, -450.0, 450.0)  # under the bridge: nothing but the void
BOUNDS = (-1950.0, 1950.0, -1230.0, 500.0)  # x_min, x_max, y_min, y_max: the heatmap's extent

START = PIECES[0]
EXIT_PAD = Rect("Exit", 1600.0, 1800.0, -200.0, 200.0)
# The routes, as waypoints after the start room (the testers pick one per run).
ROUTES: dict[str, tuple[Vec, ...]] = {
    "bridge": ((-1150.0, 0.0), (-950.0, 0.0), (-700.0, 0.0), (-420.0, 0.0), (420.0, 0.0), (700.0, 0.0),
               (950.0, 0.0), (1150.0, 0.0), (1700.0, 0.0)),
    "detour": ((-1150.0, 0.0), (-950.0, 0.0), (-710.0, -350.0), (-710.0, -1090.0), (710.0, -1090.0),
               (710.0, -350.0), (950.0, 0.0), (1150.0, 0.0), (1700.0, 0.0)),
}  # fmt: skip

# The hazards: a pendulum blade across the middle of the bridge, a spike trap in the detour.
BLADE_PIVOT = (0.0, 0.0, 320.0)  # under a low gantry over the bridge
BLADE_LENGTH = 240.0  # pivot to the middle of the blade's head
BLADE_SWING = 70.0  # degrees either side of straight down
BLADE_PERIOD = 2.6  # seconds for a full swing there and back
BLADE_REACH = 90.0  # |x| within which the head hits someone on the bridge
BLADE_HIT_ANGLE = 15.0  # the head is low enough to hit within this many degrees of straight down
SPIKES = Rect("Detour", -120.0, 120.0, -1180.0, -1000.0)
SPIKES_PERIOD, SPIKES_UP = 4.0, 0.25  # the spikes come up for SPIKES_UP seconds every SPIKES_PERIOD


def area_of(x: float, y: float) -> str:
    """The area a floor point belongs to ("" off the level); the pit counts as the bridge's."""
    for piece in PIECES:
        if piece.contains(x, y):
            return piece.area
    return HAZARD_AREA if PIT.contains(x, y) else ""


def walls(thickness: float = 20.0) -> list[tuple[Vec, Vec]]:
    """The wall segments (start, end) around every walled piece, with gaps where another piece touches it
    (doorways, the bridge's ends)."""
    segments: list[tuple[Vec, Vec]] = []
    for piece in PIECES:
        if not piece.walls:
            continue
        sides = (
            ("x", piece.x0, piece.y0, piece.y1),
            ("x", piece.x1, piece.y0, piece.y1),
            ("y", piece.y0, piece.x0, piece.x1),
            ("y", piece.y1, piece.x0, piece.x1),
        )
        for axis, at, low, high in sides:
            gaps = []
            for other in PIECES:
                if other is piece:
                    continue
                if axis == "x" and at in (other.x0, other.x1):
                    gaps.append((other.y0, other.y1))
                elif axis == "y" and at in (other.y0, other.y1):
                    gaps.append((other.x0, other.x1))
            for start, end in _subtract((low, high), gaps):
                if end - start < thickness:
                    continue
                segments.append(((at, start), (at, end)) if axis == "x" else ((start, at), (end, at)))
    return segments


def _subtract(span: tuple[float, float], gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = [span]
    for gap_low, gap_high in gaps:
        cut: list[tuple[float, float]] = []
        for low, high in pieces:
            if gap_high <= low or gap_low >= high:
                cut.append((low, high))
                continue
            if gap_low > low:
                cut.append((low, gap_low))
            if gap_high < high:
                cut.append((gap_high, high))
        pieces = cut
    return pieces
