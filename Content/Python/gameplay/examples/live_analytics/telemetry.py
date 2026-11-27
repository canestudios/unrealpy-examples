"""Playtest telemetry, collected while the game runs and summarised with pandas and NumPy.

The game appends rows (cheap: tuples into lists); the summaries turn a snapshot of them into DataFrames and
compute what the in-world displays show: where the testers walk (a heatmap), where they die (another), the
deaths per area of the level and how long the runs that reach the exit take. The summaries run in the
render process, on a snapshot, so the game thread only pays for the appends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from examples.live_analytics import level

Array = npt.NDArray[np.float64]
POSITION_COLUMNS = ["time", "run", "tester", "route", "x", "y"]
EVENT_COLUMNS = ["time", "kind", "run", "tester", "route", "x", "y", "cause"]


@dataclass(frozen=True)
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


LEVEL_BOUNDS = Bounds(*level.BOUNDS)


class Telemetry:
    """Append-only logs: positions (sampled) and events (spawns, deaths, finishes)."""

    def __init__(self) -> None:
        self.positions: list[tuple[float, int, str, str, float, float]] = []
        self.events: list[tuple[float, str, int, str, str, float, float, str]] = []

    def position(self, time: float, run: int, tester: str, route: str, x: float, y: float) -> None:
        self.positions.append((time, run, tester, route, x, y))

    def event(
        self, time: float, kind: str, run: int, tester: str, route: str, x: float, y: float, cause: str = ""
    ) -> None:
        self.events.append((time, kind, run, tester, route, x, y, cause))

    def snapshot(self) -> Snapshot:
        """Copies of the logs, safe to summarise elsewhere while the game keeps appending."""
        return Snapshot(list(self.positions), list(self.events))


@dataclass(frozen=True)
class Snapshot:
    positions: list[tuple[float, int, str, str, float, float]]
    events: list[tuple[float, str, int, str, str, float, float, str]]

    def positions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.positions, columns=POSITION_COLUMNS)

    def events_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events, columns=EVENT_COLUMNS)


def deaths_per_area(snapshot: Snapshot) -> pd.Series:
    """Deaths in each area of the level, in level.AREAS order (zero where nobody died)."""
    frame = snapshot.events_frame()
    frame = frame[frame["kind"] == "death"]
    areas = [level.area_of(x, y) for x, y in zip(frame["x"], frame["y"], strict=True)]
    counts = pd.Series(areas, dtype=object).value_counts()
    return counts.reindex(list(level.AREAS), fill_value=0).astype(int)


def finish_times(snapshot: Snapshot) -> pd.DataFrame:
    """One row per run that reached the exit: its route and how long it took (seconds)."""
    frame = snapshot.events_frame()
    starts = frame[frame["kind"] == "spawn"][["run", "time"]].rename(columns={"time": "started"})
    ends = frame[frame["kind"] == "finish"][["run", "route", "time"]]
    table = ends.merge(starts, on="run", how="inner")
    return pd.DataFrame(
        {"route": table["route"], "seconds": (table["time"] - table["started"]).astype(float)}
    ).reset_index(drop=True)


def totals(snapshot: Snapshot) -> dict[str, int]:
    """Runs started, deaths, runs that reached the exit."""
    kinds = snapshot.events_frame()["kind"].value_counts()
    return {
        "runs": int(kinds.get("spawn", 0)),
        "deaths": int(kinds.get("death", 0)),
        "finished": int(kinds.get("finish", 0)),
    }


def route_share(snapshot: Snapshot) -> dict[str, float]:
    """The share of runs that took each route."""
    frame = snapshot.events_frame()
    routes = frame[frame["kind"] == "spawn"]["route"].value_counts(normalize=True)
    return {route: float(routes.get(route, 0.0)) for route in level.ROUTES}


def heatmap(
    xs: npt.ArrayLike,
    ys: npt.ArrayLike,
    bounds: Bounds,
    cells: tuple[int, int],
    blur: float,
    saturate: float = 0.0,
) -> Array:
    """A 2D histogram of floor points (rows = y, columns = x), smoothed with a separable Gaussian and scaled
    to [0, 1] by its peak, or by `saturate` while the peak is lower (so a young heatmap is faint and
    visibly builds up instead of starting at full brightness)."""
    columns, rows = cells
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if xs.size == 0:
        return np.zeros((rows, columns))
    counts, _, _ = np.histogram2d(
        ys, xs, bins=(rows, columns), range=((bounds.y_min, bounds.y_max), (bounds.x_min, bounds.x_max))
    )
    smooth = gaussian_blur(counts, blur)
    peak = max(float(smooth.max()), saturate)
    return smooth / peak if peak > 0 else smooth


def traffic(snapshot: Snapshot, bounds: Bounds, cells: tuple[int, int], saturate: float = 0.0) -> Array:
    """Where the testers walk: their sampled positions."""
    frame = snapshot.positions_frame()
    return heatmap(frame["x"], frame["y"], bounds, cells, blur=1.0, saturate=saturate)


def deaths(snapshot: Snapshot, bounds: Bounds, cells: tuple[int, int], saturate: float = 0.0) -> Array:
    """Where the testers die."""
    frame = snapshot.events_frame()
    frame = frame[frame["kind"] == "death"]
    return heatmap(frame["x"], frame["y"], bounds, cells, blur=1.6, saturate=saturate)


def gaussian_blur(image: Array, sigma: float) -> Array:
    """A separable Gaussian blur with edge padding (NumPy only)."""
    if sigma <= 0:
        return image.astype(np.float64)
    radius = max(1, int(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(image.astype(np.float64), radius, mode="edge")
    rows = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, rows)
