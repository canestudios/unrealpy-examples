"""The pictures the playtest displays show, drawn with matplotlib (Agg, no window) in the render process:

- the dashboard on the in-world screen: deaths per area of the level and the time the runs that reach the
  exit take, under a line of totals;
- captions for the world (the labels over the heatmap, the hazard and the screen);
- the floor overlay: where the testers walk (cyan) with where they die over it (red), transparent where
  nobody went.

Each returns image file bytes (JPEG, PNG with alpha), ready for ImportBufferAsTexture2D. The
object-oriented API (Figure, no pyplot) keeps it free of global state, so it is also safe in a thread.
"""

from __future__ import annotations

import io
from typing import Any, cast

import matplotlib
import matplotlib.colors
import matplotlib.image
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from examples.live_analytics import level, telemetry

matplotlib.use("Agg")

BACKGROUND, PANEL, INK, GRID = "#0b0f17", "#121a26", "#d8e1ee", "#2a3646"
DEATH, CALM, EXIT = "#ff3b30", "#5b6b82", "#4ade80"
ROUTE_COLOURS = {"bridge": "#ffb020", "detour": "#38bdf8"}
# Where the testers walk: dark teal to bright cyan (the deaths' red stays apart from it).
WALK_MAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "walk", ["#0a4a5c", "#0fa3b1", "#5ef0f0", "#e6ffff"]
)


def _style(axes: Axes, title: str) -> None:
    axes.set_facecolor(PANEL)
    axes.set_title(title, color=INK, fontsize=30, loc="left", pad=14, fontweight="bold")
    axes.tick_params(colors=INK, labelsize=18, width=2, length=6)
    for spine in axes.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(2)
    axes.grid(color=GRID, linewidth=1.2, alpha=0.9)
    axes.set_axisbelow(True)


def dashboard(snapshot: telemetry.Snapshot, now: float, size: tuple[int, int] = (1600, 900)) -> bytes:
    """Two big charts, readable from across the level: deaths per area (the hazard's bar in red) and the time
    to reach the exit, per route. JPEG bytes."""
    width, height = size
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BACKGROUND)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(1, 2, left=0.07, right=0.98, top=0.75, bottom=0.14, wspace=0.2)
    counts = telemetry.totals(snapshot)
    header = (
        ("Runs", counts["runs"], INK, 0.07),
        ("Deaths", counts["deaths"], DEATH, 0.3),
        ("Reached exit", counts["finished"], EXIT, 0.56),
    )
    for name, value, colour, x in header:
        figure.text(x, 0.885, f"{name} {value}", color=colour, fontsize=40, fontweight="bold")
    figure.text(0.98, 0.9, f"{now:4.0f} s", color=INK, fontsize=34, ha="right")

    axes = figure.add_subplot(grid[0, 0])
    _style(axes, "Deaths per area")
    per_area = telemetry.deaths_per_area(snapshot)
    colours = [DEATH if area == level.HAZARD_AREA else CALM for area in per_area.index]
    positions = np.arange(len(per_area))
    axes.bar(positions, per_area.to_numpy(), color=colours, width=0.7)
    axes.set_xticks(positions, [area.replace(" ", "\n") for area in per_area.index])
    axes.set_ylim(0, max(4, int(per_area.max()) + 2))
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    for x, value in zip(positions, per_area.to_numpy(), strict=True):
        if value:
            axes.text(x, value + 0.15, str(value), color=INK, fontsize=26, ha="center", fontweight="bold")

    axes = figure.add_subplot(grid[0, 1])
    _style(axes, "Time to reach the exit")
    times = telemetry.finish_times(snapshot)
    bins = np.arange(4.0, 17.0, 1.0)
    stacked = [times[times["route"] == route]["seconds"].to_numpy() for route in ROUTE_COLOURS]
    axes.hist(
        cast(Any, stacked),
        bins=bins.tolist(),
        stacked=True,
        color=list(ROUTE_COLOURS.values()),
        rwidth=0.85,
        label=[f"via {route}" for route in ROUTE_COLOURS],
    )
    axes.set_xlim(bins[0], bins[-1])
    axes.set_ylim(0, max(4, int(np.histogram(times["seconds"], bins=bins)[0].max(initial=0)) + 1))
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlabel("seconds", color=INK, fontsize=20)
    axes.set_ylabel("runs", color=INK, fontsize=20)
    legend = axes.legend(loc="upper right", fontsize=20, facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    legend.get_frame().set_linewidth(2)
    buffer = io.BytesIO()  # JPEG: the engine decodes it several times faster than a PNG of this size
    figure.savefig(buffer, format="jpg", facecolor=BACKGROUND, pil_kwargs={"quality": 92})
    return buffer.getvalue()


def label_picture(text: str, colour: str = "#ffffff", size: tuple[int, int] = (1600, 200)) -> bytes:
    """A caption for the world: big bold text on a dark, rounded, half-transparent plate (PNG with alpha)."""
    width, height = size
    figure = Figure(figsize=(width / 100, height / 100), dpi=100)
    figure.patch.set_alpha(0.0)
    FigureCanvasAgg(figure)
    from matplotlib.patches import FancyBboxPatch

    plate = FancyBboxPatch(
        (0.01, 0.06),
        0.98,
        0.88,
        boxstyle="round,pad=0,rounding_size=0.08",
        transform=figure.transFigure,
        facecolor="#05070c",
        edgecolor=colour,
        linewidth=5,
        alpha=0.82,
    )
    figure.patches.append(plate)
    figure.text(
        0.5, 0.5, text, color=colour, fontsize=height * 0.36, fontweight="bold", ha="center", va="center"
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", transparent=True)
    return buffer.getvalue()


def heat_picture(walk: np.ndarray, death: np.ndarray, floor: float = 0.04) -> bytes:
    """The floor overlay as an RGBA PNG: where the testers walk in cyan (alpha rising with the traffic),
    where they die in red over it; clear where nobody went."""
    walk = np.clip(np.asarray(walk, dtype=np.float64), 0.0, 1.0)
    death = np.clip(np.asarray(death, dtype=np.float64), 0.0, 1.0)
    rgba = WALK_MAP(walk)
    rgba[..., 3] = np.where(walk > floor, 0.2 + 0.7 * np.sqrt(walk), 0.0)
    red = np.where(death > floor, 0.35 + 0.65 * np.sqrt(death), 0.0)[..., None]
    rgba[..., :3] = rgba[..., :3] * (1.0 - red) + np.array(matplotlib.colors.to_rgb(DEATH)) * red
    rgba[..., 3:] = np.maximum(rgba[..., 3:], red)
    image = (rgba * 255).astype(np.uint8)  # row 0 (y_min) on top: how the floor plane maps it
    buffer = io.BytesIO()
    matplotlib.image.imsave(buffer, image, format="png")
    return buffer.getvalue()
