"""A position-based fluid (Macklin and Mueller, "Position Based Fluids", 2013) in NVIDIA Warp: every step of
the simulation is a Python function compiled to a CUDA kernel (@wp.kernel), neighbours come from a
wp.HashGrid.

The world is in metres, z up (the game converts to centimetres). The water lives in an open box with a floor
at z = 0; it collides with the box, a wave paddle at the +x end, a gate that holds the dam until it opens,
vertical pillars, oriented boxes (ramps) and capsules (the mannequins), which the game moves every frame. The
capsules also measure the water: how much of it touches each one and how fast it flows there, which the game
turns into a push.

Runs on "cuda:0" in the game and on "cpu" in the unit tests (same kernels, Warp compiles both).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

NO_GATE = 1.0e6  # the gate's x once it has opened
RELAX = 0.01
CELL = 0.05  # metres: the drawn surface's grid
SPRAY = 16_384  # at most this many drops of spray are drawn
MAX_CAPSULES = 8
MAX_SUBSTEP = 1.0 / 60.0
WALL_FRICTION = 0.8  # per substep, for particles touching a side wall
K_CORR = 0.01


def init(cache: Path | None = None) -> None:
    """Initialise Warp once, with its kernel cache in `cache` (default: Warp's, under the user profile)."""
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(cache)
    wp.config.quiet = True
    wp.init()


# -- kernels -------------------------------------------------------------------------------------------------


@wp.func
def poly6(r2: float, h: float, k: float) -> float:
    """The poly6 kernel; k = 315 / (64 pi h^9)."""
    if r2 >= h * h:
        return 0.0
    d = h * h - r2
    return k * d * d * d


@wp.func
def spiky_grad(r: wp.vec3, h: float, k: float) -> wp.vec3:
    """The spiky kernel's gradient; k = -45 / (pi h^6)."""
    d = wp.length(r)
    if d >= h or d < 1.0e-6:
        return wp.vec3(0.0)
    s = h - d
    return r * (k * s * s / d)


@wp.kernel
def predict(x: wp.array[wp.vec3], v: wp.array[wp.vec3], p: wp.array[wp.vec3], gravity: wp.vec3, dt: float):
    i = wp.tid()
    vi = v[i] + gravity * dt
    v[i] = vi
    p[i] = x[i] + vi * dt


@wp.kernel
def lambdas(
    grid: wp.uint64,
    p: wp.array[wp.vec3],
    lam: wp.array[float],
    h: float,
    k_poly: float,
    k_spiky: float,
    rest: float,
    relax: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    pi = p[i]
    rho = float(0.0)  # noqa: UP018 - a Warp dynamic variable
    grad_i = wp.vec3(0.0)
    sum_sq = float(0.0)  # noqa: UP018 - a Warp dynamic variable
    for j in wp.hash_grid_query(grid, pi, h):
        r = pi - p[j]
        rho += poly6(wp.dot(r, r), h, k_poly)
        if j != i:
            g = spiky_grad(r, h, k_spiky) / rest
            grad_i += g
            sum_sq += wp.dot(g, g)
    c = wp.max(rho / rest - 1.0, 0.0)  # only compression: a free surface does not pull itself together
    lam[i] = -c / (sum_sq + wp.dot(grad_i, grad_i) + relax)


@wp.kernel
def deltas(
    grid: wp.uint64,
    p: wp.array[wp.vec3],
    lam: wp.array[float],
    dp: wp.array[wp.vec3],
    h: float,
    k_poly: float,
    k_spiky: float,
    rest: float,
    k_corr: float,
    w_corr: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    pi = p[i]
    li = lam[i]
    d = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, pi, h):
        if j != i:
            r = pi - p[j]
            ratio = poly6(wp.dot(r, r), h, k_poly) / w_corr
            corr = -k_corr * ratio * ratio * ratio * ratio  # artificial pressure: no clumping at the surface
            d += (li + lam[j] + corr) * spiky_grad(r, h, k_spiky)
    dp[i] = d / rest


@wp.func
def box_push(q: wp.vec3, centre: wp.vec3, half: wp.vec3, rot: wp.quat, radius: float) -> wp.vec3:
    """The correction that moves point q out of an oriented box (plus radius), or zero."""
    local = wp.quat_rotate_inv(rot, q - centre)
    a = wp.vec3(wp.abs(local[0]), wp.abs(local[1]), wp.abs(local[2])) - half - wp.vec3(radius)
    if a[0] >= 0.0 or a[1] >= 0.0 or a[2] >= 0.0:
        return wp.vec3(0.0)
    # inside: leave by the nearest face
    out = wp.vec3(0.0)
    if a[0] >= a[1] and a[0] >= a[2]:
        out = wp.vec3(-a[0] * wp.sign(local[0]), 0.0, 0.0)
    elif a[1] >= a[2]:
        out = wp.vec3(0.0, -a[1] * wp.sign(local[1]), 0.0)
    else:
        out = wp.vec3(0.0, 0.0, -a[2] * wp.sign(local[2]))
    return wp.quat_rotate(rot, out)


@wp.func
def capsule_push(q: wp.vec3, a: wp.vec3, b: wp.vec3, radius: float) -> wp.vec3:
    ab = b - a
    t = wp.clamp(wp.dot(q - a, ab) / wp.max(wp.dot(ab, ab), 1.0e-8), 0.0, 1.0)
    c = a + ab * t
    r = q - c
    d = wp.length(r)
    if d >= radius:
        return wp.vec3(0.0)
    if d < 1.0e-6:
        return wp.vec3(0.0, 0.0, radius)
    return r * ((radius - d) / d)


@wp.kernel
def collide(
    p: wp.array[wp.vec3],
    dp: wp.array[wp.vec3],
    lo: wp.vec3,
    hi: wp.vec3,
    paddle: float,
    gate: float,
    radius: float,
    pillars: wp.array[wp.vec4],
    box_centre: wp.array[wp.vec3],
    box_half: wp.array[wp.vec3],
    box_rot: wp.array[wp.quat],
    cap_a: wp.array[wp.vec3],
    cap_b: wp.array[wp.vec3],
    cap_r: wp.array[float],
):
    """Apply the density correction, then push the particle out of everything solid."""
    i = wp.tid()
    q = p[i] + dp[i]
    for k in range(pillars.shape[0]):  # vertical cylinders: x, y, radius, height
        c = pillars[k]
        rx = q[0] - c[0]
        ry = q[1] - c[1]
        d = wp.sqrt(rx * rx + ry * ry)
        reach = c[2] + radius
        if d < reach and q[2] < c[3]:
            if d < 1.0e-6:
                rx = 1.0
                d = 1.0
            q = wp.vec3(c[0] + rx / d * reach, c[1] + ry / d * reach, q[2])
    for k in range(box_centre.shape[0]):
        q = q + box_push(q, box_centre[k], box_half[k], box_rot[k], radius)
    for k in range(cap_a.shape[0]):
        q = q + capsule_push(q, cap_a[k], cap_b[k], cap_r[k] + radius)
    # Each particle stops at its own distance from a wall (0.5 to 1.5 radii): particles pushed into a corner
    # must not land on the same point, or they stack into a column that climbs the corner.
    gap = radius * (0.5 + wp.randf(wp.rand_init(7919, i)))
    left = lo[0] + gap
    right = wp.max(left, wp.min(wp.min(hi[0], paddle), gate) - gap)
    q = wp.vec3(
        wp.clamp(q[0], left, right),
        wp.clamp(q[1], lo[1] + gap, hi[1] - gap),
        wp.clamp(q[2], lo[2] + gap, hi[2] - gap),
    )
    p[i] = q


@wp.kernel
def finish(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    p: wp.array[wp.vec3],
    dt: float,
    max_speed: float,
    lo: wp.vec3,
    hi: wp.vec3,
    friction: float,
    cap_a: wp.array[wp.vec3],
    cap_b: wp.array[wp.vec3],
    cap_r: wp.array[float],
    cap_flow: wp.array[wp.vec3],
    cap_count: wp.array[float],
):
    """New velocity from the move (clamped, slowed along the side walls), new position; and what each capsule
    feels of the water."""
    i = wp.tid()
    vi = (p[i] - x[i]) / dt
    q = p[i]
    margin = 0.08
    near_x = q[0] < lo[0] + margin or q[0] > hi[0] - margin
    near_y = q[1] < lo[1] + margin or q[1] > hi[1] - margin
    if near_x or near_y:  # wall friction: no water climbing up the walls and corners
        vi = wp.vec3(vi[0] * friction, vi[1] * friction, wp.min(vi[2], 0.0) + wp.max(vi[2], 0.0) * friction)
    s = wp.length(vi)
    if s > max_speed:
        vi = vi * (max_speed / s)
    v[i] = vi
    x[i] = p[i]
    for k in range(cap_a.shape[0]):
        near = capsule_push(p[i], cap_a[k], cap_b[k], cap_r[k] + 0.15)
        if wp.length(near) > 0.0:
            wp.atomic_add(cap_flow, k, vi)
            wp.atomic_add(cap_count, k, 1.0)


@wp.kernel
def columns(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    lo: wp.vec2,
    cell: float,
    count: wp.array2d[float],
    bottom: wp.array2d[float],
    top: wp.array2d[float],
    speed: wp.array2d[float],
):
    """Per cell of the surface grid: how many particles stand in its column, the lowest and the highest one,
    their speeds."""
    i = wp.tid()
    q = x[i]
    cx = int((q[0] - lo[0]) / cell)
    cy = int((q[1] - lo[1]) / cell)
    if cx >= 0 and cy >= 0 and cx < count.shape[1] and cy < count.shape[0]:
        wp.atomic_add(count, cy, cx, 1.0)
        wp.atomic_min(bottom, cy, cx, q[2])
        wp.atomic_max(top, cy, cx, q[2])
        wp.atomic_add(speed, cy, cx, wp.length(v[i]))


@wp.kernel
def smooth(
    count: wp.array2d[float],
    bottom: wp.array2d[float],
    top: wp.array2d[float],
    speed: wp.array2d[float],
    volume: float,
    radius: float,
    out: wp.array2d[wp.vec4],
):
    """The surface the game draws, from the water each column holds (not from its highest drop, so spray and
    breaking crests do not stand up as spikes): over a 5 x 5 window, the mean depth of water on the mean
    floor under it (cm), and the mean speed (the foam)."""
    gy, gx = wp.tid()
    ny = count.shape[0]
    nx = count.shape[1]
    particles = float(0.0)  # noqa: UP018 - a Warp dynamic variable
    floor = float(0.0)  # noqa: UP018 - a Warp dynamic variable
    froth = float(0.0)  # noqa: UP018 - a Warp dynamic variable
    wet = float(0.0)  # noqa: UP018 - a Warp dynamic variable
    highest = float(0.0)  # noqa: UP018 - a Warp dynamic variable
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y = wp.clamp(gy + dy, 0, ny - 1)
            x = wp.clamp(gx + dx, 0, nx - 1)
            n = count[y, x]
            if n > 0.0:
                particles += n
                floor += bottom[y, x] - radius
                froth += speed[y, x]
                wet += 1.0
                if n > 1.0:  # a lone drop is spray
                    highest = wp.max(highest, top[y, x] + radius)
    height = float(0.0)  # noqa: UP018 - a Warp dynamic variable
    foam = float(0.0)  # noqa: UP018 - a Warp dynamic variable
    if wet >= 6.0:
        height = wp.min(wp.max(floor / wet, 0.0) + particles / 25.0 * volume, highest)
        foam = froth / particles
    out[gy, gx] = wp.vec4(height * 100.0, foam, 0.0, 0.0)


@wp.kernel
def normals(heights: wp.array2d[wp.vec4], cell: float, out: wp.array2d[wp.vec4]):
    """The smoothed surface's normals (x, y), from its heights' central differences."""
    gy, gx = wp.tid()
    ny = heights.shape[0]
    nx = heights.shape[1]
    here = heights[gy, gx]
    xl = heights[gy, wp.max(gx - 1, 0)][0]
    xr = heights[gy, wp.min(gx + 1, nx - 1)][0]
    yd = heights[wp.max(gy - 1, 0), gx][0]
    yu = heights[wp.min(gy + 1, ny - 1), gx][0]
    d = 2.0 * cell * 100.0
    n = wp.normalize(wp.vec3(-(xr - xl) / d, -(yu - yd) / d, 1.0))
    out[gy, gx] = wp.vec4(here[0], here[1], n[0], n[1])


@wp.kernel
def pack(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    scale: float,
    surface: wp.array2d[wp.vec4],
    lo: wp.vec2,
    cell: float,
    above: float,
    used: wp.array[int],
    out: wp.array[wp.vec4],
):
    """The spray's texels: the particles above the drawn surface (the rest is the surface), packed at the
    front of `out` (up to its size): position in centimetres and speed in m/s."""
    i = wp.tid()
    q = x[i]
    cx = wp.clamp(int((q[0] - lo[0]) / cell), 0, surface.shape[1] - 1)
    cy = wp.clamp(int((q[1] - lo[1]) / cell), 0, surface.shape[0] - 1)
    if q[2] * scale >= surface[cy, cx][0] + above * scale:
        slot = wp.atomic_add(used, 0, 1)
        if slot < out.shape[0]:
            out[slot] = wp.vec4(q[0] * scale, q[1] * scale, q[2] * scale, wp.length(v[i]))


# -- the scene -----------------------------------------------------------------------------------------------


@dataclass
class Tank:
    """Where the water is: the arena box, the dam, the paddle's stroke, the solid things in it."""

    lo: tuple[float, float, float] = (-6.0, -4.0, 0.0)
    hi: tuple[float, float, float] = (6.0, 4.0, 10.0)
    dam: tuple[float, float, float] = (3.1, 8.0, 1.4)  # length, width, height of the block at the -x end
    spacing: float = 0.07
    gate_opens: float = 0.25  # seconds
    paddle_stroke: float = 2.0  # metres
    paddle_period: float = 3.0  # seconds
    paddle_starts: float = 3.0
    pillars: list[tuple[float, float, float, float]] = field(default_factory=list)  # x, y, radius, height
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float], float]] = field(
        default_factory=list
    )  # centre, half extents, pitch (degrees, about y)

    def paddle(self, t: float) -> float:
        """The wave paddle's face (x) at the +x end at time t: flush with the wall, then a smooth stroke in
        and out that sends the water the dam threw there back as waves."""
        if t < self.paddle_starts:
            return self.hi[0]
        phase = (t - self.paddle_starts) / self.paddle_period
        return self.hi[0] - self.paddle_stroke * 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))

    def gate(self, t: float) -> float:
        return self.lo[0] + self.dam[0] if t < self.gate_opens else NO_GATE

    def block(self, limit: int | None = None) -> np.ndarray:
        """The dam's particles on a lattice (jittered a little), at most `limit`."""
        s = self.spacing
        length, width, height = self.dam
        xs = np.arange(self.lo[0] + s, self.lo[0] + length - s * 0.5, s)
        ys = np.arange(-width / 2.0 + s, width / 2.0 - s * 0.5, s)
        zs = np.arange(s * 0.5, height, s)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        grid = grid[np.argsort(grid[:, 2], kind="stable")]  # lowest first, so a limit keeps the bottom
        if limit is not None:
            grid = grid[:limit]
        jitter = np.random.default_rng(1).uniform(-0.05, 0.05, grid.shape) * s
        return (grid + jitter).astype(np.float32)


def _lattice(spacing: float, h: float) -> np.ndarray:
    n = int(math.ceil(h / spacing))
    offsets = np.arange(-n, n + 1) * spacing
    points = np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 3)
    return points[np.einsum("ij,ij->i", points, points) < h * h]


def rest_density(spacing: float, h: float) -> float:
    """The poly6 density of a particle inside a lattice of the given spacing (the target density)."""
    r2 = np.einsum("ij,ij->i", *(2 * [_lattice(spacing, h)]))
    return float(np.sum(315.0 / (64.0 * math.pi * h**9) * (h * h - r2) ** 3))


def stiffness(spacing: float, h: float, rest: float) -> float:
    """Sum of |grad C|^2 over a particle's neighbours at rest: the scale of the constraint's denominator."""
    r = _lattice(spacing, h)
    d = np.sqrt(np.einsum("ij,ij->i", r, r))
    r, d = r[d > 1e-9], d[d > 1e-9]
    grad = r * (-45.0 / (math.pi * h**6) * (h - d) ** 2 / d)[:, None] / rest
    return float(np.sum(grad * grad))


@dataclass
class Results:
    particles: np.ndarray  # (SPRAY, 4): the spray's x, y, z (cm), speed (m/s); unused: speed -1
    surface: np.ndarray  # (rows, columns, 4): height (cm), foam (m/s), normal x, normal y
    flow: list[tuple[tuple[float, float, float], float]]  # per capsule: mean water velocity, particles


@dataclass
class Capsule:
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    radius: float


class Fluid:
    """The particles and their step. `step(dt)` advances by dt in substeps of at most MAX_SUBSTEP seconds (and
    at least `substeps` of them), each with `iterations` density-constraint iterations."""

    def __init__(
        self,
        tank: Tank,
        *,
        limit: int | None = None,
        device: str = "cuda:0",
        substeps: int = 1,
        iterations: int = 4,
    ) -> None:
        self.tank, self.device = tank, device
        self.substeps, self.iterations = substeps, iterations
        self.h = tank.spacing * 2.0
        self.radius = tank.spacing * 0.5
        self.k_poly = 315.0 / (64.0 * math.pi * self.h**9)
        self.k_spiky = -45.0 / (math.pi * self.h**6)
        self.rest = rest_density(tank.spacing, self.h)
        scale = stiffness(tank.spacing, self.h, self.rest)
        self.relax = RELAX * scale  # constraint softness and artificial pressure, relative to the
        self.k_corr = K_CORR / scale  # constraint's own scale (so they do not depend on the spacing)
        self.w_corr = 315.0 / (64.0 * math.pi * self.h**9) * (self.h**2 - (0.2 * self.h) ** 2) ** 3
        self.max_speed = 7.0  # a dam of 1.35 m releases at about sqrt(2 g h) = 5 m/s
        self.time = 0.0
        points = tank.block(limit)
        self.count = len(points)
        self.x = wp.array(points, dtype=wp.vec3, device=device)
        self.v = wp.zeros(self.count, dtype=wp.vec3, device=device)
        self.p = wp.zeros(self.count, dtype=wp.vec3, device=device)
        self.dp = wp.zeros(self.count, dtype=wp.vec3, device=device)
        self.lam = wp.zeros(self.count, dtype=float, device=device)
        self.texels = wp.zeros(SPRAY, dtype=wp.vec4, device=device)  # the spray drawn as drops
        self.spray_used = wp.zeros(1, dtype=int, device=device)
        # The surface grid: CELL metres a cell over the tank.
        self.cell = CELL
        self.grid_size = (
            int(math.ceil((tank.hi[1] - tank.lo[1]) / CELL)),
            int(math.ceil((tank.hi[0] - tank.lo[0]) / CELL)),
        )  # rows (y), columns (x)
        self.count_grid = wp.zeros(self.grid_size, dtype=float, device=device)
        self.bottom = wp.zeros(self.grid_size, dtype=float, device=device)
        self.top = wp.zeros(self.grid_size, dtype=float, device=device)
        self.speed_grid = wp.zeros(self.grid_size, dtype=float, device=device)
        self.heights = wp.zeros(self.grid_size, dtype=wp.vec4, device=device)
        self.surface = wp.zeros(self.grid_size, dtype=wp.vec4, device=device)
        cells = max(16, int(64 * (self.count / 150_000) ** (1 / 3)) * 2)
        self.grid = wp.HashGrid(cells, cells, cells, device=device)
        pillars = tank.pillars or [(0.0, 0.0, 0.0, -1.0)]  # a zero-size pillar: the arrays may not be empty
        self.pillars = wp.array([wp.vec4(*pillar) for pillar in pillars], dtype=wp.vec4, device=device)
        boxes = tank.boxes or [((0.0, 0.0, -10.0), (0.0, 0.0, 0.0), 0.0)]
        self.box_centre = wp.array([wp.vec3(*c) for c, _, _ in boxes], dtype=wp.vec3, device=device)
        self.box_half = wp.array([wp.vec3(*half) for _, half, _ in boxes], dtype=wp.vec3, device=device)
        self.box_rot = wp.array(
            [wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(pitch)) for _, _, pitch in boxes],
            dtype=wp.quat,
            device=device,
        )
        pinned = device != "cpu"
        self.cap_a = wp.zeros(MAX_CAPSULES, dtype=wp.vec3, device=device)
        self.cap_b = wp.zeros(MAX_CAPSULES, dtype=wp.vec3, device=device)
        self.cap_r = wp.zeros(MAX_CAPSULES, dtype=float, device=device)
        self.cap_flow = wp.zeros(MAX_CAPSULES, dtype=wp.vec3, device=device)
        self.cap_count = wp.zeros(MAX_CAPSULES, dtype=float, device=device)
        self.cap_pinned = (
            wp.zeros(MAX_CAPSULES, dtype=wp.vec3, device="cpu", pinned=pinned),
            wp.zeros(MAX_CAPSULES, dtype=wp.vec3, device="cpu", pinned=pinned),
            wp.zeros(MAX_CAPSULES, dtype=float, device="cpu", pinned=pinned),
        )
        self.cap_host = tuple(array.numpy() for array in self.cap_pinned)  # views of the pinned memory
        # Two sets of host buffers: the game reads last frame's results while this frame's are on their way.
        self.results = [
            (
                wp.zeros(SPRAY, dtype=wp.vec4, device="cpu", pinned=pinned),
                wp.zeros(self.grid_size, dtype=wp.vec4, device="cpu", pinned=pinned),
                wp.zeros(MAX_CAPSULES, dtype=wp.vec3, device="cpu", pinned=pinned),
                wp.zeros(MAX_CAPSULES, dtype=float, device="cpu", pinned=pinned),
            )
            for _ in range(2)
        ]
        self.events = [wp.Event(device) if device != "cpu" else None for _ in range(2)]
        self.published: int | None = None
        self.set_capsules([])

    def set_capsules(self, capsules: list[Capsule]) -> None:
        """The mannequins' bodies this step (at most MAX_CAPSULES): written into pinned host memory and copied
        to the GPU without waiting."""
        if len(capsules) > MAX_CAPSULES:
            raise ValueError(f"at most {MAX_CAPSULES} capsules")
        self.capsule_count = len(capsules)
        a, b, r = self.cap_host
        a.fill(0.0)
        b.fill(0.0)
        a[:, 2] = b[:, 2] = -100.0  # the unused ones: far below the tank
        r.fill(0.0)
        for k, capsule in enumerate(capsules):
            a[k], b[k], r[k] = capsule.a, capsule.b, capsule.radius
        for device_array, host in zip((self.cap_a, self.cap_b, self.cap_r), self.cap_pinned, strict=True):
            wp.copy(device_array, host)

    def step(self, dt: float) -> None:
        self.cap_flow.zero_()
        self.cap_count.zero_()
        substeps = max(self.substeps, math.ceil(dt / MAX_SUBSTEP - 1e-6))
        sub = dt / substeps
        tank, n, device = self.tank, self.count, self.device
        lo, hi = wp.vec3(*tank.lo), wp.vec3(*tank.hi)
        for _ in range(substeps):
            self.time += sub
            paddle, gate = tank.paddle(self.time), tank.gate(self.time)
            wp.launch(predict, n, (self.x, self.v, self.p, wp.vec3(0.0, 0.0, -9.81), sub), device=device)
            self.dp.zero_()
            self._collide(lo, hi, paddle, gate)
            self.grid.build(self.p, self.h)
            for _ in range(self.iterations):
                wp.launch(
                    lambdas,
                    n,
                    (
                        self.grid.id,
                        self.p,
                        self.lam,
                        self.h,
                        self.k_poly,
                        self.k_spiky,
                        self.rest,
                        self.relax,
                    ),
                    device=device,
                )
                wp.launch(
                    deltas,
                    n,
                    (
                        self.grid.id,
                        self.p,
                        self.lam,
                        self.dp,
                        self.h,
                        self.k_poly,
                        self.k_spiky,
                        self.rest,
                        self.k_corr,
                        self.w_corr,
                    ),
                    device=device,
                )
                self._collide(lo, hi, paddle, gate)
                self.dp.zero_()
            wp.launch(
                finish,
                n,
                (
                    self.x,
                    self.v,
                    self.p,
                    sub,
                    self.max_speed,
                    lo,
                    hi,
                    WALL_FRICTION,
                    self.cap_a,
                    self.cap_b,
                    self.cap_r,
                    self.cap_flow,
                    self.cap_count,
                ),
                device=device,
            )

    def _collide(self, lo: wp.vec3, hi: wp.vec3, paddle: float, gate: float) -> None:
        wp.launch(
            collide,
            self.count,
            (
                self.p,
                self.dp,
                lo,
                hi,
                paddle,
                gate,
                self.radius,
                self.pillars,
                self.box_centre,
                self.box_half,
                self.box_rot,
                self.cap_a,
                self.cap_b,
                self.cap_r,
            ),
            device=self.device,
        )

    def texel_data(self) -> np.ndarray:
        """(SPRAY, 4) float32: the spray drops' x, y, z in centimetres and speed (unused ones: -1), now."""
        self.publish()
        return self.collect().particles

    def surface_data(self) -> np.ndarray:
        """(rows, columns, 4) float32: height (cm), foam (m/s), normal x and y, now."""
        self.publish()
        return self.collect().surface

    def flow(self) -> list[tuple[tuple[float, float, float], float]]:
        """Per capsule: the mean water velocity around it (m/s) and how many particles touch it, now."""
        self.publish()
        return self.collect().flow

    def publish(self) -> None:
        """Build the surface and the texels, and start copying them and the capsules' water to the host,
        without waiting."""
        slot = 0 if self.published != 0 else 1
        texels, surface, flow, count = self.results[slot]
        lo = wp.vec2(self.tank.lo[0], self.tank.lo[1])
        device = self.device
        self.count_grid.zero_()
        self.bottom.fill_(1.0e6)
        self.top.zero_()
        self.speed_grid.zero_()
        wp.launch(
            columns,
            self.count,
            (self.x, self.v, lo, self.cell, self.count_grid, self.bottom, self.top, self.speed_grid),
            device=device,
        )
        volume = self.tank.spacing**3 / self.cell**2  # metres of depth one particle adds to a cell
        wp.launch(
            smooth,
            self.grid_size,
            (self.count_grid, self.bottom, self.top, self.speed_grid, volume, self.radius, self.heights),
            device=device,
        )
        wp.launch(normals, self.grid_size, (self.heights, self.cell, self.surface), device=device)
        above = 2.0 * self.tank.spacing
        self.texels.fill_(wp.vec4(0.0, 0.0, -10000.0, -1.0))  # unused drops: hidden
        self.spray_used.zero_()
        wp.launch(
            pack,
            self.count,
            (self.x, self.v, 100.0, self.surface, lo, self.cell, above, self.spray_used, self.texels),
            device=device,
        )
        wp.copy(texels, self.texels)
        wp.copy(surface, self.surface)
        wp.copy(flow, self.cap_flow)
        wp.copy(count, self.cap_count)
        event = self.events[slot]
        if event is not None:
            wp.record_event(event)
        self.published = slot

    def collect(self) -> Results:
        """The last published results (waits until they have arrived, usually long done)."""
        assert self.published is not None, "publish() first"
        slot = self.published
        event = self.events[slot]
        if event is not None:
            wp.synchronize_event(event)
        else:
            wp.synchronize_device(self.device)
        texels, surface, flow_sums, counts = (array.numpy() for array in self.results[slot])
        flow = []
        for k in range(self.capsule_count):
            c = float(counts[k])
            mean = flow_sums[k] / c if c > 0 else np.zeros(3)
            flow.append(((float(mean[0]), float(mean[1]), float(mean[2])), c))
        return Results(texels, surface, flow)

    def positions(self) -> np.ndarray:
        return self.x.numpy()

    def velocities(self) -> np.ndarray:
        return self.v.numpy()
