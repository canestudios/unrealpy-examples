"""Fluid example: the parts that need no engine. The Warp kernels run on Warp's CPU device (the same Python
kernels the game runs on the GPU), on a small dam."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

warp = pytest.importorskip("warp")

from examples.fluid import demo, sim  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "warp"


@pytest.fixture(scope="module")
def warp_cpu() -> None:
    sim.init(CACHE)


def small_tank() -> sim.Tank:
    return sim.Tank(
        lo=(-1.0, -0.5, 0.0),
        hi=(1.0, 0.5, 3.0),
        dam=(0.6, 1.0, 0.4),
        spacing=0.05,
        gate_opens=0.05,
        paddle_starts=0.3,
        paddle_stroke=0.4,
        paddle_period=1.0,
        pillars=[(0.3, 0.0, 0.1, 1.0)],
        boxes=[((0.75, 0.0, 0.0), (0.2, 0.4, 0.1), -15.0)],
    )


def test_the_dam_is_a_lattice_inside_its_corner() -> None:
    tank = small_tank()
    block = tank.block()
    assert block.dtype == np.float32 and block.shape[1] == 3
    assert block[:, 0].min() > tank.lo[0] and block[:, 0].max() < tank.lo[0] + tank.dam[0]
    assert np.abs(block[:, 1]).max() < tank.dam[1] / 2 and block[:, 2].max() < tank.dam[2]
    low = tank.block(limit=100)
    assert len(low) == 100 and low[:, 2].max() <= block[:, 2].min() + tank.spacing  # the bottom layer first


def test_the_gate_opens_and_the_paddle_strokes() -> None:
    tank = small_tank()
    assert tank.gate(0.0) == pytest.approx(tank.lo[0] + tank.dam[0]) and tank.gate(1.0) == sim.NO_GATE
    assert tank.paddle(0.0) == tank.hi[0] == tank.paddle(tank.paddle_starts)
    half = tank.paddle_starts + tank.paddle_period / 2
    assert tank.paddle(half) == pytest.approx(tank.hi[0] - tank.paddle_stroke)


def test_rest_density_and_stiffness_scale_with_the_spacing() -> None:
    coarse, fine = sim.rest_density(0.1, 0.2), sim.rest_density(0.05, 0.1)
    assert coarse > 0 and fine == pytest.approx(coarse * 8, rel=0.05)  # number density: 1 / spacing^3
    assert sim.stiffness(0.05, 0.1, fine) > 0


@pytest.mark.usefixtures("warp_cpu")
def test_the_dam_breaks_and_the_water_stays_in_the_tank() -> None:
    tank = small_tank()
    fluid = sim.Fluid(tank, device="cpu")
    start = fluid.positions().copy()  # (on the CPU device the array is a view of the simulation)
    capsule = sim.Capsule((0.0, 0.0, 0.1), (0.0, 0.0, 0.9), 0.12)
    for _ in range(36):
        fluid.set_capsules([capsule])
        fluid.step(1.0 / 60.0)
    x, v = fluid.positions(), fluid.velocities()
    assert np.isfinite(x).all() and np.isfinite(v).all()
    r = fluid.radius
    assert x[:, 0].min() >= tank.lo[0] and x[:, 0].max() <= tank.hi[0]
    assert np.abs(x[:, 1]).max() <= tank.hi[1] and x[:, 2].min() >= 0.0
    assert x[:, 0].max() > start[:, 0].max() + 0.3  # the water ran out of the dam
    assert np.linalg.norm(v, axis=1).max() <= fluid.max_speed + 1e-3
    px, py, pr, _ = tank.pillars[0]
    assert (np.hypot(x[:, 0] - px, x[:, 1] - py) > pr + r * 0.4).all()  # nothing inside the pillar
    beside = (x[:, 2] > capsule.a[2]) & (x[:, 2] < capsule.b[2])  # level with the capsule's straight part
    assert (np.hypot(x[:, 0], x[:, 1]) > capsule.radius + r * 0.4)[beside].all()  # nor the capsule


@pytest.mark.usefixtures("warp_cpu")
def test_texels_and_the_capsules_water() -> None:
    tank = small_tank()
    fluid = sim.Fluid(tank, device="cpu")
    capsule = sim.Capsule((-0.2, 0.0, 0.1), (-0.2, 0.0, 0.9), 0.1)
    for _ in range(20):
        fluid.set_capsules([capsule])
        fluid.step(1.0 / 60.0)
    texels = fluid.texel_data().copy()
    assert texels.shape == (sim.SPRAY, 4) and texels.dtype == np.float32
    drops = texels[texels[:, 3] >= 0.0]  # the spray above the surface, packed at the front
    assert len(drops) < fluid.count and (texels[len(drops) :, 3] == -1.0).all()  # few or none yet
    positions = fluid.positions() * 100.0
    speeds = np.linalg.norm(fluid.velocities(), axis=1)
    for drop in drops[:20]:  # each drop is one of the particles, with its speed
        k = int(np.argmin(np.linalg.norm(positions - drop[:3], axis=1)))
        assert np.allclose(positions[k], drop[:3], atol=1e-3) and abs(speeds[k] - drop[3]) < 1e-4
    surface = fluid.surface_data()
    assert surface.shape == (*fluid.grid_size, 4)
    wet = surface[..., 0] > 0.0
    assert wet.any() and (~wet).any()  # water at the dam's end, none yet at the far end
    assert surface[..., 0].max() <= (fluid.positions()[:, 2].max() + fluid.radius) * 100.0 + 1e-3
    assert np.allclose(surface[..., 2] ** 2 + surface[..., 3] ** 2 <= 1.0, True)
    (((vx, _, _), touching),) = fluid.flow()
    assert touching > 0 and vx > 0.0  # the dam's water runs past it towards +x
    fluid.publish()
    results = fluid.collect()
    assert np.array_equal(results.particles, texels) and len(results.flow) == 1
    with pytest.raises(ValueError):
        fluid.set_capsules([capsule] * (sim.MAX_CAPSULES + 1))


def test_the_card_shows_a_kernel_and_the_banner_counts_particles() -> None:
    source = demo.kernel_source()
    assert source.startswith("@wp.kernel") and "def lambdas(" in source and "hash_grid_query" in source
    assert demo.banner_text(150_000) == (
        "NVIDIA Warp: 150,000-particle fluid, Python GPU kernels inside the game"
    )
    command = demo.clip_command("ffmpeg", Path("frames"), 30, Path("card.png"), Path("b.png"), Path("o.mp4"))
    graph = command[command.index("-filter_complex") + 1]
    assert "concat=n=2" in graph and command.count("-i") == 3 and command[-1] == "o.mp4"
    assert math.isclose(float(command[command.index("-t") + 1]), demo.CARD_SECONDS)
