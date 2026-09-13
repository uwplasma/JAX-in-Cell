"""The field walls, the gather, and the time loop: mirror symmetry of the wall closures,
the force on a charge sheet next to each wall, the random keys each step hands out, the
switches the implicit scheme refuses, and the bookkeeping that keeps the loop cheap."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import jaxincell._simulation as simulation_module
from jaxincell import Collisions, Domain, Simulation, Solver, Species, epsilon_0
from jaxincell._core import E_x_from_rho, apply_particle_bc, current_from_continuity, deposit, wrap_positions

L, CELLS = 1.0, 32
DX = L / CELLS
CENTRE0 = -L / 2 + DX / 2
WALLS = [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2)]           # every pair Domain accepts


def slow_ones(speed):
    """A reflection law that returns the electrons slower than about 1e6 m/s."""
    return jnp.exp(-speed ** 2 / (2 * 1e12))


def walled_simulation(**solver):
    """Electrostatic (no transverse velocity), so that a step far above the light-wave limit is stable."""
    e = Species.electrons(n=600, density=1e15, vth=(2e6, 0, 0), drift=(1e6, 0, 0), reflection=(0.0, slow_ones))
    i = Species.ions(n=600, density=1e15, electrons=e, mass_ratio=50.0, reflection=(0.0, 0.5))
    domain = Domain(length=1e-3, cells=24, dt_over_dx_c=20.0, particle_bc=("thermal", "absorbing"),
                    field_bc=("reflective", "absorbing"))
    return Simulation(domain, [e, i], Solver(filter_passes=2, **solver))


def test_the_carried_density_is_the_density_at_the_integer_time_positions():
    """The explicit step no longer deposits the density at x^n: it carries the one the
    previous step ended on. That is only right if the carried density is the deposit at
    the reconstructed positions wrap(x^{n+1/2} - dt v/2), with the weights the walls have
    left, which this checks after a run with a thermal wall, partial reflection and a
    filter."""
    sim = walled_simulation()
    d = sim.domain
    out = sim.run(40, seed=1, store_every=4)
    E, B, x_half, v, w, qm, rho, key = out.state
    x_n = wrap_positions(x_half - 0.5 * d.dt * v, w, (d.length, d.length_y, d.length_z), d.particle_bc, d.dx)
    expected = sim._smooth(deposit(x_n[:, 0], out.charge * w, d.grid[0], d.dx, d.cells, d.particle_bc))
    assert np.allclose(np.asarray(rho), np.asarray(expected), rtol=1e-12, atol=1e-12 * float(jnp.abs(expected).max()))
    assert np.allclose(np.asarray(out.rho[-1]), np.asarray(rho), rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["explicit", "implicit"])
def test_store_every_traces_the_step_once(monkeypatch, algorithm):
    """With ``store_every > 1`` the loop carries the last output along with the state
    through one scan, so the step is traced once rather than twice, which is most of the
    compile time."""
    traces = []
    method = "_explicit_step" if algorithm == "explicit" else "_implicit_step"
    original = getattr(Simulation, method)

    def counted(self, carry, extra):
        traces.append(1)
        return original(self, carry, extra)

    monkeypatch.setattr(Simulation, method, counted)
    electrons = Species.electrons(n=17, density=1e10, vth=(1e5, 0, 0))
    sim = Simulation(Domain(cells=11), [electrons], Solver(algorithm=algorithm, picard_iterations=1))
    out = sim.run(6, seed=0, store_every=3)
    assert len(traces) == 1 and out.E.shape == (2, 11, 3)
