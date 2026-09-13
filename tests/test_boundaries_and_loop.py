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


class KeyLedger:
    """Stands in for ``jax.random`` in the simulation module and records every concrete
    key that is split, folded or drawn from. Inside a scan the keys are tracers and are
    not recorded; the keys handed into a scan are."""

    def __init__(self):
        self.consumed = []

    def note(self, key):
        if not isinstance(key, jax.core.Tracer):
            self.consumed.append(tuple(np.asarray(key).ravel().tolist()))

    def split(self, key, num=2):
        self.note(key)
        return random.split(key, num)

    def fold_in(self, key, data):
        self.note(key)
        return random.fold_in(key, data)

    def uniform(self, key, *args, **kwargs):
        self.note(key)
        return random.uniform(key, *args, **kwargs)

    def normal(self, key, *args, **kwargs):
        self.note(key)
        return random.normal(key, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(random, name)


@pytest.mark.parametrize("algorithm", ["explicit", "implicit"])
def test_every_random_key_is_used_once(monkeypatch, algorithm):
    """Each key a step derives goes to one consumer -- the next step, the collisions,
    the thermal wall, the sub-steps of the implicit scheme -- and is either split or
    drawn from, once. A key used twice hands two consumers the same random numbers:
    in JAX's threefry keys ``fold_in(k, 1)`` equals ``split(k)[1]``, which is how the
    thermal wall once drew the numbers the next step's collisions drew."""
    ledger = KeyLedger()
    monkeypatch.setattr(simulation_module, "random", ledger)
    monkeypatch.setattr(Simulation, "_collide", lambda self, key, x, v, *rest: (ledger.note(key), v)[1])
    electrons = Species.electrons(n=40, density=1e6, vth=(1e6, 1e6, 1e6), drift=(-2e6, 0, 0))
    domain = Domain(length=1e-3, cells=8, particle_bc=("thermal", "reflective"), field_bc="reflective")
    sim = Simulation(domain, [electrons], Solver(algorithm=algorithm, picard_iterations=1),
                     Collisions(coulomb_log=10.0))
    carry, extra = sim.initial_state(random.PRNGKey(0))
    step = sim._explicit_step if algorithm == "explicit" else sim._implicit_step
    for _ in range(3):
        carry, _ = step(carry, extra)
    assert len(ledger.consumed) > 6
    assert len(set(ledger.consumed)) == len(ledger.consumed)


@pytest.mark.parametrize("switch", [{"filter_passes": 2}, {"field_solver": "gauss"}])
def test_the_implicit_scheme_refuses_the_switches_it_would_ignore(switch):
    """A filter and the Gauss solve both belong to the explicit scheme. The implicit one
    used to accept and silently skip them; it now says so when it is built."""
    electrons = Species.electrons(n=10, density=1e10)
    with pytest.raises(ValueError, match="implicit"):
        Simulation(Domain(), [electrons], Solver(algorithm="implicit", **switch))
    Simulation(Domain(), [electrons], Solver(algorithm="explicit", **switch))


def test_absorbed_particles_are_parked_symmetrically_off_both_grids():
    """A particle with no weight left is parked the same distance beyond either wall,
    one and a half cells, the half-width of the spline, so that neither the centred nor
    the staggered stencil reaches back into the box from there."""
    x = jnp.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
    v = jnp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nothing = (jnp.zeros(2), jnp.zeros(2))
    parked, _, w, _ = apply_particle_bc(x, v, jnp.ones(2), jnp.ones(2), (L, L, L), (2, 2), (1.0, 1.0), nothing, DX)
    assert float(parked[0, 0]) == pytest.approx(-L / 2 - 1.5 * DX)
    assert float(parked[1, 0]) == pytest.approx(L / 2 + 1.5 * DX)
    for origin in (CENTRE0, CENTRE0 + DX / 2):
        assert float(jnp.abs(deposit(parked[:, 0], jnp.ones(2), origin, DX, CELLS, (2, 2))).max()) == 0.0


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
