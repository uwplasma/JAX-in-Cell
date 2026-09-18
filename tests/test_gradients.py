"""What the derivatives of a run are, and what they are not.

Each test names the derivative it checks. They are different things, and agreement
between two of them says nothing about a third:

1. the derivative of a **fixed discretisation and a fixed realisation** -- what
   ``jax.grad`` returns, checked against forward mode and against a central difference
   small enough not to flip a branch;
2. the derivative of a **finite-time expectation** of an observable, which a finite
   sample estimates and a fixed realisation need not;
3. a **continuum** response, which neither of the above is at a fixed particle count.

The negative control is as important as the positive ones: a functional that counts
particles has a branchwise derivative of exactly zero, and more particles do not fix it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxincell import (Domain, Simulation, Solver, Source, Species, epsilon_0, mass_electron, potential,
                       quiet_start, elementary_charge as e_charge, speed_of_light as c)
from jaxincell._core import apply_particle_bc

SIGMA = np.sqrt(1.0 * e_charge / mass_electron)
DENSITY = 1e16
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
DEBYE = SIGMA / OMEGA_PE


def box(cells=32, length=None, steps_per_plasma_period=10.0, **domain):
    length = 10 * DEBYE if length is None else length
    dt = 1.0 / (steps_per_plasma_period * OMEGA_PE)
    return Domain(length=length, cells=cells, dt_over_dx_c=dt * c / (length / cells), **domain)


# --- the advertised way of building an initial state --------------------------------------

def test_quiet_start_is_differentiable_in_the_parameters_it_takes():
    """`quiet_start` builds its quadrature nodes on the host, because nothing
    differentiates with respect to a fixed node, and scales and shifts them with ordinary
    JAX values, so the thermal speed, the drift and the box length are all live. A version
    that converted them with NumPy would return the same numbers and no derivative."""
    n = 256

    def energy(vth, drift, length):
        x, v = quiet_start(n, length, vth=(vth, 0, 0), drift=(drift, 0, 0))
        return jnp.sum(v[:, 0] ** 2) + jnp.sum(x[:, 0] ** 2)

    quantiles = np.asarray(quiet_start(n, 1.0, vth=(1.0, 0, 0))[1][:, 0])
    lattice = np.asarray(quiet_start(n, 1.0)[0][:, 0])
    vth, drift, length = 1e6, 3e5, 0.01
    gradient = jax.grad(energy, argnums=(0, 1, 2))(vth, drift, length)
    # d/dvth sum (q vth + u)^2 = 2 sum q (q vth + u), and the same for the drift and the length
    assert float(gradient[0]) == pytest.approx(2 * np.sum(quantiles * (quantiles * vth + drift)), rel=1e-12)
    assert float(gradient[1]) == pytest.approx(2 * np.sum(quantiles * vth + drift), rel=1e-10)
    assert float(gradient[2]) == pytest.approx(2 * length * np.sum(lattice ** 2), rel=1e-12)
    assert np.allclose(np.asarray(jax.jit(lambda a: quiet_start(8, 0.01, vth=(a, 0, 0))[1])(1e6)),
                       np.asarray(quiet_start(8, 0.01, vth=(1e6, 0, 0))[1]))


# --- the derivative of the discrete map ----------------------------------------------------

def sheath(reflection, capacity=6000, emit=6, cells=24):
    """A small maintained sheath whose collector returns a fraction of the electrons."""
    domain = box(cells=cells, steps_per_plasma_period=1 / 0.15,
                 particle_bc="absorbing", field_bc=("open", "absorbing"))
    electrons = Species("electrons", capacity, -1.0, mass_electron, DENSITY, (np.sqrt(2) * SIGMA, 0, 0),
                        active=capacity // 3, quiet=True, reflection=(0.0, reflection),
                        source=Source(density=DENSITY, vth=(np.sqrt(2) * SIGMA,) * 3, emit=emit, model="maxwellian"))
    ions = Species("ions", capacity, 1.0, 400 * mass_electron, DENSITY, 0.0, (0.25 * SIGMA, 0, 0),
                   active=capacity // 3, quiet=True,
                   source=Source(density=DENSITY, vth=0.0, drift=(0.25 * SIGMA, 0, 0), emit=emit, model="beam"))
    return Simulation(domain, [electrons, ions], Solver(model="electrostatic"))


def test_forward_and_reverse_mode_agree_through_a_maintained_sheath():
    """Both modes differentiate the same program, so they must give the same number; the
    check is that neither the source, the wall nor the electrode closure has a rule that
    supports only one of them."""
    def measured(r):
        out = sheath(r).run(20, seed=0, store_every=20, store_particles=False)
        return jnp.mean(potential(out)[-1])

    reverse = float(jax.jit(jax.grad(measured))(0.25))
    forward = float(jax.jit(lambda r: jax.jvp(measured, (r,), (1.0,))[1])(0.25))
    assert reverse != 0.0
    assert forward == pytest.approx(reverse, rel=1e-12)


def test_the_gradient_is_the_derivative_of_the_discrete_map_at_a_small_enough_step():
    """A central difference agrees with the gradient once the step is small enough not to
    move a particle across a wall that it did not cross before. How small that is falls as
    the run lengthens -- each absorption is a branch of the program, and a long run has
    many -- which is a property of the map, not of the implementation."""
    for steps, step_size, tolerance in ((5, 1e-3, 1e-6), (20, 1e-5, 1e-6)):
        def measured(r, steps=steps):
            out = sheath(r).run(steps, seed=0, store_every=steps, store_particles=False)
            return jnp.mean(potential(out)[-1])

        f = jax.jit(measured)
        gradient = float(jax.jit(jax.grad(measured))(0.25))
        difference = float((f(0.25 + step_size) - f(0.25 - step_size)) / (2 * step_size))
        assert difference == pytest.approx(gradient, rel=tolerance)


# --- the analytical controls -----------------------------------------------------------------

def test_a_weighted_reflection_has_the_derivative_the_algebra_says():
    """A wall that returns the fraction R of each impact sends back R w and keeps (1-R) w,
    so the outgoing weight has derivative w and the collected charge -q w. The weighting is
    deterministic, which is what makes it differentiable; a Bernoulli hit-or-miss trial of
    the same mean would have no derivative at all."""
    w, q = 3.0, -e_charge

    def collected(R):
        reflection = (jnp.zeros(1), jnp.broadcast_to(R, (1,)))
        _, _, w_out, _, (arrived, kept, _) = apply_particle_bc(
            jnp.array([[0.6, 0.0, 0.0]]), jnp.array([[1e6, 0.0, 0.0]]), jnp.full(1, w), jnp.ones(1),
            (1.0, 1.0, 1.0), (2, 2), (1.0, 1.0), reflection, 0.1)
        return jnp.sum(q * kept[1]), jnp.sum(w_out)

    (charge, outgoing), gradient = jax.jvp(collected, (0.25,), (1.0,))
    assert float(charge) == pytest.approx(0.75 * w * q) and float(outgoing) == pytest.approx(0.25 * w)
    assert float(gradient[0]) == pytest.approx(-q * w, rel=1e-12)
    assert float(gradient[1]) == pytest.approx(w, rel=1e-12)


def test_the_impact_energy_a_prescribed_field_gives_and_its_derivative():
    """A particle crossing a static prescribed field arrives with K = K_0 + q(phi_0 - phi_w),
    so in a uniform E the energy the wall records is K_0 + q E d and its derivatives are q d
    and m v_0. The plasma's own field is made negligible by a tiny density, and the energy
    comes from the wall ledger, not from a separate calculation."""
    cells, length = 32, 10 * DEBYE
    start, speed = -0.25 * length, 0.4 * SIGMA
    x = jnp.zeros((1, 3)).at[0, 0].set(start)

    def arrival(field, speed):
        external = jnp.zeros((cells, 3)).at[:, 0].set(field)
        species = Species("electrons", 1, -1.0, mass_electron, 1e-12,
                          x=x, v=jnp.zeros((1, 3)).at[0, 0].set(speed))
        domain = box(cells=cells, steps_per_plasma_period=200.0,
                     particle_bc="absorbing", field_bc=("open", "absorbing"))
        out = Simulation(domain, [species], Solver(model="electrostatic"), external_E=external).run(
            900, store_every=900, store_particles=False)
        weight = 1e-12 * length                                    # density * length / n
        return out.wall.energy_in[-1, 0, 1] / weight               # J, per physical particle

    field, distance = -400.0, length / 2 - start                   # pushes an electron to the right
    expected = 0.5 * mass_electron * speed ** 2 + (-e_charge) * field * distance
    value, gradient = jax.jvp(arrival, (field, speed), (1.0, 0.0))
    assert float(value) == pytest.approx(expected, rel=2e-3)
    assert float(gradient) == pytest.approx(-e_charge * distance, rel=2e-3)
    assert float(jax.jvp(arrival, (field, speed), (0.0, 1.0))[1]) == pytest.approx(
        mass_electron * speed, rel=2e-3)


def test_collecting_a_charge_sheet_does_not_change_the_field_behind_it():
    """The electrode closure has to move the charge from the volume to the surface and not
    count it twice. A sheet drifting into the conductor leaves the field behind it alone as
    it is collected: before, the field there is the sheet's; after, it is the wall's, and
    they are the same number. A closure that deleted the charge and did not give it to the
    wall would drop the field to zero."""
    cells, length = 40, 10 * DEBYE
    domain = box(cells=cells, steps_per_plasma_period=60.0,
                 particle_bc="absorbing", field_bc=("open", "absorbing"))
    species = Species("ions", 1, 1.0, 1.67262192369e-27, 1e-12,
                      x=jnp.zeros((1, 3)).at[0, 0].set(0.2 * length), v=jnp.zeros((1, 3)).at[0, 0].set(1e5))
    out = Simulation(domain, [species], Solver(model="electrostatic")).run(2000, store_every=20)
    behind = np.asarray(out.E[:, cells // 8, 0])                   # a face well to the left of the sheet
    weight = 1e-12 * length
    expected = -e_charge * weight / epsilon_0
    collected = np.asarray(out.wall.collected)[:, 0, 1] > 0
    assert collected.any() and not collected.all()                 # it really is collected during the run
    assert np.allclose(behind, expected, rtol=2e-12)               # and the field behind never moves
    assert float(out.E[-1, -1, 0]) == pytest.approx(expected, rel=1e-12)


# --- the negative control ------------------------------------------------------------------------

def test_a_functional_that_counts_particles_has_no_branchwise_derivative():
    """For particles spread over a segment and drifting at v, the weight a wall has taken
    by time t is a sum of steps, sum_p w_p H(x_p + v t - L). Each step is flat except where
    it jumps, so the derivative of the sum with respect to v is zero almost everywhere,
    while the expectation is n v t and its derivative is n t. Autodiff returns the first,
    exactly and correctly, and it is not the second. More particles do not help: the
    branchwise derivative of a finer staircase is still zero.

    This is the reason the source emits a fixed number of particles carrying a continuous
    weight rather than a number of particles that depends on the flux, and the reason a
    sharp count is kept out of the objective of an optimisation."""
    cells, length, steps = 32, 10 * DEBYE, 300
    domain = box(cells=cells, steps_per_plasma_period=20.0,
                 particle_bc="absorbing", field_bc=("open", "absorbing"))

    def collected(drift, n):
        x, _ = quiet_start(n, length)
        species = Species("ions", n, 1.0, 1.67262192369e-27, 1e-12,
                          x=x, v=jnp.zeros((n, 3)) + jnp.array([drift, 0.0, 0.0]))
        out = Simulation(domain, [species], Solver(model="electrostatic")).run(
            steps, store_every=steps, store_particles=False)
        return out.wall.collected[-1, 0, 1]

    drift = 0.05 * SIGMA
    elapsed = steps * domain.dt
    for n in (64, 512):
        branchwise = float(jax.grad(collected)(drift, n))
        assert branchwise == 0.0                                   # exactly, at both particle counts
        # the expectation is (n_phys/L) * v * t * L = n_phys v t, and a difference over a step
        # large enough to move many particles across the wall resolves it
        h = 0.3 * drift
        response = float((collected(drift + h, n) - collected(drift - h, n)) / (2 * h))
        assert response == pytest.approx(1e-12 * elapsed, rel=0.15)
