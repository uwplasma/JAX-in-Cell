"""The field walls, the gather, and the time loop: mirror symmetry of the wall closures,
the force on a charge sheet next to each wall, the random keys each step hands out, the
switches the implicit scheme refuses, and the bookkeeping that keeps the loop cheap."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import jaxincell._simulation as simulation_module
from jaxincell import (Collisions, Domain, Simulation, Solver, Species, elementary_charge, epsilon_0,
                       mass_electron, speed_of_light as c)
from jaxincell._core import E_x_from_rho, apply_particle_bc, current_from_continuity, deposit, wrap_positions

L, CELLS = 1.0, 32
DX = L / CELLS
CENTRE0 = -L / 2 + DX / 2
WALLS = [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2)]           # every pair Domain accepts


def midpoint_pair(**solver):
    """Uniform neutral cold pair: all three currents have an analytic midpoint response."""
    omega = 1e9
    density = epsilon_0 * mass_electron * omega**2 / (2 * elementary_charge**2)
    species = [Species(name, 8, sign, mass_electron, density, sampling="low_noise")
               for name, sign in (("electrons", -1.), ("positrons", 1.))]
    sim = Simulation(Domain(length=4*c/omega, cells=4, time_step=.04/omega), species,
                     Solver(algorithm="implicit", shape_order=5, **solver))
    state, extra = sim.initial_state(random.PRNGKey(0))
    scale = mass_electron * c * omega / elementary_charge
    field = scale * jnp.array([1e-4, -2e-4, 3e-4])
    return sim, state.replace(E=jnp.broadcast_to(field, state.E.shape)), extra, omega, scale


def test_identity_midpoint_hook_preserves_the_complete_native_return():
    sim, state, extra, omega, scale = midpoint_pair()
    native = jax.jit(lambda: sim._implicit_step(state, extra))()
    explicit_none = jax.jit(lambda: sim._implicit_step(state, extra, midpoint_fields=None))()
    identity = jax.jit(lambda: sim._implicit_step(
        state, extra, midpoint_fields=lambda E, B, J: (E, B)))()
    for expected, unchanged, hooked in zip(jax.tree.leaves(native), jax.tree.leaves(explicit_none),
                                           jax.tree.leaves(identity[:2]), strict=True):
        np.testing.assert_array_equal(unchanged, expected)
        np.testing.assert_array_equal(hooked, expected)
    E_half, B_half, E_force, B_force, J_guess = identity[2]
    np.testing.assert_allclose(E_half, .5*(state.E + native[0].E), rtol=2e-13, atol=0.)
    np.testing.assert_array_equal(E_force, E_half)
    np.testing.assert_array_equal(B_force, B_half)
    np.testing.assert_allclose(J_guess/(epsilon_0*omega*scale), native[1][5]/(epsilon_0*omega*scale),
                               rtol=0., atol=2e-13)


def test_current_midpoint_hook_matches_the_cold_response_and_its_derivative():
    """Extra force -chi dt J/(2 eps0) changes the force, not the physical Maxwell current."""
    sim, state, extra, omega, scale = midpoint_pair()
    dt = sim.domain.dt

    def step(chi):
        return sim._implicit_step(state, extra,
                                  midpoint_fields=lambda E, B, J: (E-chi*dt*J/(2*epsilon_0), B))

    result, tangent = jax.jit(lambda chi: jax.jvp(step, (chi,), (jnp.ones_like(chi),)))(jnp.array(.3))
    final, output, used = result
    denominator = 1 + 1.3*(omega*dt/2)**2
    force = np.asarray(state.E) / denominator
    expected = epsilon_0*omega**2*dt/2 * force
    current_scale = epsilon_0*omega*scale
    # Fixed field/current/velocity units keep tiny cold deposits from setting their own tolerance.
    np.testing.assert_allclose(output[5]/current_scale, expected/current_scale, rtol=0., atol=2e-13)
    np.testing.assert_allclose(final.E/scale, (state.E-dt*expected/epsilon_0)/scale, rtol=0., atol=2e-13)
    np.testing.assert_allclose(final.u/c, np.asarray(state.qm)[:, None]*dt*force[0]/c, rtol=0., atol=2e-13)
    np.testing.assert_allclose(used[2]/scale, force/scale, rtol=0., atol=2e-13)
    np.testing.assert_allclose(used[4]/current_scale, output[5]/current_scale, rtol=0., atol=2e-13)
    derivative = -expected*(omega*dt/2)**2/denominator
    np.testing.assert_allclose(tangent[1][5]/current_scale, derivative/current_scale, rtol=0., atol=2e-13)


@pytest.mark.parametrize("iterations,valid", [(8, True), (1, False)])
def test_midpoint_hook_checks_the_coupled_replay_and_keeps_last_used_fields(iterations, valid):
    sim, state, extra, _, _ = midpoint_pair(picard_iterations=iterations, picard_tolerance=1e-8)

    def callback(E, B, J):
        return E-.3*sim.domain.dt*J/(2*epsilon_0), B

    result = jax.jit(lambda: sim._implicit_step(state, extra, midpoint_fields=callback))()
    assert bool(jnp.isfinite(result[0].E).all()) == valid
    assert bool(jnp.isnan(result[0].E).all()) == (not valid)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(result[2]))
    if not valid:
        np.testing.assert_array_equal(result[2][0], state.E)
        np.testing.assert_array_equal(result[2][4], jnp.zeros_like(state.E))
        assert float(jnp.max(jnp.abs(result[1][5]))) > 0.


@pytest.mark.parametrize("component", [0, 1])
def test_midpoint_hook_requires_both_grid_field_shapes(component):
    sim, state, extra, _, _ = midpoint_pair()

    def wrong_shape(E, B, J):
        fields = [E, B]
        fields[component] = fields[component][:, :2]
        return tuple(fields)

    with pytest.raises(ValueError, match="grid field shapes"):
        jax.jit(lambda: sim._implicit_step(state, extra, midpoint_fields=wrong_shape))()


@pytest.mark.parametrize("gap,vx,vy,gyro", [(.055, .2, 0., 4.), (.04, .2, 0., 10.),
                                            (.00025112075, .01, -1., .2)])
def test_magnetic_turning_before_contact_does_not_create_a_wall_impact(gap, vx, vy, gyro):
    """The held-field Boris flight stays on the pure-B circle.

    Each gap exceeds that circle's maximum excursion, including a near-grazing
    case at small gyro angle. The endpoint quadratic used to fabricate all three.
    """
    species = Species("p", 1, 1 / elementary_charge, 1., 1e-30,
                      x=jnp.array([[.5 - gap, 0., 0.]]), v=jnp.array([[vx, vy, 0.]]))
    domain = Domain(length=1., cells=16, time_step=2., particle_bc="reflective", field_bc="absorbing")
    external = jnp.zeros((16, 3)).at[:, 2].set(gyro)
    sim = Simulation(domain, [species], Solver(model="electrostatic"), external_B=external)
    st, (mass, _) = sim.initial_state(random.key(0))
    fields = jnp.array([[0., 0., 0., 0., 0., gyro]])

    def flight(x):
        return sim._wall_drift(st.key, x, st.u, st.w, st.qm, st.wall, fields, mass, 1., 0.)

    result = jax.jit(flight)(st.x)
    assert float(jnp.sum(result[-1].arrived)) == 0.
    expected = .5 - gap + (vx + gyro * vy / 2) / (1 + (gyro / 2) ** 2)
    assert float(result[0][0, 0]) == pytest.approx(expected, abs=1e-14)
    derivative = jax.jvp(flight, (st.x,), (jnp.ones_like(st.x),))[1]
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(derivative))


def test_arbitrary_held_fields_give_the_partial_boris_contact_and_its_implicit_derivative():
    rng, n = np.random.default_rng(123), 32
    side = jnp.where(jnp.arange(n) % 2, -1., 1.)
    velocity = jnp.asarray(rng.normal(0., .2, (n, 3))).at[:, 0].set(side * 1.5)
    fields = jnp.asarray(rng.normal(0., .2, (n, 6)))
    qm = jnp.asarray(rng.uniform(-2., 2., n))
    expected_time = jnp.asarray(rng.uniform(.07, .21, n))
    species = Species("p", n, 0., 1., 1.)
    sim = Simulation(Domain(length=1., cells=8, time_step=.6, particle_bc="absorbing", field_bc="absorbing"),
                     [species], Solver(model="electrostatic"), external_B=jnp.zeros((8, 3)))

    def displacement(t, F):
        pushed = sim._accelerate(velocity, F, qm, t[:, None])
        return t * sim._mean_velocity(velocity, pushed)[:, 0]

    x = jnp.zeros((n, 3)).at[:, 0].set(side / 2 - displacement(expected_time, fields))

    def contact(F):
        return sim._wall_contact(x, velocity, F, qm, .3)

    _, hit, time, wall, unresolved = jax.jit(contact)(fields)
    np.testing.assert_array_equal(hit, np.ones(n, bool))
    np.testing.assert_array_equal(wall, np.where(np.asarray(side) < 0, 0, 1))
    assert not np.any(unresolved)
    np.testing.assert_allclose(time, expected_time, rtol=1e-12, atol=0.)
    tangent = jnp.zeros_like(fields).at[:, 0].set(1.)
    slope = jax.jvp(lambda t: displacement(t, fields), (expected_time,), (jnp.ones(n),))[1]
    response = jax.jvp(lambda F: displacement(expected_time, F), (fields,), (tangent,))[1]
    derivative = jax.jvp(lambda F: contact(F)[2], (fields,), (tangent,))[1]
    np.testing.assert_allclose(derivative, -response / slope, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("gyro,gap", [(4., .01), (10., .015)])
def test_magnetic_impact_uses_the_partial_boris_circle_and_retains_its_gradient(gyro, gap):
    weight, speed = 1e-30, .2
    species = Species("p", 1, 1 / elementary_charge, 1., weight,
                      x=jnp.array([[.5 - gap, 0., 0.]]), v=jnp.array([[speed, 0., 0.]]))
    domain = Domain(length=1., cells=16, time_step=2., particle_bc="reflective", field_bc="absorbing")
    sim = Simulation(domain, [species], Solver(model="electrostatic"), external_B=jnp.zeros((16, 3)))
    st, (mass, _) = sim.initial_state(random.key(0))

    def flight(v, B):
        fields = jnp.array([[0., 0., 0., 0., 0., B]])
        return sim._wall_drift(st.key, st.x, jnp.array([[v, 0., 0.]]), st.w, st.qm,
                               st.wall, fields, mass, 1., 0.)

    wall = jax.jit(flight)(speed, gyro)[-1]
    assert float(wall.arrived[0, 1]) / weight == pytest.approx(1., rel=1e-14)
    assert float(wall.momentum[0, 1, 0]) / (2 * weight) == pytest.approx(np.sqrt(speed ** 2 - (gyro * gap) ** 2),
                                                                         rel=1e-12)

    def energy(v, B):
        return flight(v, B)[-1].energy_in[0, 1] / weight

    derivative = jax.jit(jax.grad(energy, argnums=(0, 1)))(speed, gyro)
    assert float(derivative[0]) == pytest.approx(speed, rel=1e-12)
    assert float(derivative[1]) == pytest.approx(0., abs=1e-14)

    def normal(v, B):
        return flight(v, B)[-1].momentum[0, 1, 0] / (2 * weight)

    derivative = jax.jit(jax.grad(normal, argnums=(0, 1)))(speed, gyro)
    incoming = np.sqrt(speed ** 2 - (gyro * gap) ** 2)
    assert float(derivative[0]) == pytest.approx(speed / incoming, rel=1e-11)
    assert float(derivative[1]) == pytest.approx(-gyro * gap ** 2 / incoming, rel=1e-11)


@pytest.mark.parametrize("side", [-1, 1])
def test_a_particle_resting_on_an_absorbing_wall_is_collected_before_outward_acceleration(side):
    weight = 1e-30
    species = Species("p", 1, 1 / elementary_charge, 1., weight,
                      x=jnp.array([[side / 2, 0., 0.]]), v=jnp.zeros((1, 3)))
    domain = Domain(length=1., cells=16, time_step=.1, particle_bc="absorbing", field_bc="absorbing")
    external = jnp.zeros((16, 3)).at[:, 0].set(side)
    out = Simulation(domain, [species], Solver(model="electrostatic"), external_E=external).run(1).validate()
    assert float(out.state.wall.collected[0, 0 if side < 0 else 1]) / weight == pytest.approx(1., rel=1e-14)
    assert float(jnp.sum(out.state.wall.energy_in)) == 0.
    assert float(jnp.sum(out.state.w)) == 0.


def test_a_returned_accelerated_flight_cannot_hide_a_second_contact_inside_its_endpoint():
    """After reflection the parabola dips past the opposite wall and returns inside."""
    species = Species("p", 1, 1 / elementary_charge, 1., 1e-30,
                      x=jnp.array([[.4, 0., 0.]]), v=jnp.array([[5.8, 0., 0.]]))
    domain = Domain(length=1., cells=16, time_step=2., particle_bc="reflective", field_bc="absorbing")
    sim = Simulation(domain, [species], Solver(model="electrostatic"))
    st, (mass, _) = sim.initial_state(random.key(0))
    fields = jnp.array([[12., 0., 0., 0., 0., 0.]])
    result = sim._wall_drift(st.key, st.x, st.u, st.w, st.qm, st.wall, fields, mass, 1., 0.)
    assert not np.isfinite(float(result[0][0, 0]))


@pytest.mark.parametrize("magnetic", [False, True])
def test_a_zero_restitution_return_at_rest_does_not_create_repeated_wall_contacts(magnetic):
    weight = 1e-30
    species = Species("p", 1, 0., 1., weight, reflection=.4,
                      x=jnp.array([[.4, 0., 0.]]), v=jnp.array([[.2, 0., 0.]]))
    domain = Domain(length=1., cells=16, time_step=2., particle_bc="absorbing", field_bc="absorbing",
                    restitution=0.)
    external = jnp.zeros((16, 3)) if magnetic else None
    out = Simulation(domain, [species], Solver(model="electrostatic"), external_B=external).run(1).validate()
    assert float(out.state.wall.arrived[0, 1]) / weight == pytest.approx(1., rel=1e-13)
    assert float(out.state.w[0]) / weight == pytest.approx(.4, rel=1e-13)
    assert float(out.state.x[0, 0]) == pytest.approx(.5, abs=1e-13)


def left_wall_value(F, s, bc):
    """F_{-1/2}, which the grid does not store, from the first cell of the difference equation."""
    return F[-1] if bc == (0, 0) else F[0] - DX * s[0]


def reflect_faces(F, F_left):
    """Faces mirrored about the box centre: face i+1/2 goes to face (N-2-i)+1/2, and the
    left wall face, which is not stored, becomes the stored right wall face."""
    return jnp.concatenate([F[:-1][::-1], F_left[None]])


@pytest.mark.parametrize("bc", WALLS)
def test_wall_closures_are_mirror_images_and_satisfy_gauss(bc):
    """For every pair of field walls, the Gauss solve and the continuity current of a
    charge distribution and of its mirror image, with the walls swapped, are mirror
    images: E_x and J_x change sign under the reflection. Both satisfy their difference
    equation in every cell, E_x vanishes at a reflective wall, two absorbing walls hold
    no potential difference across the box, and the periodic current carries the mean
    current it is given."""
    rng = np.random.default_rng(7)
    rho = jnp.asarray(rng.normal(size=CELLS) + 0.3)        # not neutral, so the closures matter
    drho = jnp.asarray(rng.normal(size=CELLS))
    solves = ((lambda r, b, sign: E_x_from_rho(r, DX, b), rho, rho / epsilon_0, 0.0),
              (lambda r, b, sign: current_from_continuity(0 * r, r, 1.0, DX, sign * 0.25, b), drho, -drho, 0.25))
    for field, data, source, mean in solves:
        F = field(data, bc, 1.0)
        s = source - jnp.mean(source) if bc in ((0, 0), (1, 1)) else source
        F_left = left_wall_value(F, s, bc)
        F_before = jnp.concatenate([F_left[None], F[:-1]])
        assert np.allclose(np.asarray((F - F_before) / DX), np.asarray(s), rtol=1e-12,
                           atol=1e-12 * float(jnp.abs(s).max()))
        G = field(data[::-1], bc[::-1], -1.0)             # the mean current reverses with the charges
        assert np.allclose(np.asarray(G), -np.asarray(reflect_faces(F, F_left)), rtol=1e-12,
                           atol=1e-12 * float(jnp.abs(F).max()))
        scale = float(jnp.abs(F).max())
        if bc[0] == 1:
            assert abs(float(F_left)) < 1e-12 * scale
        if bc[1] == 1:
            assert abs(float(F[-1])) < 1e-12 * scale
        if bc == (2, 2):
            assert abs(float(0.5 * F_left + jnp.sum(F[:-1]) + 0.5 * F[-1])) < 1e-12 * CELLS * scale
        if bc == (0, 0):
            assert abs(float(jnp.mean(F)) - mean) < 1e-12 * scale


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
    parked, _, w, _, _ = apply_particle_bc(x, v, jnp.ones(2), jnp.ones(2), (L, L, L), (2, 2), (1.0, 1.0), nothing, DX)
    assert float(parked[0, 0]) == pytest.approx(-L / 2 - 1.5 * DX)
    assert float(parked[1, 0]) == pytest.approx(L / 2 + 1.5 * DX)
    for origin in (CENTRE0, CENTRE0 + DX / 2):
        assert float(jnp.abs(deposit(parked[:, 0], jnp.ones(2), origin, DX, CELLS, (2, 2))).max()) == 0.0


def slow_ones(speed):
    """A reflection law that returns the electrons slower than about 1e6 m/s."""
    return jnp.exp(-speed ** 2 / (2 * 1e12))


@pytest.mark.parametrize("bc", [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3)])
def test_wrap_positions_is_the_position_map_of_the_particle_walls(bc):
    """wrap_positions repeats the position rules of apply_particle_bc to be cheaper. On
    particles beyond either wall and in all three coordinates, with and without weight,
    the two give identical positions."""
    rng = np.random.default_rng(3)
    n = 400
    x = jnp.asarray(rng.uniform(-0.6 * L, 0.6 * L, (n, 3)))
    w = jnp.asarray(np.where(rng.uniform(size=n) < 0.3, 0.0, 1.0))
    unchanged = (jnp.ones(n), jnp.ones(n))
    box = (L, 0.7, 0.9)
    full, _, _, _, _ = apply_particle_bc(x, jnp.zeros_like(x), w, jnp.ones(n), box, bc, (1.0, 1.0), unchanged, DX)
    assert np.array_equal(np.asarray(wrap_positions(x, w, box, bc, DX)), np.asarray(full))


def walled_simulation(**solver):
    """Electrostatic (no transverse velocity), so that a step far above the light-wave limit is stable."""
    e = Species.electrons(n=600, density=1e15, vth=(2e6, 0, 0), drift=(1e6, 0, 0), reflection=(0.0, slow_ones))
    i = Species.ions(n=600, density=1e15, electrons=e, mass_ratio=50.0, reflection=(0.0, 0.5))
    domain = Domain(length=1e-3, cells=24, dt_over_dx_c=20.0, particle_bc=("thermal", "absorbing"),
                    field_bc=("reflective", "absorbing"))
    return Simulation(domain, [e, i], Solver(filter_passes=2, **solver))


def test_the_carried_density_is_the_density_at_the_integer_time_positions():
    """The explicit step no longer deposits the density at x^n: it carries the one the
    previous step ended on, with physical integer-time wall positions and weights."""
    sim = walled_simulation()
    d = sim.domain
    out = sim.run(40, seed=1, store_every=4)
    state = out.state
    x_n = state.x
    expected = sim._smooth(deposit(x_n[:, 0], out.charge * state.w, d.grid[0], d.dx, d.cells, d.particle_bc))
    assert np.allclose(np.asarray(state.rho), np.asarray(expected),
                       rtol=1e-12, atol=1e-12 * float(jnp.abs(expected).max()))
    assert np.allclose(np.asarray(out.rho[-1]), np.asarray(state.rho), rtol=0, atol=0)


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


CODES = {0: "periodic", 1: "reflective", 2: "absorbing"}


def field_on_sheets(positions, bc, length=L, cells=CELLS):
    """E_x that sheets of electrons at rest, one pseudo-particle each, feel from their own
    fields and their images, gathered at their supplied physical positions. This tests
    the field/gather independently of the subsequent accelerated half flight. Returns
    the fields at the sheets and the surface charge sigma of one sheet."""
    n = len(positions)
    x = np.zeros((n, 3))
    x[:, 0] = positions
    density = 1e12
    electrons = Species.electrons(n=n, density=density).replace(x=x, v=np.zeros((n, 3)))
    walls = tuple(CODES[code] for code in bc)
    sim = Simulation(Domain(length=length, cells=cells, particle_bc=walls, field_bc=walls), [electrons], Solver())
    state, _ = sim.initial_state(random.key(0))
    sigma = electrons.charge_si * density * length / n
    return np.asarray(sim._fields_at(jnp.asarray(x), state.E, state.B, state.rho)[:, 0]), sigma


@pytest.mark.parametrize("fraction", [0.0, 0.2, 0.5, 0.77])
def test_a_charge_sheet_feels_no_force_of_its_own_in_a_periodic_box(fraction):
    """Gathering with the transpose of the deposit leaves a lone particle in a periodic
    box without a self-force, wherever it sits in its cell. Gathering E_x straight from
    the faces pushed it with up to 8 % of its own field."""
    field, sigma = field_on_sheets([CENTRE0 + (5 + fraction) * DX], (0, 0))
    assert abs(field[0]) < 1e-12 * abs(sigma) / epsilon_0


@pytest.mark.parametrize("bc", WALLS[1:])
@pytest.mark.parametrize("distance", [1.3 * DX, 5.4 * DX])
def test_a_charge_sheet_feels_its_images_as_the_one_dimensional_theory_says(bc, distance):
    """A sheet of charge sigma a distance a from the left wall, with its cloud inside the
    box, feels the field of its images (the 1D electrostatics of a sheet between two
    walls): between two symmetry planes, sigma/eps0 (1/2 - a/L), pushed from the nearer;
    from a symmetry plane towards a floating conductor, +-sigma/2 eps0 whatever a; between
    two short-circuited conductors, sigma/eps0 (a/L - 1/2), pulled to the nearer."""
    field, sigma = field_on_sheets([-L / 2 + distance], bc)
    expected = {(1, 1): 0.5 - distance / L, (1, 2): 0.5, (2, 1): -0.5, (2, 2): distance / L - 0.5}[bc]
    assert field[0] == pytest.approx(expected * sigma / epsilon_0, rel=1e-10)


@pytest.mark.parametrize("bc", WALLS)
@pytest.mark.parametrize("distance", [0.0, 0.4 * DX, 0.9 * DX, 1.6 * DX])
def test_the_force_near_a_wall_is_the_mirror_image_of_the_force_near_the_opposite_wall(bc, distance):
    """A sheet a distance a from the left wall and one a from the right wall, with the
    walls swapped, feel opposite fields -- also inside the last cell, where the cloud
    reaches beyond the wall and the value the gather assumes there matters."""
    left, _ = field_on_sheets([-L / 2 + distance], bc)
    right, sigma = field_on_sheets([L / 2 - distance], bc[::-1])
    assert abs(left[0] + right[0]) < 1e-11 * abs(sigma) / epsilon_0


@pytest.mark.parametrize("distance", [0.1 * DX, 0.7 * DX, 2.2 * DX])
def test_a_reflective_box_gathers_like_the_periodic_box_twice_as_long_with_the_images(distance):
    """Two reflective walls are two symmetry planes: the box is half of a periodic box
    twice as long holding each particle and its mirror image. The field on a sheet is the
    same in both, including within a cell of the wall, where the gather reads the mirror
    image of the first centre with the parity of E_x."""
    field, _ = field_on_sheets([-L / 2 + distance], (1, 1))
    doubled, sigma = field_on_sheets([distance, -distance], (0, 0), length=2 * L, cells=2 * CELLS)
    assert abs(field[0] - doubled[0]) < 1e-11 * abs(sigma) / epsilon_0


def lone_electron(drift, relativistic, length=1e-2):
    """One electron in a periodic box, so tenuous that its own field changes nothing."""
    electrons = Species.electrons(n=1, density=1.0, drift=drift, sampling="low_noise")
    return Simulation(Domain(length=length, cells=8), [electrons], Solver(relativistic=relativistic))


def test_speeds_past_light_are_brought_back_only_in_a_relativistic_run():
    """A relativistic run cannot represent a speed at or above c, so a drawn velocity
    past sqrt(1 - 1e-5) c is scaled back to that speed along its own direction, and not
    per component, which once let (0.9c, 0.9c) through as 1.27c. A Newtonian run has no
    such limit: its velocities, and the derivatives with respect to them, are left alone."""
    out = lone_electron((0.9 * c, 0.9 * c, 0.0), relativistic=True).run(1, seed=0)
    v = np.asarray(out.v[0, 0])
    assert np.sum(v ** 2) / c ** 2 == pytest.approx(1 - 1e-5, rel=1e-12)
    assert v[0] == pytest.approx(v[1], rel=1e-12) and v[2] == 0.0

    sim = lone_electron((1.5 * c, 0.0, 0.0), relativistic=False)
    assert float(sim.run(1, seed=0).v[0, 0, 0]) == pytest.approx(1.5 * c, rel=1e-12)
    electrons = sim.species[0]
    slope = jax.grad(lambda d: sim.replace(species=(electrons.replace(drift=(d, 0.0, 0.0)),)).run(1, seed=0).v[0, 0, 0])
    assert float(slope(1.5 * c)) == pytest.approx(1.0, rel=1e-9)


F32_RUN = """
import json, sys
import numpy as np
from jaxincell import Domain, Simulation, Solver, Species, speed_of_light as c
gammas = {}
for gamma in (100.0, 300.0):
    speed = c * np.sqrt(1 - 1 / gamma ** 2)
    electrons = Species.electrons(n=1, density=1.0, drift=(speed, 0.0, 0.0), sampling="low_noise")
    sim = Simulation(Domain(length=1e-2, cells=8, dt_over_dx_c=0.5), [electrons], Solver(relativistic=True))
    v = np.asarray(sim.run(1000, seed=0, store_every=100).v[:, 0, 0], dtype=np.float64)
    gammas[gamma] = list(1 / np.sqrt(1 - (v / c) ** 2))
print(json.dumps({"dtype": str(sim.run(1, seed=0).v.dtype), "gammas": gammas}))
"""


def test_a_relativistic_particle_keeps_its_lorentz_factor_in_single_precision():
    """A relativistic run carries u = gamma v, not v. Converting v to u and back every
    step loses a fraction gamma^2 epsilon of gamma each time, which in single precision
    walked a particle at gamma = 1000 to 1423 in a thousand steps with no field. Carried,
    u does not change at all when there is no force, so the stored velocities are
    **identical** from step to step -- which is the claim, and it is exact.

    What gamma comes out as is a different matter and is not the code's to fix: a single
    ulp of a single-precision velocity is worth 296.4 to 302.1 in gamma at gamma = 300,
    because gamma depends on v/c through 1 - (v/c)^2 and that is a difference of two
    numbers close to one. The bound below is that span, worked out here rather than
    guessed -- a fixed 0.5 % is narrower than one representable step and passes or fails
    with the rounding of the platform, which is what it did."""
    import json
    import os
    import subprocess
    import sys

    env = dict(os.environ, JAX_ENABLE_X64="0")
    result = subprocess.run([sys.executable, "-c", F32_RUN], env=env, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["dtype"] == "float32"
    for gamma, history in report["gammas"].items():
        gamma, history = float(gamma), np.array(history)
        assert np.ptp(history) == 0.0, (gamma, history)     # carried u does not drift at all
        speed = np.float32(c * np.sqrt(1 - 1 / gamma ** 2))
        span = sorted(1 / np.sqrt(1 - (np.float64(v) / c) ** 2)
                      for v in (np.nextafter(speed, np.float32(0)), np.nextafter(speed, np.float32(c))))
        assert span[0] <= history[0] <= span[1], (gamma, history[0], span)
        assert span[1] / span[0] - 1 < 0.03                 # and that span is 2 % at gamma = 300


def test_collisions_without_a_coulomb_logarithm_need_a_negative_species():
    """The default Coulomb logarithm is the one of the lightest negatively charged
    species; without one, Collisions() has to be given coulomb_log, and says so when the
    simulation is built rather than failing inside the loop."""
    ions = Species.ions(n=10, density=1e20, vth=(1e4, 0, 0))
    with pytest.raises(ValueError, match="coulomb_log"):
        Simulation(Domain(), [ions], Solver(), Collisions())
    Simulation(Domain(), [ions], Solver(), Collisions(coulomb_log=10.0))
    Simulation(Domain(), [ions, Species.electrons(n=10, density=1e20)], Solver(), Collisions())
    # a traced charge has no sign until the program runs, so construction leaves it alone
    built = jax.jit(lambda q: Simulation(Domain(), [ions.replace(charge=q)], Solver(), Collisions()).species[0].charge)
    assert float(built(elementary_charge)) == elementary_charge


def test_simulation_and_run_refuse_bad_arguments_with_value_errors():
    """Argument checks raise ValueError, which ``python -O`` keeps, where assert would vanish."""
    electrons = Species.electrons(n=10, density=1e10)
    with pytest.raises(ValueError, match="species"):
        Simulation(Domain(), [], Solver())
    with pytest.raises(ValueError, match="distinct"):
        Simulation(Domain(), [electrons, electrons], Solver())
    sim = Simulation(Domain(), [electrons], Solver())
    for steps, store_every in ((10, 3), (10, 0)):
        with pytest.raises(ValueError, match="store_every"):
            sim.run(steps, store_every=store_every)


@pytest.mark.parametrize("algorithm", ["explicit", "implicit"])
@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("restitution", [0., .5, 1.])
def test_inelastic_ballistic_wall_trajectory(algorithm, side, restitution):
    """A neutral marker reaches a stationary wall and drifts at -e*v afterwards."""
    domain = Domain(length=1., cells=8, dt_over_dx_c=8*c, particle_bc="reflective",
                    field_bc="reflective", restitution=restitution)
    positions = jnp.array([[side*s, 0., 0.] for s in (.35, .4, .45)])
    velocities = jnp.tile(jnp.array([side*.2, .03, -.04]), (3, 1))
    species = Species("neutral", 3, 0., 1., 1., x=positions, v=velocities)
    solver = Solver(algorithm=algorithm, model="electrostatic", filter_passes=0, substeps=1)
    simulation = Simulation(domain, (species,), solver)
    out = simulation.run(1).validate()
    expected = side*(.5-restitution*(np.abs(np.asarray(positions[:, 0]))+.2-.5))
    np.testing.assert_allclose(out.x[0, :, 0], expected, atol=2e-16)
    np.testing.assert_allclose(out.v[0, :, 0], -side*.2*restitution, atol=2e-16)
    np.testing.assert_allclose(out.v[0, :, 1:], velocities[:, 1:], atol=2e-16)
    wall = 0 if side < 0 else 1
    assert float((out.wall.energy_in-out.wall.energy_out)[0, 0, wall]) == pytest.approx(
        .5*.2**2*(1-restitution**2), abs=2e-16)


@pytest.mark.parametrize("side", [-1., 1.])
def test_an_outgoing_marker_exactly_on_the_wall_hits_once(side):
    x = jnp.array([[side*.5, 0., 0.]])
    v = jnp.array([[side*.2, 0., 0.]])
    ones = jnp.ones(1)
    _, reflected, _, _, hits = apply_particle_bc(x, v, ones, ones, (1., 1., 1.),
                                                 (1, 1), (.5, .5), (ones, ones), .1)
    assert float(jnp.sum(hits[0])) == 1.
    assert float(reflected[0, 0]) == -side*.1


def test_a_segment_with_unresolved_wall_crossings_cannot_look_valid():
    domain = Domain(length=1., cells=8, dt_over_dx_c=8*c, particle_bc="reflective", field_bc="reflective")
    species = Species("neutral", 1, 0., 1., 1., x=jnp.array([[.4, 0., 0.]]), v=jnp.array([[3., 0., 0.]]))
    simulation = Simulation(domain, (species,), Solver(model="electrostatic", filter_passes=0))
    out = simulation.run(2, store_every=2, store_particles=False)
    with pytest.raises(RuntimeError, match="one wall crossing"):
        out.validate()


@pytest.mark.parametrize("algorithm", ["explicit", "implicit"])
@pytest.mark.parametrize("side", [-1., 1.])
def test_elastic_wall_records_both_energies_at_the_crossing(algorithm, side):
    domain = Domain(length=1., cells=8, dt_over_dx_c=8*c, particle_bc="reflective",
                    field_bc="reflective")
    species = Species("probe", 1, .01, 1., 1e-12, x=jnp.array([[side*.35, 0., 0.]]),
                      v=jnp.array([[side*.2, .03, 0.]]))
    simulation = Simulation(domain, (species,), Solver(algorithm=algorithm, model="electrostatic", substeps=1),
                            external_E=jnp.tile(jnp.array([side*.1, 0., 0.]), (8, 1)))
    output = simulation.run(1)
    np.testing.assert_allclose(output.wall.energy_in, output.wall.energy_out, rtol=1e-13, atol=0)


def test_thermal_wall_exact_endpoint_redraws_and_records_the_same_returned_energy():
    domain = Domain(length=1., cells=8, dt_over_dx_c=8*c, particle_bc="thermal", field_bc="reflective")
    species = Species("neutral", 1, 0., 1., 1., vth=(.05, .02, .02),
                      x=jnp.array([[.3, 0., 0.]]), v=jnp.array([[.2, 0., 0.]]))
    output = Simulation(domain, (species,), Solver(model="electrostatic")).run(1, seed=1)
    assert float(output.v[0, 0, 0]) < 0
    assert float(output.v[0, 0, 0]) != -.2
    assert float(output.wall.energy_out[0, 0, 1]) == pytest.approx(
        float(jnp.sum(output.v[0, 0]**2)/2), rel=1e-13)


@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("boundary,reflection", [("reflective", 1.), ("absorbing", 0.),
                                                 ("absorbing", .4), ("thermal", 1.)])
def test_explicit_wall_events_belong_to_the_output_time(side, boundary, reflection):
    """Markers hit after, exactly at, and before t=1; no future event is consumed."""
    domain = Domain(length=1., cells=8, time_step=1., particle_bc=boundary, field_bc="reflective",
                    length_y=1., length_z=1.)
    x = jnp.array([[side*s, 0., 0.] for s in (.25, .3, .35)])
    velocity = jnp.tile(jnp.array([side*.2, .03, -.04]), (3, 1))
    species = Species("neutral", 3, 0., 1., 3., vth=(.05, .02, .02), reflection=reflection, x=x, v=velocity)
    sim = Simulation(domain, [species], Solver(model="electrostatic", filter_passes=0))
    initial, _ = sim.initial_state(random.PRNGKey(1))
    np.testing.assert_array_equal(initial.x, x)
    np.testing.assert_array_equal(initial.u, velocity)
    assert float(jnp.sum(initial.wall.arrived)) == 0
    out = sim.run(1, seed=1).validate()
    wall = 0 if side < 0 else 1
    assert float(out.t[0]) == 1.
    assert float(out.wall.arrived[0, 0, wall]) == 2.
    assert float(out.wall.collected[0, 0, wall]) == pytest.approx(2*(1-reflection), abs=1e-16)
    np.testing.assert_allclose(out.x[0, 0], [side*.45, .03, -.04], atol=1e-16)
    np.testing.assert_array_equal(out.v[0, 0], velocity[0])
    np.testing.assert_array_equal(out.state.x, out.x[-1])
    np.testing.assert_array_equal(out.state.u, out.v[-1])
    np.testing.assert_allclose(out.weight[0], [1., reflection, reflection], atol=1e-16)
    if reflection:
        # The endpoint particle has no return flight; the earlier one has 1/4 step.
        np.testing.assert_allclose(out.x[0, 1], [side*.5, .03, -.04], atol=1e-16)
        crossing = np.array([side*.5, .0225, -.03])
        np.testing.assert_allclose(out.x[0, 2], crossing + .25*np.asarray(out.v[0, 2]), atol=1e-16)
        assert side*float(out.v[0, 1, 0]) < 0
        incoming = np.sum(np.asarray(velocity[1:])**2)/2
        returned = reflection*np.sum(np.asarray(out.v[0, 1:])**2)/2
        assert float(out.wall.energy_in[0, 0, wall]) == pytest.approx(incoming, rel=1e-14)
        assert float(out.wall.energy_out[0, 0, wall]) == pytest.approx(returned, rel=1e-14)
    else:
        np.testing.assert_array_equal(out.v[0, 1:], np.zeros((2, 3)))
        np.testing.assert_array_equal(out.state.qm[1:], np.zeros(2))


def test_a_thermal_future_half_step_consumes_neither_a_draw_nor_a_wall_budget():
    domain = Domain(length=1., cells=8, time_step=1., particle_bc="thermal", field_bc="reflective",
                    length_y=1., length_z=1.)
    species = Species("neutral", 1, 0., 1., 1., vth=(.05, .02, .02),
                      x=jnp.array([[.2, 0., 0.]]), v=jnp.array([[.2, 0., 0.]]))
    sim = Simulation(domain, [species], Solver(model="electrostatic"))
    out = sim.run(2, seed=1).validate()
    assert float(out.v[0, 0, 0]) == .2
    assert float(out.wall.arrived[0, 0, 1]) == 0
    assert float(out.wall.arrived[1, 0, 1]) == 1
    assert float(out.v[1, 0, 0]) < 0
    np.testing.assert_allclose(out.x[1, 0], [.5, 0., 0.] + .5*np.asarray(out.v[1, 0]), atol=1e-16)


@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("start", [.4, .45])
def test_a_first_half_thermal_return_flies_for_the_physical_remaining_time(side, start):
    domain = Domain(length=1., cells=8, time_step=1., particle_bc="thermal", field_bc="reflective",
                    length_y=1., length_z=1.)
    species = Species("neutral", 1, 0., 1., 1., vth=(.05, .02, .02),
                      x=jnp.array([[side*start, 0., 0.]]), v=jnp.array([[side*.2, 0., 0.]]))
    out = Simulation(domain, [species], Solver(model="electrostatic")).run(1, seed=1).validate()
    remaining = 1-(.5-start)/.2
    np.testing.assert_allclose(out.x[0, 0], [side*.5, 0., 0.] + remaining*np.asarray(out.v[0, 0]), atol=1e-16)
    assert float(jnp.sum(out.wall.arrived)) == 1.


@pytest.mark.parametrize("side", [-1., 1.])
def test_relativistic_restitution_uses_the_returned_speed_for_the_remaining_flight(side):
    """Scaling proper momentum by e does not scale velocity by e at high gamma."""
    domain = Domain(length=1., cells=8, time_step=.25/c, particle_bc="reflective",
                    field_bc="reflective", restitution=.5)
    species = Species("neutral", 1, 0., 1., 1., x=jnp.array([[side*.4, 0., 0.]]),
                      v=jnp.array([[side*.8*c, 0., 0.]]))
    out = Simulation(domain, [species], Solver(model="electrostatic", relativistic=True)).run(1).validate()
    gamma = 1/np.sqrt(1-.8**2)
    u = -.5*side*gamma*.8*c
    v = u/np.sqrt(1+(u/c)**2)
    np.testing.assert_allclose(out.x[0, 0, 0], side*.5 + (.25/c-.1/(.8*c))*v, atol=1e-16)
    assert float(out.v[0, 0, 0]) == pytest.approx(v, rel=1e-14)


def test_explicit_wall_collisions_see_the_physical_endpoint_and_post_wall_weights(monkeypatch):
    domain = Domain(length=1., cells=8, time_step=1., particle_bc="absorbing", field_bc="reflective",
                    length_y=1., length_z=1.)
    species = Species("neutral", 2, 0., 1., 2., x=jnp.array([[.25, 0., 0.], [.35, 0., 0.]]),
                      v=jnp.array([[.2, 0., 0.], [.2, 0., 0.]]))
    observed = []

    def scatter(self, key, x, u, w, *rest):
        observed.append((np.asarray(x), np.asarray(w)))
        return u.at[0, 1].set(.1)

    monkeypatch.setattr(Simulation, "_collide_momenta", scatter)
    sim = Simulation(domain, [species], Solver(model="electrostatic"))
    initial, extra = sim.initial_state(random.PRNGKey(0))
    state, values = sim._explicit_step(initial, extra)
    assert observed[0][0][0, 0] == pytest.approx(.45, abs=1e-16)
    np.testing.assert_array_equal(observed[0][1], [1., 0.])
    assert float(state.u[0, 1]) == .1
    assert float(values[0][0, 1]) == 0.
