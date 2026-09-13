"""The configuration objects as pytrees and as user input, the input file, and
what leaves the package: diagnostics, openPMD, the movie writer, the version."""
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxincell import Domain, Simulation, Solver, Species
from jaxincell import mass_electron, mass_proton

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _law(speed):
    """A reflection law defined once, so that every run below shares one program."""
    return jnp.exp(-speed / 2e6)


def _wall_objective(right):
    """Field energy of a short run whose electrons stream into two absorbing walls,
    the left one following ``_law`` and the right one returning ``right``."""
    electrons = Species.electrons(n=200, density=1e15, vth=(1e6, 0.0, 0.0), drift=(2e6, 0.0, 0.0),
                                  quiet=True, reflection=(_law, right))
    domain = Domain(length=1e-3, cells=16, dt_over_dx_c=40.0, particle_bc="absorbing", field_bc="absorbing")
    return jnp.mean(Simulation(domain, [electrons], Solver()).run(20, seed=0).E[:, :, 0] ** 2)


def test_a_number_beside_a_reflection_law_is_a_differentiable_leaf():
    """In ``reflection=(law, r)`` the function is static and ``r`` is a leaf: it
    changes without recompiling, ``jax.grad`` sees it, and a traced ``r`` never
    reaches the compiled program's cache key."""
    species = Species.electrons(n=10, density=1e17, reflection=(_law, 0.3))
    leaves = jax.tree_util.tree_leaves(species)
    assert 0.3 in leaves and not any(callable(leaf) for leaf in leaves)
    other = species.replace(reflection=(_law, 0.4))
    assert jax.tree_util.tree_structure(species) == jax.tree_util.tree_structure(other)
    assert jax.tree_util.tree_map(lambda leaf: leaf, species).reflection == (_law, 0.3)
    only_law = Species.electrons(n=10, density=1e17, reflection=_law)
    assert only_law.reflection == (_law, _law) and jax.tree_util.tree_map(lambda leaf: leaf, only_law) == only_law

    from jaxincell._config import pytree_dataclass

    @pytree_dataclass(static=("label",))
    class Probe:                                   # a bare function field, which Species never stores
        label: str
        law: object
        pair: tuple

    probe = Probe("p", _law, (_law, 0.5))
    assert jax.tree_util.tree_leaves(probe) == [0.5]
    assert jax.tree_util.tree_map(lambda leaf: 2 * leaf, probe) == Probe("p", _law, (_law, 1.0))

    gradient = float(jax.grad(_wall_objective)(0.3))
    h = 1e-3
    fd = float((_wall_objective(0.3 + h) - _wall_objective(0.3 - h)) / (2 * h))
    assert fd != 0 and abs(gradient - fd) < 1e-4 * abs(fd)
    compiled = jax.jit(_wall_objective)
    first, second = float(compiled(0.3)), float(compiled(0.4))      # no leaked tracer in either call
    assert np.isfinite(first) and first != second and float(_wall_objective(0.3)) == pytest.approx(first, rel=1e-12)


def test_thermal_speed_and_drift_accept_arrays():
    """An array of three components is the three components, whichever library
    made it, and gives the same pytree as the tuple. A bare number is the x
    component alone, the direction the grid resolves."""
    spelled = Species.electrons(n=4, density=1.0, vth=(1.0, 2.0, 3.0), drift=(4.0, 5.0, 6.0))
    for array in (np.array, jnp.array):
        species = Species.electrons(n=4, density=1.0, vth=array([1.0, 2.0, 3.0]), drift=array([4.0, 5.0, 6.0]))
        assert [float(u) for u in species.vth] == [1.0, 2.0, 3.0]
        assert [float(u) for u in species.drift] == [4.0, 5.0, 6.0]
        assert jax.tree_util.tree_structure(species) == jax.tree_util.tree_structure(spelled)
    assert Species.electrons(n=4, density=1.0, vth=np.float64(2.0)).vth == (2.0, 0.0, 0.0)
    for bad in ((1.0, 2.0), np.ones(2), np.ones(4)):
        with pytest.raises(ValueError, match="vth takes a number"):
            Species.electrons(n=4, density=1.0, vth=bad)


def test_ions_derived_from_electrons_are_differentiable_in_the_electron_thermal_speed():
    def ion_vth(u):
        electrons = Species.electrons(n=4, density=1.0, vth=(u, 0.0, 0.0))
        return Species.ions(n=4, density=1.0, electrons=electrons, temperature_ratio=0.25).vth[0]

    assert float(jax.grad(ion_vth)(1e6)) == pytest.approx(np.sqrt(0.25 * mass_electron / mass_proton), rel=1e-12)
    with pytest.raises(TypeError, match="needs vth or the electron species"):
        Species.ions(n=4, density=1.0)


@pytest.mark.parametrize("build, message", [
    (lambda: Species.electrons(n=0, density=1.0), "at least one particle"),
    (lambda: Species.electrons(n=4, density=1.0, x=np.zeros((3, 3))), "x must have shape"),
    (lambda: Species.electrons(n=4, density=1.0, v=np.zeros(4)), "v must have shape"),
    (lambda: Species.electrons(n=4, density=1.0, reflection=(0.1, 0.2, 0.3)), "reflection takes"),
    (lambda: Domain(restitution=-0.5), "restitution takes a number in"),
    (lambda: Domain(particle_bc="sticky"), "particle_bc takes one of"),
    (lambda: Domain(field_bc=("periodic", "absorbing", "absorbing")), "field_bc takes one of"),
    (lambda: Domain(particle_bc=("periodic", "absorbing")), "periodic partner"),
    (lambda: Domain(cells=2), "at least four cells"),
    (lambda: Domain(field_bc="thermal"), "thermal wall"),
    (lambda: Solver(algorithm="leapfrog"), "algorithm is"),
    (lambda: Solver(field_solver="poisson"), "field_solver is"),
    (lambda: Solver(substeps=0), "at least one"),
])
def test_invalid_input_raises_a_value_error_that_python_O_keeps(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_tree_operations_stack_ensembles_and_build_in_axes():
    """JAX rebuilds a species from its leaves without re-checking them, so leaves
    stacked into an ensemble keep their leading axis, and a template of ``None``
    with integer axes stays a template instead of becoming a species of floats.
    Together they give a ``vmap`` over an ensemble of initial conditions."""
    n, length = 64, 1e-2
    domain = Domain(length=length, cells=16)
    lattice = -length / 2 + (np.arange(n) + 0.5) * length / n
    members = [Species.electrons(n=n, density=density, v=np.zeros((n, 3)), x=np.stack(
                   [lattice + amplitude * np.sin(2 * np.pi * lattice / length), 0 * lattice, 0 * lattice], axis=1))
               for density, amplitude in ((1e14, 1e-5), (2e14, -2e-5))]
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *members)
    assert stacked.x.shape == (2, n, 3) and stacked.density.shape == (2,)
    assert stacked.replace(v=jnp.zeros((2, n, 3))).v.shape == (2, n, 3)

    # only what varies is stacked; the template says which leaves carry the axis
    varying = members[0].replace(density=stacked.density, x=stacked.x)
    axes = jax.tree.map(lambda _: None, members[0]).replace(density=0, x=0)
    assert axes.density == 0 and type(axes.density) is int and axes.x == 0 and axes.vth == (None,) * 3
    assert jax.tree.map(lambda _: None, domain).replace(length=0).length == 0

    def field(species):
        return Simulation(domain, [species]).run(4, seed=0).E[-1, :, 0]

    for batched in (jax.vmap(field, in_axes=(axes,))(varying), jax.vmap(field)(stacked)):
        for member, row in zip(members, batched):
            expected = np.asarray(jax.jit(field)(member))
            assert np.allclose(np.asarray(row), expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())

    gradient = jax.grad(lambda x: jnp.sum(field(members[0].replace(x=x)) ** 2))(jnp.asarray(members[0].x))
    assert gradient.shape == (n, 3) and np.isfinite(np.asarray(gradient)).all() and float(jnp.abs(gradient).max()) > 0
