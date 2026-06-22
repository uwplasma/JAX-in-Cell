"""
    Someone needs to look through these tests and make sure they cover the
    intended cases, and that the expected results are correct.
    
    They were written using Codex and seem reasonable to me,
    but I haven't gone through the functions being tested myself to verify
    the intended behavior.
"""

import pytest
import jax.numpy as jnp
from numpy.testing import assert_allclose

from jaxincell._filters import filter_scalar_field, filter_vector_field
from jaxincell._sources import (
    calculate_charge_density,
    charge_density_BCs,
    current_density,
    current_density_periodic_CN,
    get_S2_weights_and_indices_periodic_CN,
    single_particle_charge_density,
)


def test_get_S2_weights_and_indices_periodic_CN_wraps_and_normalizes():
    """Test jaxincell._sources.get_S2_weights_and_indices_periodic_CN.

    Cases covered:
    - particles near the left and right edges wrap indices modulo grid_size.
    - weights sum to one for center, half-cell, and edge-adjacent positions.
    - the nearest-node rounding behavior is documented with hand-computable positions.
    """
    dx = 1.0
    grid_start = 0.0
    grid_size = 5

    indices, weights = get_S2_weights_and_indices_periodic_CN(
        0.0,
        dx,
        grid_start,
        grid_size,
    )
    assert_allclose(indices, jnp.array([4, 0, 1]), atol=0)
    assert_allclose(weights, jnp.array([0.125, 0.75, 0.125]))
    assert_allclose(jnp.sum(weights), 1.0)

    indices, weights = get_S2_weights_and_indices_periodic_CN(
        0.5,
        dx,
        grid_start,
        grid_size,
    )
    assert_allclose(indices, jnp.array([4, 0, 1]), atol=0)
    assert_allclose(weights, jnp.array([0.0, 0.5, 0.5]))
    assert_allclose(jnp.sum(weights), 1.0)

    indices, weights = get_S2_weights_and_indices_periodic_CN(
        -0.25,
        dx,
        grid_start,
        grid_size,
    )
    assert_allclose(indices, jnp.array([4, 0, 1]), atol=0)
    assert_allclose(weights, jnp.array([0.28125, 0.6875, 0.03125]))
    assert_allclose(jnp.sum(weights), 1.0)

    indices, weights = get_S2_weights_and_indices_periodic_CN(
        4.9,
        dx,
        grid_start,
        grid_size,
    )
    assert_allclose(indices, jnp.array([4, 0, 1]), atol=0)
    assert_allclose(weights, jnp.array([0.18, 0.74, 0.08]))
    assert_allclose(jnp.sum(weights), 1.0)


@pytest.mark.parametrize(
    "position, bc_left, bc_right, expected",
    [
        (0.25, 0, 0, (0.0, 0.0625)),
        (0.25, 1, 1, (0.0625, 0.0)),
        (0.25, 2, 2, (0.0, 0.0)),
        (2.75, 0, 0, (0.0625, 0.0)),
        (2.75, 1, 1, (0.0, 0.0625)),
        (2.75, 2, 2, (0.0, 0.0)),
    ],
)
def test_charge_density_BCs_for_particle_boundary_modes(
    position,
    bc_left,
    bc_right,
    expected,
):
    """Test jaxincell._sources.charge_density_BCs.

    Cases covered:
    - periodic boundaries exchange edge spillover charge between endpoints.
    - reflective boundaries keep spillover on the same endpoint.
    - absorbing boundaries drop spillover charge.
    """
    left_charge, right_charge = charge_density_BCs(
        bc_left,
        bc_right,
        position,
        dx=1.0,
        grid=jnp.array([0.0, 1.0, 2.0, 3.0]),
        charge=2.0,
    )

    assert_allclose(jnp.array([left_charge, right_charge]), jnp.array(expected))


def test_single_particle_charge_density_shape_function_and_boundaries():
    """Test jaxincell._sources.single_particle_charge_density.

    Cases covered:
    - particle exactly on a grid point deposits the expected quadratic S2 stencil.
    - particle halfway between grid points deposits symmetric weights.
    - nonperiodic boundary modes affect only endpoint cells.
    """
    dx = 1.0
    grid = jnp.arange(5.0)
    charge = 2.0

    on_grid = single_particle_charge_density(
        2.0,
        charge,
        dx,
        grid,
        particle_BC_left=2,
        particle_BC_right=2,
    )
    assert_allclose(on_grid, jnp.array([0.0, 0.25, 1.5, 0.25, 0.0]))

    halfway = single_particle_charge_density(
        2.5,
        charge,
        dx,
        grid,
        particle_BC_left=2,
        particle_BC_right=2,
    )
    assert_allclose(halfway, jnp.array([0.0, 0.0, 1.0, 1.0, 0.0]))

    periodic = single_particle_charge_density(0.25, charge, dx, grid, 0, 0)
    reflective = single_particle_charge_density(0.25, charge, dx, grid, 1, 1)
    absorbing = single_particle_charge_density(0.25, charge, dx, grid, 2, 2)

    assert_allclose(periodic, jnp.array([1.375, 0.5625, 0.0, 0.0, 0.0625]))
    assert_allclose(reflective, jnp.array([1.4375, 0.5625, 0.0, 0.0, 0.0]))
    assert_allclose(absorbing, jnp.array([1.375, 0.5625, 0.0, 0.0, 0.0]))


def test_calculate_charge_density_sums_particles_and_filters():
    """Test jaxincell._sources.calculate_charge_density.

    Cases covered:
    - two particles with opposite charges sum to the manual per-particle deposition.
    - filter_passes=0 returns the unfiltered deposition.
    - nonzero filter settings match filter_scalar_field applied to the unfiltered result.
    """
    dx = 1.0
    grid = jnp.arange(6.0)
    xs = jnp.array([[2.0], [3.25]])
    qs = jnp.array([[2.0], [-1.0]])

    manual = (
        single_particle_charge_density(xs[0, 0], qs[0, 0], dx, grid, 0, 0)
        + single_particle_charge_density(xs[1, 0], qs[1, 0], dx, grid, 0, 0)
    )
    deposited = calculate_charge_density(
        xs,
        qs,
        dx,
        grid,
        particle_BC_left=0,
        particle_BC_right=0,
        filter_passes=0,
        filter_alpha=0.5,
        filter_strides=(1,),
    )
    assert_allclose(deposited, manual)

    canceling = calculate_charge_density(
        jnp.array([[2.0], [2.0]]),
        jnp.array([[2.0], [-2.0]]),
        dx,
        grid,
        particle_BC_left=0,
        particle_BC_right=0,
        filter_passes=0,
        filter_alpha=0.5,
        filter_strides=(1,),
    )
    assert_allclose(canceling, jnp.zeros_like(grid))

    filtered = calculate_charge_density(
        xs,
        qs,
        dx,
        grid,
        particle_BC_left=0,
        particle_BC_right=0,
        filter_passes=2,
        filter_alpha=0.4,
        filter_strides=(1,),
        field_BC_left=1,
        field_BC_right=1,
    )
    expected_filtered = filter_scalar_field(
        manual,
        passes=2,
        alpha=0.4,
        strides=(1,),
        bc_left=1,
        bc_right=1,
    )
    assert_allclose(filtered, expected_filtered)


def test_current_density_continuity_and_transverse_components():
    """Test jaxincell._sources.current_density.

    Cases covered:
    - x-current from staggered particle positions is consistent with charge-density change over dt.
    - y and z current components equal deposited charge density times particle velocity.
    - filtering and nonperiodic boundary conditions are applied to the vector current field.
    """
    dx = 1.0
    dt = 0.2
    grid_start = 0.0
    grid = jnp.arange(8.0)
    xs_nminushalf = jnp.array([[3.0]])
    xs_n = jnp.array([[3.1]])
    xs_nplushalf = jnp.array([[3.2]])
    vs_n = jnp.array([[0.0, 0.5, -0.25]])
    qs = jnp.array([[2.0]])

    unfiltered = current_density(
        xs_nminushalf,
        xs_n,
        xs_nplushalf,
        vs_n,
        qs,
        dx,
        dt,
        grid,
        grid_start,
        particle_BC_left=0,
        particle_BC_right=0,
        filter_passes=0,
        filter_alpha=0.5,
        filter_strides=(1,),
        field_BC_left=0,
        field_BC_right=0,
    )

    rho_minus = single_particle_charge_density(
        xs_nminushalf[0, 0],
        qs[0, 0],
        dx,
        grid,
        0,
        0,
    )
    rho_plus = single_particle_charge_density(
        xs_nplushalf[0, 0],
        qs[0, 0],
        dx,
        grid,
        0,
        0,
    )
    charge_change = (rho_plus - rho_minus) / dt
    discrete_divergence = (unfiltered[:, 0] - jnp.roll(unfiltered[:, 0], 1)) / dx
    assert_allclose(charge_change + discrete_divergence, jnp.zeros_like(grid), atol=1e-12)

    rho_n = single_particle_charge_density(xs_n[0, 0], qs[0, 0], dx, grid, 0, 0)
    assert_allclose(unfiltered[:, 1], rho_n * vs_n[0, 1])
    assert_allclose(unfiltered[:, 2], rho_n * vs_n[0, 2])
    assert unfiltered.shape == (len(grid), 3)

    filtered = current_density(
        xs_nminushalf,
        xs_n,
        xs_nplushalf,
        vs_n,
        qs,
        dx,
        dt,
        grid,
        grid_start,
        particle_BC_left=1,
        particle_BC_right=1,
        filter_passes=2,
        filter_alpha=0.4,
        filter_strides=(1,),
        field_BC_left=1,
        field_BC_right=1,
    )
    unfiltered_reflective = current_density(
        xs_nminushalf,
        xs_n,
        xs_nplushalf,
        vs_n,
        qs,
        dx,
        dt,
        grid,
        grid_start,
        particle_BC_left=1,
        particle_BC_right=1,
        filter_passes=0,
        filter_alpha=0.4,
        filter_strides=(1,),
        field_BC_left=1,
        field_BC_right=1,
    )
    expected_filtered = filter_vector_field(
        unfiltered_reflective,
        passes=2,
        alpha=0.4,
        strides=(1,),
        bc_left=1,
        bc_right=1,
    )
    assert_allclose(filtered, expected_filtered)


def test_current_density_periodic_CN_accumulates_wrapped_particles():
    """Test jaxincell._sources.current_density_periodic_CN.

    Cases covered:
    - one particle matches manual S2-weighted current deposition.
    - multiple particles landing on the same grid index accumulate contributions.
    - particles outside the principal cell wrap periodically.
    """
    dx = 1.0
    grid_start = 0.0
    grid_size = 5

    one_particle = current_density_periodic_CN(
        xs_n=jnp.array([[0.0]]),
        vs_n=jnp.array([[1.0, 2.0, -3.0]]),
        qs=jnp.array([[2.0]]),
        dx=dx,
        grid_start=grid_start,
        grid_size=grid_size,
    )
    expected = jnp.zeros((grid_size, 3))
    expected = expected.at[4].set(jnp.array([0.25, 0.5, -0.75]))
    expected = expected.at[0].set(jnp.array([1.5, 3.0, -4.5]))
    expected = expected.at[1].set(jnp.array([0.25, 0.5, -0.75]))
    assert_allclose(one_particle, expected)

    wrapped_particles = current_density_periodic_CN(
        xs_n=jnp.array([[0.0], [5.0]]),
        vs_n=jnp.array([[1.0, 2.0, -3.0], [4.0, -2.0, 0.5]]),
        qs=jnp.array([[2.0], [1.0]]),
        dx=dx,
        grid_start=grid_start,
        grid_size=grid_size,
    )
    expected = jnp.zeros((grid_size, 3))
    first = jnp.array([2.0, 4.0, -6.0])
    second = jnp.array([4.0, -2.0, 0.5])
    expected = expected.at[4].add(0.125 * (first + second))
    expected = expected.at[0].add(0.75 * (first + second))
    expected = expected.at[1].add(0.125 * (first + second))
    assert_allclose(wrapped_particles, expected)
