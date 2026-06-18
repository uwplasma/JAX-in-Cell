import jax.numpy as jnp
from jax import grad
from numpy.testing import assert_allclose

from jaxincell._constants import speed_of_light
from jaxincell._particles import (
    boris_step,
    boris_step_relativistic,
    fields_to_particles_grid,
    fields_to_particles_periodic_CN,
    relativistic_rotation,
    rotation,
)
from jaxincell._sources import get_S2_weights_and_indices_periodic_CN


def expected_relativistic_e_only_step(dt, positions, velocities, charges, masses, electric_fields):
    gamma_n = 1 / jnp.sqrt(1.0 - jnp.sum((velocities / speed_of_light) ** 2, axis=1))
    p_n = gamma_n[:, None] * masses[:, None] * velocities
    p_nplus1 = p_n + charges[:, None] * electric_fields * dt
    gamma_nplus1 = jnp.sqrt(
        1.0 + jnp.sum((p_nplus1 / (masses[:, None] * speed_of_light)) ** 2, axis=1)
    )
    velocities_nplus1 = p_nplus1 / (gamma_nplus1[:, None] * masses[:, None])
    positions_nplus3_2 = positions + dt * velocities_nplus1
    return positions_nplus3_2, velocities_nplus1


def test_fields_to_particles_grid_interpolates_with_boundary_conditions():
    """Test jaxincell._particles.fields_to_particles_grid.

    Cases covered:
    - interpolate a hand-computable quadratic stencil for a particle near the grid center.
    - exercise periodic, reflective, and absorbing field boundary conditions through ghost cells.
    - verify the staggered-grid offset and grid_start logic near both domain edges.
    - cover runtime-style E-grid and B-grid staggering with grid_start != grid[0].
    - verify interpolation scales correctly when dx != 1.
    """
    dx = 1.0
    grid = jnp.arange(4.0)
    grid_start = 0.0
    field = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 10.0, -1.0],
            [5.0, 20.0, 3.0],
            [9.0, 30.0, 2.0],
        ]
    )

    centered = fields_to_particles_grid(
        jnp.array([1.0, 0.0, 0.0]),
        field,
        dx,
        grid,
        grid_start,
        field_BC_left=1,
        field_BC_right=1,
    )
    expected_centered = 0.125 * field[0] + 0.75 * field[1] + 0.125 * field[2]
    assert_allclose(centered, expected_centered)

    left_periodic = fields_to_particles_grid(
        jnp.array([-0.5, 0.0, 0.0]),
        field,
        dx,
        grid,
        grid_start,
        field_BC_left=0,
        field_BC_right=0,
    )
    left_reflective = fields_to_particles_grid(
        jnp.array([-0.5, 0.0, 0.0]),
        field,
        dx,
        grid,
        grid_start,
        field_BC_left=1,
        field_BC_right=1,
    )
    left_absorbing = fields_to_particles_grid(
        jnp.array([-0.5, 0.0, 0.0]),
        field,
        dx,
        grid,
        grid_start,
        field_BC_left=2,
        field_BC_right=2,
    )
    assert_allclose(left_periodic, 0.5 * field[-1] + 0.5 * field[0])
    assert_allclose(left_reflective, field[0])
    assert_allclose(left_absorbing, 0.5 * field[0])

    right_periodic = fields_to_particles_grid(
        jnp.array([3.5, 0.0, 0.0]),
        field,
        dx,
        grid,
        grid_start,
        field_BC_left=0,
        field_BC_right=0,
    )
    right_reflective = fields_to_particles_grid(
        jnp.array([3.5, 0.0, 0.0]),
        field,
        dx,
        grid,
        grid_start,
        field_BC_left=1,
        field_BC_right=1,
    )
    right_absorbing = fields_to_particles_grid(
        jnp.array([3.5, 0.0, 0.0]),
        field,
        dx,
        grid,
        grid_start,
        field_BC_left=2,
        field_BC_right=2,
    )
    assert_allclose(right_periodic, 0.5 * field[-1] + 0.5 * field[0])
    assert_allclose(right_reflective, field[-1])
    assert_allclose(right_absorbing, 0.5 * field[-1])

    runtime_grid = jnp.arange(4.0)
    e_grid_value = fields_to_particles_grid(
        jnp.array([1.0, 0.0, 0.0]),
        field,
        dx,
        runtime_grid + dx / 2,
        grid_start=runtime_grid[0],
        field_BC_left=1,
        field_BC_right=1,
    )
    assert_allclose(e_grid_value, 0.5 * field[0] + 0.5 * field[1])

    b_grid_value = fields_to_particles_grid(
        jnp.array([1.0, 0.0, 0.0]),
        field,
        dx,
        runtime_grid,
        grid_start=runtime_grid[0] - dx / 2,
        field_BC_left=1,
        field_BC_right=1,
    )
    assert_allclose(b_grid_value, 0.125 * field[0] + 0.75 * field[1] + 0.125 * field[2])

    dx_half = 0.5
    half_grid = dx_half * jnp.arange(4.0)
    half_dx_value = fields_to_particles_grid(
        jnp.array([0.5, 0.0, 0.0]),
        field,
        dx_half,
        half_grid,
        grid_start=half_grid[0] - dx_half / 2,
        field_BC_left=1,
        field_BC_right=1,
    )
    assert_allclose(half_dx_value, 0.125 * field[0] + 0.75 * field[1] + 0.125 * field[2])


def test_fields_to_particles_periodic_CN_wraps_indices():
    """Test jaxincell._particles.fields_to_particles_periodic_CN.

    Cases covered:
    - use particles near both domain edges to verify periodic wrapping.
    - compare the result to get_S2_weights_and_indices_periodic_CN and a manual dot product.
    - verify constant fields interpolate to the constant value.
    - verify a direct hand-computed half-cell case with dx != 1.
    """
    dx = 1.0
    grid_start = 0.0
    field = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 10.0, -1.0],
            [2.0, 20.0, -2.0],
            [3.0, 30.0, -3.0],
            [4.0, 40.0, -4.0],
        ]
    )

    for position in [0.0, -0.25, 4.9, 5.0]:
        indices, weights = get_S2_weights_and_indices_periodic_CN(
            position,
            dx,
            grid_start,
            len(field),
        )
        expected = weights @ field[indices]
        interpolated = fields_to_particles_periodic_CN(
            jnp.array([position, 0.0, 0.0]),
            field,
            dx,
            grid_start,
        )
        assert_allclose(interpolated, expected, rtol=1e-6, atol=1e-12)

    constant_field = jnp.tile(jnp.array([2.0, -3.0, 5.0]), (len(field), 1))
    constant_interpolated = fields_to_particles_periodic_CN(
        jnp.array([4.75, 0.0, 0.0]),
        constant_field,
        dx,
        grid_start,
    )
    assert_allclose(constant_interpolated, constant_field[0], rtol=1e-6, atol=1e-12)

    dx_half = 0.5
    shifted_grid_start = 2.0
    half_cell = fields_to_particles_periodic_CN(
        jnp.array([shifted_grid_start + 0.5 * dx_half, 0.0, 0.0]),
        field,
        dx_half,
        shifted_grid_start,
    )
    assert_allclose(half_cell, 0.5 * field[0] + 0.5 * field[1], rtol=1e-6, atol=1e-12)


def test_rotation_identity_and_magnetic_norm_behavior():
    """Test jaxincell._particles.rotation.

    Cases covered:
    - zero magnetic field returns the input velocity unchanged.
    - zero charge-to-mass ratio returns the input velocity unchanged.
    - nonzero magnetic field rotates velocity without changing speed for the magnetic-only step.
    - opposite charge-to-mass signs rotate in opposite directions.
    """
    dt = 0.1
    velocity = jnp.array([1.0, 2.0, -0.5])
    magnetic_field = jnp.array([0.0, 0.0, 2.0])

    assert_allclose(rotation(dt, jnp.zeros(3), velocity, 1.0), velocity, atol=0)
    assert_allclose(rotation(dt, magnetic_field, velocity, 0.0), velocity, atol=0)

    rotated_positive = rotation(dt, magnetic_field, jnp.array([1.0, 0.0, 0.0]), 1.0)
    rotated_negative = rotation(dt, magnetic_field, jnp.array([1.0, 0.0, 0.0]), -1.0)
    assert_allclose(jnp.linalg.norm(rotated_positive), 1.0, rtol=1e-6, atol=1e-12)
    assert_allclose(jnp.linalg.norm(rotated_negative), 1.0, rtol=1e-6, atol=1e-12)
    assert rotated_positive[1] < 0
    assert rotated_negative[1] > 0
    assert_allclose(rotated_positive[0], rotated_negative[0], rtol=1e-6, atol=1e-12)
    assert_allclose(rotated_positive[1], -rotated_negative[1], rtol=1e-6, atol=1e-12)


def test_boris_step_electric_and_magnetic_limits():
    """Test jaxincell._particles.boris_step.

    Cases covered:
    - E-only acceleration matches the analytic velocity and position update.
    - B-only update preserves speed to numerical tolerance.
    - batched particles with different charge-to-mass ratios keep the expected shapes.
    - zero fields advance positions linearly while preserving velocity.
    - combined E and B fields follow the Boris half-kick, rotation, half-kick sequence.
    """
    dt = 0.2
    positions = jnp.array([[0.0, 1.0, 2.0], [1.0, -1.0, 0.5]])
    velocities = jnp.array([[1.0, 0.0, 0.5], [-0.5, 0.25, 1.0]])
    q_ms = jnp.array([[2.0], [-1.0]])
    zero_fields = jnp.zeros_like(positions)

    zero_positions, zero_velocities = boris_step(
        dt,
        positions,
        velocities,
        q_ms,
        zero_fields,
        zero_fields,
    )
    assert zero_positions.shape == positions.shape
    assert zero_velocities.shape == velocities.shape
    assert_allclose(zero_velocities, velocities, atol=0)
    assert_allclose(zero_positions, positions + dt * velocities)

    electric_fields = jnp.array([[0.5, -1.0, 0.25], [1.0, 0.0, -0.5]])
    e_only_positions, e_only_velocities = boris_step(
        dt,
        positions,
        velocities,
        q_ms,
        electric_fields,
        zero_fields,
    )
    expected_velocities = velocities + q_ms * electric_fields * dt
    expected_positions = positions + dt * expected_velocities
    assert_allclose(e_only_velocities, expected_velocities, rtol=1e-6, atol=1e-12)
    assert_allclose(e_only_positions, expected_positions, rtol=1e-6, atol=1e-12)

    magnetic_fields = jnp.array([[0.0, 0.0, 2.0], [0.0, 1.5, 0.0]])
    b_only_positions, b_only_velocities = boris_step(
        dt,
        positions,
        velocities,
        q_ms,
        zero_fields,
        magnetic_fields,
    )
    assert b_only_positions.shape == positions.shape
    assert b_only_velocities.shape == velocities.shape
    assert_allclose(
        jnp.linalg.norm(b_only_velocities, axis=1),
        jnp.linalg.norm(velocities, axis=1),
        rtol=1e-6,
        atol=1e-12,
    )
    assert_allclose(b_only_positions, positions + dt * b_only_velocities, rtol=1e-6, atol=1e-12)

    combined_positions, combined_velocities = boris_step(
        dt,
        positions,
        velocities,
        q_ms,
        electric_fields,
        magnetic_fields,
    )
    expected_v_minus = velocities + q_ms * electric_fields * dt / 2
    expected_v_rot = jnp.stack(
        [
            rotation(dt, magnetic_fields[index], expected_v_minus[index], q_ms[index, 0])
            for index in range(len(positions))
        ]
    )
    expected_combined_velocities = expected_v_rot + q_ms * electric_fields * dt / 2
    expected_combined_positions = positions + dt * expected_combined_velocities
    assert_allclose(combined_velocities, expected_combined_velocities, rtol=1e-6, atol=1e-12)
    assert_allclose(combined_positions, expected_combined_positions, rtol=1e-6, atol=1e-12)


def test_relativistic_rotation_limits_and_finite_values():
    """Test jaxincell._particles.relativistic_rotation.

    Cases covered:
    - zero magnetic field leaves momentum unchanged.
    - nonzero magnetic field returns finite momentum.
    - magnetic-only rotation preserves momentum norm.
    - very small and large momentum inputs keep gamma-dependent terms finite.
    """
    dt = 0.1
    magnetic_field = jnp.array([0.0, 0.0, 2.0])
    momentum = jnp.array([1.0, 0.0, 0.25])

    assert_allclose(
        relativistic_rotation(dt, jnp.zeros(3), momentum, q=1.0, m=1.0),
        momentum,
        atol=0,
    )

    rotated_positive = relativistic_rotation(dt, magnetic_field, momentum, q=1.0, m=1.0)
    rotated_negative = relativistic_rotation(dt, magnetic_field, momentum, q=-1.0, m=1.0)
    assert bool(jnp.all(jnp.isfinite(rotated_positive)))
    assert_allclose(
        jnp.linalg.norm(rotated_positive),
        jnp.linalg.norm(momentum),
        rtol=1e-6,
        atol=1e-12,
    )
    assert rotated_positive[1] < 0
    assert rotated_negative[1] > 0
    assert_allclose(rotated_positive[0], rotated_negative[0], rtol=1e-6, atol=1e-12)
    assert_allclose(rotated_positive[1], -rotated_negative[1], rtol=1e-6, atol=1e-12)

    small_momentum = relativistic_rotation(
        dt,
        magnetic_field,
        jnp.array([1e-12, -2e-12, 3e-12]),
        q=1.0,
        m=1.0,
    )
    large_momentum = relativistic_rotation(
        dt,
        magnetic_field,
        jnp.array([0.8 * speed_of_light, -0.2 * speed_of_light, 0.1 * speed_of_light]),
        q=1.0,
        m=2.0,
    )
    assert bool(jnp.all(jnp.isfinite(small_momentum)))
    assert bool(jnp.all(jnp.isfinite(large_momentum)))


def test_boris_step_relativistic_zero_field_and_speed_limit_behavior():
    """Test jaxincell._particles.boris_step_relativistic.

    Cases covered:
    - zero E and B fields preserve velocity and advance positions linearly.
    - E-only acceleration matches the analytic relativistic momentum update.
    - B-only acceleration preserves speed through the relativistic magnetic rotation path.
    - batched particles with different charges and masses keep the expected shapes.
    - velocities remain below the speed of light.
    """
    dt = 0.01
    positions = jnp.array([[0.0, 1.0, 2.0], [1.0, -1.0, 0.5]])
    velocities = jnp.array(
        [
            [0.05 * speed_of_light, 0.0, 0.0],
            [0.0, -0.03 * speed_of_light, 0.02 * speed_of_light],
        ]
    )
    charges = jnp.array([1.0, -2.0])
    masses = jnp.array([1.0, 3.0])
    zero_fields = jnp.zeros_like(positions)

    zero_positions, zero_velocities = boris_step_relativistic(
        dt,
        positions,
        velocities,
        charges,
        masses,
        zero_fields,
        zero_fields,
    )
    assert zero_positions.shape == positions.shape
    assert zero_velocities.shape == velocities.shape
    assert_allclose(zero_velocities, velocities, rtol=1e-6, atol=1e-8)
    assert_allclose(zero_positions, positions + dt * velocities, rtol=1e-6, atol=1e-8)

    electric_fields = jnp.array([[2e8, -1e8, 0.0], [0.0, 3e8, -2e8]])
    e_only_positions, e_only_velocities = boris_step_relativistic(
        dt,
        positions,
        velocities,
        charges,
        masses,
        electric_fields,
        zero_fields,
    )
    expected_e_only_positions, expected_e_only_velocities = expected_relativistic_e_only_step(
        dt,
        positions,
        velocities,
        charges,
        masses,
        electric_fields,
    )
    assert e_only_positions.shape == positions.shape
    assert e_only_velocities.shape == velocities.shape
    assert bool(jnp.all(jnp.isfinite(e_only_positions)))
    assert bool(jnp.all(jnp.isfinite(e_only_velocities)))
    assert_allclose(e_only_velocities, expected_e_only_velocities, rtol=1e-6, atol=1e-6)
    assert_allclose(e_only_positions, expected_e_only_positions, rtol=1e-6, atol=1e-6)
    assert bool(jnp.all(jnp.linalg.norm(e_only_velocities, axis=1) < speed_of_light))

    magnetic_fields = jnp.array([[0.0, 0.0, 2.0], [0.0, -1.5, 0.5]])
    b_only_positions, b_only_velocities = boris_step_relativistic(
        dt,
        positions,
        velocities,
        charges,
        masses,
        zero_fields,
        magnetic_fields,
    )
    assert b_only_positions.shape == positions.shape
    assert b_only_velocities.shape == velocities.shape
    assert bool(jnp.all(jnp.isfinite(b_only_positions)))
    assert bool(jnp.all(jnp.isfinite(b_only_velocities)))
    assert_allclose(
        jnp.linalg.norm(b_only_velocities, axis=1),
        jnp.linalg.norm(velocities, axis=1),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(b_only_positions, positions + dt * b_only_velocities, rtol=1e-6, atol=1e-6)


def test_particle_helpers_are_differentiable_for_small_inputs():
    """Test small differentiability smoke cases for particle helpers.

    Cases covered:
    - grad through rotation with respect to velocity is finite.
    - grad through fields_to_particles_periodic_CN with respect to field values is finite.
    - grad through fields_to_particles_grid with respect to field values is finite.
    - grad through boris_step with respect to electric fields is finite.
    """
    dt = 0.1
    magnetic_field = jnp.array([0.0, 0.0, 2.0])
    velocity = jnp.array([1.0, -0.25, 0.5])

    def rotation_loss(velocity):
        return jnp.sum(rotation(dt, magnetic_field, velocity, 1.0) ** 2)

    rotation_gradient = grad(rotation_loss)(velocity)
    assert rotation_gradient.shape == velocity.shape
    assert bool(jnp.all(jnp.isfinite(rotation_gradient)))

    field = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
    )

    def interpolation_loss(field):
        return jnp.sum(
            fields_to_particles_periodic_CN(
                jnp.array([0.25, 0.0, 0.0]),
                field,
                dx=1.0,
                grid_start=0.0,
            )
        )

    interpolation_gradient = grad(interpolation_loss)(field)
    assert interpolation_gradient.shape == field.shape
    assert bool(jnp.all(jnp.isfinite(interpolation_gradient)))
    assert float(jnp.linalg.norm(interpolation_gradient)) > 0

    grid = jnp.arange(4.0)

    def grid_interpolation_loss(field):
        return jnp.sum(
            fields_to_particles_grid(
                jnp.array([1.0, 0.0, 0.0]),
                field,
                dx=1.0,
                grid=grid,
                grid_start=0.0,
                field_BC_left=1,
                field_BC_right=1,
            )
        )

    grid_interpolation_gradient = grad(grid_interpolation_loss)(field)
    assert grid_interpolation_gradient.shape == field.shape
    assert bool(jnp.all(jnp.isfinite(grid_interpolation_gradient)))
    assert float(jnp.linalg.norm(grid_interpolation_gradient)) > 0

    positions = jnp.array([[0.0, 1.0, 2.0], [1.0, -1.0, 0.5]])
    velocities = jnp.array([[1.0, 0.0, 0.5], [-0.5, 0.25, 1.0]])
    q_ms = jnp.array([[2.0], [-1.0]])
    magnetic_fields = jnp.array([[0.0, 0.0, 2.0], [0.0, 1.5, 0.0]])
    electric_fields = jnp.array([[0.5, -1.0, 0.25], [1.0, 0.0, -0.5]])

    def boris_loss(electric_fields):
        positions_next, velocities_next = boris_step(
            dt,
            positions,
            velocities,
            q_ms,
            electric_fields,
            magnetic_fields,
        )
        return jnp.sum(positions_next) + jnp.sum(velocities_next)

    boris_gradient = grad(boris_loss)(electric_fields)
    assert boris_gradient.shape == electric_fields.shape
    assert bool(jnp.all(jnp.isfinite(boris_gradient)))
    assert float(jnp.linalg.norm(boris_gradient)) > 0
