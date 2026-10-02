import pytest
from jaxincell._boundary_conditions import (
    set_BC_single_particle, set_BC_particles,
    set_BC_single_particle_positions, set_BC_positions,
    field_ghost_cells_E, field_ghost_cells_B, field_2_ghost_cells)
from jaxincell._constants import speed_of_light
import jax.numpy as jnp

def test_set_BC_single_particle_periodic():
    x_n = jnp.array([0.1, 0.5, 0.8])
    v_n = jnp.array([1.0, 1.0, 1.0])
    q = 1.0
    q_m = 1.0
    dx = 0.1
    grid = jnp.linspace(0.0, 1, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 0
    BC_right = 0

    x_n_updated, v_n_updated, q_updated, q_m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(x_n_updated, jnp.array([0.1, 0.5, 0.8])), "Periodic BC failed for position"
    assert jnp.allclose(v_n_updated, jnp.array([1.0, -1.0, -1.0])), "Periodic BC failed for velocity"
    assert q_updated == 1.0, "Periodic BC failed for charge"
    assert q_m_updated == 1.0, "Periodic BC failed for charge-to-mass ratio"

def test_set_BC_single_particle_reflective():
    x_n = jnp.array([-1.1, 1.0, 1.0])
    v_n = jnp.array([1.0, 1.0, 1.0])
    q = 1.0
    q_m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 1
    BC_right = 1

    x_n_updated, v_n_updated, q_updated, q_m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(x_n_updated, jnp.array([-0.9, -1.0, -1.0])), "Reflective BC failed for position"
    assert jnp.allclose(v_n_updated, jnp.array([-1.0, 1.0, 1.0])), "Reflective BC failed for velocity"
    assert q_updated == 1.0, "Reflective BC failed for charge"
    assert q_m_updated == 1.0, "Reflective BC failed for charge-to-mass ratio"

def test_set_BC_single_particle_absorbing():
    x_n = jnp.array([1.1, 1.0, 1.0])
    v_n = jnp.array([1.0, 1.0, 1.0])
    q = 1.0
    q_m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 2
    BC_right = 2

    x_n_updated, v_n_updated, q_updated, q_m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(x_n_updated, jnp.array([1.3, -1.0, -1.0])), "Absorbing BC failed for position"
    assert jnp.allclose(v_n_updated, jnp.array([0.0, 0.0, 0.0])), "Absorbing BC failed for velocity"
    assert q_updated == 0.0, "Absorbing BC failed for charge"
    assert q_m_updated == 0.0, "Absorbing BC failed for charge-to-mass ratio"

def test_set_BC_single_particle_mixed():
    x_n = jnp.array([-1.1, 1.0, 1.0])
    v_n = jnp.array([1.0, 1.0, 1.0])
    q = 1.0
    q_m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 1
    BC_right = 2

    x_n_updated, v_n_updated, q_updated, q_m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(x_n_updated, jnp.array([-0.9, -1.0, -1.0])), "Mixed BC failed for position"
    assert jnp.allclose(v_n_updated, jnp.array([-1.0, 1.0, 1.0])), "Mixed BC failed for velocity"
    assert q_updated == 1.0, "Mixed BC failed for charge"
    assert q_m_updated == 1.0, "Mixed BC failed for charge-to-mass ratio"

def test_set_BC_particles_periodic():
    xs_n = jnp.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    vs_n = jnp.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    qs = jnp.array([1.0, -1.0])
    ms = jnp.array([1.0, 1.0])
    q_ms = jnp.array([1.0, -1.0])
    dx = 0.1
    grid = jnp.linspace(-5.0, 5.0, 100)
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0
    BC_left = 0
    BC_right = 0

    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(xs_n_updated, xs_n)
    assert jnp.allclose(vs_n_updated, vs_n)
    assert jnp.allclose(qs_updated, qs)
    assert jnp.allclose(ms_updated, ms)
    assert jnp.allclose(q_ms_updated, q_ms)

def test_set_BC_particles_reflective():
    xs_n = jnp.array([[6.0, 2.0, 3.0], [-6.0, -2.0, -3.0]])
    vs_n = jnp.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    qs = jnp.array([1.0, -1.0])
    ms = jnp.array([1.0, 1.0])
    q_ms = jnp.array([1.0, -1.0])
    dx = 0.1
    grid = jnp.linspace(-5.0, 5.0, 100)
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0
    BC_left = 1
    BC_right = 1

    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    expected_xs_n = jnp.array([[4.0, 2.0, 3.0], [-4.0, -2.0, -3.0]])
    expected_vs_n = jnp.array([[-0.1, 0.2, 0.3], [0.1, -0.2, -0.3]])

    assert jnp.allclose(xs_n_updated, expected_xs_n)
    assert jnp.allclose(vs_n_updated, expected_vs_n)
    assert jnp.allclose(qs_updated, qs)
    assert jnp.allclose(ms_updated, ms)
    assert jnp.allclose(q_ms_updated, q_ms)

def test_set_BC_particles_absorbing():
    xs_n = jnp.array([[6.0, 2.0, 3.0], [-6.0, -2.0, -3.0]])
    vs_n = jnp.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    qs = jnp.array([1.0, -1.0])
    ms = jnp.array([1.0, 1.0])
    q_ms = jnp.array([1.0, -1.0])
    dx = 0.1
    grid = jnp.linspace(-5.0, 5.0, 100)
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0
    BC_left = 2
    BC_right = 2

    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    expected_xs_n = jnp.array([[grid[-1] + 3 * dx, 2.0, 3.0], [grid[0] - 1.5 * dx, -2.0, -3.0]])
    expected_vs_n = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected_qs = jnp.array([0.0, 0.0])
    expected_q_ms = jnp.array([0.0, 0.0])

    assert jnp.allclose(xs_n_updated, expected_xs_n)
    assert jnp.allclose(vs_n_updated, expected_vs_n)
    assert jnp.allclose(qs_updated, expected_qs)
    assert jnp.allclose(ms_updated, ms)
    assert jnp.allclose(q_ms_updated, expected_q_ms)

def test_set_BC_single_particle_periodic():
    x_n = jnp.array([1.0, 1.0, 1.0])
    v_n = jnp.array([1.0, 1.0, 1.0])
    q = 1.0
    q_m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 0
    BC_right = 0

    x_n_updated, v_n_updated, q_updated, q_m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(x_n_updated, jnp.array([1.0, -1.0, -1.0])), "Periodic BC failed for position"
    assert jnp.allclose(v_n_updated, jnp.array([1.0, 1.0, 1.0])), "Periodic BC failed for velocity"
    assert q_updated == 1.0, "Periodic BC failed for charge"
    assert q_m_updated == 1.0, "Periodic BC failed for charge-to-mass ratio"

def test_set_BC_single_particle_reflective():
    x_n = jnp.array([-1.1, 1.0, 1.0])
    v_n = jnp.array([1.0, 1.0, 1.0])
    q = 1.0
    q_m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 1
    BC_right = 1

    x_n_updated, v_n_updated, q_updated, q_m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(x_n_updated, jnp.array([-0.9, -1.0, -1.0])), "Reflective BC failed for position"
    assert jnp.allclose(v_n_updated, jnp.array([-1.0, 1.0, 1.0])), "Reflective BC failed for velocity"
    assert q_updated == 1.0, "Reflective BC failed for charge"
    assert q_m_updated == 1.0, "Reflective BC failed for charge-to-mass ratio"

def test_set_BC_single_particle_absorbing():
    x_n = jnp.array([1.1, 1.0, 1.0])
    v_n = jnp.array([1.0, 1.0, 1.0])
    q = 1.0
    q_m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 2
    BC_right = 2

    x_n_updated, v_n_updated, q_updated, q_m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(x_n_updated, jnp.array([1.3, -1.0, -1.0])), "Absorbing BC failed for position"
    assert jnp.allclose(v_n_updated, jnp.array([0.0, 0.0, 0.0])), "Absorbing BC failed for velocity"
    assert q_updated == 0.0, "Absorbing BC failed for charge"
    assert q_m_updated == 0.0, "Absorbing BC failed for charge-to-mass ratio"

def test_set_BC_single_particle_mixed():
    x_n = jnp.array([-1.1, 1.0, 1.0])
    v_n = jnp.array([1.0, 1.0, 1.0])
    q = 1.0
    q_m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 1
    BC_right = 2

    x_n_updated, v_n_updated, q_updated, q_m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(x_n_updated, jnp.array([-0.9, -1.0, -1.0])), "Mixed BC failed for position"
    assert jnp.allclose(v_n_updated, jnp.array([-1.0, 1.0, 1.0])), "Mixed BC failed for velocity"
    assert q_updated == 1.0, "Mixed BC failed for charge"
    assert q_m_updated == 1.0, "Mixed BC failed for charge-to-mass ratio"

def test_set_BC_particles_periodic():
    xs_n = jnp.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    vs_n = jnp.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    qs = jnp.array([1.0, -1.0])
    ms = jnp.array([1.0, 1.0])
    q_ms = jnp.array([1.0, -1.0])
    dx = 0.1
    grid = jnp.linspace(-5.0, 5.0, 100)
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0
    BC_left = 0
    BC_right = 0

    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    assert jnp.allclose(xs_n_updated, xs_n)
    assert jnp.allclose(vs_n_updated, vs_n)
    assert jnp.allclose(qs_updated, qs)
    assert jnp.allclose(ms_updated, ms)
    assert jnp.allclose(q_ms_updated, q_ms)

def test_set_BC_particles_reflective():
    xs_n = jnp.array([[6.0, 2.0, 3.0], [-6.0, -2.0, -3.0]])
    vs_n = jnp.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    qs = jnp.array([1.0, -1.0])
    ms = jnp.array([1.0, 1.0])
    q_ms = jnp.array([1.0, -1.0])
    dx = 0.1
    grid = jnp.linspace(-5.0, 5.0, 100)
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0
    BC_left = 1
    BC_right = 1

    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    expected_xs_n = jnp.array([[4.0, 2.0, 3.0], [-4.0, -2.0, -3.0]])
    expected_vs_n = jnp.array([[-0.1, 0.2, 0.3], [0.1, -0.2, -0.3]])

    assert jnp.allclose(xs_n_updated, expected_xs_n)
    assert jnp.allclose(vs_n_updated, expected_vs_n)
    assert jnp.allclose(qs_updated, qs)
    assert jnp.allclose(ms_updated, ms)
    assert jnp.allclose(q_ms_updated, q_ms)

def test_set_BC_particles_absorbing():
    xs_n = jnp.array([[6.0, 2.0, 3.0], [-6.0, -2.0, -3.0]])
    vs_n = jnp.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    qs = jnp.array([1.0, -1.0])
    ms = jnp.array([1.0, 1.0])
    q_ms = jnp.array([1.0, -1.0])
    dx = 0.1
    grid = jnp.linspace(-5.0, 5.0, 100)
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0
    BC_left = 2
    BC_right = 2

    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
    )

    expected_xs_n = jnp.array([[grid[-1] + 3 * dx, 2.0, 3.0], [grid[0] - 1.5 * dx, -2.0, -3.0]])
    expected_vs_n = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected_qs = jnp.array([0.0, 0.0])
    expected_q_ms = jnp.array([0.0, 0.0])

    assert jnp.allclose(xs_n_updated, expected_xs_n)
    assert jnp.allclose(vs_n_updated, expected_vs_n)
    assert jnp.allclose(qs_updated, expected_qs)
    assert jnp.allclose(ms_updated, ms)
    assert jnp.allclose(q_ms_updated, expected_q_ms)

def test_set_BC_single_particle_positions():
    x_n = jnp.array([1.0, 1.0, 1.0])
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0

    x_n_updated = set_BC_single_particle_positions(x_n, box_size_x, box_size_y, box_size_z)

    assert jnp.allclose(x_n_updated, jnp.array([1.0, -1.0, -1.0])), "Periodic BC failed for position"

def test_set_BC_positions():
    xs_n = jnp.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0
    xs_n_updated = set_BC_positions(xs_n, 1, 0.1, jnp.linspace(-5.0, 5.0, 100), box_size_x, box_size_y, box_size_z, 0, 0)
    
    assert jnp.allclose(xs_n_updated, xs_n), "Periodic BC failed for positions"

def test_set_BC_single_particle_positions():
    x_n = jnp.array([1.0, 1.0, 1.0])
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 0
    BC_right = 0

    x_n_updated = set_BC_single_particle_positions(x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right)
    assert jnp.allclose(x_n_updated, jnp.array([1.0, -1.0, -1.0])), "Periodic BC failed for position"

    BC_left = 1
    BC_right = 1
    x_n = jnp.array([-1.1, 1.0, 1.0])
    x_n_updated = set_BC_single_particle_positions(x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right)
    assert jnp.allclose(x_n_updated, jnp.array([-0.9, -1.0, -1.0])), "Reflective BC failed for position"

    BC_left = 2
    BC_right = 2
    x_n = jnp.array([-1.1, 1.0, 1.0])
    x_n_updated = set_BC_single_particle_positions(x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right)
    assert jnp.allclose(x_n_updated, jnp.array([grid[0] - 1.5 * dx, -1.0, -1.0])), "Absorbing BC failed for position"

def test_field_ghost_cells_E():
    field_BC_left = 0
    field_BC_right = 0
    E_field = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B_field = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    field_ghost_cell_L, field_ghost_cell_R = field_ghost_cells_E(field_BC_left, field_BC_right, E_field, B_field)
    assert jnp.allclose(field_ghost_cell_L, E_field[-1]), "Periodic BC failed for left ghost cell"
    assert jnp.allclose(field_ghost_cell_R, E_field[0]), "Periodic BC failed for right ghost cell"

    field_BC_left = 1
    field_BC_right = 1
    field_ghost_cell_L, field_ghost_cell_R = field_ghost_cells_E(field_BC_left, field_BC_right, E_field, B_field)
    assert jnp.allclose(field_ghost_cell_L, E_field[0]), "Reflective BC failed for left ghost cell"
    assert jnp.allclose(field_ghost_cell_R, E_field[-1]), "Reflective BC failed for right ghost cell"

    field_BC_left = 2
    field_BC_right = 2
    field_ghost_cell_L, field_ghost_cell_R = field_ghost_cells_E(field_BC_left, field_BC_right, E_field, B_field)
    expected_L = jnp.array([0, -2 * speed_of_light * B_field[0, 2] - E_field[0, 1], 2 * speed_of_light * B_field[0, 1] - E_field[0, 2]])
    expected_R = jnp.array([0, 3 * E_field[-1, 1] - 2 * speed_of_light * B_field[-1, 2], 3 * E_field[-1, 2] + 2 * speed_of_light * B_field[-1, 1]])
    assert jnp.allclose(field_ghost_cell_L, expected_L), "Absorbing BC failed for left ghost cell"
    assert jnp.allclose(field_ghost_cell_R, expected_R), "Absorbing BC failed for right ghost cell"

def test_field_ghost_cells_B():
    field_BC_left = 0
    field_BC_right = 0
    B_field = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    E_field = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    field_ghost_cell_L, field_ghost_cell_R = field_ghost_cells_B(field_BC_left, field_BC_right, B_field, E_field)
    assert jnp.allclose(field_ghost_cell_L, B_field[-1]), "Periodic BC failed for left ghost cell"
    assert jnp.allclose(field_ghost_cell_R, B_field[0]), "Periodic BC failed for right ghost cell"

    field_BC_left = 1
    field_BC_right = 1
    field_ghost_cell_L, field_ghost_cell_R = field_ghost_cells_B(field_BC_left, field_BC_right, B_field, E_field)
    assert jnp.allclose(field_ghost_cell_L, B_field[0]), "Reflective BC failed for left ghost cell"
    assert jnp.allclose(field_ghost_cell_R, B_field[-1]), "Reflective BC failed for right ghost cell"

    field_BC_left = 2
    field_BC_right = 2
    field_ghost_cell_L, field_ghost_cell_R = field_ghost_cells_B(field_BC_left, field_BC_right, B_field, E_field)
    expected_L = jnp.array([0, 3 * B_field[0, 1] - (2 / speed_of_light) * E_field[0, 2], 3 * B_field[0, 2] + (2 / speed_of_light) * E_field[0, 1]])
    expected_R = jnp.array([0, -(2 / speed_of_light) * E_field[-1, 2] - B_field[-1, 1], (2 / speed_of_light) * E_field[-1, 1] - B_field[-1, 2]])
    assert jnp.allclose(field_ghost_cell_L, expected_L), "Absorbing BC failed for left ghost cell"
    assert jnp.allclose(field_ghost_cell_R, expected_R), "Absorbing BC failed for right ghost cell"

def test_field_2_ghost_cells():
    field_BC_left = 0
    field_BC_right = 0
    field = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    field_ghost_cell_L2, field_ghost_cell_L1, field_ghost_cell_R = field_2_ghost_cells(field_BC_left, field_BC_right, field)
    assert jnp.allclose(field_ghost_cell_L2, field[-2]), "Periodic BC failed for left ghost cell L2"
    assert jnp.allclose(field_ghost_cell_L1, field[-1]), "Periodic BC failed for left ghost cell L1"
    assert jnp.allclose(field_ghost_cell_R, field[0]), "Periodic BC failed for right ghost cell"

    field_BC_left = 1
    field_BC_right = 1
    field_ghost_cell_L2, field_ghost_cell_L1, field_ghost_cell_R = field_2_ghost_cells(field_BC_left, field_BC_right, field)
    assert jnp.allclose(field_ghost_cell_L2, field[1]), "Reflective BC failed for left ghost cell L2"
    assert jnp.allclose(field_ghost_cell_L1, field[0]), "Reflective BC failed for left ghost cell L1"
    assert jnp.allclose(field_ghost_cell_R, field[-1]), "Reflective BC failed for right ghost cell"

    field_BC_left = 2
    field_BC_right = 2
    field_ghost_cell_L2, field_ghost_cell_L1, field_ghost_cell_R = field_2_ghost_cells(field_BC_left, field_BC_right, field)
    assert jnp.allclose(field_ghost_cell_L2, jnp.array([0, 0, 0])), "Absorbing BC failed for left ghost cell L2"
    assert jnp.allclose(field_ghost_cell_L1, jnp.array([0, 0, 0])), "Absorbing BC failed for left ghost cell L1"
    assert jnp.allclose(field_ghost_cell_R, jnp.array([0, 0, 0])), "Absorbing BC failed for right ghost cell"

if __name__ == "__main__":
    pytest.main()