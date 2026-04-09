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
    m = 1.0
    dx = 0.1
    grid = jnp.linspace(0.0, 1, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 0
    BC_right = 0

    x_n_updated, v_n_updated, q_updated, q_m_updated, _ = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
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
    m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 1
    BC_right = 1

    x_n_updated, v_n_updated, q_updated, q_m_updated, _ = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
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
    m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 2
    BC_right = 2

    x_n_updated, v_n_updated, q_updated, q_m_updated, _ = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
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
    m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 1
    BC_right = 2

    x_n_updated, v_n_updated, q_updated, q_m_updated, _ = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right
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
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
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
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
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
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
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
    m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 0
    BC_right = 0

    x_n_updated, v_n_updated, q_updated, q_m_updated, _ = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
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
    m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 1
    BC_right = 1

    x_n_updated, v_n_updated, q_updated, q_m_updated, _ = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
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
    m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 2
    BC_right = 2

    x_n_updated, v_n_updated, q_updated, q_m_updated, _ = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
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
    m = 1.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    BC_left = 1
    BC_right = 2

    x_n_updated, v_n_updated, q_updated, q_m_updated, _ = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
    )

    assert jnp.allclose(x_n_updated, jnp.array([-0.9, -1.0, -1.0])), "Mixed BC failed for position"
    assert jnp.allclose(v_n_updated, jnp.array([-1.0, 1.0, 1.0])), "Mixed BC failed for velocity"
    assert q_updated == 1.0, "Mixed BC failed for charge"
    assert q_m_updated == 1.0, "Mixed BC failed for charge-to-mass ratio"

def test_set_BC_single_particle_bc3():
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    q = 2.0
    q_m = 0.5
    m = 4.0

    # --- weight=0.5: half absorbed, half reflected ---
    x_n = jnp.array([-1.1, 0.0, 0.0])
    v_n = jnp.array([-1.0, 0.5, 0.3])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=3, BC_right=0, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
    )
    assert jnp.allclose(x_n_updated, jnp.array([-0.9, 0.0, 0.0])), "weight=0.5 left: position not reflected"
    assert jnp.allclose(v_n_updated, jnp.array([1.0, 0.5, 0.3])),   "weight=0.5 left: vx not flipped"
    assert jnp.allclose(q_updated, q * 0.5),                         "weight=0.5 left: charge wrong"
    assert jnp.allclose(q_m_updated, q_m),                           "weight=0.5 left: q_m must not change"
    assert jnp.allclose(m_updated, m * 0.5),                         "weight=0.5 left: mass wrong"

    x_n = jnp.array([1.1, 0.0, 0.0])
    v_n = jnp.array([1.0, 0.5, 0.3])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=0, BC_right=3, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
    )
    assert jnp.allclose(x_n_updated, jnp.array([0.9, 0.0, 0.0])),  "weight=0.5 right: position not reflected"
    assert jnp.allclose(v_n_updated, jnp.array([-1.0, 0.5, 0.3])), "weight=0.5 right: vx not flipped"
    assert jnp.allclose(q_updated, q * 0.5),                        "weight=0.5 right: charge wrong"
    assert jnp.allclose(q_m_updated, q_m),                          "weight=0.5 right: q_m must not change"
    assert jnp.allclose(m_updated, m * 0.5),                        "weight=0.5 right: mass wrong"

    # --- weight=0.25: 75% absorbed, 25% reflected ---
    x_n = jnp.array([-1.1, 0.0, 0.0])
    v_n = jnp.array([-1.0, 0.5, 0.3])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=3, BC_right=0, mixed_BC_weight=0.25, COR_left=1.0, COR_right=1.0
    )
    assert jnp.allclose(x_n_updated, jnp.array([-0.9, 0.0, 0.0])), "weight=0.25 left: position not reflected"
    assert jnp.allclose(v_n_updated, jnp.array([1.0, 0.5, 0.3])),   "weight=0.25 left: vx not flipped"
    assert jnp.allclose(q_updated, q * 0.25),                        "weight=0.25 left: charge wrong"
    assert jnp.allclose(q_m_updated, q_m),                           "weight=0.25 left: q_m must not change"
    assert jnp.allclose(m_updated, m * 0.25),                        "weight=0.25 left: mass wrong"

    x_n = jnp.array([1.1, 0.0, 0.0])
    v_n = jnp.array([1.0, 0.5, 0.3])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=0, BC_right=3, mixed_BC_weight=0.25, COR_left=1.0, COR_right=1.0
    )
    assert jnp.allclose(x_n_updated, jnp.array([0.9, 0.0, 0.0])),   "weight=0.25 right: position not reflected"
    assert jnp.allclose(v_n_updated, jnp.array([-1.0, 0.5, 0.3])),  "weight=0.25 right: vx not flipped"
    assert jnp.allclose(q_updated, q * 0.25),                        "weight=0.25 right: charge wrong"
    assert jnp.allclose(q_m_updated, q_m),                           "weight=0.25 right: q_m must not change"
    assert jnp.allclose(m_updated, m * 0.25),                        "weight=0.25 right: mass wrong"

    # --- weight=0.75: 25% absorbed, 75% reflected ---
    x_n = jnp.array([-1.1, 0.0, 0.0])
    v_n = jnp.array([-1.0, 0.5, 0.3])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=3, BC_right=0, mixed_BC_weight=0.75, COR_left=1.0, COR_right=1.0
    )
    assert jnp.allclose(x_n_updated, jnp.array([-0.9, 0.0, 0.0])), "weight=0.75 left: position not reflected"
    assert jnp.allclose(v_n_updated, jnp.array([1.0, 0.5, 0.3])),   "weight=0.75 left: vx not flipped"
    assert jnp.allclose(q_updated, q * 0.75),                        "weight=0.75 left: charge wrong"
    assert jnp.allclose(q_m_updated, q_m),                           "weight=0.75 left: q_m must not change"
    assert jnp.allclose(m_updated, m * 0.75),                        "weight=0.75 left: mass wrong"

    x_n = jnp.array([1.1, 0.0, 0.0])
    v_n = jnp.array([1.0, 0.5, 0.3])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=0, BC_right=3, mixed_BC_weight=0.75, COR_left=1.0, COR_right=1.0
    )
    assert jnp.allclose(x_n_updated, jnp.array([0.9, 0.0, 0.0])),   "weight=0.75 right: position not reflected"
    assert jnp.allclose(v_n_updated, jnp.array([-1.0, 0.5, 0.3])),  "weight=0.75 right: vx not flipped"
    assert jnp.allclose(q_updated, q * 0.75),                        "weight=0.75 right: charge wrong"
    assert jnp.allclose(q_m_updated, q_m),                           "weight=0.75 right: q_m must not change"
    assert jnp.allclose(m_updated, m * 0.75),                        "weight=0.75 right: mass wrong"


def test_set_BC_single_particle_bc4():
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0
    q = 2.0
    q_m = 0.5
    m = 4.0

    # --- Left wall: |vx|=0.6, max_vx=1.0 → weight=0.4 ---
    x_n = jnp.array([-1.1, 0.0, 0.0])
    v_n = jnp.array([-0.6, 0.5, 0.3])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=4, BC_right=0, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0, max_vx=1.0
    )
    assert jnp.allclose(x_n_updated, jnp.array([-0.9, 0.0, 0.0])),  "bc4 left: position not reflected"
    assert jnp.allclose(v_n_updated, jnp.array([0.6, 0.5, 0.3])),   "bc4 left: vx not flipped"
    assert jnp.allclose(q_updated, q * 0.4),                         "bc4 left: charge wrong"
    assert jnp.allclose(q_m_updated, q_m),                           "bc4 left: q_m must not change"
    assert jnp.allclose(m_updated, m * 0.4),                         "bc4 left: mass wrong"

    # --- Right wall: |vx|=0.8, max_vx=1.0 → weight=0.2 ---
    x_n = jnp.array([1.1, 0.0, 0.0])
    v_n = jnp.array([0.8, 0.5, 0.3])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=0, BC_right=4, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0, max_vx=1.0
    )
    assert jnp.allclose(x_n_updated, jnp.array([0.9, 0.0, 0.0])),   "bc4 right: position not reflected"
    assert jnp.allclose(v_n_updated, jnp.array([-0.8, 0.5, 0.3])),  "bc4 right: vx not flipped"
    assert jnp.allclose(q_updated, q * 0.2),                         "bc4 right: charge wrong"
    assert jnp.allclose(q_m_updated, q_m),                           "bc4 right: q_m must not change"
    assert jnp.allclose(m_updated, m * 0.2),                         "bc4 right: mass wrong"

    # --- Left wall: |vx|=1.0, max_vx=2.0 → weight=0.5 (tests normalization scaling) ---
    x_n = jnp.array([-1.1, 0.0, 0.0])
    v_n = jnp.array([-1.0, 0.5, 0.3])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=4, BC_right=0, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0, max_vx=2.0
    )
    assert jnp.allclose(q_updated, q * 0.5),   "bc4 max_vx=2.0: charge wrong"
    assert jnp.allclose(q_m_updated, q_m),      "bc4 max_vx=2.0: q_m must not change"
    assert jnp.allclose(m_updated, m * 0.5),   "bc4 max_vx=2.0: mass wrong"

    # --- Inside domain: no BC applied ---
    x_n = jnp.array([0.0, 0.0, 0.0])
    v_n = jnp.array([0.5, 0.5, 0.5])
    x_n_updated, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=4, BC_right=4, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0, max_vx=1.0
    )
    assert jnp.allclose(q_updated, q),    "bc4 inside: charge should not change"
    assert jnp.allclose(q_m_updated, q_m), "bc4 inside: q_m should not change"
    assert jnp.allclose(m_updated, m),    "bc4 inside: mass should not change"


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
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
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
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
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
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
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

def test_set_BC_particles_bc3():
    # Three particles: one hits right wall, one hits left wall, one stays inside.
    xs_n = jnp.array([[6.0, 0.0, 0.0], [-6.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    vs_n = jnp.array([[1.0, 0.5, 0.3], [-1.0, 0.5, 0.3], [0.5, 0.5, 0.5]])
    qs   = jnp.array([2.0, 2.0, 2.0])
    ms   = jnp.array([4.0, 4.0, 4.0])
    q_ms = jnp.array([0.5, 0.5, 0.5])
    dx = 0.1
    grid = jnp.linspace(-5.0, 5.0, 100)
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0

    # --- weight=0.5 ---
    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=3, BC_right=3, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
    )
    assert jnp.allclose(xs_n_updated, jnp.array([[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])), "weight=0.5: positions wrong"
    assert jnp.allclose(vs_n_updated, jnp.array([[-1.0, 0.5, 0.3], [1.0, 0.5, 0.3], [0.5, 0.5, 0.5]])), "weight=0.5: velocities wrong"
    assert jnp.allclose(qs_updated,   jnp.array([2.0 * 0.5, 2.0 * 0.5, 2.0])),  "weight=0.5: charges wrong"
    assert jnp.allclose(ms_updated,   jnp.array([4.0 * 0.5, 4.0 * 0.5, 4.0])),  "weight=0.5: masses wrong"
    assert jnp.allclose(q_ms_updated, q_ms),                                      "weight=0.5: q_m must not change"

    # --- weight=0.25: 75% absorbed ---
    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=3, BC_right=3, mixed_BC_weight=0.25, COR_left=1.0, COR_right=1.0
    )
    assert jnp.allclose(qs_updated,   jnp.array([2.0 * 0.25, 2.0 * 0.25, 2.0])), "weight=0.25: charges wrong"
    assert jnp.allclose(ms_updated,   jnp.array([4.0 * 0.25, 4.0 * 0.25, 4.0])), "weight=0.25: masses wrong"
    assert jnp.allclose(q_ms_updated, q_ms),                                       "weight=0.25: q_m must not change"

    # --- weight=0.75: 25% absorbed ---
    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=3, BC_right=3, mixed_BC_weight=0.75, COR_left=1.0, COR_right=1.0
    )
    assert jnp.allclose(qs_updated,   jnp.array([2.0 * 0.75, 2.0 * 0.75, 2.0])), "weight=0.75: charges wrong"
    assert jnp.allclose(ms_updated,   jnp.array([4.0 * 0.75, 4.0 * 0.75, 4.0])), "weight=0.75: masses wrong"
    assert jnp.allclose(q_ms_updated, q_ms),                                       "weight=0.75: q_m must not change"


def test_set_BC_particles_bc4():
    # Three particles: one hits right wall (fastest), one hits left wall (slower), one inside.
    # max_vx is auto-computed as max(|vx|) = 2.0 (from particle 0).
    xs_n = jnp.array([[6.0, 0.0, 0.0], [-6.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    vs_n = jnp.array([[2.0, 0.5, 0.3], [-1.0, 0.5, 0.3], [0.5, 0.5, 0.5]])
    qs   = jnp.array([2.0, 2.0, 2.0])
    ms   = jnp.array([4.0, 4.0, 4.0])
    q_ms = jnp.array([0.5, 0.5, 0.5])
    dx = 0.1
    grid = jnp.linspace(-5.0, 5.0, 100)
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0

    xs_n_updated, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=4, BC_right=4, mixed_BC_weight=0.5, COR_left=1.0, COR_right=1.0
    )
    # max_vx = 2.0
    # Particle 0 (hit right, vx=2.0): weight = clamp(1 - 2.0/2.0, 0, 1) = 0.0
    # Particle 1 (hit left, vx=-1.0 → flipped to 1.0): weight = clamp(1 - 1.0/2.0, 0, 1) = 0.5
    # Particle 2 (inside): no change
    assert jnp.allclose(xs_n_updated, jnp.array([[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])), "bc4 batch: positions wrong"
    assert jnp.allclose(vs_n_updated, jnp.array([[-2.0, 0.5, 0.3], [1.0, 0.5, 0.3], [0.5, 0.5, 0.5]])),  "bc4 batch: velocities wrong"
    assert jnp.allclose(qs_updated,   jnp.array([2.0 * 0.0, 2.0 * 0.5, 2.0]), atol=1e-5),  "bc4 batch: charges wrong"
    assert jnp.allclose(ms_updated,   jnp.array([4.0 * 0.0, 4.0 * 0.5, 4.0]), atol=1e-5),  "bc4 batch: masses wrong"
    assert jnp.allclose(q_ms_updated, q_ms),                                      "bc4 batch: q_m must not change"


def test_set_BC_single_particle_COR():
    q = 2.0
    q_m = 0.5
    m = 4.0
    dx = 0.1
    grid = jnp.linspace(-1.0, 1.0, 10)
    box_size_x = 2.0
    box_size_y = 2.0
    box_size_z = 2.0

    # --- BC=1, left wall, COR_left=0.5: vx flipped and halved ---
    x_n = jnp.array([-1.1, 0.0, 0.0])
    v_n = jnp.array([1.0, 0.5, 0.3])
    _, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z,
        BC_left=1, BC_right=1, mixed_BC_weight=0.5, COR_left=0.5, COR_right=1.0
    )
    assert jnp.allclose(v_n_updated[0], -0.5 * 1.0), "COR left BC=1: vx should be -COR*vx"
    assert jnp.allclose(v_n_updated[1:], v_n[1:]),    "COR left BC=1: vy, vz unchanged"
    assert jnp.allclose(q_updated, q),                "COR left BC=1: charge unchanged"
    assert jnp.allclose(q_m_updated, q_m),            "COR left BC=1: q_m unchanged"
    assert jnp.allclose(m_updated, m),                "COR left BC=1: mass unchanged"

    # --- BC=1, right wall, COR_right=0.8: vx flipped and scaled ---
    x_n = jnp.array([1.1, 0.0, 0.0])
    v_n = jnp.array([-1.0, 0.5, 0.3])
    _, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z,
        BC_left=1, BC_right=1, mixed_BC_weight=0.5, COR_left=1.0, COR_right=0.8
    )
    assert jnp.allclose(v_n_updated[0], -0.8 * (-1.0)), "COR right BC=1: vx should be -COR*vx"
    assert jnp.allclose(v_n_updated[1:], v_n[1:]),       "COR right BC=1: vy, vz unchanged"

    # --- BC=3, left wall, COR_left=0.5: vx scaled AND q/m halved independently ---
    x_n = jnp.array([-1.1, 0.0, 0.0])
    v_n = jnp.array([1.0, 0.5, 0.3])
    _, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z,
        BC_left=3, BC_right=0, mixed_BC_weight=0.5, COR_left=0.5, COR_right=1.0
    )
    assert jnp.allclose(v_n_updated[0], -0.5 * 1.0), "COR+BC3: vx should be -COR*vx"
    assert jnp.allclose(q_updated, q * 0.5),          "COR+BC3: charge halved by mixed_BC_weight"
    assert jnp.allclose(m_updated, m * 0.5),          "COR+BC3: mass halved by mixed_BC_weight"
    assert jnp.allclose(q_m_updated, q_m),            "COR+BC3: q_m unchanged"

    # --- Inside domain: COR has no effect ---
    x_n = jnp.array([0.0, 0.0, 0.0])
    v_n = jnp.array([1.0, 0.5, 0.3])
    _, v_n_updated, q_updated, q_m_updated, m_updated = set_BC_single_particle(
        x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z,
        BC_left=1, BC_right=1, mixed_BC_weight=0.5, COR_left=0.5, COR_right=0.5
    )
    assert jnp.allclose(v_n_updated, v_n), "COR inside: velocity unchanged"
    assert jnp.allclose(q_updated, q),     "COR inside: charge unchanged"


def test_set_BC_particles_COR():
    # Two particles both hitting walls with COR != 1
    xs_n = jnp.array([[6.0, 0.0, 0.0], [-6.0, 0.0, 0.0]])
    vs_n = jnp.array([[1.0, 0.5, 0.3], [-1.0, 0.5, 0.3]])
    qs   = jnp.array([2.0, 2.0])
    ms   = jnp.array([4.0, 4.0])
    q_ms = jnp.array([0.5, 0.5])
    dx = 0.1
    grid = jnp.linspace(-5.0, 5.0, 100)
    box_size_x = 10.0
    box_size_y = 10.0
    box_size_z = 10.0

    _, vs_n_updated, qs_updated, ms_updated, q_ms_updated = set_BC_particles(
        xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z,
        BC_left=1, BC_right=1, mixed_BC_weight=0.5, COR_left=0.6, COR_right=0.4
    )
    # Particle 0 hits right wall: vx = -COR_right * 1.0 = -0.4
    assert jnp.allclose(vs_n_updated[0, 0], -0.4), "COR batch: right wall vx wrong"
    assert jnp.allclose(vs_n_updated[0, 1:], vs_n[0, 1:]), "COR batch: vy,vz unchanged right"
    # Particle 1 hits left wall: vx = -COR_left * (-1.0) = 0.6
    assert jnp.allclose(vs_n_updated[1, 0], 0.6), "COR batch: left wall vx wrong"
    assert jnp.allclose(vs_n_updated[1, 1:], vs_n[1, 1:]), "COR batch: vy,vz unchanged left"
    # Charges and masses unchanged for BC=1
    assert jnp.allclose(qs_updated, qs),   "COR batch BC=1: charges unchanged"
    assert jnp.allclose(ms_updated, ms),   "COR batch BC=1: masses unchanged"
    assert jnp.allclose(q_ms_updated, q_ms), "COR batch BC=1: q_m unchanged"


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

    BC_left = 3
    BC_right = 3
    # Particle exits left wall with BC_left=3: reflected like BC=1
    x_n = jnp.array([-1.1, 0.0, 0.0])
    x_n_updated = set_BC_single_particle_positions(
        x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=3, BC_right=0
    )
    assert jnp.allclose(x_n_updated, jnp.array([-0.9, 0.0, 0.0])), "Mixed BC positions left: position not reflected"

    # Particle exits right wall with BC_right=3: reflected like BC=1
    x_n = jnp.array([1.1, 0.0, 0.0])
    x_n_updated = set_BC_single_particle_positions(
        x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=0, BC_right=3
    )
    assert jnp.allclose(x_n_updated, jnp.array([0.9, 0.0, 0.0])), "Mixed BC positions right: position not reflected"

    # BC=4 reflects position identically to BC=1
    x_n = jnp.array([-1.1, 0.0, 0.0])
    x_n_updated = set_BC_single_particle_positions(
        x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=4, BC_right=0
    )
    assert jnp.allclose(x_n_updated, jnp.array([-0.9, 0.0, 0.0])), "BC=4 positions left: position not reflected"

    x_n = jnp.array([1.1, 0.0, 0.0])
    x_n_updated = set_BC_single_particle_positions(
        x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left=0, BC_right=4
    )
    assert jnp.allclose(x_n_updated, jnp.array([0.9, 0.0, 0.0])), "BC=4 positions right: position not reflected"

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