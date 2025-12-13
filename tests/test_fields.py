# tests/test_fields.py
import pytest
import numpy as np

import jax
import jax.numpy as jnp

from jaxincell._fields import (
    E_from_Gauss_1D_FFT,
    E_from_Poisson_1D_FFT,
    E_from_Gauss_1D_Cartesian,
    curlE,
    curlB,
    field_update,
    field_update1,
    field_update2,
)
from jaxincell._boundary_conditions import (
    field_ghost_cells_E,
    field_ghost_cells_B,
    field_2_ghost_cells,
    set_BC_single_particle,
    set_BC_particles,
    set_BC_positions,
)
from jaxincell._constants import epsilon_0, speed_of_light


# -----------------------------
# small helpers
# -----------------------------
def _zero_mean(x):
    return x - jnp.mean(x)

def _project_zero_and_nyquist(rho):
    """Project out k=0 and Nyquist (even N) to match FFT Gauss/Poisson solvers."""
    n = rho.shape[0]
    rk = jnp.fft.fft(rho)
    rk = rk.at[0].set(0.0)
    if n % 2 == 0:
        rk = rk.at[n // 2].set(0.0)
    return jnp.fft.ifft(rk).real

def _finite_diff_grad_scalar(fun, x, idx, eps=1e-6):
    """
    Central finite difference for scalar fun(x) with respect to x[idx].
    x: 1D array
    """
    x = np.array(x, dtype=np.float64)
    xp = x.copy()
    xm = x.copy()
    xp[idx] += eps
    xm[idx] -= eps
    fp = float(fun(jnp.array(xp)))
    fm = float(fun(jnp.array(xm)))
    return (fp - fm) / (2 * eps)


# ============================================================
# FFT field solvers
# ============================================================
def test_gauss_fft_satisfies_gauss_law_periodic_zero_mean():
    nx = 32
    dx = 0.01
    key = jax.random.PRNGKey(0)

    rho = jax.random.normal(key, (nx,))
    rho = _zero_mean(rho)  # k=0 is special; make it clean
    rho = _project_zero_and_nyquist(rho)

    E = E_from_Gauss_1D_FFT(rho, dx)

    dE_dx = (E - jnp.roll(E, 1)) / dx
    assert jnp.allclose(dE_dx, rho / epsilon_0, rtol=1e-6, atol=1e-6)


def test_poisson_fft_matches_gauss_fft_for_zero_mean_rho():
    nx = 32
    dx = 0.02
    key = jax.random.PRNGKey(1)

    rho = jax.random.normal(key, (nx,))
    rho = _zero_mean(rho)

    E_gauss = E_from_Gauss_1D_FFT(rho, dx)
    E_pois = E_from_Poisson_1D_FFT(rho, dx)

    assert jnp.allclose(E_gauss, E_pois, rtol=1e-6, atol=1e-6)


def test_cartesian_gauss_matches_fft_up_to_constant_offset():
    nx = 16
    dx = 0.03
    key = jax.random.PRNGKey(2)

    rho = jax.random.normal(key, (nx,))
    rho = _zero_mean(rho)
    rho = _project_zero_and_nyquist(rho)

    E_fft = E_from_Gauss_1D_FFT(rho, dx)
    E_cart = E_from_Gauss_1D_Cartesian(rho, dx)

    # Solutions can differ by a constant; compare after removing mean.
    assert jnp.allclose(_zero_mean(E_cart), _zero_mean(E_fft), rtol=1e-4, atol=1e-4)


# ============================================================
# Gradient checks (FFT solver)
# ============================================================
def test_poisson_fft_grad_matches_finite_difference():
    nx = 32
    dx = 0.01
    key = jax.random.PRNGKey(3)

    rho0 = jax.random.normal(key, (nx,))
    rho0 = _zero_mean(rho0)

    def loss(rho):
        E = E_from_Poisson_1D_FFT(rho, dx)
        return jnp.sum(E**2)

    g = jax.grad(loss)(rho0)

    # finite-diff check a few coordinates
    rho0_np = np.array(rho0, dtype=np.float64)
    for idx in (0, 7, 19):
        g_fd = _finite_diff_grad_scalar(loss, rho0_np, idx, eps=1e-6)
        assert np.isfinite(g_fd)
        assert np.allclose(float(g[idx]), g_fd, rtol=5e-3, atol=5e-3)


# ============================================================
# Field ghost cells (boundary_conditions)
# ============================================================
@pytest.mark.parametrize("bcL,bcR", [(0,0), (1,1), (2,2), (3,3), (0,1), (1,2), (2,0)])
def test_field_ghost_cells_E_B_all_branches(bcL, bcR):
    G = 8
    key = jax.random.PRNGKey(4)
    E = jax.random.normal(key, (G, 3))
    B = jax.random.normal(key, (G, 3))

    gL, gR = field_ghost_cells_E(bcL, bcR, E, B)
    hL, hR = field_ghost_cells_B(bcL, bcR, B, E)

    # Periodic
    if bcL == 0:
        assert jnp.allclose(gL, E[-1])
        assert jnp.allclose(hL, B[-1])
    if bcR == 0:
        assert jnp.allclose(gR, E[0])
        assert jnp.allclose(hR, B[0])

    # Reflective
    if bcL == 1:
        assert jnp.allclose(gL, E[0])
        assert jnp.allclose(hL, B[0])
    if bcR == 1:
        assert jnp.allclose(gR, E[-1])
        assert jnp.allclose(hR, B[-1])

    # Absorbing: check exact formulas (from your _boundary_conditions.py)
    if bcL == 2:
        expect_gL = jnp.array([0.0,
                               -2 * speed_of_light * B[0, 2] - E[0, 1],
                               2 * speed_of_light * B[0, 1] - E[0, 2]])
        expect_hL = jnp.array([0.0,
                               3 * B[0, 1] - (2 / speed_of_light) * E[0, 2],
                               3 * B[0, 2] + (2 / speed_of_light) * E[0, 1]])
        assert jnp.allclose(gL, expect_gL)
        assert jnp.allclose(hL, expect_hL)

    if bcR == 2:
        expect_gR = jnp.array([0.0,
                               3 * E[-1, 1] - 2 * speed_of_light * B[-1, 2],
                               3 * E[-1, 2] + 2 * speed_of_light * B[-1, 1]])
        expect_hR = jnp.array([0.0,
                               -(2 / speed_of_light) * E[-1, 2] - B[-1, 1],
                               (2 / speed_of_light) * E[-1, 1] - B[-1, 2]])
        assert jnp.allclose(gR, expect_gR)
        assert jnp.allclose(hR, expect_hR)

    # Custom (3) returns zeros in your implementation
    if bcL == 3:
        assert jnp.allclose(gL, jnp.zeros((3,)))
        assert jnp.allclose(hL, jnp.zeros((3,)))
    if bcR == 3:
        assert jnp.allclose(gR, jnp.zeros((3,)))
        assert jnp.allclose(hR, jnp.zeros((3,)))


def test_field_2_ghost_cells_branches():
    G = 6
    field = jnp.arange(G * 3, dtype=jnp.float32).reshape(G, 3)

    # periodic
    L2, L1, R = field_2_ghost_cells(0, 0, field)
    assert jnp.allclose(L2, field[-2])
    assert jnp.allclose(L1, field[-1])
    assert jnp.allclose(R, field[0])

    # reflective
    L2, L1, R = field_2_ghost_cells(1, 1, field)
    assert jnp.allclose(L2, field[1])
    assert jnp.allclose(L1, field[0])
    assert jnp.allclose(R, field[-1])

    # absorbing/custom -> zeros in your code
    L2, L1, R = field_2_ghost_cells(2, 2, field)
    assert jnp.allclose(L2, jnp.zeros((3,)))
    assert jnp.allclose(L1, jnp.zeros((3,)))
    assert jnp.allclose(R, jnp.zeros((3,)))


# ============================================================
# curlE / curlB correctness (periodic)
# ============================================================
def test_curlE_periodic_matches_backward_difference():
    G = 16
    dx = 0.1
    dt = 0.01

    # Construct fields with known gradients
    x = jnp.arange(G, dtype=jnp.float32)
    E = jnp.zeros((G, 3), dtype=jnp.float32)
    B = jnp.zeros((G, 3), dtype=jnp.float32)

    E = E.at[:, 1].set(2.0 * x)   # Ey
    E = E.at[:, 2].set(-3.0 * x)  # Ez

    out = curlE(E, B, dx, dt, field_BC_left=0, field_BC_right=0)

    # backward diff with periodic wrap: dF/dx[i] = (F[i] - F[i-1]) / dx
    dEz = (E[:, 2] - jnp.roll(E[:, 2], 1)) / dx
    dEy = (E[:, 1] - jnp.roll(E[:, 1], 1)) / dx

    expected = jnp.stack([jnp.zeros(G), -dEz, dEy], axis=1)
    assert out.shape == (G, 3)
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_curlB_periodic_matches_forward_difference_after_center_shift():
    G = 16
    dx = 0.2
    dt = 0.01

    x = jnp.arange(G, dtype=jnp.float32)
    B = jnp.zeros((G, 3), dtype=jnp.float32)
    E = jnp.zeros((G, 3), dtype=jnp.float32)

    # Only By, Bz matter for curlB result
    B = B.at[:, 1].set(1.5 * x)
    B = B.at[:, 2].set(-0.5 * x)

    out = curlB(B, E, dx, dt, field_BC_left=0, field_BC_right=0)

    # From your implementation: after ghosting, B_field is rolled by -1, then backward diff on that
    # which is equivalent to forward diff on the original periodic array:
    # dF/dx[i] = (F[i+1] - F[i]) / dx
    dBz = (jnp.roll(B[:, 2], -1) - B[:, 2]) / dx
    dBy = (jnp.roll(B[:, 1], -1) - B[:, 1]) / dx
    expected = jnp.stack([jnp.zeros(G), -dBz, dBy], axis=1)

    assert out.shape == (G, 3)
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)


# ============================================================
# Gradient check (curlE)
# ============================================================
def test_curlE_grad_matches_finite_difference():
    G = 12
    dx = 0.1
    dt = 0.01
    key = jax.random.PRNGKey(5)

    E0 = jax.random.normal(key, (G, 3))
    B0 = jnp.zeros((G, 3))

    # scalar loss for grad
    def loss(E):
        c = curlE(E, B0, dx, dt, field_BC_left=0, field_BC_right=0)
        return jnp.sum(c**2)

    g = jax.grad(loss)(E0)

    # finite diff on one entry (avoid doing too many for runtime)
    E0_np = np.array(E0, dtype=np.float64)
    i, j = 3, 2  # Ez at i=3
    eps = 1e-6

    def loss_np(E_flat):
        E = E_flat.reshape(G, 3)
        return float(loss(jnp.array(E)))

    E_flat = E0_np.reshape(-1)
    idx_flat = i * 3 + j
    fd = _finite_diff_grad_scalar(lambda ef: loss_np(np.array(ef, dtype=np.float64)), E_flat, idx_flat, eps=eps)

    assert np.isfinite(fd)
    assert np.allclose(float(g[i, j]), fd, rtol=5e-3, atol=5e-3)


# ============================================================
# field_update / field_update1 / field_update2 consistency in a regime where they should match
# ============================================================
def test_field_update_variants_match_when_curls_zero():
    """
    If E and B are uniform (no spatial gradients), curlE=curlB=0,
    so all update orderings must agree and only the current term updates E.
    """
    G = 10
    dx = 0.1
    dt = 0.05

    E = jnp.ones((G, 3)) * 0.0
    B = jnp.ones((G, 3)) * 0.0
    j = jnp.ones((G, 3)) * 2.5

    bcL, bcR = 0, 0

    E_a, B_a = field_update(E, B, dx, dt, j, bcL, bcR)
    E_b, B_b = field_update1(E, B, dx, dt, j, bcL, bcR)
    E_c, B_c = field_update2(E, B, dx, dt, j, bcL, bcR)

    # expected: E += dt * ( -j/epsilon_0 ), B unchanged
    E_expect = E + dt * (-(j / epsilon_0))
    B_expect = B

    assert jnp.allclose(E_a, E_expect)
    assert jnp.allclose(E_b, E_expect)
    assert jnp.allclose(E_c, E_expect)
    assert jnp.allclose(B_a, B_expect)
    assert jnp.allclose(B_b, B_expect)
    assert jnp.allclose(B_c, B_expect)


def test_field_update_grad_wrt_current_is_correct_in_zero_curl_case():
    G = 6
    dx = 0.2
    dt = 0.03
    bcL, bcR = 0, 0

    E = jnp.zeros((G, 3))
    B = jnp.zeros((G, 3))
    j0 = jnp.ones((G, 3)) * 1.2

    def scalar_out(j_in):
        E_new, B_new = field_update(E, B, dx, dt, j_in, bcL, bcR)
        return jnp.sum(E_new)  # linear in j in this regime

    gj = jax.grad(scalar_out)(j0)
    # d/dj sum(E - dt*j/eps0) = -dt/eps0 for every entry
    assert jnp.allclose(gj, jnp.ones_like(j0) * (-(dt / epsilon_0)))


# ============================================================
# Particle BCs (brief but covers BC branches)
# ============================================================
def test_set_BC_single_particle_periodic_wrap():
    dx = 0.1
    box_x, box_y, box_z = 1.0, 2.0, 3.0
    grid = jnp.linspace(-box_x / 2, box_x / 2, 11)

    x = jnp.array([-0.6, 1.2, -1.7])  # x out left, y/z out but periodic in y/z
    v = jnp.array([1.0, 2.0, 3.0])
    q = 1.0
    q_m = 2.0

    x2, v2, q2, qm2 = set_BC_single_particle(
        x, v, q, q_m, dx, grid, box_x, box_y, box_z, BC_left=0, BC_right=0
    )

    # x periodic wrap into [-0.5,0.5)
    assert -box_x / 2 <= float(x2[0]) <= box_x / 2
    # y/z always periodic
    assert -box_y / 2 <= float(x2[1]) <= box_y / 2
    assert -box_z / 2 <= float(x2[2]) <= box_z / 2
    # velocity unchanged for periodic
    assert jnp.allclose(v2, v)
    assert q2 == q
    assert qm2 == q_m


def test_set_BC_single_particle_reflective_flips_vx():
    dx = 0.1
    box_x, box_y, box_z = 1.0, 2.0, 3.0
    grid = jnp.linspace(-box_x / 2, box_x / 2, 11)

    x = jnp.array([0.8, 0.0, 0.0])  # out right
    v = jnp.array([1.0, 2.0, 3.0])
    q = 1.0
    q_m = 2.0

    x2, v2, q2, qm2 = set_BC_single_particle(
        x, v, q, q_m, dx, grid, box_x, box_y, box_z, BC_left=1, BC_right=1
    )

    # reflective right: x -> box_x - x
    assert np.isclose(float(x2[0]), box_x - float(x[0]))
    # vx flips sign
    assert np.isclose(float(v2[0]), -float(v[0]))
    assert np.isclose(float(v2[1]), float(v[1]))
    assert np.isclose(float(v2[2]), float(v[2]))
    assert q2 == q
    assert qm2 == q_m


def test_set_BC_single_particle_absorbing_zeroes_charge_and_velocity():
    dx = 0.1
    box_x, box_y, box_z = 1.0, 2.0, 3.0
    grid = jnp.linspace(-box_x / 2, box_x / 2, 11)

    x = jnp.array([0.8, 0.0, 0.0])  # out right
    v = jnp.array([1.0, 2.0, 3.0])
    q = 1.0
    q_m = 2.0

    x2, v2, q2, qm2 = set_BC_single_particle(
        x, v, q, q_m, dx, grid, box_x, box_y, box_z, BC_left=2, BC_right=2
    )

    # absorbing right: x placed beyond grid, velocity zero, charge zero
    assert np.isclose(float(v2[0]), 0.0) and np.isclose(float(v2[1]), 0.0) and np.isclose(float(v2[2]), 0.0)
    assert np.isclose(float(q2), 0.0)
    assert np.isclose(float(qm2), 0.0)


def test_set_BC_particles_and_positions_vectorized_runs():
    dx = 0.1
    box_x, box_y, box_z = 1.0, 2.0, 3.0
    grid = jnp.linspace(-box_x / 2, box_x / 2, 11)

    xs = jnp.array([
        [-0.6, 0.0, 0.0],  # out left
        [0.2,  0.0, 0.0],  # in
        [0.8,  0.0, 0.0],  # out right
    ])
    vs = jnp.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    qs = jnp.array([1.0, 1.0, 1.0])
    ms = jnp.array([1.0, 1.0, 1.0])
    qms = qs / ms

    xs2, vs2, qs2, ms2, qms2 = set_BC_particles(xs, vs, qs, ms, qms, dx, grid, box_x, box_y, box_z, 0, 0)
    assert xs2.shape == xs.shape
    assert vs2.shape == vs.shape
    assert qs2.shape == qs.shape
    assert ms2.shape == ms.shape
    assert qms2.shape == qms.shape

    xs3 = set_BC_positions(xs, qs, dx, grid, box_x, box_y, box_z, 0, 0)
    assert xs3.shape == xs.shape
