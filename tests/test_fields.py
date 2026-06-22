"""
    Someone needs to look through these tests and make sure they cover the
    intended cases, and that the expected results are correct.

    They were written using Codex and I did go through at a high level
    to make sure the tests are reasonable, but a more in depth review
    would be valuable just in case.
"""

import pytest
import jax.numpy as jnp
from jax import grad
from numpy.testing import assert_allclose

from jaxincell._constants import epsilon_0, speed_of_light
from jaxincell._fields import (
    E_from_Gauss_1D_Cartesian,
    E_from_Gauss_1D_FFT,
    E_from_Poisson_1D_FFT,
    curlB,
    curlE,
    field_update,
    field_update1,
    field_update2,
)


def test_E_from_Gauss_1D_FFT_zero_mode_and_shape():
    """Test jaxincell._fields.E_from_Gauss_1D_FFT.

    Cases covered:
    - zero charge density returns a finite zero electric field with matching shape.
    - a balanced sinusoidal charge density produces finite values with no NaNs.
    - sine and cosine Fourier modes get the expected amplitude and phase.
    - the spectral derivative of E satisfies Gauss' law for nonzero modes.
    - the k=0 protection branch is exercised without leaking a DC field.
    """
    grid_size = 16
    dx = 0.25
    x = dx * jnp.arange(grid_size)
    wavenumber = 2 * jnp.pi * 2 / (grid_size * dx)

    zero_charge = jnp.zeros(grid_size)
    zero_field = E_from_Gauss_1D_FFT(zero_charge, dx)
    assert zero_field.shape == zero_charge.shape
    assert bool(jnp.all(jnp.isfinite(zero_field)))
    assert_allclose(zero_field, jnp.zeros_like(zero_field), atol=0)

    constant_charge = epsilon_0 * jnp.ones(grid_size)
    dc_field = E_from_Gauss_1D_FFT(constant_charge, dx)
    assert_allclose(dc_field, jnp.zeros_like(dc_field), atol=1e-12)

    sinusoidal_charge = epsilon_0 * jnp.sin(wavenumber * x)
    sinusoidal_field = E_from_Gauss_1D_FFT(sinusoidal_charge, dx)
    expected_field = -jnp.cos(wavenumber * x) / wavenumber
    assert bool(jnp.all(jnp.isfinite(sinusoidal_field)))
    assert_allclose(sinusoidal_field, expected_field, rtol=1e-6, atol=1e-12)

    cosine_charge = 3.5 * epsilon_0 * jnp.cos(wavenumber * x)
    cosine_field = E_from_Gauss_1D_FFT(cosine_charge, dx)
    expected_cosine_field = 3.5 * jnp.sin(wavenumber * x) / wavenumber
    assert_allclose(cosine_field, expected_cosine_field, rtol=1e-6, atol=1e-12)

    kx = jnp.fft.fftfreq(grid_size, d=dx) * 2 * jnp.pi
    spectral_divergence = 1j * kx * jnp.fft.fft(sinusoidal_field)
    spectral_charge_density = jnp.fft.fft(sinusoidal_charge) / epsilon_0
    assert_allclose(
        spectral_divergence[1:],
        spectral_charge_density[1:],
        rtol=1e-6,
        atol=1e-10,
    )


def test_E_from_Poisson_1D_FFT_zero_average_and_sinusoid():
    """Test jaxincell._fields.E_from_Poisson_1D_FFT.

    Cases covered:
    - zero charge density returns a finite zero electric field.
    - a single Fourier-mode charge density gives the expected phase-shifted field.
    - sine and cosine Fourier modes agree with the Gauss FFT solver for zero-mean charge.
    - the phi_k[0] zero-average branch removes the potential DC mode.
    """
    grid_size = 16
    dx = 0.25
    x = dx * jnp.arange(grid_size)
    wavenumber = 2 * jnp.pi * 2 / (grid_size * dx)

    zero_charge = jnp.zeros(grid_size)
    zero_field = E_from_Poisson_1D_FFT(zero_charge, dx)
    assert zero_field.shape == zero_charge.shape
    assert bool(jnp.all(jnp.isfinite(zero_field)))
    assert_allclose(zero_field, jnp.zeros_like(zero_field), atol=0)

    sinusoidal_charge = epsilon_0 * jnp.sin(wavenumber * x)
    expected_field = -jnp.cos(wavenumber * x) / wavenumber
    sinusoidal_field = E_from_Poisson_1D_FFT(sinusoidal_charge, dx)
    assert_allclose(sinusoidal_field, expected_field, rtol=1e-6, atol=1e-12)
    assert_allclose(
        sinusoidal_field,
        E_from_Gauss_1D_FFT(sinusoidal_charge, dx),
        rtol=1e-6,
        atol=1e-12,
    )

    cosine_charge = 3.5 * epsilon_0 * jnp.cos(wavenumber * x)
    cosine_field = E_from_Poisson_1D_FFT(cosine_charge, dx)
    expected_cosine_field = 3.5 * jnp.sin(wavenumber * x) / wavenumber
    assert_allclose(cosine_field, expected_cosine_field, rtol=1e-6, atol=1e-12)
    assert_allclose(
        cosine_field,
        E_from_Gauss_1D_FFT(cosine_charge, dx),
        rtol=1e-6,
        atol=1e-12,
    )

    offset_charge = epsilon_0 * (7.0 + jnp.sin(wavenumber * x))
    offset_field = E_from_Poisson_1D_FFT(offset_charge, dx)
    assert_allclose(offset_field, expected_field, rtol=1e-6, atol=1e-12)


def test_E_from_Gauss_1D_Cartesian_solves_discrete_divergence():
    """Test jaxincell._fields.E_from_Gauss_1D_Cartesian.

    Cases covered:
    - a small hand-computable charge-density vector solves the lower-triangular divergence system.
    - output has the same grid length as the input.
    - values remain finite for balanced positive and negative charges.
    """
    dx = 0.5
    charge_density = epsilon_0 * jnp.array([1.0, -0.25, 0.5, -1.25])

    electric_field = E_from_Gauss_1D_Cartesian(charge_density, dx)
    divergence_matrix = (
        jnp.diag(jnp.ones(len(charge_density)))
        - jnp.diag(jnp.ones(len(charge_density) - 1), k=-1)
    )

    assert electric_field.shape == charge_density.shape
    assert bool(jnp.all(jnp.isfinite(electric_field)))
    assert_allclose(
        divergence_matrix @ electric_field,
        (dx / epsilon_0) * charge_density,
        rtol=1e-6,
        atol=1e-12,
    )
    assert_allclose(electric_field, jnp.array([0.5, 0.375, 0.625, 0.0]))


def test_curlE_boundary_condition_cases():
    """Test jaxincell._fields.curlE.

    Cases covered:
    - periodic, reflective, and absorbing field boundary conditions use the expected ghost cells.
    - constant E_y/E_z fields produce zero curl.
    - linear E_y/E_z fields produce the expected finite-difference signs.
    - asymmetric absorbing boundaries exercise the coupled E/B ghost-cell formula with dx != 1.
    """
    dx = 1.0
    dt = 0.1
    constant_E = jnp.tile(jnp.array([0.0, 3.0, -2.0]), (3, 1))
    zero_B = jnp.zeros_like(constant_E)

    assert_allclose(
        curlE(constant_E, zero_B, dx, dt, field_BC_left=1, field_BC_right=1),
        jnp.zeros_like(constant_E),
        atol=0,
    )

    linear_E = jnp.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 3.0, 5.0],
            [0.0, 7.0, 11.0],
        ]
    )

    reflective_expected = jnp.array(
        [
            [0.0, -0.0, 0.0],
            [0.0, -3.0, 2.0],
            [0.0, -6.0, 4.0],
        ]
    )
    assert_allclose(curlE(linear_E, zero_B, dx, dt, 1, 1), reflective_expected)

    periodic_expected = jnp.array(
        [
            [0.0, 9.0, -6.0],
            [0.0, -3.0, 2.0],
            [0.0, -6.0, 4.0],
        ]
    )
    assert_allclose(curlE(linear_E, zero_B, dx, dt, 0, 0), periodic_expected)

    absorbing_expected = jnp.array(
        [
            [0.0, -4.0, 2.0],
            [0.0, -3.0, 2.0],
            [0.0, -6.0, 4.0],
        ]
    )
    assert_allclose(curlE(linear_E, zero_B, dx, dt, 2, 2), absorbing_expected)

    dx_half = 0.5
    coupled_B = zero_B.at[0].set(jnp.array([0.0, 1e-9, -2e-9]))
    absorbing_left_ghost = jnp.array(
        [
            0.0,
            -2 * speed_of_light * coupled_B[0, 2] - linear_E[0, 1],
            2 * speed_of_light * coupled_B[0, 1] - linear_E[0, 2],
        ]
    )
    asymmetric_expected = jnp.array(
        [
            [
                0.0,
                -(linear_E[0, 2] - absorbing_left_ghost[2]) / dx_half,
                (linear_E[0, 1] - absorbing_left_ghost[1]) / dx_half,
            ],
            [
                0.0,
                -(linear_E[1, 2] - linear_E[0, 2]) / dx_half,
                (linear_E[1, 1] - linear_E[0, 1]) / dx_half,
            ],
            [
                0.0,
                -(linear_E[2, 2] - linear_E[1, 2]) / dx_half,
                (linear_E[2, 1] - linear_E[1, 1]) / dx_half,
            ],
        ]
    )
    assert_allclose(
        curlE(linear_E, coupled_B, dx_half, dt, field_BC_left=2, field_BC_right=1),
        asymmetric_expected,
        rtol=1e-6,
        atol=1e-12,
    )


def test_curlB_boundary_condition_cases():
    """Test jaxincell._fields.curlB.

    Cases covered:
    - periodic, reflective, and absorbing field boundary conditions use the expected ghost cells.
    - the internal roll by -1 is captured with a hand-computable staggered-grid example.
    - constant B_y/B_z fields produce zero curl where the chosen boundary permits it.
    - asymmetric absorbing boundaries exercise the coupled B/E ghost-cell formula with dx != 1.
    """
    dx = 1.0
    dt = 0.1
    constant_B = jnp.tile(jnp.array([0.0, 3.0, -2.0]), (3, 1))
    zero_E = jnp.zeros_like(constant_B)

    assert_allclose(
        curlB(constant_B, zero_E, dx, dt, field_BC_left=1, field_BC_right=1),
        jnp.zeros_like(constant_B),
        atol=0,
    )

    linear_B = jnp.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 3.0, 5.0],
            [0.0, 7.0, 11.0],
        ]
    )

    reflective_expected = jnp.array(
        [
            [0.0, -3.0, 2.0],
            [0.0, -6.0, 4.0],
            [0.0, -0.0, 0.0],
        ]
    )
    assert_allclose(curlB(linear_B, zero_E, dx, dt, 1, 1), reflective_expected)

    periodic_expected = jnp.array(
        [
            [0.0, -3.0, 2.0],
            [0.0, -6.0, 4.0],
            [0.0, 9.0, -6.0],
        ]
    )
    assert_allclose(curlB(linear_B, zero_E, dx, dt, 0, 0), periodic_expected)

    absorbing_expected = jnp.array(
        [
            [0.0, -3.0, 2.0],
            [0.0, -6.0, 4.0],
            [0.0, 22.0, -14.0],
        ]
    )
    assert_allclose(curlB(linear_B, zero_E, dx, dt, 2, 2), absorbing_expected)

    dx_half = 0.5
    coupled_E = zero_E.at[-1].set(jnp.array([0.0, 1.5e6, -0.5e6]))
    absorbing_right_ghost = jnp.array(
        [
            0.0,
            -(2 / speed_of_light) * coupled_E[-1, 2] - linear_B[-1, 1],
            (2 / speed_of_light) * coupled_E[-1, 1] - linear_B[-1, 2],
        ]
    )
    asymmetric_expected = jnp.array(
        [
            [
                0.0,
                -(linear_B[1, 2] - linear_B[0, 2]) / dx_half,
                (linear_B[1, 1] - linear_B[0, 1]) / dx_half,
            ],
            [
                0.0,
                -(linear_B[2, 2] - linear_B[1, 2]) / dx_half,
                (linear_B[2, 1] - linear_B[1, 1]) / dx_half,
            ],
            [
                0.0,
                -(absorbing_right_ghost[2] - linear_B[2, 2]) / dx_half,
                (absorbing_right_ghost[1] - linear_B[2, 1]) / dx_half,
            ],
        ]
    )
    assert_allclose(
        curlB(linear_B, coupled_E, dx_half, dt, field_BC_left=1, field_BC_right=2),
        asymmetric_expected,
        rtol=1e-6,
        atol=1e-12,
    )


@pytest.mark.parametrize("update_function", [field_update, field_update1, field_update2])
def test_field_update_variants_zero_sources_and_current_response(update_function):
    """Test jaxincell._fields.field_update, field_update1, and field_update2.

    Cases covered:
    - zero E, B, and current leave both fields unchanged.
    - zero curl with nonzero current updates only E by the Ampere current term.
    - output shapes match the input field shapes for all update variants.
    """
    dx = 1.0
    dt = 0.2
    zero_fields = jnp.zeros((4, 3))
    zero_current = jnp.zeros_like(zero_fields)

    electric_field, magnetic_field = update_function(
        zero_fields,
        zero_fields,
        dx,
        dt,
        zero_current,
        field_BC_left=1,
        field_BC_right=1,
    )
    assert electric_field.shape == zero_fields.shape
    assert magnetic_field.shape == zero_fields.shape
    assert_allclose(electric_field, zero_fields, atol=0)
    assert_allclose(magnetic_field, zero_fields, atol=0)

    constant_current = epsilon_0 * jnp.tile(jnp.array([1.0, -2.0, 3.0]), (4, 1))
    electric_field, magnetic_field = update_function(
        zero_fields,
        zero_fields,
        dx,
        dt,
        constant_current,
        field_BC_left=1,
        field_BC_right=1,
    )
    expected_E = -dt * constant_current / epsilon_0
    assert_allclose(electric_field, expected_E)
    assert_allclose(magnetic_field, zero_fields, atol=0)


def test_field_update_variants_call_low_level_curls_consistently():
    """Test orchestration in jaxincell._fields.field_update, field_update1, and field_update2.

    Cases covered:
    - compare each update variant against explicit calls to curlE and curlB.
    - verify the sequencing difference between simultaneous update, E-first update, and B-first update.
    - cover nonperiodic boundary conditions through the full update path.
    - cover periodic boundary conditions through a nontrivial full update path.
    """
    dx = 0.5
    dt = 1e-6
    field_BC_left = 2
    field_BC_right = 1
    E_fields = 1e-3 * jnp.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, -1.5, 0.5],
            [0.0, 0.25, -0.75],
            [0.0, 2.0, -1.0],
        ]
    )
    B_fields = 1e-9 * jnp.array(
        [
            [0.0, -2.0, 1.0],
            [0.0, 0.5, 3.0],
            [0.0, 1.5, -1.0],
            [0.0, -0.25, 2.0],
        ]
    )
    current = epsilon_0 * jnp.array(
        [
            [0.0, 0.5, -1.0],
            [0.0, -0.25, 0.75],
            [0.0, 1.25, -0.5],
            [0.0, -0.75, 0.25],
        ]
    )

    simultaneous_E, simultaneous_B = field_update(
        E_fields,
        B_fields,
        dx,
        dt,
        current,
        field_BC_left,
        field_BC_right,
    )
    assert simultaneous_E.shape == E_fields.shape
    assert simultaneous_B.shape == B_fields.shape

    curl_E_initial = curlE(E_fields, B_fields, dx, dt, field_BC_left, field_BC_right)
    curl_B_initial = curlB(B_fields, E_fields, dx, dt, field_BC_left, field_BC_right)
    expected_simultaneous_B = B_fields - dt * curl_E_initial
    expected_simultaneous_E = E_fields + dt * (
        (speed_of_light**2) * curl_B_initial - (current / epsilon_0)
    )
    assert_allclose(simultaneous_E, expected_simultaneous_E, rtol=1e-6, atol=1e-12)
    assert_allclose(simultaneous_B, expected_simultaneous_B, rtol=1e-6, atol=1e-12)

    e_first_E, e_first_B = field_update1(
        E_fields,
        B_fields,
        dx,
        dt,
        current,
        field_BC_left,
        field_BC_right,
    )
    assert e_first_E.shape == E_fields.shape
    assert e_first_B.shape == B_fields.shape

    expected_e_first_E = E_fields + dt * (
        (speed_of_light**2) * curl_B_initial - (current / epsilon_0)
    )
    expected_e_first_B = B_fields - dt * curlE(
        expected_e_first_E,
        B_fields,
        dx,
        dt,
        field_BC_left,
        field_BC_right,
    )
    assert_allclose(e_first_E, expected_e_first_E, rtol=1e-6, atol=1e-12)
    assert_allclose(e_first_B, expected_e_first_B, rtol=1e-6, atol=1e-12)

    b_first_E, b_first_B = field_update2(
        E_fields,
        B_fields,
        dx,
        dt,
        current,
        field_BC_left,
        field_BC_right,
    )
    assert b_first_E.shape == E_fields.shape
    assert b_first_B.shape == B_fields.shape

    expected_b_first_B = B_fields - dt * curl_E_initial
    expected_b_first_E = E_fields + dt * (
        (speed_of_light**2)
        * curlB(
            expected_b_first_B,
            E_fields,
            dx,
            dt,
            field_BC_left,
            field_BC_right,
        )
        - (current / epsilon_0)
    )
    assert_allclose(b_first_E, expected_b_first_E, rtol=1e-6, atol=1e-12)
    assert_allclose(b_first_B, expected_b_first_B, rtol=1e-6, atol=1e-12)

    periodic_E, periodic_B = field_update(
        E_fields,
        B_fields,
        dx,
        dt,
        current,
        field_BC_left=0,
        field_BC_right=0,
    )
    periodic_curl_E = curlE(E_fields, B_fields, dx, dt, 0, 0)
    periodic_curl_B = curlB(B_fields, E_fields, dx, dt, 0, 0)
    expected_periodic_B = B_fields - dt * periodic_curl_E
    expected_periodic_E = E_fields + dt * (
        (speed_of_light**2) * periodic_curl_B - (current / epsilon_0)
    )
    assert_allclose(periodic_E, expected_periodic_E, rtol=1e-6, atol=1e-12)
    assert_allclose(periodic_B, expected_periodic_B, rtol=1e-6, atol=1e-12)

    assert not bool(jnp.allclose(e_first_B, simultaneous_B, rtol=0, atol=0))
    assert not bool(jnp.allclose(b_first_E, simultaneous_E, rtol=0, atol=0))


def test_field_functions_are_differentiable_for_small_inputs():
    """Test small differentiability smoke cases for field functions.

    Cases covered:
    - grad through E_from_Poisson_1D_FFT with respect to charge density is finite.
    - grad through field_update with respect to current density is finite and matches the Ampere term.
    """
    grid_size = 8
    dx = 0.25
    x = dx * jnp.arange(grid_size)
    wavenumber = 2 * jnp.pi / (grid_size * dx)
    charge_density = epsilon_0 * (
        jnp.sin(wavenumber * x) + 0.5 * jnp.cos(2 * wavenumber * x)
    )

    def poisson_loss(charge_density):
        electric_field = E_from_Poisson_1D_FFT(charge_density, dx)
        return jnp.sum(electric_field**2)

    poisson_gradient = grad(poisson_loss)(charge_density)
    assert poisson_gradient.shape == charge_density.shape
    assert bool(jnp.all(jnp.isfinite(poisson_gradient)))
    assert float(jnp.linalg.norm(poisson_gradient)) > 0

    dt = 1e-6
    E_fields = 1e-3 * jnp.array(
        [
            [0.0, 1.0, -2.0],
            [0.0, 0.5, 0.75],
            [0.0, -1.0, 0.25],
        ]
    )
    B_fields = 1e-9 * jnp.array(
        [
            [0.0, -0.5, 1.0],
            [0.0, 0.25, -0.75],
            [0.0, 1.5, 0.5],
        ]
    )
    current = epsilon_0 * jnp.ones_like(E_fields)

    def field_update_loss(current):
        electric_field, _ = field_update(
            E_fields,
            B_fields,
            dx,
            dt,
            current,
            field_BC_left=1,
            field_BC_right=1,
        )
        return jnp.sum(electric_field)

    current_gradient = grad(field_update_loss)(current)
    assert current_gradient.shape == current.shape
    assert bool(jnp.all(jnp.isfinite(current_gradient)))
    assert_allclose(
        current_gradient,
        -(dt / epsilon_0) * jnp.ones_like(current),
        rtol=1e-6,
        atol=1e-12,
    )
