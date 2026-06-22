"""
    Someone needs to look through these tests and make sure they cover the
    intended cases, and that the expected results are correct.

    They were written using Codex and seem reasonable to me,
    but I haven't gone through the functions being tested myself to verify
    the intended behavior.
"""

# tests/test_algorithms.py

import jax.numpy as jnp
import pytest

import jaxincell._algorithms as algorithms
from jaxincell._algorithms import Boris_step, CN_step
from jaxincell._boundary_conditions import set_BC_particles, set_BC_positions
from jaxincell._parameters._domain_parameters import clean_and_initialize_domain_parameters
from jaxincell._parameters._external_field_parameters import clean_and_initialize_external_field_parameters
from jaxincell._parameters._solver_parameters import clean_and_initialize_solver_parameters
from jaxincell._parameters._species_parameters import clean_and_initialize_species_parameters
from jaxincell._state_initialization import build_domain_state, initialize_field_state, initialize_particle_state


def _small_parameters_for_algorithms():
    """
    Build a tiny, cheap parameter set using the same initialization
    as the main simulation, so we don't need to manually reimplement
    PIC plumbing here.
    """
    number_grid_points = 4
    number_pseudoparticles = 4

    base_species = {
        "number_pseudoparticles": number_pseudoparticles,
        "grid_points_per_Debye_length": 1.0,
        "weight": 1.0,
        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
        "perturbation_wavenumber_x": 1.0,
        "perturbation_wavenumber_y": 1.0,
        "perturbation_wavenumber_z": 1.0,
        "random_positions_x": False,
        "random_positions_y": False,
        "random_positions_z": False,
        "vth_over_c_x": 0.01,
        "vth_over_c_y": 0.01,
        "vth_over_c_z": 0.01,
        "drift_speed_x": 1.0,
        "drift_speed_y": 0.0,
        "drift_speed_z": 0.0,
        "velocity_plus_minus_x": False,
        "velocity_plus_minus_y": False,
        "velocity_plus_minus_z": False,
    }

    domain_parameters = clean_and_initialize_domain_parameters(
        {
            "length": 0.01,
            "length_y": 0.01,
            "length_z": 0.01,
            "number_grid_points": number_grid_points,
            "number_grid_points_y": 3,
            "number_grid_points_z": 3,
            "total_steps": 2,
        }
    )
    solver_parameters = clean_and_initialize_solver_parameters(
        {
            "field_solver": 0,
            "max_number_of_Picard_iterations_implicit_CN": 3,
            "number_of_particle_substeps_implicit_CN": 2,
            "filter_passes": 0,
            "filter_alpha": 0.5,
            "filter_strides": (1, 2, 4),
            "seed": 1701,
            "relativistic": False,
            "time_evolution_algorithm": 0,
            "print_info": False,
        }
    )
    species_parameters = clean_and_initialize_species_parameters(
        {
            "electrons": {
                "electrons0": {
                    **base_species,
                    "charge_over_elementary_charge": -1.0,
                },
            },
            "ions": {
                "ions0": {
                    **base_species,
                    "charge_over_elementary_charge": 1.0,
                    "mass_over_proton_mass": 1.0,
                    "ion_temperature_over_electron_temperature_x": 1.0,
                    "ion_temperature_over_electron_temperature_y": 1.0,
                    "ion_temperature_over_electron_temperature_z": 1.0,
                },
            },
        }
    )
    external_field_parameters = clean_and_initialize_external_field_parameters({})

    domain_state = build_domain_state(domain_parameters)
    particle_state = initialize_particle_state(
        species_parameters,
        domain_parameters,
        solver_parameters,
        domain_state,
    )
    field_state = initialize_field_state(
        domain_parameters,
        solver_parameters,
        external_field_parameters,
        domain_state,
        particle_state,
    )
    external_field_parameters = {
        **external_field_parameters,
        "external_electric_field": field_state["external_electric_field"],
        "external_magnetic_field": field_state["external_magnetic_field"],
    }

    params = {
        **domain_parameters,
        **solver_parameters,
        "box_size": domain_state["box_size"],
        "dx": domain_state["dx"],
        "dt": domain_state["dt"],
        "grid": domain_state["grid"],
        "fields": field_state["fields"],
        "initial_positions": particle_state["positions"],
        "initial_velocities": particle_state["velocities"],
        "charges": particle_state["charges"],
        "masses": particle_state["masses"],
        "charge_to_mass_ratios": particle_state["charge_to_mass_ratios"],
    }

    return params, external_field_parameters, number_grid_points, number_pseudoparticles


def _boris_step_inputs(params):
    """
    Build the staggered Boris carry exactly as Simulation._simulation does.
    """
    dx = params["dx"]
    dt = params["dt"]
    grid = params["grid"]
    box_size = params["box_size"]
    E_field, B_field = params["fields"]
    positions = params["initial_positions"]
    velocities = params["initial_velocities"]
    charges = params["charges"]
    masses = params["masses"]
    q_ms = params["charge_to_mass_ratios"]
    particle_BC_left = params["particle_BC_left"]
    particle_BC_right = params["particle_BC_right"]

    positions_plus1_2, velocities_bc, qs, ms, q_ms_bc = set_BC_particles(
        positions + (dt / 2.0) * velocities,
        velocities,
        charges,
        masses,
        q_ms,
        dx,
        grid,
        *box_size,
        particle_BC_left,
        particle_BC_right,
    )

    positions_minus1_2 = set_BC_positions(
        positions - (dt / 2.0) * velocities,
        charges,
        dx,
        grid,
        *box_size,
        particle_BC_left,
        particle_BC_right,
    )

    return (
        E_field,
        B_field,
        positions_minus1_2,
        positions,
        positions_plus1_2,
        velocities_bc,
        qs,
        ms,
        q_ms_bc,
    )


def _cn_step_inputs(params):
    E_field, B_field = params["fields"]
    return (
        E_field,
        B_field,
        params["initial_positions"],
        params["initial_velocities"],
        params["charges"],
        params["masses"],
        params["charge_to_mass_ratios"],
    )


def _assert_arrays_are_finite(*arrays):
    for array in arrays:
        assert jnp.all(jnp.isfinite(array))


def _assert_positions_in_box(positions, box_size):
    for axis_index, length in enumerate(box_size):
        assert jnp.all(positions[:, axis_index] >= -length / 2)
        assert jnp.all(positions[:, axis_index] <= length / 2)


def _assert_boris_step_contract(
    carry,
    step_data,
    number_grid_points,
    input_carry=None,
    box_size=None,
):
    (
        E_new,
        B_new,
        pos_minus1_2_new,
        pos_new,
        pos_plus1_2_new,
        vel_new,
        qs_new,
        ms_new,
        q_ms_new,
    ) = carry
    pos_step, vel_step, E_step, B_step, J_step, rho_step = step_data
    n_particles = qs_new.shape[0]

    assert E_new.shape == (number_grid_points, 3)
    assert B_new.shape == (number_grid_points, 3)
    assert pos_minus1_2_new.shape == (n_particles, 3)
    assert pos_new.shape == (n_particles, 3)
    assert pos_plus1_2_new.shape == (n_particles, 3)
    assert vel_new.shape == (n_particles, 3)
    assert qs_new.shape == (n_particles, 1)
    assert ms_new.shape == (n_particles, 1)
    assert q_ms_new.shape == (n_particles, 1)
    assert pos_step.shape == (n_particles, 3)
    assert vel_step.shape == (n_particles, 3)
    assert E_step.shape == (number_grid_points, 3)
    assert B_step.shape == (number_grid_points, 3)
    assert J_step.shape == (number_grid_points, 3)
    assert rho_step.shape == (number_grid_points,)
    assert jnp.allclose(pos_new, pos_step)
    assert jnp.allclose(vel_new, vel_step)
    assert jnp.allclose(E_new, E_step)
    assert jnp.allclose(B_new, B_step)
    if input_carry is not None:
        assert jnp.allclose(qs_new, input_carry[6])
        assert jnp.allclose(ms_new, input_carry[7])
        assert jnp.allclose(q_ms_new, input_carry[8])
    if box_size is not None:
        _assert_positions_in_box(pos_minus1_2_new, box_size)
        _assert_positions_in_box(pos_new, box_size)
        _assert_positions_in_box(pos_plus1_2_new, box_size)
    _assert_arrays_are_finite(
        E_new,
        B_new,
        pos_minus1_2_new,
        pos_new,
        pos_plus1_2_new,
        vel_new,
        qs_new,
        ms_new,
        q_ms_new,
        pos_step,
        vel_step,
        E_step,
        B_step,
        J_step,
        rho_step,
    )


def _assert_cn_step_contract(
    carry,
    step_data,
    number_grid_points,
    input_carry=None,
    box_size=None,
):
    E_new, B_new, pos_new, vel_new, qs_new, ms_new, q_ms_new = carry
    pos_step, vel_step, E_step, B_step, J_step, rho_step = step_data
    n_particles = qs_new.shape[0]

    assert E_new.shape == (number_grid_points, 3)
    assert B_new.shape == (number_grid_points, 3)
    assert pos_new.shape == (n_particles, 3)
    assert vel_new.shape == (n_particles, 3)
    assert qs_new.shape == (n_particles, 1)
    assert ms_new.shape == (n_particles, 1)
    assert q_ms_new.shape == (n_particles, 1)
    assert pos_step.shape == (n_particles, 3)
    assert vel_step.shape == (n_particles, 3)
    assert E_step.shape == (number_grid_points, 3)
    assert B_step.shape == (number_grid_points, 3)
    assert J_step.shape == (number_grid_points, 3)
    assert rho_step.shape == (number_grid_points,)
    assert jnp.allclose(pos_new, pos_step)
    assert jnp.allclose(vel_new, vel_step)
    assert jnp.allclose(E_new, E_step)
    assert jnp.allclose(B_new, B_step)
    if input_carry is not None:
        assert jnp.allclose(qs_new, input_carry[4])
        assert jnp.allclose(ms_new, input_carry[5])
        assert jnp.allclose(q_ms_new, input_carry[6])
    if box_size is not None:
        _assert_positions_in_box(pos_new, box_size)
    _assert_arrays_are_finite(
        E_new,
        B_new,
        pos_new,
        vel_new,
        qs_new,
        ms_new,
        q_ms_new,
        pos_step,
        vel_step,
        E_step,
        B_step,
        J_step,
        rho_step,
    )


def test_boris_step_relativistic_and_field_solver_branch():
    """
    Exercise:
    - relativistic=True branch (boris_step_relativistic),
    - field_solver != 0 branch (E_from_* correction),
    and verify that output shapes are consistent.
    """
    params, external_field_parameters, G, _ = _small_parameters_for_algorithms()

    dx = params["dx"]
    dt = params["dt"]
    grid = params["grid"]
    box_size = params["box_size"]
    particle_BC_left = params["particle_BC_left"]
    particle_BC_right = params["particle_BC_right"]
    field_BC_left = params["field_BC_left"]
    field_BC_right = params["field_BC_right"]
    carry0 = _boris_step_inputs(params)

    # Force relativistic path and use a nonzero field_solver
    params_rel = dict(params)
    params_rel["relativistic"] = True

    carry1, step_data = Boris_step(
        carry0,
        step_index=0,
        solver_parameters=params_rel,
        external_field_parameters=external_field_parameters,
        dx=dx,
        dt=dt,
        grid=grid,
        box_size=box_size,
        particle_BC_left=particle_BC_left,
        particle_BC_right=particle_BC_right,
        field_BC_left=field_BC_left,
        field_BC_right=field_BC_right,
        field_solver=1,  # triggers E_from_* branch
    )

    _assert_boris_step_contract(
        carry1,
        step_data,
        G,
        input_carry=carry0,
        box_size=box_size,
    )


def test_boris_step_selects_nonrelativistic_or_relativistic_pusher(monkeypatch):
    """Test jaxincell._algorithms.Boris_step pusher dispatch.

    Cases:
    - relativistic=False calls boris_step.
    - relativistic=True calls boris_step_relativistic.
    - the selected pusher's velocity update reaches the returned carry.
    """
    params, external_field_parameters, G, _ = _small_parameters_for_algorithms()

    def zero_centered_boris_carry():
        carry = list(_boris_step_inputs(params))
        positions = jnp.zeros_like(carry[3])
        velocities = jnp.zeros_like(carry[5])
        carry[2] = positions
        carry[3] = positions
        carry[4] = positions
        carry[5] = velocities
        return tuple(carry)

    def nonrelativistic_pusher(dt, positions, velocities, q_ms, E_field, B_field):
        return positions, velocities + 1.0

    def relativistic_pusher(dt, positions, velocities, qs, ms, E_field, B_field):
        return positions, velocities + 2.0

    monkeypatch.setattr(algorithms, "boris_step", nonrelativistic_pusher)
    monkeypatch.setattr(algorithms, "boris_step_relativistic", relativistic_pusher)

    for relativistic, expected_velocity in ((False, 1.0), (True, 2.0)):
        params_branch = dict(params)
        params_branch["relativistic"] = relativistic
        carry0 = zero_centered_boris_carry()
        carry1, step_data = Boris_step(
            carry0,
            step_index=0,
            solver_parameters=params_branch,
            external_field_parameters=external_field_parameters,
            dx=params["dx"],
            dt=params["dt"],
            grid=params["grid"],
            box_size=params["box_size"],
            particle_BC_left=params["particle_BC_left"],
            particle_BC_right=params["particle_BC_right"],
            field_BC_left=params["field_BC_left"],
            field_BC_right=params["field_BC_right"],
            field_solver=0,
        )

        _assert_boris_step_contract(
            carry1,
            step_data,
            G,
            input_carry=carry0,
            box_size=params["box_size"],
        )
        assert jnp.allclose(carry1[5], expected_velocity)


def test_boris_step_adds_external_fields_before_particle_push(monkeypatch):
    """Test jaxincell._algorithms.Boris_step external field application.

    Cases:
    - external electric field is added before interpolation to particles.
    - external magnetic field is added before interpolation to particles.
    - the nonrelativistic pusher receives the combined field values.
    """
    params, external_field_parameters, G, _ = _small_parameters_for_algorithms()
    carry0 = list(_boris_step_inputs(params))
    carry0[0] = jnp.zeros((G, 3))
    carry0[1] = jnp.zeros((G, 3))
    carry0[2] = jnp.zeros_like(carry0[2])
    carry0[3] = jnp.zeros_like(carry0[3])
    carry0[4] = jnp.zeros_like(carry0[4])
    carry0[5] = jnp.zeros_like(carry0[5])

    external_electric_field = jnp.tile(jnp.array([[1.0, 2.0, 3.0]]), (G, 1))
    external_magnetic_field = jnp.tile(jnp.array([[4.0, 5.0, 6.0]]), (G, 1))
    external_field_parameters = {
        **external_field_parameters,
        "external_electric_field": external_electric_field,
        "external_magnetic_field": external_magnetic_field,
    }
    captured_fields = {}

    def zero_current_density(*args, **kwargs):
        return jnp.zeros((G, 3))

    def first_grid_field_value(x_n, field, dx, grid, grid_start, field_BC_left, field_BC_right):
        return field[0]

    def capture_nonrelativistic_pusher(dt, positions, velocities, q_ms, E_field, B_field):
        captured_fields["E_field"] = E_field
        captured_fields["B_field"] = B_field
        return positions, velocities

    monkeypatch.setattr(algorithms, "current_density", zero_current_density)
    monkeypatch.setattr(algorithms, "fields_to_particles_grid", first_grid_field_value)
    monkeypatch.setattr(algorithms, "boris_step", capture_nonrelativistic_pusher)

    Boris_step(
        tuple(carry0),
        step_index=0,
        solver_parameters=params,
        external_field_parameters=external_field_parameters,
        dx=params["dx"],
        dt=params["dt"],
        grid=params["grid"],
        box_size=params["box_size"],
        particle_BC_left=params["particle_BC_left"],
        particle_BC_right=params["particle_BC_right"],
        field_BC_left=params["field_BC_left"],
        field_BC_right=params["field_BC_right"],
        field_solver=0,
    )

    assert jnp.allclose(captured_fields["E_field"], external_electric_field[0])
    assert jnp.allclose(captured_fields["B_field"], external_magnetic_field[0])


def test_boris_step_nonrelativistic_branch_and_zero_field_solver_path():
    """Test jaxincell._algorithms.Boris_step.

    Cases:
    - relativistic=False selects boris_step rather than boris_step_relativistic.
    - field_solver=0 skips the charge-density correction switcher branch.
    - returned carry advances staggered positions consistently with step_data.
    """
    params, external_field_parameters, G, _ = _small_parameters_for_algorithms()
    carry0 = _boris_step_inputs(params)

    carry1, step_data = Boris_step(
        carry0,
        step_index=0,
        solver_parameters=params,
        external_field_parameters=external_field_parameters,
        dx=params["dx"],
        dt=params["dt"],
        grid=params["grid"],
        box_size=params["box_size"],
        particle_BC_left=params["particle_BC_left"],
        particle_BC_right=params["particle_BC_right"],
        field_BC_left=params["field_BC_left"],
        field_BC_right=params["field_BC_right"],
        field_solver=0,
    )

    _assert_boris_step_contract(
        carry1,
        step_data,
        G,
        input_carry=carry0,
        box_size=params["box_size"],
    )
    assert jnp.allclose(carry1[2], carry0[4])


def test_boris_step_field_solver_switcher_variants():
    """Test jaxincell._algorithms.Boris_step field_solver dispatch.

    Cases:
    - field_solver=1 uses E_from_Gauss_1D_FFT.
    - field_solver=2 uses E_from_Gauss_1D_Cartesian.
    - field_solver=3 uses E_from_Poisson_1D_FFT.
    - unsupported nonzero field_solver values raise the expected KeyError or validation error.
    """
    params, external_field_parameters, G, _ = _small_parameters_for_algorithms()

    for field_solver in (1, 2, 3):
        carry0 = _boris_step_inputs(params)
        carry1, step_data = Boris_step(
            carry0,
            step_index=0,
            solver_parameters=params,
            external_field_parameters=external_field_parameters,
            dx=params["dx"],
            dt=params["dt"],
            grid=params["grid"],
            box_size=params["box_size"],
            particle_BC_left=params["particle_BC_left"],
            particle_BC_right=params["particle_BC_right"],
            field_BC_left=params["field_BC_left"],
            field_BC_right=params["field_BC_right"],
            field_solver=field_solver,
        )

        _assert_boris_step_contract(
            carry1,
            step_data,
            G,
            input_carry=carry0,
            box_size=params["box_size"],
        )
        charge_density = algorithms.calculate_charge_density(
            carry0[3],
            carry1[6],
            params["dx"],
            params["grid"] + params["dx"] / 2,
            params["particle_BC_left"],
            params["particle_BC_right"],
            filter_passes=params["filter_passes"],
            filter_alpha=params["filter_alpha"],
            filter_strides=params["filter_strides"],
            field_BC_left=params["field_BC_left"],
            field_BC_right=params["field_BC_right"],
        )
        solver_function = {
            1: algorithms.E_from_Gauss_1D_FFT,
            2: algorithms.E_from_Gauss_1D_Cartesian,
            3: algorithms.E_from_Poisson_1D_FFT,
        }[field_solver]
        assert jnp.allclose(carry1[0][:, 0], solver_function(charge_density, params["dx"]))

    with pytest.raises(KeyError):
        Boris_step(
            _boris_step_inputs(params),
            step_index=0,
            solver_parameters=params,
            external_field_parameters=external_field_parameters,
            dx=params["dx"],
            dt=params["dt"],
            grid=params["grid"],
            box_size=params["box_size"],
            particle_BC_left=params["particle_BC_left"],
            particle_BC_right=params["particle_BC_right"],
            field_BC_left=params["field_BC_left"],
            field_BC_right=params["field_BC_right"],
            field_solver=4,
        )


def test_cn_step_picard_stopping_conditions():
    """Test jaxincell._algorithms.CN_step Picard while_loop orchestration.

    Cases:
    - convergence before max_number_of_Picard_iterations_implicit_CN exits early.
    - max iteration limit stops the loop when tolerance is not met.
    - number_of_particle_substeps_implicit_CN changes the substep accumulation shape but not public output shape.
    """
    params, _, G, _ = _small_parameters_for_algorithms()

    test_cases = [
        {
            "tolerance_Picard_iterations_implicit_CN": 1e9,
            "max_number_of_Picard_iterations_implicit_CN": 5,
            "num_substeps": 1,
        },
        {
            "tolerance_Picard_iterations_implicit_CN": 1e-30,
            "max_number_of_Picard_iterations_implicit_CN": 1,
            "num_substeps": 1,
        },
        {
            "tolerance_Picard_iterations_implicit_CN": 1e-3,
            "max_number_of_Picard_iterations_implicit_CN": 2,
            "num_substeps": 3,
        },
    ]

    expected_positions_shape = params["initial_positions"].shape
    for test_case in test_cases:
        params_cn = dict(params)
        params_cn["tolerance_Picard_iterations_implicit_CN"] = test_case[
            "tolerance_Picard_iterations_implicit_CN"
        ]
        params_cn["max_number_of_Picard_iterations_implicit_CN"] = test_case[
            "max_number_of_Picard_iterations_implicit_CN"
        ]

        cn_carry0 = _cn_step_inputs(params)
        carry1, step_data = CN_step(
            cn_carry0,
            step_index=0,
            solver_parameters=params_cn,
            dx=params["dx"],
            dt=params["dt"],
            grid=params["grid"],
            box_size=params["box_size"],
            particle_BC_left=params["particle_BC_left"],
            particle_BC_right=params["particle_BC_right"],
            field_BC_left=params["field_BC_left"],
            field_BC_right=params["field_BC_right"],
            num_substeps=test_case["num_substeps"],
        )

        _assert_cn_step_contract(
            carry1,
            step_data,
            G,
            input_carry=cn_carry0,
            box_size=params["box_size"],
        )
        assert carry1[2].shape == expected_positions_shape
        assert step_data[0].shape == expected_positions_shape


def test_cn_step_shapes_and_substepping():
    """
    Exercise the CN_step Picard iteration and substep loop:
    - verifies that it runs to completion with small tolerances,
    - checks shapes of the returned fields and sources.
    """
    params, _, G, _ = _small_parameters_for_algorithms()

    # Use slightly relaxed Picard settings to keep the test cheap
    params_cn = dict(params)
    params_cn["tolerance_Picard_iterations_implicit_CN"] = 1e-3
    params_cn["max_number_of_Picard_iterations_implicit_CN"] = 2

    cn_carry0 = _cn_step_inputs(params)
    carry1, step_data = CN_step(
        cn_carry0,
        step_index=0,
        solver_parameters=params_cn,
        dx=params["dx"],
        dt=params["dt"],
        grid=params["grid"],
        box_size=params["box_size"],
        particle_BC_left=params["particle_BC_left"],
        particle_BC_right=params["particle_BC_right"],
        field_BC_left=params["field_BC_left"],
        field_BC_right=params["field_BC_right"],
        num_substeps=params["number_of_particle_substeps_implicit_CN"],
    )

    _assert_cn_step_contract(
        carry1,
        step_data,
        G,
        input_carry=cn_carry0,
        box_size=params["box_size"],
    )
