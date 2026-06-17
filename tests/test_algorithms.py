# tests/test_algorithms.py

import jax.numpy as jnp
import pytest

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
    E_field, B_field = params["fields"]
    positions = params["initial_positions"]
    velocities = params["initial_velocities"]
    charges = params["charges"]
    masses = params["masses"]
    q_ms = params["charge_to_mass_ratios"]

    particle_BC_left = params["particle_BC_left"]
    particle_BC_right = params["particle_BC_right"]
    field_BC_left = params["field_BC_left"]
    field_BC_right = params["field_BC_right"]

    # Build half-step / staggered positions exactly as in simulation(...)
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

    carry0 = (
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

    # Unpack new carry
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
    ) = carry1

    n_particles = qs_new.shape[0]

    # Shape checks
    assert E_new.shape == (G, 3)
    assert B_new.shape == (G, 3)
    assert pos_minus1_2_new.shape == (n_particles, 3)
    assert pos_new.shape == (n_particles, 3)
    assert pos_plus1_2_new.shape == (n_particles, 3)
    assert vel_new.shape == (n_particles, 3)
    assert qs_new.shape == (n_particles, 1)
    assert ms_new.shape == (n_particles, 1)
    assert q_ms_new.shape == (n_particles, 1)

    # step_data = (positions, velocities, E_field, B_field, J, charge_density)
    pos_step, vel_step, E_step, B_step, J_step, rho_step = step_data
    assert pos_step.shape == (n_particles, 3)
    assert vel_step.shape == (n_particles, 3)
    assert E_step.shape == (G, 3)
    assert B_step.shape == (G, 3)
    assert J_step.shape == (G, 3)
    assert rho_step.shape == (G,)


@pytest.mark.skip(reason="scaffold only")
def test_boris_step_nonrelativistic_branch_and_zero_field_solver_path():
    """Test jaxincell._algorithms.Boris_step.

    Cases to implement:
    - relativistic=False selects boris_step rather than boris_step_relativistic.
    - field_solver=0 skips the charge-density correction switcher branch.
    - returned carry advances staggered positions consistently with step_data.
    """


@pytest.mark.skip(reason="scaffold only")
def test_boris_step_field_solver_switcher_variants():
    """Test jaxincell._algorithms.Boris_step field_solver dispatch.

    Cases to implement:
    - field_solver=1 uses E_from_Gauss_1D_FFT.
    - field_solver=2 uses E_from_Gauss_1D_Cartesian.
    - field_solver=3 uses E_from_Poisson_1D_FFT.
    - unsupported nonzero field_solver values raise the expected KeyError or validation error.
    """


@pytest.mark.skip(reason="scaffold only")
def test_cn_step_picard_stopping_conditions():
    """Test jaxincell._algorithms.CN_step Picard while_loop orchestration.

    Cases to implement:
    - convergence before max_number_of_Picard_iterations_implicit_CN exits early.
    - max iteration limit stops the loop when tolerance is not met.
    - number_of_particle_substeps_implicit_CN changes the substep accumulation shape but not public output shape.
    """


def test_cn_step_shapes_and_substepping():
    """
    Exercise the CN_step Picard iteration and substep loop:
    - verifies that it runs to completion with small tolerances,
    - checks shapes of the returned fields and sources.
    """
    params, _, G, _ = _small_parameters_for_algorithms()

    dx = params["dx"]
    dt = params["dt"]
    grid = params["grid"]
    box_size = params["box_size"]
    particle_BC_left = params["particle_BC_left"]
    particle_BC_right = params["particle_BC_right"]
    field_BC_left = params["field_BC_left"]
    field_BC_right = params["field_BC_right"]
    num_substeps = params["number_of_particle_substeps_implicit_CN"]

    E_field, B_field = params["fields"]
    positions = params["initial_positions"]
    velocities = params["initial_velocities"]
    qs = params["charges"]
    ms = params["masses"]
    q_ms = params["charge_to_mass_ratios"]

    # Use slightly relaxed Picard settings to keep the test cheap
    params_cn = dict(params)
    params_cn["tolerance_Picard_iterations_implicit_CN"] = 1e-3
    params_cn["max_number_of_Picard_iterations_implicit_CN"] = 2

    carry0 = (E_field, B_field, positions, velocities, qs, ms, q_ms)

    carry1, step_data = CN_step(
        carry0,
        step_index=0,
        solver_parameters=params_cn,
        dx=dx,
        dt=dt,
        grid=grid,
        box_size=box_size,
        particle_BC_left=particle_BC_left,
        particle_BC_right=particle_BC_right,
        field_BC_left=field_BC_left,
        field_BC_right=field_BC_right,
        num_substeps=num_substeps,
    )

    # Unpack updated carry
    E_new, B_new, pos_new, vel_new, qs_new, ms_new, q_ms_new = carry1
    n_particles = qs_new.shape[0]

    assert E_new.shape == (G, 3)
    assert B_new.shape == (G, 3)
    assert pos_new.shape == (n_particles, 3)
    assert vel_new.shape == (n_particles, 3)
    assert qs_new.shape == (n_particles, 1)
    assert ms_new.shape == (n_particles, 1)
    assert q_ms_new.shape == (n_particles, 1)

    # step_data = (positions_plus1, velocities_plus1, E_field, B_field, J, charge_density)
    pos_step, vel_step, E_step, B_step, J_step, rho_step = step_data
    assert pos_step.shape == (n_particles, 3)
    assert vel_step.shape == (n_particles, 3)
    assert E_step.shape == (G, 3)
    assert B_step.shape == (G, 3)
    assert J_step.shape == (G, 3)
    assert rho_step.shape == (G,)
