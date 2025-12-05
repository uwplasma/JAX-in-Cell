# tests/test_algorithms.py

import jax.numpy as jnp

from jaxincell._simulation import initialize_particles_fields
from jaxincell._algorithms import Boris_step, CN_step
from jaxincell._boundary_conditions import set_BC_particles, set_BC_positions


def _small_parameters_for_algorithms():
    """
    Build a tiny, cheap parameter set using the same initialization
    as the main simulation, so we don't need to manually reimplement
    PIC plumbing here.
    """
    input_parameters = {
        "length": 0.01,
        "grid_points_per_Debye_length": 1.0,
        "print_info": False,
        "species": [],          # <-- important to avoid KeyError in initialize_particles_fields
    }
    number_grid_points = 4
    number_pseudoelectrons = 4

    params = initialize_particles_fields(
        input_parameters=input_parameters,
        number_grid_points=number_grid_points,
        number_pseudoelectrons=number_pseudoelectrons,
        number_pseudoparticles_species=(),         # empty tuple, consistent with species=[]
        total_steps=2,
        max_number_of_Picard_iterations_implicit_CN=3,
        number_of_particle_substeps_implicit_CN=2,
    )
    return params, number_grid_points, number_pseudoelectrons


def test_boris_step_relativistic_and_field_solver_branch():
    """
    Exercise:
    - relativistic=True branch (boris_step_relativistic),
    - field_solver != 0 branch (E_from_* correction),
    and verify that output shapes are consistent.
    """
    params, G, _ = _small_parameters_for_algorithms()

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
        parameters=params_rel,
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


def test_cn_step_shapes_and_substepping():
    """
    Exercise the CN_step Picard iteration and substep loop:
    - verifies that it runs to completion with small tolerances,
    - checks shapes of the returned fields and sources.
    """
    params, G, _ = _small_parameters_for_algorithms()

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
        parameters=params_cn,
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
