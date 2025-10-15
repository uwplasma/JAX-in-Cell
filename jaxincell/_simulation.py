import jax.numpy as jnp

from functools import partial
from jax_tqdm import scan_tqdm

from jax import lax, jit, config
config.update("jax_enable_x64", True)

from ._diagnostics import diagnostics
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._algorithms import Boris_step, CN_step
from ._initialization import initialize_particles_fields

__all__ = ["simulation"]

@partial(jit, static_argnames=['number_grid_points', 'number_pseudoelectrons', 'total_steps', 'field_solver', "time_evolution_algorithm",
                               "max_number_of_Picard_iterations_implicit_CN","number_of_particle_substeps_implicit_CN"])
def simulation(input_parameters={}, number_grid_points=100, number_pseudoelectrons=3000, total_steps=1000, 
               field_solver=0,positions=None, velocities=None,time_evolution_algorithm=0,max_number_of_Picard_iterations_implicit_CN=7, number_of_particle_substeps_implicit_CN=2):
    """
    Run a plasma physics simulation using a Particle-In-Cell (PIC) method in JAX.

    This function simulates the evolution of a plasma system by solving for particle motion
    (electrons and ions) and self-consistent electromagnetic fields on a grid. It uses the 
    Boris algorithm for particle updates and a leapfrog scheme for field updates.

    Parameters:
    ----------
    user_parameters : dict
        User-defined parameters for the simulation. These can include:
        - Physical parameters: box size, number of particles, thermal velocities.
        - Numerical parameters: grid resolution, time step size.
        - Boundary conditions for particles and fields.
        - Random seed for reproducibility.

    Returns:
    -------
    output : dict
    """
    # **Initialize simulation parameters**
    parameters = initialize_particles_fields(input_parameters, number_grid_points=number_grid_points,
                                             number_pseudoelectrons=number_pseudoelectrons, total_steps=total_steps,
                                             max_number_of_Picard_iterations_implicit_CN=max_number_of_Picard_iterations_implicit_CN,
                                             number_of_particle_substeps_implicit_CN=number_of_particle_substeps_implicit_CN)

    # Extract parameters for convenience
    dx = parameters["dx"]
    dt = parameters["dt"]
    grid = parameters["grid"]
    box_size = parameters["box_size"]
    E_field, B_field = parameters["fields"]
    field_BC_left = parameters["field_BC_left"]
    field_BC_right = parameters["field_BC_right"]
    particle_BC_left = parameters["particle_BC_left"]
    particle_BC_right = parameters["particle_BC_right"]

    # **Use provided positions/velocities if given, otherwise use defaults**
    if positions is None:
        positions = parameters["initial_positions"]
    if velocities is None:
        velocities = parameters["initial_velocities"]

    # Ensure the provided positions/velocities match the expected shape
    if positions.shape != parameters["initial_positions"].shape:
        raise ValueError(f"Expected positions shape {parameters['initial_positions'].shape}, got {positions.shape}")
    if velocities.shape != parameters["initial_velocities"].shape:
        raise ValueError(f"Expected velocities shape {parameters['initial_velocities'].shape}, got {velocities.shape}")

    # Leapfrog integration: positions at half-step before the start
    positions_plus1_2, velocities, qs, ms, q_ms = set_BC_particles(
        positions + (dt / 2) * velocities, velocities,
        parameters["charges"], parameters["masses"], parameters["charge_to_mass_ratios"],
        dx, grid, *box_size, particle_BC_left, particle_BC_right)
    
    positions_minus1_2 = set_BC_positions(
        positions - (dt / 2) * velocities,
        parameters["charges"], dx, grid, *box_size,
        particle_BC_left, particle_BC_right)
    
    if time_evolution_algorithm == 0:
        initial_carry = (E_field, B_field, positions_minus1_2, positions,
                         positions_plus1_2, velocities, qs, ms, q_ms)
        step_func = lambda carry, step_index: Boris_step(
            carry, step_index, parameters, dx, dt, grid, box_size,
            particle_BC_left, particle_BC_right, field_BC_left, field_BC_right, field_solver)
    else:
        initial_carry = (E_field, B_field, positions,
                         velocities, qs, ms, q_ms)
        step_func = lambda carry, step_index: CN_step(
            carry, step_index, parameters, dx, dt, grid, box_size,
            particle_BC_left, particle_BC_right, field_BC_left, field_BC_right,
            parameters["number_of_particle_substeps_implicit_CN"])

    @scan_tqdm(total_steps)
    def simulation_step(carry, step_index):
        return step_func(carry, step_index)
 
    # Run simulation
    _, results = lax.scan(simulation_step, initial_carry, jnp.arange(total_steps))

    # Unpack results
    positions_over_time, velocities_over_time, electric_field_over_time, \
    magnetic_field_over_time, current_density_over_time, charge_density_over_time = results
    
    # **Output results**
    temporary_output = {
        "position_electrons": positions_over_time[ :, :number_pseudoelectrons, :],
        "velocity_electrons": velocities_over_time[:, :number_pseudoelectrons, :],
        "mass_electrons":     parameters["masses"][   :number_pseudoelectrons],
        "charge_electrons":   parameters["charges"][  :number_pseudoelectrons],
        "position_ions":      positions_over_time[ :, number_pseudoelectrons:, :],
        "velocity_ions":      velocities_over_time[:, number_pseudoelectrons:, :],
        "mass_ions":          parameters["masses"][   number_pseudoelectrons:],
        "charge_ions":        parameters["charges"][  number_pseudoelectrons:],
        "electric_field":  electric_field_over_time,
        "magnetic_field":  magnetic_field_over_time,
        "current_density": current_density_over_time,
        "charge_density":  charge_density_over_time,
        "number_grid_points":     number_grid_points,
        "number_pseudoelectrons": number_pseudoelectrons,
        "total_steps": total_steps,
        "time_array":  jnp.linspace(0, total_steps * dt, total_steps),
    }
    
    output = {**temporary_output, **parameters}

    diagnostics(output)
    
    return output