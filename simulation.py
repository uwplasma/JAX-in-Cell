from jax import jit
from jax import lax
from jax import vmap
from tqdm import tqdm
import jax.numpy as jnp
from jax.random import normal
from functools import partial
from jax.debug import print as jprint
from jax.random import PRNGKey, uniform
from particles import fields_to_particles_grid, boris_step
from sources import current_density, calculate_charge_density
from boundary_conditions import set_BC_positions, set_BC_particles
from fields import field_update1, field_update2, E_from_Poisson_equation
from constants import speed_of_light, epsilon_0, charge_electron, charge_proton, mass_electron, mass_proton

def initialize_simulation_parameters(user_parameters):
    """
    Initialize the simulation parameters for a particle-in-cell simulation, 
    combining user-provided values with predefined defaults. This function 
    ensures all required parameters are set and automatically calculates 
    derived parameters based on the inputs.

    The function uses lambda functions to define derived parameters that 
    depend on other parameters. These lambda functions are evaluated after 
    merging user-provided parameters with the defaults, ensuring derived 
    parameters are consistent with any overrides.

    Parameters:
    ----------
    user_parameters : dict
        Dictionary containing user-specified parameters. Any parameter not provided
        will default to predefined values.

    Returns:
    -------
    parameters : dict
        Dictionary containing all simulation parameters, with user-provided values
        overriding defaults.
    """
    # Define all default parameters in a single dictionary
    default_parameters = {
        # Basic simulation settings
        "length": 1e-2,                       # Dimensions of the simulation box
        "amplitude_perturbation_x": 0.1,      # Amplitude of sinusoidal perturbation in x
        "wavenumber_perturbation_x": lambda p: 2 * jnp.pi / p["length"], # Wavenumber of sinusoidal perturbation in x
        "grid_points_per_Debye_length": 9,    # dx over Debye length
        "vth_electrons_over_c": 0.05,         # Thermal velocity of electrons over speed of light
        "CFL_factor": 0.5,                    # dt * speed_of_light / dx
        "seed": 1701,                         # Random seed for reproducibility
        
        # Boundary conditions
        "particle_BC_left": 0,                # Left boundary condition for particles
        "particle_BC_right": 0,               # Right boundary condition for particles
        "field_BC_left": 0,                   # Left boundary condition for fields
        "field_BC_right": 0,                  # Right boundary condition for fields
        
        # External fields (initialized to zero)
        "external_electric_field_value": 0, 
        "external_magnetic_field_value": 0,
        
    }

    # Merge user-provided parameters into the default dictionary
    parameters = {**default_parameters, **user_parameters}
    
    # Compute derived parameters based on user-provided or default values
    for key, value in parameters.items():
        if callable(value):  # If the value is a lambda function, compute it
            parameters[key] = value(parameters)

    return parameters

def initialize_particles_fields(parameters_float, number_grid_points=50, number_pseudoelectrons=500):
    """
    Initialize particles and electromagnetic fields for a Particle-in-Cell simulation.
    
    This function generates particle positions, velocities, charges, masses, and 
    charge-to-mass ratios, as well as the initial electric and magnetic fields. It 
    combines user-provided parameters with default values and calculates derived quantities.

    Parameters:
    ----------
    user_parameters : dict
        Dictionary of user-specified simulation parameters. Can include:
        - Physical parameters (e.g., box size, number of particles, thermal velocities).
        - Numerical parameters (e.g., grid resolution, CFL factor).
        - Boundary conditions and random seed for reproducibility.

    Returns:
    -------
    parameters : dict
        Updated dictionary containing:
        - Particle positions and velocities (electrons and ions).
        - Particle charges, masses, and charge-to-mass ratios.
        - Initial electric and magnetic fields.
    """
    # Merge user parameters with defaults
    parameters = initialize_simulation_parameters(parameters_float)

    # Simulation box dimensions
    length = parameters["length"]
    box_size = (length, length, length)

    # Random key generator for reproducibility
    random_key = PRNGKey(parameters["seed"])
    
    # **Particle Positions**
    # Electron positions: Add random y, z positions to x initialized by perturbation

    electron_xs = jnp.linspace(-length / 2, length / 2, number_pseudoelectrons)
    electron_xs-= (parameters["amplitude_perturbation_x"] / parameters["wavenumber_perturbation_x"]) * jnp.sin(parameters["wavenumber_perturbation_x"] * electron_xs)
    electron_ys = uniform(random_key, shape=(number_pseudoelectrons,), minval=-box_size[1] / 2, maxval=box_size[1] / 2)
    electron_zs = uniform(random_key, shape=(number_pseudoelectrons,), minval=-box_size[2] / 2, maxval=box_size[2] / 2)
    electron_positions = jnp.stack((electron_xs, electron_ys, electron_zs), axis=1)

    # Ion positions: Add random y, z positions to uniform grid x positions
    ion_xs = jnp.linspace(-length / 2, length / 2, number_pseudoelectrons)
    ion_ys = uniform(random_key, shape=(number_pseudoelectrons,), minval=-box_size[1] / 2, maxval=box_size[1] / 2)
    ion_zs = uniform(random_key, shape=(number_pseudoelectrons,), minval=-box_size[2] / 2, maxval=box_size[2] / 2)
    ion_positions = jnp.stack((ion_xs, ion_ys, ion_zs), axis=1)

    particle_positions = jnp.concatenate((electron_positions, ion_positions))
    
    # **Particle Velocities**
    # Thermal velocities (Maxwell-Boltzmann distribution)
    vth_electrons = parameters["vth_electrons_over_c"] * speed_of_light
    vth_ions = vth_electrons * jnp.sqrt(mass_electron / mass_proton)
    v_electrons = vth_electrons / jnp.sqrt(2) * normal(random_key, shape=(number_pseudoelectrons, 3))
    v_ions = vth_ions / jnp.sqrt(2) * normal(random_key, shape=(number_pseudoelectrons, 3))
    particle_velocities = jnp.concatenate((v_electrons, v_ions))

    # **Particle Charges and Masses**
    # Pseudoparticle weights -> density of real particles = number_pseudoelectrons * weight / length, put in terms of Debye length
    Debye_length_per_dx = 1 / parameters["grid_points_per_Debye_length"]
    weight = (
        epsilon_0
        * mass_electron
        * speed_of_light**2
        / charge_electron**2
        * number_grid_points**2
        / length
        / (2 * number_pseudoelectrons)
        * parameters["vth_electrons_over_c"]**2
        / Debye_length_per_dx**2
    )
    charges = jnp.concatenate((
        charge_electron * weight * jnp.ones((number_pseudoelectrons, 1)),
        charge_proton * weight * jnp.ones((number_pseudoelectrons, 1))
    ), axis=0)
    masses = jnp.concatenate((
        mass_electron * weight * jnp.ones((number_pseudoelectrons, 1)),
        mass_proton * weight * jnp.ones((number_pseudoelectrons, 1))
    ), axis=0)
    charge_to_mass_ratios = charges / masses
    
    # **Fields Initialization**
    # Grid setup
    dx = length / number_grid_points
    grid = jnp.linspace(-length / 2 + dx / 2, length / 2 - dx / 2, number_grid_points)
    dt = parameters["CFL_factor"] * dx / speed_of_light
    
    # Electric field initialized to same perturbation as particle positions
    # E_field = jnp.zeros((grid.size, 3))
    # for i in range(grid.size):
    #     E_field = E_field.at[i, 0].set(
    #         q_electron * number_pseudoelectrons * A * jnp.sin(k * (grid[i] + dx / 2)) / (k * L * epsilon_0)
    #     )
    
    # Magnetic field initialized to zero
    B_field = jnp.zeros((grid.size, 3))
    
    # Electric field initialization using Poisson's equation
    E_field_x = E_from_Poisson_equation(particle_positions, charges, dx, grid, 0, 0)
    E_field = jnp.stack((E_field_x, jnp.zeros_like(grid), jnp.zeros_like(grid)), axis=1)
    
    fields = (E_field, B_field)

    # **Update parameters**
    parameters.update({
        "weight": weight,
        "particle_positions": particle_positions,
        "particle_velocities": particle_velocities,
        "charges": charges,
        "masses": masses,
        "charge_to_mass_ratios": charge_to_mass_ratios,
        "fields": fields,
        "grid": grid,
        "dx": dx,
        "dt": dt,
        "box_size": box_size,
        "external_electric_field": parameters["external_electric_field_value"]*jnp.ones((number_grid_points, 3)),
        "external_magnetic_field": parameters["external_magnetic_field_value"]*jnp.ones((number_grid_points, 3)),
        "number_pseudoelectrons": number_pseudoelectrons,
        "number_grid_points": number_grid_points,
    })
    
    return parameters


@partial(jit, static_argnames=['number_grid_points', 'number_pseudoelectrons', 'total_steps'])
def simulation(parameters_float, number_grid_points=50, number_pseudoelectrons=500, total_steps=350):
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
        - Numerical parameters: grid resolution, CFL factor, time step size.
        - Boundary conditions for particles and fields.
        - Random seed for reproducibility.

    Returns:
    -------
    output : dict
        Contains the following simulation results:
        - "E_field": Time evolution of the electric field on the grid.
        - "E_energy": Time evolution of the total electric field energy.
        - "charge_density": Time evolution of the charge density on the grid.
    """

    # **Initialize simulation parameters**
    parameters = initialize_particles_fields(parameters_float, number_grid_points=number_grid_points, number_pseudoelectrons=number_pseudoelectrons)

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

    # **Initialize particle positions and velocities**
    positions = parameters["particle_positions"]
    velocities = parameters["particle_velocities"]

    # Leapfrog integration: positions at half-step before the start
    positions_plus1_2, velocities, qs, ms, q_ms = set_BC_particles(
        positions + (dt / 2) * velocities, velocities,
        parameters["charges"], parameters["masses"], parameters["charge_to_mass_ratios"],
        dx, grid, *box_size, particle_BC_left, particle_BC_right
    )
    positions_minus1_2 = set_BC_positions(
        positions - (dt / 2) * velocities,
        parameters["charges"], dx, grid, *box_size,
        particle_BC_left, particle_BC_right
    )

    # **Preallocate arrays for diagnostics**
    E_field_over_time = jnp.zeros((total_steps, grid.size, 3))
    E_energy_over_time = jnp.zeros((total_steps,))
    charge_density_over_time = jnp.zeros((total_steps, grid.size))

    initial_carry = (
        E_field, B_field, positions_minus1_2, positions,
        positions_plus1_2, velocities, qs, ms, q_ms,  # Include particle properties
        (E_field_over_time, E_energy_over_time, charge_density_over_time)
    )

    def simulation_step(carry, step_index):
        # Unpack carry
        (
            E_field, B_field, positions_minus1_2, positions,
            positions_plus1_2, velocities, qs, ms, q_ms,  # Unpack particle properties
            diagnostics
        ) = carry

        # Compute current density
        J = current_density(
            positions_minus1_2, positions, positions_plus1_2, velocities, qs,
            dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right
        )

        # First half-step field update
        E_field, B_field = field_update1(
            E_field, B_field, dx, dt / 2, J, field_BC_left, field_BC_right
        )

        # Add external fields
        total_E = E_field + parameters["external_electric_field"]
        total_B = B_field + parameters["external_magnetic_field"]

        # Interpolate fields to particle positions
        E_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
            x_n, total_E, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right
        ))(positions_plus1_2)
        B_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
            x_n, total_B, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right
        ))(positions_plus1_2)

        # Particle update: Boris pusher
        positions_plus3_2, velocities_plus1 = boris_step(
            dt, positions_plus1_2, velocities, q_ms, E_field_at_x, B_field_at_x
        )

        # Apply boundary conditions
        positions_plus3_2, velocities_plus1, qs, ms, q_ms = set_BC_particles(
            positions_plus3_2, velocities_plus1, qs, ms, q_ms, dx, grid,
            *box_size, particle_BC_left, particle_BC_right
        )
        positions_plus1 = set_BC_positions(
            positions_plus3_2 - (dt / 2) * velocities_plus1, qs, dx, grid, *box_size,
            particle_BC_left, particle_BC_right
        )

        # Second half-step field update
        J = current_density(
            positions_plus1_2, positions_plus1, positions_plus3_2, velocities_plus1,
            qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right
        )
        E_field, B_field = field_update2(
            E_field, B_field, dx, dt / 2, J, field_BC_left, field_BC_right
        )

        # Update positions and velocities
        positions_minus1_2, positions_plus1_2 = positions_plus1_2, positions_plus3_2
        velocities = velocities_plus1
        positions = positions_plus1

        # Diagnostics
        E_field_over_time, E_energy_over_time, charge_density_over_time = diagnostics
        E_field_over_time = E_field_over_time.at[step_index].set(E_field)
        E_energy_over_time = E_energy_over_time.at[step_index].set(
            jnp.sum(0.5 * epsilon_0 * vmap(jnp.dot)(E_field, E_field) / dx)
        )
        charge_density_over_time = charge_density_over_time.at[step_index].set(
            calculate_charge_density(positions, qs, dx, grid, particle_BC_left, particle_BC_right)
        )

        # Return updated carry
        carry = (
            E_field, B_field, positions_minus1_2, positions,
            positions_plus1_2, velocities, qs, ms, q_ms,  # Update particle properties
            (E_field_over_time, E_energy_over_time, charge_density_over_time)
        )
        return carry, None

    # Run simulation using lax.scan
    final_carry, _ = lax.scan(simulation_step, initial_carry, jnp.arange(total_steps))

    # Extract diagnostics
    _, _, _, _, _, _, _, _, _, diagnostics = final_carry
    E_field_over_time, E_energy_over_time, charge_density_over_time = diagnostics


    # **Output results**
    output = {
        "E_field": E_field_over_time,
        "E_energy": E_energy_over_time,
        "charge_density": charge_density_over_time,
        "time_array": jnp.linspace(0, total_steps*dt, total_steps),
        "number_grid_points": number_grid_points,
        "number_pseudoelectrons": number_pseudoelectrons,
        "total_steps": total_steps,
    }

    return {**output, **parameters}
