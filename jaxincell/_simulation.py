import jax.numpy as jnp
from jax.lax import cond
from functools import partial
from jax.random import normal
from jax import lax, jit, vmap, config
from jax.debug import print as jprint
from jax.random import PRNGKey, uniform
from ._particles import fields_to_particles_grid, boris_step
from ._sources import current_density, calculate_charge_density,current_density_CN
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._fields import field_update, E_from_Gauss_1D_Cartesian, E_from_Gauss_1D_FFT, E_from_Poisson_1D_FFT
from ._constants import speed_of_light, epsilon_0, charge_electron, charge_proton, mass_electron, mass_proton
from ._diagnostics import diagnostics
from jax_tqdm import scan_tqdm
try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib
config.update("jax_enable_x64", True)

__all__ = ["initialize_simulation_parameters", "initialize_particles_fields", "simulation", "load_parameters"]

def load_parameters(input_file):
    """
    Load parameters from a TOML file.

    Parameters:
    ----------
    input_file : str
        Path to the TOML file containing simulation parameters.

    Returns:
    -------
    parameters : dict
        Dictionary containing simulation parameters.
    """
    parameters = tomllib.load(open(input_file, "rb"))
    input_parameters = parameters['input_parameters']
    solver_parameters = parameters['solver_parameters']
    return input_parameters, solver_parameters

def initialize_simulation_parameters(user_parameters={}):
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
        "length": 1e-2,                           # Dimensions of the simulation box
        "amplitude_perturbation_x": 5e-4,          # Amplitude of sinusoidal (sin) perturbation in x
        "wavenumber_electrons": 8,    # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
        "wavenumber_ions": 0,    # Wavenumber of sinusoidal ion density perturbation in x (factor of 2pi/length)
        "grid_points_per_Debye_length": 2,        # dx over Debye length
        "vth_electrons_over_c": 0.05,             # Thermal velocity of electrons over speed of light
        "ion_temperature_over_electron_temperature": 0.01, # Temperature ratio of ions to electrons
        "timestep_over_spatialstep_times_c": 1.0, # dt * speed_of_light / dx
        "seed": 1701,                             # Random seed for reproducibility
        "electron_drift_speed": 0,                # Drift speed of electrons
        "ion_drift_speed":      0,                # Drift speed of ions
        "velocity_plus_minus_electrons": False,   # create two groups of electrons moving in opposite directions
        "velocity_plus_minus_ions": False,        # create two groups of electrons moving in opposite directions
        "print_info": True,                       # Print information about the simulation
        
        # Boundary conditions
        "particle_BC_left":  0,                   # Left boundary condition for particles
        "particle_BC_right": 0,                   # Right boundary condition for particles
        "field_BC_left":     0,                   # Left boundary condition for fields
        "field_BC_right":    0,                   # Right boundary condition for fields
        
        # External fields (initialized to zero)
        "external_electric_field_amplitude": 0, # Amplitude of sinusoidal (cos) perturbation in x
        "external_electric_field_wavenumber": 0, # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
        "external_magnetic_field_amplitude": 0, # Amplitude of sinusoidal (cos) perturbation in x
        "external_magnetic_field_wavenumber": 0, # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
    }

    # Merge user-provided parameters into the default dictionary
    parameters = {**default_parameters, **user_parameters}
    
    # Compute derived parameters based on user-provided or default values
    for key, value in parameters.items():
        if callable(value):  # If the value is a lambda function, compute it
            parameters[key] = value(parameters)

    return parameters

def initialize_particles_fields(input_parameters={}, number_grid_points=50, number_pseudoelectrons=500, total_steps=350):
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
        - Numerical parameters (e.g., grid resolution, timestep).
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
    parameters = initialize_simulation_parameters(input_parameters)

    # Simulation box dimensions
    length = parameters["length"]
    box_size = (length, length, length)

    # Random key generator for reproducibility
    random_key = PRNGKey(parameters["seed"])
    
    # **Particle Positions**
    wavenumber_perturbation_x_electrons = parameters["wavenumber_electrons"] * 2 * jnp.pi / length
    electron_xs = jnp.linspace(-length / 2, length / 2, number_pseudoelectrons)
    electron_xs+= parameters["amplitude_perturbation_x"] * jnp.sin(wavenumber_perturbation_x_electrons * electron_xs)
    electron_ys = uniform(random_key, shape=(number_pseudoelectrons,), minval=-box_size[1] / 2, maxval=box_size[1] / 2)
    electron_zs = uniform(random_key, shape=(number_pseudoelectrons,), minval=-box_size[2] / 2, maxval=box_size[2] / 2)
    electron_positions = jnp.stack((electron_xs, electron_ys, electron_zs), axis=1)

    # Ion positions: Add random y, z positions to uniform grid x positions
    wavenumber_perturbation_x_ions = parameters["wavenumber_ions"] * 2 * jnp.pi / length
    ion_xs = jnp.linspace(-length / 2, length / 2, number_pseudoelectrons)
    ion_xs+= parameters["amplitude_perturbation_x"] * jnp.sin(wavenumber_perturbation_x_ions * ion_xs)
    ion_ys = uniform(random_key, shape=(number_pseudoelectrons,), minval=-box_size[1] / 2, maxval=box_size[1] / 2)
    ion_zs = uniform(random_key, shape=(number_pseudoelectrons,), minval=-box_size[2] / 2, maxval=box_size[2] / 2)
    ion_positions = jnp.stack((ion_xs, ion_ys, ion_zs), axis=1)

    positions = jnp.concatenate((electron_positions, ion_positions))
    
    # **Particle Velocities**
    # Thermal velocities (Maxwell-Boltzmann distribution)
    vth_electrons = parameters["vth_electrons_over_c"] * speed_of_light
    v_electrons_x = vth_electrons / jnp.sqrt(2) * normal(random_key, shape=(number_pseudoelectrons, )) + parameters["electron_drift_speed"]
    v_electrons_x = jnp.where(parameters["velocity_plus_minus_electrons"], v_electrons_x * (-1) ** jnp.arange(0, number_pseudoelectrons), v_electrons_x)
    v_electrons_y = jnp.zeros((number_pseudoelectrons, ))
    v_electrons_z = jnp.zeros((number_pseudoelectrons, ))
    electron_velocities = jnp.stack((v_electrons_x, v_electrons_y, v_electrons_z), axis=1)
    vth_ions = jnp.sqrt(jnp.abs(parameters["ion_temperature_over_electron_temperature"])) * vth_electrons * jnp.sqrt(jnp.abs(mass_electron / mass_proton))
    v_ions_x = vth_ions / jnp.sqrt(2) * normal(random_key, shape=(number_pseudoelectrons, )) + parameters["ion_drift_speed"]
    v_ions_x = jnp.where(parameters["velocity_plus_minus_ions"], v_ions_x * (-1) ** jnp.arange(0, number_pseudoelectrons), v_ions_x)
    v_ions_y = jnp.zeros((number_pseudoelectrons, ))
    v_ions_z = jnp.zeros((number_pseudoelectrons, ))
    ion_velocities = jnp.stack((v_ions_x, v_ions_y, v_ions_z), axis=1)
    
    velocities = jnp.concatenate((electron_velocities, ion_velocities))

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
        charge_proton   * weight * jnp.ones((number_pseudoelectrons, 1))
    ), axis=0)
    masses = jnp.concatenate((
        mass_electron * weight * jnp.ones((number_pseudoelectrons, 1)),
        mass_proton   * weight * jnp.ones((number_pseudoelectrons, 1))
    ), axis=0)
    charge_to_mass_ratios = charges / masses

    # Grid setup
    dx = length / number_grid_points
    grid = jnp.linspace(-length / 2 + dx / 2, length / 2 - dx / 2, number_grid_points)
    dt = parameters["timestep_over_spatialstep_times_c"] * dx / speed_of_light

    # Print information about the simulation
    plasma_frequency = jnp.sqrt(number_pseudoelectrons * weight * charge_electron**2)/jnp.sqrt(mass_electron)/jnp.sqrt(epsilon_0)/jnp.sqrt(length)

    cond(parameters["print_info"],
        lambda _: jprint((
            # f"Number of pseudoelectrons: {number_pseudoelectrons}\n"
            # f"Number of grid points: {number_grid_points}\n"
            "Length of the simulation box: {:.2e} Debye lengths\n"
            "Density of electrons: {:.2e} m^-3\n"
            "Ion temperature: {:.2e} eV\n"
            "Electron temperature: {:.2e} eV\n"
            "Debye length: {:.2e} m\n"
            "Wavenumber * Debye length: {:.2e}\n"
            "Pseudoparticles per cell: {:.2e}\n"
            "Steps at each plasma frequency: {}\n"
            "Total time: {} / plasma frequency\n"
            "Number of particles on a Debye cube: {:.2e}\n"
            "Charge x External electric field x Debye Length / Temperature: {:.2e}\n"
        ),length/(Debye_length_per_dx*dx),
          number_pseudoelectrons * weight / length,
          -parameters["ion_temperature_over_electron_temperature"] * vth_electrons**2 * mass_electron / 2 / charge_electron,
          -mass_electron * vth_electrons**2 / 2 / charge_electron,
          Debye_length_per_dx*dx,
          wavenumber_perturbation_x_electrons*Debye_length_per_dx*dx,
          number_pseudoelectrons / number_grid_points,
          1/(plasma_frequency * dt),
          dt * plasma_frequency * total_steps,
          number_pseudoelectrons * weight / length * (Debye_length_per_dx*dx)**3,
          -charge_electron * parameters["external_electric_field_amplitude"] * Debye_length_per_dx*dx / (mass_electron * vth_electrons**2 / 2),
        ), lambda _: None, operand=None)
    
    # **Fields Initialization**
    # Electric field initialized to same perturbation as particle positions
    # E_field = jnp.zeros((grid.size, 3))
    # for i in range(grid.size):
    #     E_field = E_field.at[i, 0].set(
    #         -charge_electron * weight * number_pseudoelectrons * parameters["amplitude_perturbation_x"] * jnp.sin(wavenumber_perturbation_x * (grid[i] + dx / 2)) / (length * epsilon_0)
    #     )
    # Magnetic field initialized to zero
    B_field = jnp.zeros((grid.size, 3))
    
    # Electric field initialization using Poisson's equation
    charge_density = calculate_charge_density(positions, charges, dx, grid, parameters["particle_BC_left"], parameters["particle_BC_right"])
    # E_field_x = E_from_Gauss_1D_Cartesian(charge_density, dx)
    E_field_x = E_from_Gauss_1D_FFT(charge_density, dx)
    # E_field_x = E_from_Poisson_1D_FFT(charge_density, dx)
    E_field = jnp.stack((E_field_x, jnp.zeros_like(grid), jnp.zeros_like(grid)), axis=1)
    
    fields = (E_field, B_field)
    
    external_E_field_x = parameters["external_electric_field_amplitude"] * jnp.cos(parameters["external_electric_field_wavenumber"] * jnp.linspace(-jnp.pi, jnp.pi, number_grid_points))
    external_B_field_x = parameters["external_magnetic_field_amplitude"] * jnp.cos(parameters["external_magnetic_field_wavenumber"] * jnp.linspace(-jnp.pi, jnp.pi, number_grid_points))

    # **Update parameters**
    parameters.update({
        "weight": weight,
        "initial_positions": positions,
        "initial_velocities": velocities,
        "charges": charges,
        "masses": masses,
        "charge_to_mass_ratios": charge_to_mass_ratios,
        "fields": fields,
        "grid": grid,
        "dx": dx,
        "dt": dt,
        "box_size": box_size,
        "external_electric_field": jnp.array([external_E_field_x, jnp.zeros((number_grid_points,)), jnp.zeros((number_grid_points,))]).T,
        "external_magnetic_field": jnp.array([external_B_field_x, jnp.zeros((number_grid_points,)), jnp.zeros((number_grid_points,))]).T,
        "number_pseudoelectrons": number_pseudoelectrons,
        "number_grid_points": number_grid_points,
        "plasma_frequency": plasma_frequency,
    })
    
    return parameters


@partial(jit, static_argnames=['number_grid_points', 'number_pseudoelectrons', 'total_steps', 'field_solver'])
def simulation(input_parameters={}, number_grid_points=100, number_pseudoelectrons=3000, total_steps=1000, field_solver=0):
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
                                             number_pseudoelectrons=number_pseudoelectrons, total_steps=total_steps)

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
    positions  = parameters["initial_positions"]
    velocities = parameters["initial_velocities"]

    # Leapfrog integration: positions at half-step before the start
    positions_plus1_2, velocities, qs, ms, q_ms = set_BC_particles(
        positions + (dt / 2) * velocities, velocities,
        parameters["charges"], parameters["masses"], parameters["charge_to_mass_ratios"],
        dx, grid, *box_size, particle_BC_left, particle_BC_right)
    
    positions_minus1_2 = set_BC_positions(
        positions - (dt / 2) * velocities,
        parameters["charges"], dx, grid, *box_size,
        particle_BC_left, particle_BC_right)

    # initial_carry = (
    #     E_field, B_field, positions_minus1_2, positions,
    #     positions_plus1_2, velocities, qs, ms, q_ms,)

    initial_carry = (
     E_field, B_field, positions,
     velocities, qs, ms, q_ms,)
    #Add external fields
    # total_E = E_field + parameters["external_electric_field"]
    # total_B = B_field + parameters["external_magnetic_field"]
    #  # Interpolate fields to particle positions
    # E_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
    #     x_n, total_E, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(positions_plus1_2)
    
    # B_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
    #     x_n, total_B, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right))(positions_plus1_2)

    # # Particle update: Boris pusher
    # positions_plus3_2, velocities_plus1 = boris_step(
    #     dt, positions_plus1_2, velocities, q_ms, E_field_at_x, B_field_at_x)

    # # Apply boundary conditions
    # positions_plus3_2, velocities_plus1, qs, ms, q_ms = set_BC_particles(
    #     positions_plus3_2, velocities_plus1, qs, ms, q_ms, dx, grid,
    #     *box_size, particle_BC_left, particle_BC_right)
    
    # positions_plus1 = set_BC_positions(positions_plus3_2 - (dt / 2) * velocities_plus1,
    #                                     qs, dx, grid, *box_size, particle_BC_left, particle_BC_right)
    
    # positions_plus1_2 = 0.5 * (positions + positions_plus1)
    # velocities_plus1_2 = 0.5 * (velocities + velocities_plus1)

    # positions_plus1_2, velocities_plus1_2, qs, ms, q_ms = set_BC_particles(
    # positions_plus1_2, velocities_plus1_2, qs, ms, q_ms, dx, grid,
    # *box_size, particle_BC_left, particle_BC_right)

    # initial_carry = (
    #  E_field, B_field, positions,positions_plus1_2,
    #  velocities,velocities_plus1_2, qs, ms, q_ms,)
    
    # @scan_tqdm(total_steps)
    # def simulation_step(carry, step_index):
    #     (E_field, B_field, positions_minus1_2, positions,
    #      positions_plus1_2, velocities, qs, ms, q_ms) = carry
        
    #     # Add external fields
    #     total_E = E_field + parameters["external_electric_field"]
    #     total_B = B_field + parameters["external_magnetic_field"]

    #     # Interpolate fields to particle positions
    #     E_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
    #         x_n, total_E, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(positions_plus1_2)
        
    #     B_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
    #         x_n, total_B, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right))(positions_plus1_2)

    #     # Particle update: Boris pusher
    #     positions_plus3_2, velocities_plus1 = boris_step(
    #         dt, positions_plus1_2, velocities, q_ms, E_field_at_x, B_field_at_x)

    #     # Apply boundary conditions
    #     positions_plus3_2, velocities_plus1, qs, ms, q_ms = set_BC_particles(
    #         positions_plus3_2, velocities_plus1, qs, ms, q_ms, dx, grid,
    #         *box_size, particle_BC_left, particle_BC_right)
        
    #     positions_plus1 = set_BC_positions(positions_plus3_2 - (dt / 2) * velocities_plus1,
    #                                        qs, dx, grid, *box_size, particle_BC_left, particle_BC_right)

    #     if field_solver != 0:
    #         charge_density = calculate_charge_density(positions, qs, dx, grid + dx / 2, particle_BC_left, particle_BC_right)
    #         switcher = {
    #             1: E_from_Gauss_1D_FFT,
    #             2: E_from_Gauss_1D_Cartesian,
    #             3: E_from_Poisson_1D_FFT,
    #         }
    #         E_field = E_field.at[:,0].set(switcher[field_solver](charge_density, dx))
    #         J = 0
    #     else:
    #         J = current_density(positions_plus1_2, positions_plus1, positions_plus3_2, velocities_plus1,
    #                             qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
    #         E_field, B_field = field_update(E_field, B_field, dx, dt, J, field_BC_left, field_BC_right)

    #     # Update positions and velocities
    #     positions_minus1_2, positions_plus1_2 = positions_plus1_2, positions_plus3_2
    #     velocities = velocities_plus1
    #     positions = positions_plus1

    #     # Prepare state for the next step
    #     carry = (E_field, B_field, positions_minus1_2, positions,
    #              positions_plus1_2, velocities, qs, ms, q_ms)

    #     # Collect data for storage
    #     charge_density = calculate_charge_density(positions, qs, dx, grid, particle_BC_left, particle_BC_right)
    #     step_data = (positions, velocities, E_field, B_field, J, charge_density)
        
    #     return carry, step_data
    
    # @jit
    # def current_density_CN(positions_half, velocities_half, qs, dx, dt, grid, grid_start, BC_left, BC_right):
    #     def current_contribution(x_p, v_p, q):
    #         contributions = jnp.zeros((len(grid), 3))
    #         L = grid[-1] - grid[0] + dx  # Total grid length
            
    #         # Base contribution from the particle
    #         for i in range(len(grid)):
    #             x_i = grid[i]
    #             distance = jnp.abs(x_p - x_i)
    #             cond1 = distance <= dx/2
    #             cond2 = (dx/2 < distance) & (distance <= 3*dx/2)
    #             S = jnp.where(cond1, (0.75 - (x_p - x_i)**2 / dx**2),
    #                         jnp.where(cond2, 0.5 * (1.5 - distance/dx)**2, 0.0))

    #             contributions = contributions.at[i].add(q * v_p * S / dx)
            
    #         # Left boundary handling
    #         distance_to_left = x_p - grid[0]
    #         is_near_left = distance_to_left <= 3*dx/2
            
    #         # Periodic BC (wrap to right)
    #         x_p_wrapped_left = x_p + L
    #         contrib_left_periodic = jnp.zeros_like(grid)
    #         for i in range(len(grid)):
    #             x_i = grid[i]
    #             distance = jnp.abs(x_p_wrapped_left - x_i)
    #             cond1 = distance <= dx/2
    #             cond2 = (dx/2 < distance) & (distance <= 3*dx/2)
    #             S = jnp.where(cond1, (0.75 - (x_p_wrapped_left - x_i)**2 / dx**2),
    #                         jnp.where(cond2, 0.5 * (1.5 - distance/dx)**2, 0.0))
    #             contrib_left_periodic = contrib_left_periodic.at[i].add(q * v_p * S / dx)
    #         contributions += jnp.where((BC_left == 0) & is_near_left, contrib_left_periodic, 0.0)
            
    #         # # Reflective BC (mirror position and reverse velocity)
    #         # x_p_mirrored_left = 2 * grid[0] - x_p
    #         # v_p_mirrored_left = -v_p
    #         # contrib_left_reflective = jnp.zeros_like(grid)
    #         # for i in range(len(grid)):
    #         #     x_i = grid[i]
    #         #     distance = jnp.abs(x_p_mirrored_left - x_i)
    #         #     cond1 = distance <= dx/2
    #         #     cond2 = (dx/2 < distance) & (distance <= 3*dx/2)
    #         #     S = jnp.where(cond1, (0.75 - (x_p_mirrored_left - x_i)**2 / dx**2),
    #         #                 jnp.where(cond2, 0.5 * (1.5 - distance/dx)**2, 0.0))
    #         #     contrib_left_reflective = contrib_left_reflective.at[i].add(q * v_p_mirrored_left * S / dx)
    #         # contributions += jnp.where((BC_left == 1) & is_near_left, contrib_left_reflective, 0.0)
            
    #         # Right boundary handling
    #         distance_to_right = grid[-1] - x_p
    #         is_near_right = distance_to_right <= 3*dx/2
            
    #         # Periodic BC (wrap to left)
    #         x_p_wrapped_right = x_p - L
    #         contrib_right_periodic = jnp.zeros_like(grid)
    #         for i in range(len(grid)):
    #             x_i = grid[i]
    #             distance = jnp.abs(x_p_wrapped_right - x_i)
    #             cond1 = distance <= dx/2
    #             cond2 = (dx/2 < distance) & (distance <= 3*dx/2)
    #             S = jnp.where(cond1, (0.75 - (x_p_wrapped_right - x_i)**2 / dx**2),
    #                         jnp.where(cond2, 0.5 * (1.5 - distance/dx)**2, 0.0))
    #             contrib_right_periodic = contrib_right_periodic.at[i].add(q * v_p * S / dx)
    #         contributions += jnp.where((BC_right == 0) & is_near_right, contrib_right_periodic, 0.0)
            
    #         # # Reflective BC (mirror position and reverse velocity)
    #         # x_p_mirrored_right = 2 * grid[-1] - x_p
    #         # v_p_mirrored_right = -v_p
    #         # contrib_right_reflective = jnp.zeros_like(grid)
    #         # for i in range(len(grid)):
    #         #     x_i = grid[i]
    #         #     distance = jnp.abs(x_p_mirrored_right - x_i)
    #         #     cond1 = distance <= dx/2
    #         #     cond2 = (dx/2 < distance) & (distance <= 3*dx/2)
    #         #     S = jnp.where(cond1, (0.75 - (x_p_mirrored_right - x_i)**2 / dx**2),
    #         #                 jnp.where(cond2, 0.5 * (1.5 - distance/dx)**2, 0.0))
    #         #     contrib_right_reflective = contrib_right_reflective.at[i].add(q * v_p_mirrored_right * S / dx)
    #         # contributions += jnp.where((BC_right == 1) & is_near_right, contrib_right_reflective, 0.0)
            
    #         return contributions
        
    #     # Vectorize over particles
    #     contributions = vmap(current_contribution)(positions_half, velocities_half, qs)
    #     return jnp.sum(contributions, axis=0)
    # from jax import debug
    # @scan_tqdm(total_steps)
    # def simulation_step(carry, step_index):
    #     (E_field, B_field, positions,
    #      velocities, qs, ms, q_ms) = carry
        
    #     dt = parameters["dt"]

    #     # Initial guess for E, B and position at n+1 (using previous step)
    #     E_new = E_field
    #     B_new = B_field
    #     E_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
    #         x_n, E_new, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(positions)
    #     B_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
    #         x_n, B_new, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right))(positions)
    #     positions_new, velocities_new = boris_step(
    #         dt, positions, velocities, q_ms, E_field_at_x , B_field_at_x)

    #     #positions_new, velocities_new =  positions, velocities

    #     # def objective_function(state):
    #     # # def iteration_body(state):
    #     #     E_new,B_new, positions_new,velocities_new = state
    #         # Get the original shapes
    #     E_shape = E_new.shape
    #     B_shape = B_new.shape
    #     pos_shape = positions_new.shape
    #     vel_shape = velocities_new.shape

    #     # Get the number of elements in each array
    #     E_size = E_new.size
    #     B_size = B_new.size
    #     pos_size = positions_new.size
    #     vel_size = velocities_new.size

    #     def objective_function(state, q_ms,qs, ms):
    #         E_new = state[:E_size].reshape(E_shape)
    #         B_new = state[E_size:E_size + B_size].reshape(B_shape)
    #         positions_new = state[E_size + B_size:E_size + B_size + pos_size].reshape(pos_shape)
    #         velocities_new = state[E_size + B_size + pos_size:].reshape(vel_shape)
    #         E_avg = 0.5 * (E_field + E_new)
    #         B_avg = 0.5 * (B_field + B_new)
            
    #         x_half = 0.5 * (positions_new + positions)
    #         E_at_half = vmap(lambda x_n: fields_to_particles_grid(
    #             x_n, E_avg, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(x_half)
    #         B_at_half = vmap(lambda x_n: fields_to_particles_grid(
    #             x_n, B_avg, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right))(x_half)

    #         velocities_new = velocities + (q_ms * E_at_half) * dt
    #         v_half = 0.5 * (velocities + velocities_new)
    #         positions_new = positions + v_half * dt

    #         positions_new, velocities_new, qs, ms, q_ms = set_BC_particles(
    #             positions_new, velocities_new, qs, ms, q_ms, dx, grid,
    #             *box_size, particle_BC_left, particle_BC_right)

    #         positions_plus1_2 = 0.5 * (positions + positions_new)
    #         v_half = 0.5 * (velocities + velocities_new)
    #         J = current_density_CN( positions_plus1_2, v_half,
    #             qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)

    #         E_next = E_field - (dt / epsilon_0) * (J-jnp.sum(J)/len(J))
    #         delta_E = jnp.max(jnp.abs(E_next - E_new))/jnp.max(jnp.abs(E_next))
    #         debug.print("delta_E: {}", delta_E)
    #         # converged = delta_E < tolerance
    #         # return E_new,B_new, positions_new,velocities_new , qs, ms, q_ms ,iteration + 1, converged
    #         return delta_E

    #     # Initialize loop variables
    #     E_new = E_field
    #     iteration = 0
    #     converged = False


    #     # E_new,B_new, positions_new,velocities_new , _, _, _, _, _ = lax.while_loop(lambda state: jnp.logical_and(state[7] < max_iter, jnp.logical_not(state[8])),
    #     #                                 iteration_body,
    #     #                                 (E_new,B_new, positions_new,velocities_new , qs, ms, q_ms,iteration, converged))
    #     from jax.scipy.optimize import minimize
    #     initial_state_1 = jnp.ravel(jnp.array([jnp.ravel(E_new),jnp.ravel(B_new)]))
    #     initial_state_2 = jnp.ravel(jnp.array([jnp.ravel(positions_new),jnp.ravel(velocities_new)]))
    #     initial_state = jnp.ravel(jnp.concatenate([initial_state_1, initial_state_2]))
    #     solution = minimize(objective_function, initial_state, method="BFGS", tol=1e-8, args=(q_ms,qs, ms), options={"maxiter":20})
    #     debug.print("Optimization terminated: success = {}, status = {}", 
    #         solution.success, solution.status)
    #     debug.print("Tolerance reached: {}", solution.tol if hasattr(solution, "tol") else "N/A")
    #     debug.print("Number of iterations: {}", solution.nit)
    #     from jax import grad
    #     grad_objective = grad(objective_function)
    #     grad_values = grad_objective(initial_state, q_ms, qs, ms)
    #     debug.print("Gradient values: {}", grad_values)

    #     E_new = solution.x[:E_size].reshape(E_shape)
    #     B_new = solution.x[E_size:E_size + B_size].reshape(B_shape)
    #     positions_new = solution.x[E_size + B_size:E_size + B_size + pos_size].reshape(pos_shape)
    #     velocities_new = solution.x[E_size + B_size + pos_size:].reshape(vel_shape)

    #     # Update fields after iteration
    #     E_field = E_new
    #     B_field = B_new  # Assuming B update is handled similarly if needed

    #     # Final positions and velocities after convergence
    #     positions_plus1= positions_new
    #     velocities_plus1 = velocities_new
    #     positions_plus1_2 = 0.5 * (positions + positions_plus1)
    #     # Charge density with quadratic shape
    #     charge_density = calculate_charge_density(positions, qs, dx, grid, particle_BC_left, particle_BC_right)
    #     J = current_density(positions, positions_plus1_2, positions_plus1, velocities_new,
    #             qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
        
    #     # Prepare next state
    #     carry = (E_field, B_field, positions_plus1,
    #             velocities_plus1, qs, ms, q_ms)
        
    #     # Collect data
    #     step_data = (positions_plus1, velocities_plus1, E_field, B_field, J, charge_density)
        
    #     return carry, step_data

    # # Run simulation
    # _, results = lax.scan(simulation_step, initial_carry, jnp.arange(total_steps))

    # # Unpack results
    # positions_over_time, velocities_over_time, electric_field_over_time, \
    # magnetic_field_over_time, current_density_over_time, charge_density_over_time = results
    
    # # **Output results**
    # temporary_output = {
    #     # "all_positions":  positions_over_time,
    #     # "all_velocities": velocities_over_time,
    #     "position_electrons": positions_over_time[ :, :number_pseudoelectrons, :],
    #     "velocity_electrons": velocities_over_time[:, :number_pseudoelectrons, :],
    #     "mass_electrons":     parameters["masses"][   :number_pseudoelectrons],
    #     "charge_electrons":   parameters["charges"][  :number_pseudoelectrons],
    #     "position_ions":      positions_over_time[ :, number_pseudoelectrons:, :],
    #     "velocity_ions":      velocities_over_time[:, number_pseudoelectrons:, :],
    #     "mass_ions":          parameters["masses"][   number_pseudoelectrons:],
    #     "charge_ions":        parameters["charges"][  number_pseudoelectrons:],
    #     "electric_field":  electric_field_over_time,
    #     "magnetic_field":  magnetic_field_over_time,
    #     "current_density": current_density_over_time,
    #     "charge_density":  charge_density_over_time,
    #     "number_grid_points":     number_grid_points,
    #     "number_pseudoelectrons": number_pseudoelectrons,
    #     "total_steps": total_steps,
    #     "time_array":  jnp.linspace(0, total_steps * dt, total_steps),
    # }
    
    # output = {**temporary_output, **parameters}

    # diagnostics(output)
    
    # return output

    from jax import debug
    @scan_tqdm(total_steps)
    def simulation_step(carry, step_index):
        (E_field, B_field, positions,
         velocities, qs, ms, q_ms) = carry
        
        # dt = parameters["dt"]
        # n_iterations = 10  # Number of iterations for Picard iteration
        # positions_new=positions
        # velocities_new=velocities
        # E_new=E_field
        # # Start the iteration loop
        # for iteration in range(n_iterations):
        #     x_half = 0.5 * (positions + positions_new)
        #     v_half = 0.5 * (velocities + velocities_new)

        #     J = current_density_CN(x_half, v_half,
        #         qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
        #     E_next = E_field - (dt / epsilon_0) * (J-jnp.sum(J)/len(J))
        #     E_avg = 0.5 * (E_field + E_next)
        #     B_avg = B_field 
        #     E_at_half = vmap(lambda x_n: fields_to_particles_grid(
        #         x_n, E_avg, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(x_half)
        #     B_at_half = vmap(lambda x_n: fields_to_particles_grid(
        #         x_n, B_avg, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right))(x_half)

        #     velocities_new = velocities + (q_ms * E_at_half) * dt
        #     velocities_plus1_2 = 0.5 * (velocities + velocities_new)
        #     positions_new = positions + velocities_plus1_2 * dt

        #     positions_new, velocities_new, qs, ms, q_ms = set_BC_particles(
        #         positions_new, velocities_new, qs, ms, q_ms, dx, grid,
        #         *box_size, particle_BC_left, particle_BC_right)
            
        #     delta_E = jnp.abs(jnp.mean(E_next - E_new))/jnp.max(jnp.abs(E_next))
        #     debug.print("delta_E: {}", delta_E)
        #     E_new=E_next
        #     positions_new=positions_new
        #     velocities_new=velocities_new

        dt = parameters["dt"]
        n_iterations = 1  # Number of iterations for Picard iteration
        E_new=E_field
        B_new=B_field
        positions_new=positions
        
        # Start the iteration loop
        for iteration in range(n_iterations):
            E_avg = 0.5 * (E_field + E_new)
            B_avg = 0.5 * (B_field + B_new)
            
            x_half = 0.5 * (positions_new + positions)
            E_at_half = vmap(lambda x_n: fields_to_particles_grid(
                x_n, E_avg, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(x_half)
            B_at_half = vmap(lambda x_n: fields_to_particles_grid(
                x_n, B_avg, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right))(x_half)

            velocities_new = velocities + (q_ms * E_at_half) * dt
            velocities_plus1_2 = 0.5 * (velocities + velocities_new)
            positions_new = positions + velocities_plus1_2 * dt
            positions_plus1_2 = 0.5 * (positions + positions_new)

            positions_new, velocities_new, qs, ms, q_ms = set_BC_particles(
                positions_new, velocities_new, qs, ms, q_ms, dx, grid,
                *box_size, particle_BC_left, particle_BC_right)
            
            positions_plus1_2, velocities_plus1_2, qs, ms, q_ms = set_BC_particles(
                positions_plus1_2, velocities_plus1_2, qs, ms, q_ms, dx, grid,
                *box_size, particle_BC_left, particle_BC_right)

            
            J = current_density_CN( positions_plus1_2, velocities_plus1_2,
                qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)

            E_next = E_field - (dt / epsilon_0) * (J-jnp.sum(J, axis=0)/len(J))
            delta_E = jnp.abs(jnp.mean(E_next - E_new))/jnp.max(jnp.abs(E_next))
            # debug.print("delta_E: {}", delta_E)
            E_new=E_next
            positions_new=positions_new

        # Update fields after iteration
        E_field = E_new
        B_field = B_field  # Assuming B update is handled similarly if needed

        # Final positions and velocities after convergence
        positions_plus1= positions_new
        velocities_plus1 = velocities_new
        # Charge density with quadratic shape
        charge_density = calculate_charge_density(positions, qs, dx, grid, particle_BC_left, particle_BC_right)
        
        # Prepare next state
        carry = (E_field, B_field, positions_plus1,
                velocities_plus1, qs, ms, q_ms)
        
        # Collect data
        step_data = (positions_plus1, velocities_plus1, E_field, B_field, J, charge_density)
        
        return carry, step_data


    # Run simulation
    _, results = lax.scan(simulation_step, initial_carry, jnp.arange(total_steps))

    # Unpack results
    positions_over_time, velocities_over_time, electric_field_over_time, \
    magnetic_field_over_time, current_density_over_time, charge_density_over_time = results
    
    # **Output results**
    temporary_output = {
        # "all_positions":  positions_over_time,
        # "all_velocities": velocities_over_time,
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

    # @scan_tqdm(total_steps)
    # def simulation_step(carry, step_index):
    #     (E_field, B_field, positions, positions_plus1_2, 
    #     velocities, velocities_plus1_2, qs, ms, q_ms) = carry
    #     dt = parameters["dt"]
    #     n_iterations = 1  # Number of iterations for Picard iteration
        
    #     # Start the iteration loop
    #     for iteration in range(n_iterations):
    #         # First half step
    #         J_plus1_2 = current_density_CN(positions_plus1_2, velocities_plus1_2,
    #                                         qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
    #         E_plus_1 = E_field - (dt / epsilon_0) * (J_plus1_2 - jnp.sum(J_plus1_2) / len(J_plus1_2))
    #         B_plus_1 = B_field
    #         E_avg = 0.5 * (E_field + E_plus_1)
    #         B_avg = 0.5 * (B_field + B_plus_1)
            
    #         E_at_half = vmap(lambda x_n: fields_to_particles_grid(
    #             x_n, E_avg, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(positions_plus1_2)
    #         B_at_half = vmap(lambda x_n: fields_to_particles_grid(
    #             x_n, B_avg, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right))(positions_plus1_2)
            
    #         velocities_plus1 = velocities + (q_ms * E_at_half) * dt
    #         velocities_plus1_2 = 0.5 * (velocities + velocities_plus1)
    #         positions_plus1 = positions + velocities_plus1_2 * dt
    #         positions_plus1_2 = 0.5 * (positions + positions_plus1)

    #         positions_next, velocities_next, qs, ms, q_ms = set_BC_particles(
    #             positions_plus1, velocities_plus1, qs, ms, q_ms, dx, grid,
    #             *box_size, particle_BC_left, particle_BC_right)

    #         # # Second half step
    #         # J = current_density_CN(positions_next, velocities_next,
    #         #                         qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)

    #         # if field_solver != 0:
    #         #     charge_density = calculate_charge_density(positions_plus1_2, qs, dx, grid + dx / 2, particle_BC_left, particle_BC_right)
    #         #     switcher = {
    #         #         1: E_from_Gauss_1D_FFT,
    #         #         2: E_from_Gauss_1D_Cartesian,
    #         #         3: E_from_Poisson_1D_FFT,
    #         #     }
    #         #     E_half = E_field.at[:, 0].set(switcher[field_solver](charge_density, dx))
    #         #     J_half = 0
    #         # else:
    #         #     J_half = J_plus1_2
    #         #     E_half, B_half = field_update(E_field, B_field, dx, dt / 2, J_half, field_BC_left, field_BC_right)

    #         # E_new = E_half - (dt / epsilon_0) * (J - jnp.sum(J) / len(J))
    #         # B_new = B_avg
    #         # E_avg = 0.5 * (E_field + E_new)
    #         # B_avg = 0.5 * (B_field + B_new)

    #         # E_at_half = vmap(lambda x_n: fields_to_particles_grid(
    #         #     x_n, E_avg, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(positions_next)
    #         # B_at_half = vmap(lambda x_n: fields_to_particles_grid(
    #         #     x_n, B_avg, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right))(positions_next)
            
    #         # velocities_new = velocities_plus1_2 + (q_ms * E_at_half) * dt
    #         # positions_new = positions_plus1_2 + velocities_new * dt

    #         # positions_new, velocities_new, qs, ms, q_ms = set_BC_particles(
    #         #     positions_new, velocities_new, qs, ms, q_ms, dx, grid,
    #         #     *box_size, particle_BC_left, particle_BC_right)
    #         #Second half step using boris
    #         positions_plus2, velocities_plus3_2= boris_step(
    #             dt, positions_next, velocities_plus1_2, q_ms, E_at_half, B_at_half)
    #         positions_plus3_2=0.5*(positions_plus1+positions_plus2)

    #         positions_new, velocities_new, qs, ms, q_ms = set_BC_particles(
    #                 positions_plus3_2, velocities_plus3_2, qs, ms, q_ms, dx, grid,
    #                 *box_size, particle_BC_left, particle_BC_right)            

    #         # Charge density with quadratic shape
    #         charge_density = calculate_charge_density(positions_new, qs, dx, grid, particle_BC_left, particle_BC_right)

    #         # Update posistion and velocity with the new values for the next iteration
    #         velocities_plus1_2 = velocities_plus1_2
    #         positions_plus1_2 = positions_plus1_2

    #     # Prepare next state after the iteration loop
    #     carry = (E_plus_1, B_plus_1, positions_next, positions_new,
    #             velocities_next, velocities_new, qs, ms, q_ms)

    #     # Collect data after all iterations
    #     step_data = (positions_next, velocities_next, E_plus_1, B_plus_1, J_plus1_2, charge_density)
        
    #     return carry, step_data
    # # Run simulation
    # _, results = lax.scan(simulation_step, initial_carry, jnp.arange(total_steps))

    # # Unpack results
    # positions_over_time, velocities_over_time, electric_field_over_time, \
    # magnetic_field_over_time, current_density_over_time, charge_density_over_time = results
    
    # # **Output results**
    # temporary_output = {
    #     # "all_positions":  positions_over_time,
    #     # "all_velocities": velocities_over_time,
    #     "position_electrons": positions_over_time[ :, :number_pseudoelectrons, :],
    #     "velocity_electrons": velocities_over_time[:, :number_pseudoelectrons, :],
    #     "mass_electrons":     parameters["masses"][   :number_pseudoelectrons],
    #     "charge_electrons":   parameters["charges"][  :number_pseudoelectrons],
    #     "position_ions":      positions_over_time[ :, number_pseudoelectrons:, :],
    #     "velocity_ions":      velocities_over_time[:, number_pseudoelectrons:, :],
    #     "mass_ions":          parameters["masses"][   number_pseudoelectrons:],
    #     "charge_ions":        parameters["charges"][  number_pseudoelectrons:],
    #     "electric_field":  electric_field_over_time,
    #     "magnetic_field":  magnetic_field_over_time,
    #     "current_density": current_density_over_time,
    #     "charge_density":  charge_density_over_time,
    #     "number_grid_points":     number_grid_points,
    #     "number_pseudoelectrons": number_pseudoelectrons,
    #     "total_steps": total_steps,
    #     "time_array":  jnp.linspace(0, total_steps * dt, total_steps),
    # }
    
    # output = {**temporary_output, **parameters}

    # diagnostics(output)
    
    # return output