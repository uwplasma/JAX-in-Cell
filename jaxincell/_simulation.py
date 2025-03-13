import jax.numpy as jnp
from jax.lax import cond
from functools import partial
from jax.random import normal
from jax import lax, jit, vmap, config
from jax.debug import print as jprint
from jax.random import PRNGKey, uniform
from ._particles import fields_to_particles_grid, boris_step
from ._sources import current_density, calculate_charge_density
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._fields import field_update, E_from_Gauss_1D_Cartesian, E_from_Gauss_1D_FFT, E_from_Poisson_1D_FFT
from ._constants import speed_of_light, epsilon_0, elementary_charge, mass_electron, mass_proton
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
        "amplitude_perturbation_x": 1e-7,         # Amplitude of sinusoidal (sin) perturbation in x
        "wavenumber_electrons": 8,    # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
        "wavenumber_ions": 0,         # Wavenumber of sinusoidal ion density perturbation in x (factor of 2pi/length)
        "grid_points_per_Debye_length": 2,        # dx over Debye length
        "vth_electrons_over_c": 0.05,             # Thermal velocity of electrons over speed of light
        "ion_temperature_over_electron_temperature": 0.01, # Temperature ratio of ions to electrons
        "timestep_over_spatialstep_times_c": 1.0, # dt * speed_of_light / dx
        "seed": 1701,                             # Random seed for reproducibility
        "electron_drift_speed": 100000000.0,                # Drift speed of electrons
        "ion_drift_speed":      0,                # Drift speed of ions
        "velocity_plus_minus_electrons": True,    # create two groups of electrons moving in opposite directions
        "velocity_plus_minus_ions": False,        # create two groups of electrons moving in opposite directions
        "print_info": True,                       # Print information about the simulation
        "electron_charge_over_elementary_charge": -1, # Electron charge in units of the elementary charge
        "ion_charge_over_elementary_charge": 1,   # Ion charge in units of the elementary charge
        "ion_mass_over_proton_mass": 1,           # Ion mass in units of the proton mass

        # Boundary conditions
        "particle_BC_left":  0,                   # Left boundary condition for particles
        "particle_BC_right": 0,                   # Right boundary condition for particles
        "field_BC_left":     0,                   # Left boundary condition for fields
        "field_BC_right":    0,                   # Right boundary condition for fields
        
        # External fields (initialized to zero)
        "external_electric_field_amplitude": 0,   # Amplitude of sinusoidal (cos) perturbation in x
        "external_electric_field_wavenumber": 0,  # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
        "external_magnetic_field_amplitude": 0,   # Amplitude of sinusoidal (cos) perturbation in x
        "external_magnetic_field_wavenumber": 0,  # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
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

    # **Particle Charges and Masses**
    charge_electrons = parameters["electron_charge_over_elementary_charge"] * elementary_charge
    charge_ions      = parameters["ion_charge_over_elementary_charge"]      * elementary_charge
    mass_electrons   = mass_electron
    mass_ions        = parameters["ion_mass_over_proton_mass"] * mass_proton

    # Pseudoparticle weights -> density of real particles = number_pseudoelectrons * weight / length, put in terms of Debye length
    Debye_length_per_dx = 1 / parameters["grid_points_per_Debye_length"]
    weight = (
        epsilon_0
        * mass_electrons
        * speed_of_light**2
        / charge_electrons**2
        * number_grid_points**2
        / length
        / (2 * number_pseudoelectrons)
        * parameters["vth_electrons_over_c"]**2
        / Debye_length_per_dx**2
    )
    charges = jnp.concatenate((
        charge_electrons * weight * jnp.ones((number_pseudoelectrons, 1)),
        charge_ions   * weight * jnp.ones((number_pseudoelectrons, 1))
    ), axis=0)
    masses = jnp.concatenate((
        mass_electrons * weight * jnp.ones((number_pseudoelectrons, 1)),
        mass_ions      * weight * jnp.ones((number_pseudoelectrons, 1))
    ), axis=0)
    charge_to_mass_ratios = charges / masses

    # **Particle Velocities**
    # Thermal velocities (Maxwell-Boltzmann distribution)
    vth_electrons = parameters["vth_electrons_over_c"] * speed_of_light
    v_electrons_x = vth_electrons / jnp.sqrt(2) * normal(random_key, shape=(number_pseudoelectrons, )) + parameters["electron_drift_speed"]
    v_electrons_x = jnp.where(parameters["velocity_plus_minus_electrons"], v_electrons_x * (-1) ** jnp.arange(0, number_pseudoelectrons), v_electrons_x)
    v_electrons_y = jnp.zeros((number_pseudoelectrons, ))
    v_electrons_z = jnp.zeros((number_pseudoelectrons, ))
    electron_velocities = jnp.stack((v_electrons_x, v_electrons_y, v_electrons_z), axis=1)
    vth_ions = jnp.sqrt(jnp.abs(parameters["ion_temperature_over_electron_temperature"])) * vth_electrons * jnp.sqrt(jnp.abs(mass_electrons / mass_ions))
    v_ions_x = vth_ions / jnp.sqrt(2) * normal(random_key, shape=(number_pseudoelectrons, )) + parameters["ion_drift_speed"]
    v_ions_x = jnp.where(parameters["velocity_plus_minus_ions"], v_ions_x * (-1) ** jnp.arange(0, number_pseudoelectrons), v_ions_x)
    v_ions_y = jnp.zeros((number_pseudoelectrons, ))
    v_ions_z = jnp.zeros((number_pseudoelectrons, ))
    ion_velocities = jnp.stack((v_ions_x, v_ions_y, v_ions_z), axis=1)
    
    velocities = jnp.concatenate((electron_velocities, ion_velocities))

    # Grid setup
    dx = length / number_grid_points
    grid = jnp.linspace(-length / 2 + dx / 2, length / 2 - dx / 2, number_grid_points)
    dt = parameters["timestep_over_spatialstep_times_c"] * dx / speed_of_light

    # Print information about the simulation
    plasma_frequency = jnp.sqrt(number_pseudoelectrons * weight * charge_electrons**2)/jnp.sqrt(mass_electrons)/jnp.sqrt(epsilon_0)/jnp.sqrt(length)

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
          -parameters["ion_temperature_over_electron_temperature"] * vth_electrons**2 * mass_electrons / 2 / charge_electrons,
          -mass_electrons * vth_electrons**2 / 2 / charge_electrons,
          Debye_length_per_dx*dx,
          wavenumber_perturbation_x_electrons*Debye_length_per_dx*dx,
          number_pseudoelectrons / number_grid_points,
          1/(plasma_frequency * dt),
          dt * plasma_frequency * total_steps,
          number_pseudoelectrons * weight / length * (Debye_length_per_dx*dx)**3,
          -charge_electrons * parameters["external_electric_field_amplitude"] * Debye_length_per_dx*dx / (mass_electrons * vth_electrons**2 / 2),
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

    initial_carry = (
        E_field, B_field, positions_minus1_2, positions,
        positions_plus1_2, velocities, qs, ms, q_ms,)

    @scan_tqdm(total_steps)
    def simulation_step(carry, step_index):
        (E_field, B_field, positions_minus1_2, positions,
         positions_plus1_2, velocities, qs, ms, q_ms) = carry
        
        # Add external fields
        total_E = E_field + parameters["external_electric_field"]
        total_B = B_field + parameters["external_magnetic_field"]

        # Interpolate fields to particle positions
        E_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
            x_n, total_E, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(positions_plus1_2)
        
        B_field_at_x = vmap(lambda x_n: fields_to_particles_grid(
            x_n, total_B, dx, grid, grid[0] - dx / 2, field_BC_left, field_BC_right))(positions_plus1_2)

        # Particle update: Boris pusher
        positions_plus3_2, velocities_plus1 = boris_step(
            dt, positions_plus1_2, velocities, q_ms, E_field_at_x, B_field_at_x)

        # Apply boundary conditions
        positions_plus3_2, velocities_plus1, qs, ms, q_ms = set_BC_particles(
            positions_plus3_2, velocities_plus1, qs, ms, q_ms, dx, grid,
            *box_size, particle_BC_left, particle_BC_right)
        
        positions_plus1 = set_BC_positions(positions_plus3_2 - (dt / 2) * velocities_plus1,
                                           qs, dx, grid, *box_size, particle_BC_left, particle_BC_right)

        if field_solver != 0:
            charge_density = calculate_charge_density(positions, qs, dx, grid + dx / 2, particle_BC_left, particle_BC_right)
            switcher = {
                1: E_from_Gauss_1D_FFT,
                2: E_from_Gauss_1D_Cartesian,
                3: E_from_Poisson_1D_FFT,
            }
            E_field = E_field.at[:,0].set(switcher[field_solver](charge_density, dx))
            J = 0
        else:
            J = current_density(positions_plus1_2, positions_plus1, positions_plus3_2, velocities_plus1,
                                qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
            E_field, B_field = field_update(E_field, B_field, dx, dt, J, field_BC_left, field_BC_right)

        # Update positions and velocities
        positions_minus1_2, positions_plus1_2 = positions_plus1_2, positions_plus3_2
        velocities = velocities_plus1
        positions = positions_plus1

        # Prepare state for the next step
        carry = (E_field, B_field, positions_minus1_2, positions,
                 positions_plus1_2, velocities, qs, ms, q_ms)

        # Collect data for storage
        charge_density = calculate_charge_density(positions, qs, dx, grid, particle_BC_left, particle_BC_right)
        step_data = (positions, velocities, E_field, B_field, J, charge_density)
        
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