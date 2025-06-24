import jax.numpy as jnp

from jax.lax import cond
from functools import partial
from jax_tqdm import scan_tqdm
from jax.debug import print as jprint
from jax import lax, jit, vmap, config
from jax.random import PRNGKey, uniform, normal

from ._diagnostics import diagnostics
from ._sources import current_density, calculate_charge_density
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._particles import fields_to_particles_grid, boris_step, boris_step_relativistic
from ._constants import speed_of_light, epsilon_0, elementary_charge, mass_electron, mass_proton
from ._fields import (field_update, E_from_Gauss_1D_Cartesian, E_from_Gauss_1D_FFT,
                      E_from_Poisson_1D_FFT, field_update1, field_update2)
from ._algorithms import Boris_step, CN_step

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
        "random_positions_x": False,  # Use random positions in x for particles
        "random_positions_y": True,  # Use random positions in y for particles
        "random_positions_z": True,  # Use random positions in z for particles
        "amplitude_perturbation_x": 1e-7,         # Amplitude of sinusoidal (sin) perturbation in x
        "amplitude_perturbation_y": 0,            # Amplitude of sinusoidal (sin) perturbation in y
        "amplitude_perturbation_z": 0,            # Amplitude of sinusoidal (sin) perturbation in z
        "wavenumber_electrons_x": 8,    # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
        "wavenumber_electrons_y": 0,    # Wavenumber of sinusoidal electron density perturbation in y (factor of 2pi/length)
        "wavenumber_electrons_z": 0,    # Wavenumber of sinusoidal electron density perturbation in z (factor of 2pi/length)
        "wavenumber_ions_x": 0,         # Wavenumber of sinusoidal ion density perturbation in x (factor of 2pi/length)
        "wavenumber_ions_y": 0,         # Wavenumber of sinusoidal ion density perturbation in y (factor of 2pi/length)
        "wavenumber_ions_z": 0,         # Wavenumber of sinusoidal ion density perturbation in z (factor of 2pi/length)
        "grid_points_per_Debye_length": 2,        # dx over Debye length
        "vth_electrons_over_c_x": 0.05,           # Thermal velocity of electrons over speed of light in the x direction
        "vth_electrons_over_c_y": 0.0,            # Thermal velocity of electrons over speed of light in the y direction
        "vth_electrons_over_c_z": 0.0,            # Thermal velocity of electrons over speed of light in the z direction
        "ion_temperature_over_electron_temperature_x": 1, # Temperature ratio of ions to electrons in the x direction
        "ion_temperature_over_electron_temperature_y": 1, # Temperature ratio of ions to electrons in the y direction
        "ion_temperature_over_electron_temperature_z": 1, # Temperature ratio of ions to electrons in the z direction
        "timestep_over_spatialstep_times_c": 1.0,   # dt * speed_of_light / dx
        "seed": 1701,                               # Random seed for reproducibility
        "electron_drift_speed_x": 1e8,              # Drift speed of electrons in the x direction
        "electron_drift_speed_y": 0,              # Drift speed of electrons in the y direction
        "electron_drift_speed_z": 0,              # Drift speed of electrons in the z direction
        "ion_drift_speed_x":      0,                # Drift speed of ions in the x direction
        "ion_drift_speed_y":      0,                # Drift speed of ions in the y direction
        "ion_drift_speed_z":      0,                # Drift speed of ions in the z direction
        "velocity_plus_minus_electrons_x": True,    # create two groups of electrons moving in opposite directions in the x direction
        "velocity_plus_minus_electrons_y": False,   # create two groups of electrons moving in opposite directions in the y direction
        "velocity_plus_minus_electrons_z": False,   # create two groups of electrons moving in opposite directions in the z direction
        "velocity_plus_minus_ions_x": False,        # create two groups of electrons moving in opposite directions in the x direction
        "velocity_plus_minus_ions_y": False,        # create two groups of electrons moving in opposite directions in the y direction
        "velocity_plus_minus_ions_z": False,        # create two groups of electrons moving in opposite directions in the z direction
        "print_info": True,                       # Print information about the simulation
        "electron_charge_over_elementary_charge": -1, # Electron charge in units of the elementary charge
        "ion_charge_over_elementary_charge": 1,   # Ion charge in units of the elementary charge
        "ion_mass_over_proton_mass": 1,           # Ion mass in units of the proton mass
        "relativistic": False,                    # Use relativistic Boris pusher
        "tolerance_Picard_iterations_implicit_CN": 1e-6, # Tolerance for Picard iterations in implicit Crank-Nicholson method

        # Boundary conditions
        "particle_BC_left":  0,                   # Left boundary condition for particles
        "particle_BC_right": 0,                   # Right boundary condition for particles
        "field_BC_left":     0,                   # Left boundary condition for fields
        "field_BC_right":    0,                   # Right boundary condition for fields
        
        # External fields (initialized to zero)
        "external_electric_field_amplitude":  0,   # Amplitude of sinusoidal (cos) perturbation in x
        "external_electric_field_wavenumber": 0,  # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
        "external_magnetic_field_amplitude":  0,   # Amplitude of sinusoidal (cos) perturbation in x
        "external_magnetic_field_wavenumber": 0,  # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
        
        "weight": 0,
    }

    # Merge user-provided parameters into the default dictionary
    parameters = {**default_parameters, **user_parameters}
    
    # Compute derived parameters based on user-provided or default values
    for key, value in parameters.items():
        if callable(value):  # If the value is a lambda function, compute it
            parameters[key] = value(parameters)

    return parameters

def initialize_particles_fields(input_parameters={}, number_grid_points=50, number_pseudoelectrons=500, total_steps=350
                                ,max_number_of_Picard_iterations_implicit_CN=7, number_of_particle_substeps_implicit_CN=2):
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
    seed = parameters["seed"]
    
    # **Particle Positions**
    # Use random positions in x if requested, otherwise linspace
    electron_xs = lax.cond(parameters["random_positions_x"],
        lambda _: uniform(PRNGKey(seed+1), shape=(number_pseudoelectrons,), minval=-box_size[0] / 2, maxval=box_size[0] / 2),
        lambda _: jnp.linspace(-length / 2, length / 2, number_pseudoelectrons), operand=None)
    wavenumber_perturbation_x_electrons = parameters["wavenumber_electrons_x"] * 2 * jnp.pi / length
    electron_xs+= parameters["amplitude_perturbation_x"] * jnp.sin(wavenumber_perturbation_x_electrons * electron_xs)
    electron_ys = lax.cond(parameters["random_positions_y"],
        lambda _: uniform(PRNGKey(seed+2), shape=(number_pseudoelectrons,), minval=-box_size[1] / 2, maxval=box_size[1] / 2),
        lambda _: jnp.linspace(-length / 2, length / 2, number_pseudoelectrons), operand=None)
    wavenumber_perturbation_y_electrons = parameters["wavenumber_electrons_y"] * 2 * jnp.pi / length
    electron_ys+= parameters["amplitude_perturbation_y"] * jnp.sin(wavenumber_perturbation_y_electrons * electron_ys)
    electron_zs = lax.cond(parameters["random_positions_z"],
        lambda _: uniform(PRNGKey(seed+3), shape=(number_pseudoelectrons,), minval=-box_size[2] / 2, maxval=box_size[2] / 2),
        lambda _: jnp.linspace(-length / 2, length / 2, number_pseudoelectrons), operand=None)
    wavenumber_perturbation_z_electrons = parameters["wavenumber_electrons_z"] * 2 * jnp.pi / length
    electron_zs+= parameters["amplitude_perturbation_z"] * jnp.sin(wavenumber_perturbation_z_electrons * electron_zs)
    electron_positions = jnp.stack((electron_xs, electron_ys, electron_zs), axis=1)

    # Ion positions: Add random y, z positions to uniform grid x positions
    ion_xs = lax.cond(parameters["random_positions_x"],
        lambda _: uniform(PRNGKey(seed+1), shape=(number_pseudoelectrons,), minval=-box_size[0] / 2, maxval=box_size[0] / 2),
        lambda _: jnp.linspace(-length / 2, length / 2, number_pseudoelectrons), operand=None)
    wavenumber_perturbation_x_ions = parameters["wavenumber_ions_x"] * 2 * jnp.pi / length
    ion_xs+= parameters["amplitude_perturbation_x"] * jnp.sin(wavenumber_perturbation_x_ions * ion_xs)
    ion_ys = lax.cond(parameters["random_positions_y"],
        lambda _: uniform(PRNGKey(seed+2), shape=(number_pseudoelectrons,), minval=-box_size[1] / 2, maxval=box_size[1] / 2),
        lambda _: jnp.linspace(-length / 2, length / 2, number_pseudoelectrons), operand=None)
    wavenumber_perturbation_y_ions = parameters["wavenumber_ions_y"] * 2 * jnp.pi / length
    ion_ys+= parameters["amplitude_perturbation_y"] * jnp.sin(wavenumber_perturbation_y_ions * ion_ys)
    ion_zs = lax.cond(parameters["random_positions_z"],
        lambda _: uniform(PRNGKey(seed+3), shape=(number_pseudoelectrons,), minval=-box_size[2] / 2, maxval=box_size[2] / 2),
        lambda _: jnp.linspace(-length / 2, length / 2, number_pseudoelectrons), operand=None)
    wavenumber_perturbation_z_ions = parameters["wavenumber_ions_z"] * 2 * jnp.pi / length
    ion_zs+= parameters["amplitude_perturbation_z"] * jnp.sin(wavenumber_perturbation_z_ions * ion_zs)
    ion_positions = jnp.stack((ion_xs, ion_ys, ion_zs), axis=1)

    positions = jnp.concatenate((electron_positions, ion_positions))

    # **Particle Charges and Masses**
    charge_electrons = parameters["electron_charge_over_elementary_charge"] * elementary_charge
    charge_ions      = parameters["ion_charge_over_elementary_charge"]      * elementary_charge
    mass_electrons   = mass_electron
    mass_ions        = parameters["ion_mass_over_proton_mass"] * mass_proton

    # Pseudoparticle weights -> density of real particles = number_pseudoelectrons * weight / length, put in terms of Debye length
    vth_electrons_over_c = jnp.max(jnp.array([
        parameters["vth_electrons_over_c_x"],
        parameters["vth_electrons_over_c_y"],
        parameters["vth_electrons_over_c_z"]])
    )
    vth_electrons = vth_electrons_over_c * speed_of_light

    Debye_length_per_dx = 1 / parameters["grid_points_per_Debye_length"]
    weight = (
        epsilon_0
        * mass_electrons
        * speed_of_light**2
        / charge_electrons**2
        * number_grid_points**2
        / length
        / (2 * number_pseudoelectrons)
        * vth_electrons_over_c**2
        / Debye_length_per_dx**2
    )
    weight = jnp.where(parameters["weight"]==0, weight, parameters["weight"])
    Debye_length_per_dx = jnp.where(vth_electrons_over_c==0, 0, 1 / (jnp.sqrt(
                            weight
                            / epsilon_0
                            / mass_electrons
                            * length
                            * (2 * number_pseudoelectrons))
                            / speed_of_light
                            * (-charge_electrons)
                            / number_grid_points
                            / (vth_electrons_over_c)
    ))
    
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
    # Electron thermal velocities and drift speeds
    v_electrons_x = parameters["vth_electrons_over_c_x"] * speed_of_light / jnp.sqrt(2) * normal(PRNGKey(seed+7), shape=(number_pseudoelectrons, )) + parameters["electron_drift_speed_x"]
    v_electrons_x = jnp.where(parameters["velocity_plus_minus_electrons_x"], v_electrons_x * (-1) ** jnp.arange(0, number_pseudoelectrons), v_electrons_x)
    v_electrons_y = parameters["vth_electrons_over_c_y"] * speed_of_light / jnp.sqrt(2) * normal(PRNGKey(seed+8), shape=(number_pseudoelectrons, )) + parameters["electron_drift_speed_y"]
    v_electrons_y = jnp.where(parameters["velocity_plus_minus_electrons_y"], v_electrons_y * (-1) ** jnp.arange(0, number_pseudoelectrons), v_electrons_y)
    v_electrons_z = parameters["vth_electrons_over_c_z"] * speed_of_light / jnp.sqrt(2) * normal(PRNGKey(seed+9), shape=(number_pseudoelectrons, )) + parameters["electron_drift_speed_z"]
    v_electrons_z = jnp.where(parameters["velocity_plus_minus_electrons_z"], v_electrons_z * (-1) ** jnp.arange(0, number_pseudoelectrons), v_electrons_z)
    electron_velocities = jnp.stack((v_electrons_x, v_electrons_y, v_electrons_z), axis=1)
    
    # Ion thermal velocities and drift speeds
    vth_ions_x = jnp.sqrt(jnp.abs(parameters["ion_temperature_over_electron_temperature_x"])) * parameters["vth_electrons_over_c_x"] * speed_of_light * jnp.sqrt(jnp.abs(mass_electrons / mass_ions))
    vth_ions_y = jnp.sqrt(jnp.abs(parameters["ion_temperature_over_electron_temperature_y"])) * parameters["vth_electrons_over_c_y"] * speed_of_light * jnp.sqrt(jnp.abs(mass_electrons / mass_ions))
    vth_ions_z = jnp.sqrt(jnp.abs(parameters["ion_temperature_over_electron_temperature_z"])) * parameters["vth_electrons_over_c_z"] * speed_of_light * jnp.sqrt(jnp.abs(mass_electrons / mass_ions))
    v_ions_x = vth_ions_x / jnp.sqrt(2) * normal(PRNGKey(seed+10), shape=(number_pseudoelectrons, )) + parameters["ion_drift_speed_x"]
    v_ions_x = jnp.where(parameters["velocity_plus_minus_ions_x"], v_ions_x * (-1) ** jnp.arange(0, number_pseudoelectrons), v_ions_x)
    v_ions_y = vth_ions_y / jnp.sqrt(2) * normal(PRNGKey(seed+11), shape=(number_pseudoelectrons, )) + parameters["ion_drift_speed_y"]
    v_ions_y = jnp.where(parameters["velocity_plus_minus_ions_y"], v_ions_y * (-1) ** jnp.arange(0, number_pseudoelectrons), v_ions_y)
    v_ions_z = vth_ions_z / jnp.sqrt(2) * normal(PRNGKey(seed+12), shape=(number_pseudoelectrons, )) + parameters["ion_drift_speed_z"]
    v_ions_z = jnp.where(parameters["velocity_plus_minus_ions_z"], v_ions_z * (-1) ** jnp.arange(0, number_pseudoelectrons), v_ions_z)
    ion_velocities = jnp.stack((v_ions_x, v_ions_y, v_ions_z), axis=1)
    
    # Combine electron and ion velocities
    velocities = jnp.concatenate((electron_velocities, ion_velocities))
    # Cap velocities at 99% the speed of light
    speed_limit = 0.99 * speed_of_light
    velocities = jnp.where(jnp.abs(velocities) >= speed_limit, jnp.sign(velocities) * speed_limit, velocities)

    # Grid setup
    dx = length / number_grid_points
    grid = jnp.linspace(-length / 2 + dx / 2, length / 2 - dx / 2, number_grid_points)
    dt = parameters["timestep_over_spatialstep_times_c"] * dx / speed_of_light

    # Print information about the simulation
    plasma_frequency = jnp.sqrt(number_pseudoelectrons * weight * charge_electrons**2)/jnp.sqrt(mass_electrons)/jnp.sqrt(epsilon_0)/jnp.sqrt(length)
    relativistic_gamma_factor = 1 / jnp.sqrt(1 - jnp.sum(velocities**2, axis=1) / speed_of_light**2)

    cond(parameters["print_info"],
        lambda _: jprint((
            # f"Number of pseudoelectrons: {number_pseudoelectrons}\n"
            # f"Number of grid points: {number_grid_points}\n"
            "Length of the simulation box: {} Debye lengths or {} Skin Depths\n"
            "Density of electrons: {} m^-3\n"
            "Electron temperature: {} eV\n"
            "Ion temperature / Electron temperature: {}\n"
            "Debye length: {} m\n"
            "Skin depth: {} m\n"
            "Wavenumber * Debye length: {}\n" 
            "Pseudoparticles per cell: {}\n"
            "Pseudoparticle weight: {}\n"
            "Steps at each plasma frequency: {}\n"
            "Total time: {} / plasma frequency\n"
            "Number of particles on a Debye cube: {}\n"
            "Relativistic gamma factor: Maximum {}, Average {}\n"
            "Charge x External electric field x Debye Length / Temperature: {}\n"
        ),length/(Debye_length_per_dx*dx),
          length/(speed_of_light/plasma_frequency),
          number_pseudoelectrons * weight / length,
          mass_electrons * vth_electrons**2 / 2 / (-charge_electrons),
          parameters["ion_temperature_over_electron_temperature_x"],
          Debye_length_per_dx*dx,
          speed_of_light/plasma_frequency,
          wavenumber_perturbation_x_electrons*Debye_length_per_dx*dx,
          number_pseudoelectrons / number_grid_points,
          weight,
          1/(plasma_frequency * dt),
          dt * plasma_frequency * total_steps,
          number_pseudoelectrons * weight / length * (Debye_length_per_dx*dx)**3,
          jnp.max(relativistic_gamma_factor), jnp.mean(relativistic_gamma_factor),
          -charge_electrons * parameters["external_electric_field_amplitude"] * Debye_length_per_dx*dx / (mass_electrons * vth_electrons**2 / 2),
        ), lambda _: None, operand=None)
    
    # **Fields Initialization**
    B_field = jnp.zeros((grid.size, 3))
    E_field = jnp.zeros((grid.size, 3))
    
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
        "max_initial_vth_electrons": vth_electrons,
        "max_number_of_Picard_iterations_implicit_CN": max_number_of_Picard_iterations_implicit_CN, 
        "number_of_particle_substeps_implicit_CN": number_of_particle_substeps_implicit_CN,
    })
    
    return parameters


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
        initial_carry = (
            E_field, B_field, positions_minus1_2, positions,
            positions_plus1_2, velocities, qs, ms, q_ms,
        )
        step_func = lambda carry, step_index: Boris_step(
            carry, step_index, parameters, dx, dt, grid, box_size,
            particle_BC_left, particle_BC_right, field_BC_left, field_BC_right, field_solver
        )
    else:
        initial_carry = (
            E_field, B_field, positions,
            velocities, qs, ms, q_ms,
        )
        step_func = lambda carry, step_index: CN_step(
            carry, step_index, parameters, dx, dt, grid, box_size,
            particle_BC_left, particle_BC_right, field_BC_left, field_BC_right,
            parameters["number_of_particle_substeps_implicit_CN"]
        )

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