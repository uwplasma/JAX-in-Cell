import jax.numpy as jnp
from jax.lax import cond
from functools import partial
from jax_tqdm import scan_tqdm
from jax.debug import print as jprint
from jax import lax, jit, config
from jax.random import PRNGKey, uniform, normal

from ._sources import calculate_charge_density
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._constants import speed_of_light, epsilon_0, elementary_charge, mass_electron, mass_proton
from ._fields import E_from_Gauss_1D_Cartesian
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
    # Interface for additional species and/or particle populations
    try:
        # Nest within main struct to avoid changing top-level internal API
        input_parameters['species'] = parameters['species']
    except:
        input_parameters['species'] = []
    # Convert TOML array -> Python tuple to make hashable static argument, as
    # required by Jax
    try:
        solver_parameters['number_pseudoparticles_species'] = tuple(solver_parameters['number_pseudoparticles_species'])
    except KeyError:
        solver_parameters['number_pseudoparticles_species'] = ()
    G = solver_parameters["number_grid_points"]
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

        # Filtering parameters for current and charge density (digital smoothing)
        "filter_passes": 5,       # number of passes of the digital filter applied to ρ and J
        "filter_alpha": 0.5,      # filter strength (0 < alpha < 1)
        "filter_strides": (1, 2, 4),  # multi-scale strides for filtering

        "weight": 0,
    }

    # Merge user-provided parameters into the default dictionary
    parameters = {**default_parameters, **user_parameters}

    # Compute derived parameters based on user-provided or default values
    for key, value in parameters.items():
        if callable(value):  # If the value is a lambda function, compute it
            parameters[key] = value(parameters)

    return parameters

def initialize_particles_fields(input_parameters={}, number_grid_points=50, number_pseudoelectrons=500,
                                number_pseudoparticles_species=None, total_steps=350
                                ,max_number_of_Picard_iterations_implicit_CN=20, number_of_particle_substeps_implicit_CN=2):
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

    # Introduce additional particle species
    for ii, species in enumerate(parameters['species']):
        plists = make_particles(species_parameters=species,
                                Nprt=number_pseudoparticles_species[ii],
                                box_size=box_size, weight=weight, seed=seed,
                                rng_index=ii)
        positions  = jnp.concatenate((positions,  plists['positions']))
        velocities = jnp.concatenate((velocities, plists['velocities']))
        charges    = jnp.concatenate((charges,    plists['charges']), axis=0)
        masses     = jnp.concatenate((masses,     plists['masses']), axis=0)

    # After done adding all species
    charge_to_mass_ratios = charges / masses

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
    
    charge_density = calculate_charge_density(positions, charges, dx, grid, parameters["particle_BC_left"], parameters["particle_BC_right"],
                                              parameters["filter_passes"], parameters["filter_alpha"], parameters["filter_strides"],
                                              field_BC_left=parameters["field_BC_left"], field_BC_right=parameters["field_BC_right"])
    # Initial E field from charge density via Gauss's law
    E_field_x = E_from_Gauss_1D_Cartesian(charge_density, dx)
    E_field = jnp.stack((E_field_x, jnp.zeros_like(grid), jnp.zeros_like(grid)), axis=1)
    fields = (E_field, B_field)
    # --- External fields (arrays if provided at top-level; else zeros)
    G = number_grid_points

    secB = input_parameters.get("external_magnetic_field")
    secE = input_parameters.get("external_electric_field")
    # print(secB["B"])
    if isinstance(secB, dict) and "B" in secB:
        parameters["external_magnetic_field"] = jnp.asarray(secB["B"], dtype=jnp.float32)
    else:
        parameters["external_magnetic_field"] = jnp.zeros((G, 3), dtype=jnp.float32)

    if isinstance(secE, dict) and "E" in secE:
        parameters["external_electric_field"] = jnp.asarray(secE["E"], dtype=jnp.float32)
    else:
        parameters["external_electric_field"] = jnp.zeros((G, 3), dtype=jnp.float32)

    # external_E_field_x = parameters["external_electric_field_amplitude"] * jnp.cos(parameters["external_electric_field_wavenumber"] * jnp.linspace(-jnp.pi, jnp.pi, number_grid_points))
    # external_B_field_x = parameters["external_magnetic_field_amplitude"] * jnp.cos(parameters["external_magnetic_field_wavenumber"] * jnp.linspace(-jnp.pi, jnp.pi, number_grid_points))

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
        "external_electric_field": parameters["external_electric_field"],
        # "external_magnetic_field": jnp.array([external_B_field_x, jnp.zeros((number_grid_points,)), jnp.zeros((number_grid_points,))]).T,
        "external_magnetic_field": parameters["external_magnetic_field"],
        "number_pseudoelectrons": number_pseudoelectrons,
        "number_grid_points": number_grid_points,
        "plasma_frequency": plasma_frequency,
        "max_initial_vth_electrons": vth_electrons,
        "max_number_of_Picard_iterations_implicit_CN": max_number_of_Picard_iterations_implicit_CN,
        "number_of_particle_substeps_implicit_CN": number_of_particle_substeps_implicit_CN,
    })

    return parameters

def make_particles(species_parameters, Nprt, box_size, weight, seed, rng_index):
    """
    Generate Nprt total particles of a user-requested species with specified
    charge, mass, and space/velocity distribution.

    Parameters:
    ----------
    species_parameters : dict
        Dictionary of user-specified species parameters.
    Nprt : int
        Total number of pseudoparticles in the domain
    box_size : tuple
        Domain size in x,y,z
    weight : float
        Top-level pseudoelectron weight
    seed : int
        Top-level random number generator seed used for entire simulation
    rng_index : int
        Species or particle population index in [0,1,2,3,...]
        Use a unique index value for each population.
        This index is used to advance the random seed and so avoid spurious
        correlation between different particle positions and velocities.
        See https://docs.jax.dev/en/latest/random-numbers.html

    Returns:
    -------
    plist : dict
        Dictionary with lists of positions, velocities, charges, masses.
    """
    _p = species_parameters
    charge = _p["charge_over_elementary_charge"] * elementary_charge
    mass   = _p["mass_over_proton_mass"] * mass_proton
    vth_x  = _p["vth_over_c_x"] * speed_of_light
    vth_y  = _p["vth_over_c_y"] * speed_of_light
    vth_z  = _p["vth_over_c_z"] * speed_of_light

    # This code is brittle; it depends on hard-coded offsets to the RNG seed
    # within initialize_particles_fields(...)
    assert rng_index >= 0
    local_seed = seed+12 + rng_index*6
    # Separate position/velocity seeds allow different ion and electron
    # populations to be inited with identical space positions, but
    # uncorrelated velocity distributions
    seed_pos = lax.cond(_p['seed_position_override'],
                        lambda _: _p['seed_position'],
                        lambda _: local_seed, operand=None)
    seed_vel = local_seed

    out = dict()

    # **Particle Positions**

    xs = lax.cond(_p["random_positions_x"],
        lambda _: uniform(PRNGKey(seed_pos+1), shape=(Nprt,),
                          minval=-box_size[0] / 2, maxval=box_size[0] / 2),
        lambda _: jnp.linspace(-box_size[0] / 2, box_size[0] / 2, Nprt), operand=None)
    wavenumber_perturbation_x = _p["wavenumber_perturbation_x"] * 2 * jnp.pi / box_size[0]
    xs += _p["amplitude_perturbation_x"] * jnp.sin(wavenumber_perturbation_x * xs)

    ys = lax.cond(_p["random_positions_y"],
        lambda _: uniform(PRNGKey(seed_pos+2), shape=(Nprt,),
                          minval=-box_size[1] / 2, maxval=box_size[1] / 2),
        lambda _: jnp.linspace(-box_size[1] / 2, box_size[1] / 2, Nprt), operand=None)
    wavenumber_perturbation_y = _p["wavenumber_perturbation_y"] * 2 * jnp.pi / box_size[1]
    ys += _p["amplitude_perturbation_y"] * jnp.sin(wavenumber_perturbation_y * ys)

    zs = lax.cond(_p["random_positions_z"],
        lambda _: uniform(PRNGKey(seed_pos+3), shape=(Nprt,),
                          minval=-box_size[2] / 2, maxval=box_size[2] / 2),
        lambda _: jnp.linspace(-box_size[2] / 2, box_size[2] / 2, Nprt), operand=None)
    wavenumber_perturbation_z = _p["wavenumber_perturbation_z"] * 2 * jnp.pi / box_size[2]
    zs += _p["amplitude_perturbation_z"] * jnp.sin(wavenumber_perturbation_z * zs)

    out['positions'] = jnp.stack((xs, ys, zs), axis=1)

    # **Particle Charges and Masses**

    out['charges'] = charge * weight * _p['weight_ratio'] * jnp.ones((Nprt, 1))
    out['masses']  = mass   * weight * _p['weight_ratio'] * jnp.ones((Nprt, 1))

    # **Particle Velocities**

    v_x = vth_x/jnp.sqrt(2) * normal(PRNGKey(seed_vel+4), shape=(Nprt,))
    v_y = vth_y/jnp.sqrt(2) * normal(PRNGKey(seed_vel+5), shape=(Nprt,))
    v_z = vth_z/jnp.sqrt(2) * normal(PRNGKey(seed_vel+6), shape=(Nprt,))
    v_x += _p["drift_speed_x"]
    v_y += _p["drift_speed_y"]
    v_z += _p["drift_speed_z"]

    out['velocities'] = jnp.stack((v_x, v_y, v_z), axis=1)

    return out

@partial(jit, static_argnames=['number_grid_points', 'number_pseudoelectrons', 'number_pseudoparticles_species', 'total_steps', 'field_solver', "time_evolution_algorithm",
                               "max_number_of_Picard_iterations_implicit_CN","number_of_particle_substeps_implicit_CN"])
def simulation(input_parameters={}, number_grid_points=100, number_pseudoelectrons=3000,
               number_pseudoparticles_species=None, total_steps=1000,
               field_solver=0,positions=None, velocities=None,time_evolution_algorithm=0,max_number_of_Picard_iterations_implicit_CN=20, number_of_particle_substeps_implicit_CN=2):
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

    # For simulation(...) parameters specified via Python, not parsed TOML file
    if 'species' not in input_parameters:
        input_parameters['species'] = []
    if not number_pseudoparticles_species:
        number_pseudoparticles_species = ()
    else:
        number_pseudoparticles_species = tuple(number_pseudoparticles_species)
    assert len(number_pseudoparticles_species) == len(input_parameters['species'])

    # **Initialize simulation parameters**
    parameters = initialize_particles_fields(input_parameters, number_grid_points=number_grid_points,
                                             number_pseudoelectrons=number_pseudoelectrons,
                                             number_pseudoparticles_species=number_pseudoparticles_species,
                                             total_steps=total_steps,
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
        ## segregate ions/electrons in non-jitted method outside simulation(...)
        ## so we can make use of dynamically constructed arrays
        #"position_electrons": positions_over_time[ :, :number_pseudoelectrons, :],
        #"velocity_electrons": velocities_over_time[:, :number_pseudoelectrons, :],
        #"mass_electrons":     parameters["masses"][   :number_pseudoelectrons],
        #"charge_electrons":   parameters["charges"][  :number_pseudoelectrons],
        #"position_ions":      positions_over_time[ :, number_pseudoelectrons:, :],
        #"velocity_ions":      velocities_over_time[:, number_pseudoelectrons:, :],
        #"mass_ions":          parameters["masses"][   number_pseudoelectrons:],
        #"charge_ions":        parameters["charges"][  number_pseudoelectrons:],
        "positions": positions_over_time,
        "velocities": velocities_over_time,
        "masses": parameters["masses"],
        "charges": parameters["charges"],
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

    # diagnostics(output)

    return output