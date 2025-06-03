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
import sys
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
    jprint("Weight of pseudoparticles: {}", weight)
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
    dt = parameters["timestep_over_spatialstep_times_c"] * dx *5 / speed_of_light

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
    initial_charge_density = calculate_charge_density(positions, qs, dx, grid, particle_BC_left, particle_BC_right)
    initial_carry = (
     E_field, B_field, positions,
     velocities, qs, ms, q_ms,)
    # Add external fields
    # total_E = E_field + parameters["external_electric_field"]
    # total_B = B_field + parameters["external_magnetic_field"]


    def integrate(y, dx): return 0.5 * (jnp.asarray(dx) * (y[..., 1:] + y[..., :-1])).sum(-1)
    
    def compute_divergence(E, dx):
        div_E = jnp.zeros_like(E)  # Initialize array for divergence

        # Use JAX's `.at[]` to modify the array
        div_E = div_E.at[1:-1].set((E[2:] - E[:-2]) / (2 * dx))
        div_E = div_E.at[0].set((E[1] - E[0]) / dx)
        div_E = div_E.at[-1].set((E[-1] - E[-2]) / dx)

        return div_E

    # from jax import debug
    # @scan_tqdm(total_steps)
    # def simulation_step(carry, step_index):
    #     (E_field, B_field, positions,
    #      velocities, qs, ms, q_ms) = carry

    #     dt = parameters["dt"]
    #     length = parameters["length"]
    #     n_iterations = 7  # Number of iterations for Picard iteration
    #     num_substeps=1
    #     E_new=E_field
    #     B_new=B_field
    #     positions_new=positions
    #     velocities_new=velocities
    #     # positions_sub1=positions
    #     # positions_sub1_2_all = jnp.zeros((num_substeps,) + positions.shape)
    #     positions_sub1_2_all = jnp.tile(positions[None, ...], (num_substeps, 1, 1))
    #     debug.print(" positions = {}",  positions[0:5])
    #     debug.print(" velocities = {}",  velocities[0:5])
    #     debug.print(" E_field = {}",  E_field[0:5])
    #     # Start the iteration loop
    #     for iteration in range(n_iterations):
    #         E_avg = 0.5 * (E_field + E_new)
    #         B_avg = 0.5 * (B_field + B_new)
   
    #         # # Loop through substeps
    #         positions_sub=positions
    #         velocities_sub = velocities
    #         # positions_sub1_2=0.5*(positions+positions_sub1)
    
    #         J = jnp.zeros((len(grid), 3))
    #         dtau = dt / num_substeps
    #         for step in range(num_substeps):
    #             # Compute the substep positions and velocities
    #             # t_end = (step + 1) / num_substeps
    #             E_at_half = vmap(lambda x_n: fields_to_particles_grid(
    #                 x_n, E_avg, dx, grid + dx / 2, grid[0], field_BC_left, field_BC_right))(positions_sub1_2_all[step])
    #             # debug.print(" pos_stag_prev[0] = {}", positions_sub1_2_all[step][0:5])
    #             velocities_subnew = velocities_sub + (q_ms * E_at_half) * dtau
    #             # debug.print("  substep {}: E_mid[0] = {}", step, E_at_half[0:5])

    #             # debug.print("  substep {}: velocities_subnew[0] = {}", step, velocities_subnew[0:5])

    #             # debug.print("  substep {}: velocities_sub[0] = {}", step, velocities_sub[0:5])
    #             velocities_plus1_2 = 0.5 * (velocities_sub + velocities_subnew)
    #             positions_subnew = positions_sub + velocities_plus1_2 * dtau
    #             # debug.print("  substep {}: positions_subnew[0] = {}", step, positions_subnew[0:5])
    #             positions_subnew, velocities_plus1_2, qs, ms, q_ms = set_BC_particles(
    #             positions_subnew, velocities_plus1_2, qs, ms, q_ms, dx, grid,
    #             *box_size, particle_BC_left, particle_BC_right)
    #             positions_sub1_2 = set_BC_positions(positions_subnew - (dtau / 2) * velocities_plus1_2,
    #                                             qs, dx, grid, *box_size, particle_BC_left, particle_BC_right)
    #             positions_sub1_2_all = positions_sub1_2_all.at[step].set(positions_sub1_2)
    #             # debug.print("  pos_stag_new[0] = {}",  positions_sub1_2[0:5])
    #             # Now call the current_density function for each substep
    #             J_substep = current_density(positions_sub, positions_sub1_2,positions_subnew, velocities_plus1_2,
    #                 qs, dx, dtau, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
    #             # J_substep = current_density_CN(positions_sub1_2, velocities_plus1_2,
    #             # qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
    #             # debug.print("  substep {}: J_substep sum = {}", step, jnp.sum(J_substep))
    #             positions_sub=positions_subnew
    #             velocities_sub=velocities_subnew

    #             # Accumulate the result for each substep
    #             J += J_substep*dtau
    #         J=J/dt

    #         # J = current_density_CN(positions_plus1_2, velocities_plus1_2,
    #         #     qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
    #         mean_J = jnp.array([integrate(J[:,0], dx=dx), integrate(J[:,1], dx=dx), integrate(J[:,2], dx=dx)])/length
    #         # debug.print("Picard it {}: mean_J = {}", iteration, mean_J)
    #         E_next = E_field - (dt / epsilon_0) * (J-mean_J)
            
    #         delta_E = jnp.abs(jnp.mean(E_next - E_new))/jnp.max(jnp.abs(E_next))
    #         # debug.print("mean_J: {}", mean_J)
    #         # debug.print("Picard it {}: E_next[0] = {}", iteration, E_next[0:5])
    #         # debug.print("delta_E: {}", delta_E)


    #         E_new=E_next
    #         positions_new=positions_subnew
    #         velocities_new=velocities_subnew
            
    #         # E_new=E_next
    #         # positions_new=positions_new
    #         # velocities_new=velocities_new

    #     # Update fields after iteration
    #     E_field = E_new
    #     B_field = B_field  # Assuming B update is handled similarly if needed

    #     # Final positions and velocities after convergence
    #     positions_plus1= positions_new
    #     velocities_plus1 = velocities_new
    #     # Charge density with quadratic shape
    #     charge_density = calculate_charge_density(positions_new, qs, dx, grid, particle_BC_left, particle_BC_right)
    #     # charge_err = jnp.max(compute_divergence(E_new[:,0], dx)* epsilon_0 - charge_density+initial_charge_density)
    #     # debug.print("charge_err: {}", charge_err) 
    #     #debug.print("charge_den: {}",  jnp.mean(jnp.abs(charge_density))) 


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

        dt = parameters["dt"]
        length = parameters["length"]
        n_iterations = 7  # Number of iterations for Picard iteration
        num_substeps=2
        E_new=E_field
        B_new=B_field
        # positions_new=positions
        positions_new=positions+ (dt) * velocities
        velocities_new=velocities
        # debug.print(" positions = {}",  positions[0:5])
        # debug.print(" velocities = {}",  velocities[0:5])
        # debug.print(" E_field = {}",  E_field[0:5])
        # one substep of your leapfrog+current‐deposit

        # --- before your Picard scan, initialize the array of half-step positions ---
        positions_sub1_2_all_init = jnp.tile(positions[None, ...], (num_substeps, 1, 1))
        # debug.print(" positions = {}",  positions[0:5])
        # one micro‐step: reads & writes the per-step pos_stag array
        def picard_step(pic_carry, _):
            E_new, pos_fix, pos_prev, vel_fix, vel_prev, qs_prev, ms_prev, q_ms_prev, pos_stag_arr = pic_carry
            
            # recompute midpoint field for *this* Picard iteration
            E_avg = 0.5 * (E_field + E_new)
            # debug.print("  E_avg[0] = {}" , E_avg[0:5])
            dtau  = dt / num_substeps

            # now define substep_loop so it captures THIS E_avg
            def substep_loop(sub_carry, step_idx):
                pos_sub, vel_sub, qs_sub, ms_sub, q_ms_sub, pos_stag_arr = sub_carry

                # grab the staggered pos for this substep
                pos_stag_prev = pos_stag_arr[step_idx]
                # debug.print("  substep {}:pos_stag_prev[0] = {}", step_idx, pos_stag_prev[0:5])
                # field at half‑step uses this iteration’s E_avg
                E_mid = vmap(lambda x:
                    fields_to_particles_grid(
                        x, E_avg, dx, grid + dx/2, grid[0],
                        field_BC_left, field_BC_right)
                )(pos_stag_prev)
                
                # leapfrog update
                vel_new = vel_sub + (q_ms_sub * E_mid) * dtau
                # debug.print("  substep {}: E_mid[0] = {}" ,step_idx, E_mid[0:5])
                # debug.print("  substep {}: velocities_subnew[0] = {}", step_idx, vel_new[0:5])
                # debug.print("  substep {}: velocities_sub[0] = {}", step_idx, vel_sub[0:5])

                vel_mid = 0.5 * (vel_sub + vel_new)
                pos_new = pos_sub + vel_mid * dtau
                # debug.print("  substep {}: positions_subnew[0] = {}", step_idx, pos_new[0:5])
                # enforce BCs
                pos_new, vel_mid, qs_new, ms_new, q_ms_new = set_BC_particles(
                    pos_new, vel_mid, qs_sub, ms_sub, q_ms_sub,
                    dx, grid, *box_size, particle_BC_left, particle_BC_right
                )

                # compute & write back new staggered pos
                pos_stag_new = set_BC_positions(
                    pos_new - 0.5*dtau*vel_mid,
                    qs_new, dx, grid,
                    *box_size, particle_BC_left, particle_BC_right
                )
                # debug.print("  substep {}: pos_stag_new[0] = {}", step_idx, pos_stag_new[0:5])
                pos_stag_arr = pos_stag_arr.at[step_idx].set(pos_stag_new)

                # deposit current
                J_sub = current_density(
                    pos_sub, pos_stag_new, pos_new, vel_mid, qs_new, dx, dtau, grid,
                    grid[0] - dx/2, particle_BC_left, particle_BC_right
                )
                # debug.print("  substep {}: J_substep sum = {}", step_idx, jnp.sum(J_sub))
                return (pos_new, vel_new, qs_new, ms_new, q_ms_new, pos_stag_arr), J_sub * dtau

            # pack initial substep carry (including full pos_stag array)
            sub_init = (
                pos_fix, vel_fix,
                qs_prev, ms_prev, q_ms_prev,
                pos_stag_arr
            )

            # run through all substeps
            (pos_final, vel_final, qs_final, ms_final, q_ms_final, pos_stag_arr), J_accs = lax.scan(
                substep_loop,
                sub_init,
                jnp.arange(num_substeps)
            )
            # debug.print("J_accs: {}",J_accs) 
            # assemble J and update E
            J_iter = jnp.sum(J_accs, axis=0) / dt
            # debug.print("J_accs0: {}",J_accs[0][0]) 
            # debug.print("J_accs1: {}",J_accs[1][25]) 
            # debug.print("J_iter: {}",J_iter.shape) 
            mean_J = jnp.mean(J_iter, axis=0)
            # mean_J = jnp.array([
            #     integrate(J_iter[:,0], dx=dx),
            #     integrate(J_iter[:,1], dx=dx),
            #     integrate(J_iter[:,2], dx=dx),
            # ]) / length
            # debug.print("length: {}",length)
            # debug.print(" mean_J = {}", mean_J)

            E_next = E_field - (dt / epsilon_0) * (J_iter - mean_J)
            # debug.print("E_next[0] = {}", E_next[0:5])
            delta_E = jnp.abs(jnp.mean(E_next - E_new))/jnp.max(jnp.abs(E_next))
            # debug.print("delta_E: {}", delta_E)
            return (
                (E_next,pos_fix, pos_final, vel_fix, vel_final, qs_final, ms_final, q_ms_final, pos_stag_arr),
                J_iter
            )


        # run n_iterations Picard sweeps and grab the last J
        picard_init = (E_new, positions, positions_new, velocities,velocities_new, qs, ms, q_ms, positions_sub1_2_all_init)
        (E_new, pos_fix, positions_new, vel_fix,velocities_new, qs_new, ms_new, q_ms_new,_), J_all = lax.scan(
            picard_step,
            picard_init,
            None,
            length=n_iterations
        )
        J = J_all[-1]

        # Update fields after iteration
        E_field = E_new
        B_field = B_field  # Assuming B update is handled similarly if needed

        # Final positions and velocities after convergence
        positions_plus1= positions_new
        velocities_plus1 = velocities_new
        # Charge density with quadratic shape
        charge_density = calculate_charge_density(positions_new, qs, dx, grid, particle_BC_left, particle_BC_right)
        # charge_err = jnp.max(compute_divergence(E_new[:,0], dx)* epsilon_0 - charge_density+initial_charge_density)
        # debug.print("charge_err: {}", charge_err) 
        #debug.print("charge_den: {}",  jnp.mean(jnp.abs(charge_density))) 

        # debug.print("  E[0] = {}" , E_field[0:5])
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
