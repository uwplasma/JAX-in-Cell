## Weibel_instability.py
# Example of plasma oscillations of electrons
from jaxincell import plot
from jaxincell import simulation
from jax import block_until_ready

input_parameters = {
"length"                       : 3e-1,    # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 0,       # amplitude of sinusoidal perturbation in x
"wavenumber_electrons_x"       : 0,       # wavenumber of perturbation in z
"grid_points_per_Debye_length" : 0.7,     # dx over Debye length
"velocity_plus_minus_electrons_z": False, # create two groups of electrons moving in opposite directions
"velocity_plus_minus_electrons_x": False, # create two groups of electrons moving in opposite directions
"random_positions_x": True,  # Use random positions in x for particles
"random_positions_y": True,  # Use random positions in y for particles
"random_positions_z": True,  # Use random positions in z for particles
"electron_drift_speed_x": 0, # Drift speed of electrons in x direction
"electron_drift_speed_z": 0, # Drift speed of electrons in z direction
"ion_temperature_over_electron_temperature_x": 1, #not 10!, # Temperature of ions over temperature of electrons
"print_info"                   : True,    # print information about the simulation
"vth_electrons_over_c_x": 0.01,           # Thermal velocity of electrons over speed of light
"vth_electrons_over_c_z": 0.10,           # Thermal velocity of electrons over speed of light
"timestep_over_spatialstep_times_c": 0.5, # dt * speed_of_light / dx
"relativistic": False,  # Use relativistic equations of motion
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 201,  # Number of grid points
    "number_pseudoelectrons" : 3000, # Number of pseudoelectrons
    "total_steps"            : 6000, # Total number of time steps
    "time_evolution_algorithm": 0,  # Algorithm to evolve particles in time - 0: Boris, 1: Implicit_Crank Nicholson (electrostatic)
}

output = block_until_ready(simulation(input_parameters, **solver_parameters))

plot(output, direction="xz")  # Plot the results in x and z direction
