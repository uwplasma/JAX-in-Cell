## Langmuir_wave.py
# Example of plasma oscillations of electrons
import os, sys;
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from plot import plot_results
from simulation import simulation
from diagnostics import diagnostics
import jax.numpy as jnp

input_parameters = {
"length"                       : 1,  # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 0.01,  # amplitude of sinusoidal perturbation in x
"wavenumber_electrons": 1, # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
"grid_points_per_Debye_length" : 3,     # dx over Debye length
"vth_electrons_over_c"         : 0.1,  # thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature": 1e-9, # Temperature of ions over temperature of electrons
"timestep_over_spatialstep_times_c": 0.5, # dt * speed_of_light / dx
"print_info"                   : True,  # print information about the simulation
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 33,  # Number of grid points
    "number_pseudoelectrons" : 3000, # Number of pseudoelectrons
    "total_steps"            : 1000, # Total number of time steps
}

output = simulation(input_parameters, **solver_parameters)
diagnostics(output)
plot_results(output)
