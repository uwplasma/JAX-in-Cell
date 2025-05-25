## Weibel_instability.py
# Example of plasma oscillations of electrons
from jaxincell import plot
from jaxincell import simulation
from jax import block_until_ready

input_parameters = {
"length"                       : 1e-2,  # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 0,  # amplitude of sinusoidal perturbation in x
"grid_points_per_Debye_length" : 0.3,     # dx over Debye length
"velocity_plus_minus_electrons_z": True,    # create two groups of electrons moving in opposite directions
"electron_drift_speed_z": 1e8, # Drift speed of electrons in z direction
"vth_electrons_over_c_z": 1e-2,             # Thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature": 1e-6, # Temperature of ions over temperature of electrons
"print_info"                   : True,  # print information about the simulation
"vth_electrons_over_c_x": 0,             # Thermal velocity of electrons over speed of light
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 33,  # Number of grid points
    "number_pseudoelectrons" : 3000, # Number of pseudoelectrons
    "total_steps"            : 300, # Total number of time steps
}

output = block_until_ready(simulation(input_parameters, **solver_parameters))

plot(output)
