## main.py
# Main script to run the simulation, plot the results and compute the frequency of the charge density oscillations
import time
from plot import plot_results
from simulation import simulation
from diagnostics import diagnostics
from jax import block_until_ready

input_parameters = {
"length"                       : 1e-2,  # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 5e-4,  # amplitude of sinusoidal perturbation in x
"wavenumber_electrons": 8, # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
"grid_points_per_Debye_length" : 2,     # dx over Debye length
"vth_electrons_over_c"         : 0.05,  # thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature": 1e-2, # Temperature of ions over temperature of electrons
"timestep_over_spatialstep_times_c": 1.0, # dt * speed_of_light / dx
"electron_drift_speed"         : 0,     # drift speed of electrons
"velocity_plus_minus_electrons": False, # create two groups of electrons moving in opposite directions
"print_info"                   : True,  # print information about the simulation
"external_electric_field_amplitude": 0, # External electric field value (V/m)
"external_electric_field_wavenumber": 0,  # External electric Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
"amplitude_perturbation_x"     : 1e-7,  # Two-Stream (amplitude of sinusoidal perturbation in x)
"electron_drift_speed"         : 1e8,   # Two-Stream (drift speed of electrons)
"velocity_plus_minus_electrons": True,  # Two-Stream (create two groups of electrons moving in opposite directions)
# "wavenumber_electrons": 1,  # Plasma Oscillations (Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length))
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 100,  # Number of grid points
    "number_pseudoelectrons" : 3000, # Number of pseudoelectrons
    "total_steps"            : 1000, # Total number of time steps
}

n_simulations = 1 # Check that first simulation takes longer due to compilation

# Run the simulation
for i in range(n_simulations):
    if i>0: input_parameters["print_info"] = False
    start = time.time()
    output = block_until_ready(simulation(input_parameters, **solver_parameters))
    print(f"Run #{i+1}: Wall clock time: {time.time()-start}s")

# Populate output with diagnostics and print frequencies
diagnostics(output)

# Plot the results
plot_results(output)
