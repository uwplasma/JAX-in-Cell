## main.py
# Main script to run the simulation, plot the results and compute the frequency of the charge density oscillations
from simulation import simulation
from diagnostics import diagnostics
from plot import plot_results

parameters_float = {
"length"                       : 1e-2, # Dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 0.1,  # Amplitude of sinusoidal perturbation in x
"grid_points_per_Debye_length" : 9,    # dx over Debye length
"vth_electrons_over_c"         : 0.05, # Thermal velocity of electrons over speed of light
"CFL_factor"  : 0.5,  # dt * speed_of_light / dx
"seed"        : 1701, # Random seed for reproducibility
}

number_grid_points     = 20  # Number of grid points
number_pseudoelectrons = 500 # Number of pseudoelectrons
total_steps            = 20 # Total number of time steps

# Run the simulation
output = simulation(parameters_float, number_grid_points=number_grid_points, number_pseudoelectrons=number_pseudoelectrons, total_steps=total_steps)

# Plot the results
plot_results(output)

# Compute frequency of the charge density oscillations
# frequency = diagnostics()