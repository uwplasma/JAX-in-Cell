## main.py
# Main script to run the simulation, plot the results and compute the frequency of the charge density oscillations
from simulation import simulation
from diagnostics import diagnostics
from plot import plot_results
import time
from jax import block_until_ready
import jax.numpy as jnp

parameters_float = {
"length"                       : 1e-2, # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 1e-1, # amplitude of sinusoidal perturbation in x
"grid_points_per_Debye_length" : 9,    # dx over Debye length
"vth_electrons_over_c"         : 0.05, # thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature": 1, # Temperature of ions over temperature of electrons
"CFL_factor"           : 0.5, # dt * speed_of_light / dx
# "electron_drift_speed" : 0,               # drift speed of electrons
# "velocity_plus_minus_electrons"  : False, # create two groups of electrons moving in opposite directions
"electron_drift_speed" : 1e8,            # Two-Stream (drift speed of electrons)
"velocity_plus_minus_electrons"  : True, # Two-Stream (create two groups of electrons moving in opposite directions)
}

number_grid_points     = 20  # Number of grid points
number_pseudoelectrons = 2000 # Number of pseudoelectrons
total_steps            = 400 # Total number of time steps

n_simulations = 1

# Run the simulation
for i in range(n_simulations):
    start = time.time()
    output = block_until_ready(simulation(parameters_float, number_grid_points=number_grid_points, number_pseudoelectrons=number_pseudoelectrons, total_steps=total_steps))
    print(f"Run #{i+1}:Wall clock time: {time.time()-start}s")#, simulation time: {output['time_array'][-1]:.2e}s has a mean E energy of {jnp.mean(output['E_energy']):.2e}")

# Populate output with diagnostics and print frequencies
diagnostics(output)

# Plot the results
plot_results(output)
