## main.py
# Main script to run the simulation, plot the results and compute the frequency of the charge density oscillations
import time
from plot import plot_results
from simulation import simulation
from diagnostics import diagnostics
from jax import block_until_ready

parameters_float = {
"length"                       : 1e-2, # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 1e-6, # amplitude of sinusoidal perturbation in x
"wavenumber_perturbation_x_factor": 1,  # Wavenumber of sinusoidal (sin) perturbation in x (factor of 2pi/length)
"grid_points_per_Debye_length" : 5,    # dx over Debye length
"vth_electrons_over_c"         : 0.05, # thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature": 1, # Temperature of ions over temperature of electrons
"CFL_factor"           : 0.5, # dt * speed_of_light / dx
# "electron_drift_speed" : 0,               # drift speed of electrons
# "velocity_plus_minus_electrons"  : False, # create two groups of electrons moving in opposite directions
"electron_drift_speed" : 1e8,            # Two-Stream (drift speed of electrons)
"velocity_plus_minus_electrons"  : True, # Two-Stream (create two groups of electrons moving in opposite directions)
"print_info"           : True, # print information about the simulation
"external_electric_field_amplitude": 0, # External electric field value (V/m)
"external_electric_field_wavenumber_perturbation_x_factor": 0,  # External electric Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
}

number_grid_points     = 20  # Number of grid points
number_pseudoelectrons = 2000 # Number of pseudoelectrons
total_steps            = 500 # Total number of time steps
fraction_random_particles = 0.02 # Fraction of particles to be randomly distributed

n_simulations = 1

# Run the simulation
for i in range(n_simulations):
    if i>0: parameters_float["print_info"] = False
    start = time.time()
    output = block_until_ready(simulation(parameters_float, number_grid_points=number_grid_points,
                                          number_pseudoelectrons=number_pseudoelectrons, total_steps=total_steps,
                                          fraction_random_particles=fraction_random_particles))
    print(f"Run #{i+1}:Wall clock time: {time.time()-start}s")#, simulation time: {output['time_array'][-1]:.2e}s has a mean E energy of {jnp.mean(output['E_energy']):.2e}")

# Populate output with diagnostics and print frequencies
diagnostics(output)

# Plot the results
plot_results(output)
