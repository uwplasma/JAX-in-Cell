## Langmuir_wave.py
# Example of plasma oscillations of electrons
from jaxincell import plot
from jaxincell import simulation, diagnostics
import jax.numpy as jnp
from jax import block_until_ready

input_parameters = {
"length"                       : 1,  # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 0.01,  # amplitude of sinusoidal perturbation in x
"wavenumber_electrons_x": 1, # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
"grid_points_per_Debye_length" : 3,     # dx over Debye length
"vth_electrons_over_c_x"         : 0.05,  # thermal velocity of electrons over speed of light
"velocity_plus_minus_electrons_x": False,    # create two groups of electrons moving in opposite directions
"ion_temperature_over_electron_temperature_x": 1e-9, # Temperature of ions over temperature of electrons
"timestep_over_spatialstep_times_c": 0.5, # dt * speed_of_light / dx
"print_info"                   : True,  # print information about the simulation
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT,
    "number_grid_points"     : 33,  # Number of grid points
    "number_pseudoelectrons" : 3000, # Number of pseudoelectrons
    "total_steps"            : 1000, # Total number of time steps
}

output = block_until_ready(simulation(input_parameters, **solver_parameters))

# Post-process: segregate ions/electrons, compute energies, compute FFT
diagnostics(output)

print(f"Dominant FFT frequency (f): {output['dominant_frequency']} Hz")
print(f"Plasma frequency (w_p):     {output['plasma_frequency']} Hz")
print(f"Error: {jnp.abs(output['dominant_frequency'] - output['plasma_frequency']) / output['plasma_frequency'] * 100:.2f}%")

plot(output)
