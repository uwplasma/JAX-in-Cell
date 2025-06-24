## Landau_damping.py
# Example of electric field damping in a plasma
from jaxincell import plot
from jaxincell import simulation
import jax.numpy as jnp
from jax import block_until_ready

input_parameters = {
"length"                       : 1e-2,  # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 4e-3,  # amplitude of sinusoidal perturbation in x
"wavenumber_electrons_x": 8, # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
"velocity_plus_minus_electrons_x": False, # create two groups of electrons moving in opposite directions
"grid_points_per_Debye_length" : 10,     # dx over Debye length
"vth_electrons_over_c_x"         : 0.05,  # thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature_x": 1e-9, # Temperature of ions over temperature of electrons
"timestep_over_spatialstep_times_c": 3, # dt * speed_of_light / dx
"print_info"                   : True,  # print information about the simulation
"tolerance_Picard_iterations_implicit_CN": 1e-5, # Tolerance for Picard iterations
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 81,  # Number of grid points
    "number_pseudoelectrons" : 3000, # Number of pseudoelectrons
    "total_steps"            : 200, # Total number of time steps
    "time_evolution_algorithm": 1,  # Algorithm to evolve particles in time - 0: Boris, 1: Implicit_Crank Nicholson
    "max_number_of_Picard_iterations_implicit_CN": 30, # Maximum number of iterations for Picard iteration converging
    "number_of_particle_substeps_implicit_CN": 2, # The number of substep for one time eletric field update
}

output = block_until_ready(simulation(input_parameters, **solver_parameters))

print(f"Dominant FFT frequency (f): {output['dominant_frequency']} Hz")
print(f"Plasma frequency (w_p):     {output['plasma_frequency']} Hz")
print(f"Error: {jnp.abs(output['dominant_frequency'] - output['plasma_frequency']) / output['plasma_frequency'] * 100:.2f}%")

plot(output)
