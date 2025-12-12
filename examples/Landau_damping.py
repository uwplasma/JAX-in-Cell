## Landau_damping.py
# Example of electric field damping in a plasma
from jaxincell import plot
from jaxincell import simulation, diagnostics
import jax.numpy as jnp
from jax import block_until_ready

input_parameters = {
    "length"                       : 1,  # dimensions of the simulation box in (x, y, z)
    "amplitude_perturbation_x"     : 0.025,  # amplitude of sinusoidal perturbation in x
    "wavenumber_electrons_x": 1.02, # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
    "velocity_plus_minus_electrons_x": False, # create two groups of electrons moving in opposite directions
    "grid_points_per_Debye_length" : 0.4,     # dx over Debye length
    "vth_electrons_over_c_x"         : 0.35,  # thermal velocity of electrons over speed of light
    "ion_temperature_over_electron_temperature_x": 1e-9, # Temperature of ions over temperature of electrons
    "timestep_over_spatialstep_times_c": 1, # dt * speed_of_light / dx
    "print_info"                   : True,  # print information about the simulation
    "tolerance_Picard_iterations_implicit_CN": 1e-5, # Tolerance for Picard iterations
    "electron_drift_speed_x": 0.0, # Drift speed of electrons in x direction
    "ion_mass_over_proton_mass": 1e9,
    "relativistic": False, # Use relativistic equations of motion
    "filter_passes": 5, # Number of passes of the smoothing filter to apply to the charge density
    "filter_alpha": 0.5, # Smoothing filter parameter
    "filter_strides": [1, 2, 4], # Strides for multi-scale smoothing
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 32,  # Number of grid points
    "number_pseudoelectrons" : 40000, # Number of pseudoelectrons
    "total_steps"            : 300, # Total number of time steps
    "time_evolution_algorithm": 0,  # Algorithm to evolve particles in time - 0: Boris, 1: Implicit_Crank Nicholson
    "max_number_of_Picard_iterations_implicit_CN": 20, # Maximum number of iterations for Picard iteration converging
    "number_of_particle_substeps_implicit_CN": 1, # The number of substep for one time eletric field update
}

output = block_until_ready(simulation(input_parameters, **solver_parameters))

# Post-process: segregate ions/electrons, compute energies, compute FFT
diagnostics(output)

print(f"Dominant FFT frequency (f): {output['dominant_frequency']} Hz")
print(f"Plasma frequency (w_p):     {output['plasma_frequency']} Hz")
print(f"Error: {jnp.abs(output['dominant_frequency'] - output['plasma_frequency']) / output['plasma_frequency'] * 100:.2f}%")

plot(output)