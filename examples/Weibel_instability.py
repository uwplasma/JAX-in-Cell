## Weibel_instability.py
# Example of plasma oscillations of electrons
from jaxincell import plot
from jaxincell import simulation, diagnostics
from jax import block_until_ready

input_parameters = {
"length"                       : 3e-1,    # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 0,       # amplitude of sinusoidal perturbation in x
"wavenumber_electrons_x"       : 0,       # wavenumber of perturbation in z
"grid_points_per_Debye_length" : 1.1,     # dx over Debye length
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
"timestep_over_spatialstep_times_c": 1.0, # dt * speed_of_light / dx
"relativistic": False,  # Use relativistic equations of motion
"tolerance_Picard_iterations_implicit_CN": 1e-12, # Tolerance for Picard iterations
"filter_passes": 5,       # number of passes of the digital filter applied to ρ and J
"filter_alpha": 0.5,      # filter strength (0 < alpha < 1)
"filter_strides": (1, 2, 4),  # multi-scale strides for filtering
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 150,  # Number of grid points
    "number_pseudoelectrons" : 3000, # Number of pseudoelectrons
    "total_steps"            : 2500, # Total number of time steps
    "time_evolution_algorithm": 0,  # Algorithm to evolve particles in time - 0: Boris, 1: Implicit_Crank Nicholson
    "max_number_of_Picard_iterations_implicit_CN": 20, # Maximum number of iterations for Picard iteration converging
    "number_of_particle_substeps_implicit_CN": 1, # The number of substep for one time eletric field update
}

output = block_until_ready(simulation(input_parameters, **solver_parameters))

# Post-process: segregate ions/electrons, compute energies, compute FFT
diagnostics(output)

plot(output, direction="xz", animation_interval=1)  # Plot the results in x and z direction
# Save the animation as an mp4 file (takes longer)
# plot(output, save_mp4="weibel_instability.mp4", direction="xz", fps=50, dpi=150, save_dpi=60, save_crf=32, save_stride=5, show=False)