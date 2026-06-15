## Weibel_instability.py
# Example of plasma oscillations of electrons
from jaxincell import plot
from jaxincell import Simulation, diagnostics
from jax import block_until_ready

parameters = {
    "domain_parameters": {
        "length": 3e-1,
        "timestep_over_spatialstep_times_c": 1.0,
        "number_grid_points": 150,
        "total_steps": 2500,
    },
    "species_parameters": {
        "electrons": {
            "electrons0": {
                "number_pseudoparticles": 3000,
                "grid_points_per_Debye_length": 1.1,
                "perturbation_amplitude_x": 0,
                "perturbation_wavenumber_x": 0,
                "random_positions_x": True,
                "random_positions_y": True,
                "random_positions_z": True,
                "vth_over_c_x": 0.01,
                "vth_over_c_z": 0.10,
                "drift_speed_x": 0,
                "drift_speed_z": 0,
                "velocity_plus_minus_x": False,
                "velocity_plus_minus_z": False,
            },
        },
        "ions": {
            "ions0": {
                "number_pseudoparticles": 3000,
                "grid_points_per_Debye_length": 1.1,
                "random_positions_x": True,
                "random_positions_y": True,
                "random_positions_z": True,
                "vth_over_c_x": "_electrons0",
                "vth_over_c_y": "_electrons0",
                "vth_over_c_z": "_electrons0",
                "ion_temperature_over_electron_temperature_x": 1,
            },
        },
    },
    "solver_parameters": {
        "field_solver": 0,
        "time_evolution_algorithm": 0,
        "max_number_of_Picard_iterations_implicit_CN": 20,
        "number_of_particle_substeps_implicit_CN": 1,
        "tolerance_Picard_iterations_implicit_CN": 1e-12,
        "relativistic": False,
        "filter_passes": 5,
        "filter_alpha": 0.5,
        "filter_strides": (1, 2, 4),
        "print_info": True,
    },
}

sim = Simulation(parameters)
output = block_until_ready(sim.run())

# Post-process: segregate ions/electrons, compute energies, compute FFT
diagnostics(output)

plot(output, direction="xz", animation_interval=1)  # Plot the results in x and z direction
# Save the animation as an mp4 file (takes longer)
# plot(output, save_mp4="weibel_instability.mp4", direction="xz", fps=50, dpi=150, save_dpi=60, save_crf=32, save_stride=5, show=False)
