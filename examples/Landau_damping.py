## Landau_damping.py
# Example of electric field damping in a plasma
from jaxincell import plot
from jaxincell import Simulation, diagnostics
import jax.numpy as jnp
from jax import block_until_ready

parameters = {
    "domain_parameters": {
        "length": 1,
        "timestep_over_spatialstep_times_c": 1,
        "number_grid_points": 32,
        "total_steps": 300,
    },
    "species_parameters": {
        "electrons": {
            "electrons0": {
                "number_pseudoparticles": 40000,
                "grid_points_per_Debye_length": 0.4,
                "perturbation_amplitude_x": 0.025,
                "perturbation_wavenumber_x": 1.02,
                "vth_over_c_x": 0.35,
                "drift_speed_x": 0.0,
                "velocity_plus_minus_x": False,
            },
        },
        "ions": {
            "ions0": {
                "number_pseudoparticles": 40000,
                "grid_points_per_Debye_length": 0.4,
                "mass_over_proton_mass": 1e9,
                "vth_over_c_x": "_electrons0",
                "vth_over_c_y": "_electrons0",
                "vth_over_c_z": "_electrons0",
                "ion_temperature_over_electron_temperature_x": 1e-9,
            },
        },
    },
    "solver_parameters": {
        "field_solver": 0,
        "time_evolution_algorithm": 0,
        "max_number_of_Picard_iterations_implicit_CN": 20,
        "number_of_particle_substeps_implicit_CN": 1,
        "tolerance_Picard_iterations_implicit_CN": 1e-5,
        "relativistic": False,
        "filter_passes": 5,
        "filter_alpha": 0.5,
        "filter_strides": [1, 2, 4],
        "print_info": True,
    },
}

sim = Simulation(parameters)
output = block_until_ready(sim.run())

# Post-process: segregate ions/electrons, compute energies, compute FFT
diagnostics(output)

print(f"Dominant FFT frequency (f): {output['dominant_frequency']} Hz")
print(f"Plasma frequency (w_p):     {output['plasma_frequency']} Hz")
print(f"Error: {jnp.abs(output['dominant_frequency'] - output['plasma_frequency']) / output['plasma_frequency'] * 100:.2f}%")

plot(output)
