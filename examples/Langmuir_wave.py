## Langmuir_wave.py
# Example of plasma oscillations of electrons
from jaxincell import plot
from jaxincell import Simulation, diagnostics
import jax.numpy as jnp
from jax import block_until_ready

parameters = {
    "domain_parameters": {
        "length": 1,
        "timestep_over_spatialstep_times_c": 0.5,
        "number_grid_points": 33,
        "total_steps": 1000,
    },
    "species_parameters": {
        "electrons": {
            "electrons0": {
                "number_pseudoparticles": 3000,
                "grid_points_per_Debye_length": 3,
                "perturbation_amplitude_x": 0.01,
                "perturbation_wavenumber_x": 1,
                "vth_over_c_x": 0.05,
                "velocity_plus_minus_x": False,
            },
        },
        "ions": {
            "ions0": {
                "number_pseudoparticles": 3000,
                "grid_points_per_Debye_length": 3,
                "vth_over_c_x": "_electrons0",
                "vth_over_c_y": "_electrons0",
                "vth_over_c_z": "_electrons0",
                "ion_temperature_over_electron_temperature_x": 1e-9,
            },
        },
    },
    "solver_parameters": {
        "field_solver": 0,
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
