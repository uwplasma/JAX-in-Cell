#!/usr/bin/env python
"""
bump-on-tail.py
Weak electron beam on tail of bulk Maxwellian electron distribution drives
slowly-growing Langmuir waves with Im(omega) << omega_pe.
"""

import os
from datetime import datetime
from jax import block_until_ready
from jaxincell import plot, simulation, load_parameters, diagnostics

input_file = 'bump-on-tail.toml'
current_directory = os.path.dirname(os.path.abspath(__file__))
input_toml_path = os.path.join(current_directory, input_file)

input_parameters, solver_parameters = load_parameters(input_toml_path)

simulation_type = 2
if simulation_type == 0:
    print("Running bump-on-tail with periodic BCs (default)...")
    save_mp4_name = "bump_on_tail_explicit_periodic.mp4"
elif simulation_type == 1:
    print("Running bump-on-tail with reflective BCs...")
    input_parameters["particle_BC_left"] = 1
    input_parameters["particle_BC_right"] = 1
    input_parameters["field_BC_left"] = 1
    input_parameters["field_BC_right"] = 1
    save_mp4_name = "bump_on_tail_explicit_reflective.mp4"
elif simulation_type == 2:
    print("Running bump-on-tail with implicit field solver...")
    solver_parameters["time_evolution_algorithm"] = 1
    save_mp4_name = "bump_on_tail_implicit_periodic.mp4"
elif simulation_type == 3:
    print("Running bump-on-tail with digital filtering...")
    input_parameters["filter_passes"] = 5
    save_mp4_name = "bump_on_tail_explicit_periodic_filtered.mp4"

# Run the simulation
started = datetime.now()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print("Simulation done, elapsed:", datetime.now()-started)

# Post-process: segregate ions/electrons, compute energies, compute FFT
print("Post-processing...")
diagnostics(output)

# Plot the results
print("Plotting results (might take a minute)...")
plot(output, animation_interval=20)
# Save the animation as an mp4 file (takes longer)
# plot(output, save_mp4=save_mp4_name, animation_interval=20, fps=50, dpi=60, show=True)

# Save the output to a file
# import numpy as np
# np.savez("simulation_output.npz", **output)

# # Load the output from the file
# data = np.load("simulation_output.npz", allow_pickle=True)
# output2 = dict(data)
