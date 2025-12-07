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

# Run the simulation
started = datetime.now()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print("Simulation done, elapsed:", datetime.now()-started)

# Post-process: segregate ions/electrons, compute energies, compute FFT
print("Post-processing...")
diagnostics(output)

# Plot the results
print("Plotting results (might take a minute)...")
plot(output)

# Save the output to a file
# import numpy as np
# np.savez("simulation_output.npz", **output)

# # Load the output from the file
# data = np.load("simulation_output.npz", allow_pickle=True)
# output2 = dict(data)
