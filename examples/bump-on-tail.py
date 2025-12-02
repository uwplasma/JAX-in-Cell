#!/usr/bin/env python
"""
bump-on-tail.py
Weak electron beam on tail of bulk Maxwellian electron distribution drives
slowly-growing Langmuir waves with Im(omega) << omega_pe.
"""

import numpy as np
from datetime import datetime
from jax import config
# config.update("jax_traceback_filtering", "off")
from jax import block_until_ready
from jaxincell import plot, simulation, load_parameters, diagnostics

input_parameters, solver_parameters = load_parameters('bump-on-tail.toml')

# Run the simulation
started = datetime.now()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print("Simulation done, elapsed:", datetime.now()-started)

# Post-process: segregate ions/electrons, compute energies, compute FFT
diagnostics(output)

# Plot the results
plot(output)

# Save the output to a file
np.savez("simulation_output.npz", **output)

# # Load the output from the file
# data = np.load("simulation_output.npz", allow_pickle=True)
# output2 = dict(data)
