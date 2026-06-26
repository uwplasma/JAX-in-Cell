# Example script to run the simulation and plot the results
import os
import time
from jax import block_until_ready
from jaxincell import plot, Simulation, load_parameters, diagnostics

# Read from input.toml (assuming it's in the same directory as this script)
input_file = 'input.toml'
current_directory = os.path.dirname(os.path.abspath(__file__))
input_toml_path = os.path.join(current_directory, input_file)

parameters = load_parameters(input_toml_path)

n_simulations = 1 # >1 to check that first simulation takes longer due to JIT compilation

# Run the simulation
for i in range(n_simulations):
    if i > 0:
        parameters.setdefault("solver_parameters", {})["print_info"] = False
    sim = Simulation(parameters)
    start = time.time()
    output = block_until_ready(sim.run())
    print(f"Run #{i+1}: Wall clock time: {time.time()-start}s")

# Post-process: segregate ions/electrons, compute energies, compute FFT
diagnostics(output)

# Plot the results
plot(output, animation_interval=5)
# Save the animation as an mp4 file (takes longer)
# plot(output, save_mp4="two_stream.mp4", fps=50, dpi=150, save_dpi=60, save_crf=32, save_stride=5, show=False)

# # Save the output to a file
# import numpy as np
# np.savez("simulation_output.npz", **output)

# # Load the output from the file
# import pickle
# data = np.load("simulation_output.npz", allow_pickle=True)
# output2 = dict(data)
