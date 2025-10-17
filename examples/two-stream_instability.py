# Example script to run the simulation and plot the results
import time
from jax import block_until_ready
from jaxincell import plot, simulation, load_parameters

# Read from input.toml
input_parameters, solver_parameters = load_parameters('input.toml')

n_simulations = 1 # >1 to check that first simulation takes longer due to JIT compilation

# Run the simulation
for i in range(n_simulations):
    if i>0: input_parameters["print_info"] = False
    start = time.time()
    output = block_until_ready(simulation(input_parameters, **solver_parameters))
    print(f"Run #{i+1}: Wall clock time: {time.time()-start}s")

# Plot the results
plot(output, direction="x")

# 1) Waves + spectrum + energy
# from jaxincell import wave_spectrum_movie
# wave_spectrum_movie(output, direction="x", save_path=None, fps=60)

# 2) Phase space (both species)
# from jaxincell import phase_space_movie
# phase_space_movie(output, direction="x", species="both", points_per_species=250, save_path=None, fps=60)

# 3) Particle box (both species)
# from jaxincell import particle_box_movie
# particle_box_movie( output, direction="x", trail_len=20, n_electrons=100, n_ions=100,
#     show_field=True, field_alpha=0.32, field_cmap="coolwarm", save_path=None,  fps=60)

# Save the output to a file
# import numpy as np
# np.savez("simulation_output.npz", **output)

# Load the output from the file
# import pickle
# import json
# data = np.load("simulation_output.npz", allow_pickle=True)
# output2 = dict(data)
