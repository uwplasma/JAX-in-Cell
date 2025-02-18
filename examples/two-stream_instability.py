# Example script to run the simulation and plot the results
import time
import tomllib
from jax import block_until_ready
from jaxincell import plot
from jaxincell import simulation

# Read from input.toml
parameters = tomllib.load(open('input.toml', "rb"))
input_parameters = parameters['input_parameters']
solver_parameters = parameters['solver_parameters']

n_simulations = 3 # Check that first simulation takes longer due to JIT compilation

# Run the simulation
for i in range(n_simulations):
    if i>0: input_parameters["print_info"] = False
    start = time.time()
    output = block_until_ready(simulation(input_parameters, **solver_parameters))
    print(f"Run #{i+1}: Wall clock time: {time.time()-start}s")

# Plot the results
plot(output)
