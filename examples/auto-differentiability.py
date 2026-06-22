## differentiability.py
# Calculate derivatives of outputs with respect to inputs
import os, sys;
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, block_until_ready
from jaxincell import Simulation, load_parameters

# Read from input.toml (assuming it's in the same directory as this script)
input_file = 'input.toml'
current_directory = os.path.dirname(os.path.abspath(__file__))
input_toml_path = os.path.join(current_directory, input_file)

parameters = load_parameters(input_toml_path)

parameters.setdefault("domain_parameters", {})
parameters.setdefault("species_parameters", {})
parameters.setdefault("solver_parameters", {})

parameters["solver_parameters"]["print_info"] = False
parameters["domain_parameters"]["total_steps"] = 400
parameters["domain_parameters"]["number_grid_points"] = 60
parameters["species_parameters"]["electrons"]["electrons0"]["number_pseudoparticles"] = 3000
parameters["species_parameters"]["ions"]["ions0"]["number_pseudoparticles"] = 3000

sim = Simulation(parameters)
total_steps = sim.domain_parameters["total_steps"]

def mean_electric_field(electron_drift_speed):
    runtime_input_parameters = {
        "electrons": {
            "electrons0": {
                "drift_speed_x": electron_drift_speed,
            },
        },
    }
    output = sim.run(runtime_input_parameters)
    electric_field = jnp.mean(output['electric_field'][:, :, 0], axis=1)
    mean_E = jnp.mean(electric_field[total_steps//2:])
    return mean_E

# Calculate the derivative of the plasma frequency with respect to the wavenumber
electron_drift_speed = 1e8
epsilon_array = [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
E_field = block_until_ready(mean_electric_field(electron_drift_speed))
print(f"Mean electric field at drift speed {electron_drift_speed} m/s: {E_field}")
start = time.time()
derivative_JAX = block_until_ready(grad(mean_electric_field)(electron_drift_speed))
time_JAX_derivative = time.time()-start

derivative_analytical_array = []
time_analytical_derivative = []
for epsilon in epsilon_array:
    print(f"Calculating derivative for epsilon = {epsilon}")
    start = time.time()
    E_field_plus_epsilon = block_until_ready(mean_electric_field(electron_drift_speed+epsilon))
    time_analytical_derivative.append(time.time()-start)
    derivative_analytical = (E_field_plus_epsilon - E_field) / epsilon
    derivative_analytical_array.append(derivative_analytical)

# Plot the results
plt.figure(figsize=(6, 4))
plt.plot(epsilon_array, [derivative_JAX]*len(epsilon_array), label="JAX derivative")
plt.plot(epsilon_array, derivative_analytical_array, 'o-', label="Numerical derivative")
plt.xlabel(r"$\epsilon$", fontsize=20)
plt.ylabel(r"${d \left< E \right> }/{d v_{\text{drift}}}$", fontsize=20)
plt.legend(fontsize=14)
plt.xscale('log')
plt.tight_layout()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("differentiability.png", dpi=300)
plt.show()
