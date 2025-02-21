## differentiability.py
# Calculate derivatives of outputs with respect to inputs
import os, sys;
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, grad, lax, block_until_ready, debug
from jaxincell import simulation, load_parameters

# Read from input.toml
input_parameters, solver_parameters = load_parameters('input.toml')

@jit
def mean_electric_field(electron_drift_speed):
    input_parameters["electron_drift_speed"] = electron_drift_speed
    output = block_until_ready(simulation(input_parameters, **solver_parameters))
    electric_field = jnp.mean(output['electric_field'][:, :, 0], axis=1)
    mean_E = jnp.mean(lax.slice(electric_field, [solver_parameters["total_steps"]//2], [solver_parameters["total_steps"]]))
    return mean_E

# Calculate the derivative of the plasma frequency with respect to the wavenumber
electron_drift_speed = 1e8
epsilon_array = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
E_field = mean_electric_field(electron_drift_speed)
start = time.time()
derivative_JAX = grad(mean_electric_field)(electron_drift_speed)
time_JAX_derivative = time.time()-start

derivative_analytical_array = []
time_analytical_derivative = []
for epsilon in epsilon_array:
    debug.print("Calculating derivative for epsilon = {}",epsilon)
    start = time.time()
    E_field_plus_epsilon = mean_electric_field(electron_drift_speed+epsilon)
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
