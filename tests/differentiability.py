## differentiability.py
# Calculate derivatives of outputs with respect to inputs
import os, sys;
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulation import simulation
from jax import jit, grad, lax, block_until_ready
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

input_parameters = {
"length"                       : 1e-2,  # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 5e-4,  # amplitude of sinusoidal perturbation in x
"wavenumber_perturbation_x_factor": 8, # Wavenumber of sinusoidal (sin) perturbation in x (factor of 2pi/length)
"grid_points_per_Debye_length" : 2,     # dx over Debye length
"vth_electrons_over_c"         : 0.05,  # thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature": 1e-2, # Temperature of ions over temperature of electrons
"timestep_over_spatialstep_times_c": 1.0, # dt * speed_of_light / dx
"electron_drift_speed"         : 0,     # drift speed of electrons
"velocity_plus_minus_electrons": False, # create two groups of electrons moving in opposite directions
"print_info"                   : False,  # print information about the simulation
"external_electric_field_amplitude": 0, # External electric field value (V/m)
"external_electric_field_wavenumber_perturbation_x_factor": 0,  # External electric Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
"amplitude_perturbation_x"     : 1e-7,  # Two-Stream (amplitude of sinusoidal perturbation in x)
"electron_drift_speed"         : 1e8,   # Two-Stream (drift speed of electrons)
"velocity_plus_minus_electrons": True,  # Two-Stream (create two groups of electrons moving in opposite directions)
# "wavenumber_perturbation_x_factor": 1,  # Plasma Oscillations (Wavenumber of sinusoidal (sin) perturbation in x (factor of 2pi/length))
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 100,  # Number of grid points
    "number_pseudoelectrons" : 2000, # Number of pseudoelectrons
    "total_steps"            : 1000, # Total number of time steps
}

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
    print(f"Calculating derivative for epsilon = {epsilon}")
    
    start = time.time()
    E_field_plus_epsilon = mean_electric_field(electron_drift_speed+epsilon)
    time_analytical_derivative.append(time.time()-start)
    derivative_analytical = (E_field_plus_epsilon - E_field) / epsilon
    derivative_analytical_array.append(derivative_analytical)

# Plot the results
plt.figure(figsize=(6, 4))
plt.plot(epsilon_array, [derivative_JAX]*len(epsilon_array), label="JAX derivative")
plt.plot(epsilon_array, derivative_analytical_array, '*-', label="Numerical derivative")
plt.xlabel(r"$\epsilon$", fontsize=20)
plt.ylabel(r"${d \left< E \right> }/{d v_{\text{drift}}}$", fontsize=20)
plt.legend(fontsize=14)
plt.xscale('log')
plt.tight_layout()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("tests/differentiability.png", dpi=300)
plt.show()
