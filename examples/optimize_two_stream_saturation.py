## two_stream_saturation.py
# Optimize the non-linear saturation of the two_stream_instabiltity to be as small as possible
import os, sys;
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax.numpy as jnp
from jax import jit, grad
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from jaxincell import simulation, diagnostics, epsilon_0, load_parameters

# Read from input.toml
input_parameters, solver_parameters = load_parameters('input.toml')

steps_to_average = 800 # only take the mean of these last steps
minimum_Ti = -2
maximum_Ti = 3
nTi = 15
max_iterations_optimization = 20
x0_optimization = 3.0
learning_rate = 0.05
############### -------- #################
## Objective function to minimize - nonlinear saturation value (mean of long time electric field energy)
@jit
def objective_function(Ti):
    params = input_parameters.copy()
    params["ion_temperature_over_electron_temperature"] = Ti
    output = simulation(params, **solver_parameters)
    abs_E_squared              = jnp.sum(output['electric_field']**2, axis=-1)
    integral_E_squared         = jnp.trapezoid(abs_E_squared, dx=output['dx'], axis=-1)
    energy = (epsilon_0/2) * integral_E_squared
    return jnp.mean(energy[-steps_to_average:])
jac = jit(grad(objective_function))
############### -------- #################
print(f'Perform a first run to see one objective function')
input_parameters["ion_temperature_over_electron_temperature"] = x0_optimization
output = simulation(input_parameters, **solver_parameters)
diagnostics(output, print_to_terminal=False)
objective = objective_function(x0_optimization)
plt.figure(figsize=(8,6))
plt.plot(output['time_array']*output['plasma_frequency'],output['electric_field_energy'], label='Electric Field Energy')
plt.axhline(y=objective, color='r', linestyle='-', label='Mean value used for optimization')
plt.axvline(x=output['time_array'][-steps_to_average]*output['plasma_frequency'], color='b', linestyle='-', label='Initial time used for mean value')
plt.xlabel(r't (1/$\omega_{pe}$)',fontsize=18)
plt.ylabel(r'$\int |E|^2 dx$',fontsize=18)
plt.legend(fontsize=16)
plt.yscale('log')
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.tight_layout()
plt.savefig('objective_function_onerun_twostream.pdf', dpi=300)
plt.show()
############### -------- #################
ion_temperature_over_electron_temperature = jnp.logspace(start=minimum_Ti,stop=maximum_Ti,num=nTi)
energy_array =[]
print(f'Perform a parameter scan with {len(ion_temperature_over_electron_temperature)} values to see the variation of the objective function')
for i, Ti in enumerate(ion_temperature_over_electron_temperature):
    objective = objective_function(Ti)
    print(f' Iteration {i}/{len(ion_temperature_over_electron_temperature)} with Ti/Te={Ti:.3f} has objective={objective:.3f}')
    energy_array.append(objective)
############### -------- #################
print(f'Perform a simple optimization with {max_iterations_optimization} iterations')
## Using Least Squares
res = least_squares(objective_function, x0=x0_optimization, diff_step=learning_rate, verbose=2, max_nfev=max_iterations_optimization)
optimized_Ti = res.x[0]
## Using OPTAX
# import optax
# import time
# optimizer = optax.adam(learning_rate)
# params = jnp.array([x0_optimization])
# opt_state = optimizer.init(params)
# for i in range(max_iterations_optimization):
#   start_time = time.time()
#   gradient = grad(objective_function)(params)
#   updates, opt_state = optimizer.update(gradient, opt_state)
#   params = optax.apply_updates(params, updates)
#   print(f' Optimization iteration {i}/{max_iterations_optimization} took {(time.time()-start_time):.3f}s, has Ti={params[0]:3f}')
# optimized_Ti = params[0]
## Solution
print(f' Minimum at Ti={optimized_Ti:.3f} with objective={objective_function(optimized_Ti):.3f}')
############### -------- #################
plt.figure(figsize=(8,6))
plt.plot(ion_temperature_over_electron_temperature, energy_array, label='Objective function landscape')
plt.axvline(x=x0_optimization, color='b', linestyle='-', label='Initial optimization condition')
plt.axvline(x=optimized_Ti, color='r', linestyle='-', label='Optimization result')
plt.xlabel(r'$T_i/T_e$',fontsize=18)
plt.ylabel(r'$\left<\int |E|^2 dx\right>$',fontsize=18)
plt.xscale('log')
plt.legend(fontsize=16)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.tight_layout()
plt.savefig('objective_function_landscape_twostream.pdf', dpi=300)
plt.show()



