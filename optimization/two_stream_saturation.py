## two_stream_saturation.py
# Optimize the non-linear saturation of the two_stream_instabiltity to be as small as possible
import os, sys;
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulation import simulation
from diagnostics import diagnostics
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

input_parameters = {
"length"                       : 1e-2,  # dimensions of the simulation box in (x, y, z)
"wavenumber_perturbation_x_factor": 8, # Wavenumber of sinusoidal (sin) perturbation in x (factor of 2pi/length)
"grid_points_per_Debye_length" : 2,     # dx over Debye length
"vth_electrons_over_c"         : 0.05,  # thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature": 1e-2, # Temperature of ions over temperature of electrons
"timestep_over_spatialstep_times_c": 1.0, # dt * speed_of_light / dx
"print_info"                   : False,  # print information about the simulation
"external_electric_field_amplitude": 0, # External electric field value (V/m)
"external_electric_field_wavenumber_perturbation_x_factor": 0,  # External electric Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
"amplitude_perturbation_x"     : 1e-7,  # Two-Stream (amplitude of sinusoidal perturbation in x)
"electron_drift_speed"         : 1e8,   # Two-Stream (drift speed of electrons)
"velocity_plus_minus_electrons": True,  # Two-Stream (create two groups of electrons moving in opposite directions)
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 30,  # Number of grid points
    "number_pseudoelectrons" : 2000, # Number of pseudoelectrons
    "total_steps"            : 1000, # Total number of time steps
}

initial_time_over_wpe = 30
minimum_Ti = -2
maximum_Ti = 3
nTi = 10
max_evals_opt = 10
x0_optimization = 0.1

def objective_function(Ti=input_parameters["ion_temperature_over_electron_temperature"]):
    input_parameters["ion_temperature_over_electron_temperature"] = Ti
    output = simulation(input_parameters, **solver_parameters)
    diagnostics(output, print_to_terminal=False)
    start_index = jnp.argwhere((output['time_array']*output['plasma_frequency'])>initial_time_over_wpe).min()
    return jnp.mean(output['electric_field_energy'][start_index:])

print(f'Perform a first run to see one objective function')
output = simulation(input_parameters, **solver_parameters)
diagnostics(output, print_to_terminal=False)
objective = objective_function()
plt.figure(figsize=(8,6))
plt.plot(output['time_array']*output['plasma_frequency'],output['electric_field_energy'], label='Electric Field Energy')
plt.axhline(y=objective, color='r', linestyle='-', label='Mean value used for optimization')
plt.axvline(x=initial_time_over_wpe, color='b', linestyle='-', label='Initial time used for mean value')
plt.xlabel(r't (1/$\omega_{pe}$)',fontsize=18)
plt.ylabel(r'$\int |E|^2 dx$',fontsize=18)
plt.legend(fontsize=16)
plt.yscale('log')
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.tight_layout()
plt.show()

ion_temperature_over_electron_temperature = jnp.logspace(start=minimum_Ti,stop=maximum_Ti,num=nTi)
energy_array =[]
print(f'Perform a parameter scan with {len(ion_temperature_over_electron_temperature)} values to see the variation of the objective function')
for i, Ti in enumerate(ion_temperature_over_electron_temperature):
    objective = objective_function(Ti)
    print(f' Iteration {i}/{len(ion_temperature_over_electron_temperature)} with Ti/Te={Ti} has objective={objective}')
    energy_array.append(objective)

print(f'Perform a simple optimization with {max_evals_opt} function evaluations')
res = least_squares(objective_function, x0=x0_optimization, verbose=2, max_nfev=max_evals_opt)
print(f' Minimum at Ti={res.x} with objective={objective_function(res.x)}')

plt.figure(figsize=(8,6))
plt.plot(ion_temperature_over_electron_temperature, energy_array, label='Objective function landscape')
plt.axvline(x=res.x, color='b', linestyle='-', label='Initial optimization condition')
plt.axvline(x=res.x, color='r', linestyle='-', label='Optimization result')
plt.xlabel(r'$T_i/T_e$',fontsize=18)
plt.ylabel(r'$\left<\int |E|^2 dx\right>$',fontsize=18)
plt.xscale('log')
plt.legend(fontsize=16)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.tight_layout()
plt.show()



