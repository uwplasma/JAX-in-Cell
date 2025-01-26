## scaling_time.py
# NOTE: run this script from the root directory with `python tests/scaling_time.py`
import os, sys;
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
from simulation import simulation
from jax import block_until_ready
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

parameters_float = {
"length"                       : 1e-2,  # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 1e-4,  # amplitude of sinusoidal perturbation in x
"wavenumber_perturbation_x_factor": 10, # Wavenumber of sinusoidal (sin) perturbation in x (factor of 2pi/length)
"grid_points_per_Debye_length" : 2,     # dx over Debye length
"vth_electrons_over_c"         : 0.05,  # thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature": 1e-2, # Temperature of ions over temperature of electrons
"CFL_factor"                   : 0.5,   # dt * speed_of_light / dx
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

field_solver           = 0    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
number_grid_points     = 100  # Number of grid points
number_pseudoelectrons = 1500 # Number of pseudoelectrons
total_steps            = 400  # Total number of time steps

print(f"Run 1 simulation for JIT compilation")
block_until_ready(simulation(parameters_float, number_grid_points=number_grid_points,
                  number_pseudoelectrons=number_pseudoelectrons, total_steps=total_steps, field_solver=field_solver))

grid_points_list     = [number_grid_points, number_grid_points*2, number_grid_points*4, number_grid_points*8]
pseudoelectrons_list = [number_pseudoelectrons, number_pseudoelectrons*2, number_pseudoelectrons*4, number_pseudoelectrons*8]
steps_list           = [total_steps, total_steps*2, total_steps*4, total_steps*8]

print('\n\nMeasure time vs number of grid points')
times_grid_points = []
for grid_points in grid_points_list:
    start = time.time()
    output = block_until_ready(simulation(parameters_float, number_grid_points=grid_points+1, field_solver=field_solver,
                                          number_pseudoelectrons=number_pseudoelectrons, total_steps=total_steps))
    elapsed_time = time.time() - start
    times_grid_points.append(elapsed_time)
    print(f"Grid points: {grid_points}, Time: {elapsed_time}s")
    
print('\n\nMeasure time vs number of pseudoelectrons')
times_pseudoelectrons = []
for pseudoelectrons in pseudoelectrons_list:
    start = time.time()
    output = block_until_ready(simulation(parameters_float, number_grid_points=number_grid_points+2, field_solver=field_solver,
                                          number_pseudoelectrons=pseudoelectrons, total_steps=total_steps))
    elapsed_time = time.time() - start
    times_pseudoelectrons.append(elapsed_time)
    print(f"Pseudoelectrons: {pseudoelectrons}, Time: {elapsed_time}s")

print('\n\nMeasure time vs number of steps')
times_steps = []
for steps in steps_list:
    start = time.time()
    output = block_until_ready(simulation(parameters_float, number_grid_points=number_grid_points+3, field_solver=field_solver,
                                          number_pseudoelectrons=number_pseudoelectrons, total_steps=steps))
    elapsed_time = time.time() - start
    times_steps.append(elapsed_time)
    print(f"Steps: {steps}, Time: {elapsed_time}s")

# Plotting the results
fig, axs = plt.subplots(1, 3, figsize=(8, 4))
# Plot time vs number of grid points
axs[0].plot(grid_points_list, times_grid_points, marker='o', label='Simulation Time')
axs[0].set_xlabel('Number of Grid Points')
axs[0].set_ylabel('Wall Clock Time (s)')
slope, intercept, r_value, p_value, std_err = linregress(np.log(grid_points_list), np.log(times_grid_points))
fit_line = [slope * x + intercept for x in np.log(grid_points_list)]
axs[0].plot(grid_points_list, np.exp(fit_line), 'k--', label=f'Linear Fit')
axs[0].legend(loc='upper left')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].grid(True, which="both", ls="-")

# Plot time vs number of pseudoelectrons
axs[1].plot(pseudoelectrons_list, times_pseudoelectrons, marker='o', label='Simulation Time')
axs[1].set_xlabel('Number of Particles')
axs[1].set_ylabel('Wall Clock Time (s)')
slope, intercept, r_value, p_value, std_err = linregress(np.log(pseudoelectrons_list), np.log(times_pseudoelectrons))
fit_line = [slope * x + intercept for x in np.log(pseudoelectrons_list)]
axs[1].plot(pseudoelectrons_list, np.exp(fit_line), 'k--', label=f'Linear Fit')
axs[1].legend(loc='upper left')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].grid(True, which="both", ls="-")

# Plot time vs number of steps
axs[2].plot(steps_list, times_steps, marker='o', label='Simulation Time')
axs[2].set_xlabel('Number of Time Steps')
axs[2].set_ylabel('Wall Clock Time (s)')
slope, intercept, r_value, p_value, std_err = linregress(np.log(steps_list), np.log(times_steps))
fit_line = [slope * x + intercept for x in np.log(steps_list)]
axs[2].plot(steps_list, np.exp(fit_line), 'k--', label=f'Linear Fit')
axs[2].legend(loc='upper left')
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].grid(True, which="both", ls="-")

# Set the same y-axis limits for all plots
all_times = times_grid_points + times_pseudoelectrons + times_steps
for ax in axs: ax.set_ylim([min(all_times), max(all_times)])

plt.tight_layout()
plt.savefig('tests/scaling_time.pdf', dpi=300)
plt.show()



