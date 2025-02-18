## scaling_time.py
import os, sys;
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
from jaxincell import simulation
from jaxincell import diagnostics
from jax import block_until_ready
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.stats import linregress
import matplotlib

####################################################################################################

input_parameters = {
"length"                       : 1e-2,  # dimensions of the simulation box in (x, y, z)
"amplitude_perturbation_x"     : 5e-4,  # amplitude of sinusoidal perturbation in x
"wavenumber_electrons": 8, # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
"grid_points_per_Debye_length" : 2,     # dx over Debye length
"vth_electrons_over_c"         : 0.05,  # thermal velocity of electrons over speed of light
"ion_temperature_over_electron_temperature": 1e-2, # Temperature of ions over temperature of electrons
"timestep_over_spatialstep_times_c": 0.2, # dt * speed_of_light / dx
"electron_drift_speed"         : 0,     # drift speed of electrons
"velocity_plus_minus_electrons": False, # create two groups of electrons moving in opposite directions
"print_info"                   : False,  # print information about the simulation
"external_electric_field_amplitude": 0, # External electric field value (V/m)
"external_electric_field_wavenumber": 0,  # External electric Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
"amplitude_perturbation_x"     : 1e-7,  # Two-Stream (amplitude of sinusoidal perturbation in x)
"electron_drift_speed"         : 1e8,   # Two-Stream (drift speed of electrons)
"velocity_plus_minus_electrons": True,  # Two-Stream (create two groups of electrons moving in opposite directions)
# "wavenumber_electrons": 1,  # Plasma Oscillations (Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length))
}

solver_parameters = {
    "field_solver"           : 1,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT, 
    "number_grid_points"     : 50,   # Number of grid points
    "number_pseudoelectrons" : 1500, # Number of pseudoelectrons
    "total_steps"            : 2000, # Total number of time steps
}

increase_factor = 1.5
grid_points_list     = [int(solver_parameters["number_grid_points"] * (increase_factor ** i)) for i in range(4)]
pseudoelectrons_list = [int(solver_parameters["number_pseudoelectrons"] * (increase_factor ** i)) for i in range(4)]
steps_list           = [int(solver_parameters["total_steps"] * (increase_factor ** i)) for i in range(4)]
timestep_list      = [input_parameters["timestep_over_spatialstep_times_c"] * (increase_factor ** i) for i in range(4)]

####################################################################################################

# Function to compute the maximum relative energy error
def max_relative_energy_error(output):
    diagnostics(output, print_to_terminal=False)
    relative_energy_error = jnp.abs((output["total_energy"] - output["total_energy"][0]) / output["total_energy"][0])
    return jnp.max(relative_energy_error)

# Fuction to measure time and error
def measure_time_and_error(parameter_list, param_name):
    times = []
    max_relative_errors = []
    for j, param in enumerate(parameter_list):
        start = time.time()
        if param_name in solver_parameters:
            old_param = solver_parameters[param_name]
            solver_parameters[param_name] = param
            solver_parameters['number_pseudoelectrons'] = int(solver_parameters['number_pseudoelectrons']+j)
            output = block_until_ready(simulation(input_parameters, **solver_parameters))
            solver_parameters[param_name] = old_param
        if param_name in input_parameters:
            old_param = input_parameters[param_name]
            # if 'timestep_over_spatialstep_times_c' in param_name:
            #     old_time_steps = solver_parameters['total_steps']
            #     old_grid_points = solver_parameters['number_grid_points']
            #     solver_parameters['total_steps'] = int(solver_parameters['total_steps'] * param / timestep_list[0])
            #     solver_parameters['number_grid_points'] = int(solver_parameters['number_grid_points'] * param / input_parameters['timestep_over_spatialstep_times_c'])
            input_parameters[param_name] = param
            solver_parameters['number_pseudoelectrons'] = int(solver_parameters['number_pseudoelectrons']+j)
            output = block_until_ready(simulation(input_parameters, **solver_parameters))
            input_parameters[param_name] = old_param
            # if 'timestep_over_spatialstep_times_c' in param_name:
            #     solver_parameters['total_steps'] = old_time_steps
            #     solver_parameters['number_grid_points'] = old_grid_points
        elapsed_time = time.time() - start
        times.append(elapsed_time)
        max_relative_errors.append(max_relative_energy_error(output))
        print(f"{param_name.capitalize()}: {param}, Time: {elapsed_time}s")
    return times, max_relative_errors

# Function to plot results
def plot_results(ax, x_data, y_data, xlabel, ylabel):
    fmt = matplotlib.ticker.StrMethodFormatter("{x:g}")
    ax.plot(x_data, y_data, marker='o', label='Simulation')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    slope, intercept, r_value, p_value, std_err = linregress(jnp.log(jnp.array(x_data)), jnp.log(jnp.array(y_data)))
    fit_line = [slope * x + intercept for x in jnp.log(jnp.array(x_data))]
    ax.plot(x_data, jnp.exp(jnp.array(fit_line)), 'k--', label=f'Linear Fit')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_minor_formatter(fmt)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis="x", which="both", rotation=60)

####################################################################################################

print(f"Run 1 simulation for JIT compilation")
base_simulation = block_until_ready(simulation(input_parameters, **solver_parameters))

print('\n\nMeasure time vs number of grid points')
times_grid_points, max_grid_points_relative_energy_error_array = measure_time_and_error(grid_points_list, 'number_grid_points')

print('\n\nMeasure time vs number of pseudoelectrons')
times_pseudoelectrons, max_pseudoelectrons_relative_energy_error_array = measure_time_and_error(pseudoelectrons_list, 'number_pseudoelectrons')

print('\n\nMeasure time vs number of steps')
times_steps, max_steps_relative_energy_error_array = measure_time_and_error(steps_list, 'total_steps')

print('\n\nMeasure time vs timestep factor')
times_timestep, max_timestep_relative_energy_error_array = measure_time_and_error(timestep_list, 'timestep_over_spatialstep_times_c')

####################################################################################################

# Function to plot and save results
def plot_and_save_results(x_data_list, y_data_list, x_labels, y_labels, file_name):
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    for i, (x_data, y_data, x_label, y_label) in enumerate(zip(x_data_list, y_data_list, x_labels, y_labels)):
        plot_results(axs[i], x_data, y_data, x_label, y_label)
        if i > 0:
            axs[i].axes.get_yaxis().set_visible(False)
        axs[i].legend()
    all_y_data = [item for sublist in y_data_list for item in sublist]
    for ax in axs:
        ax.set_ylim([0.9 * min(all_y_data), 1.1 * max(all_y_data)])
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)

# Data for plotting
x_data_list = [grid_points_list, pseudoelectrons_list, steps_list, timestep_list]
time_y_data_list = [times_grid_points, times_pseudoelectrons, times_steps, times_timestep]
error_y_data_list = [max_grid_points_relative_energy_error_array, max_pseudoelectrons_relative_energy_error_array, max_steps_relative_energy_error_array, max_timestep_relative_energy_error_array]
x_labels = ['Number of Grid Points', 'Number of Particles', 'Number of Time Steps', 'Time Step Over Spatial Step Times c']
time_y_label = 'Wall Clock Time (s)'
error_y_label = 'Max Relative Energy Error'

# Plotting the time scaling
plot_and_save_results(x_data_list, time_y_data_list, x_labels, [time_y_label] * 4, 'tests/scaling_time.pdf')

# Plotting the relative energy error
plot_and_save_results(x_data_list, error_y_data_list, x_labels, [error_y_label] * 4, 'tests/scaling_energy_error.pdf')

plt.show()