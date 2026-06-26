## scaling_time.py
import time
from copy import deepcopy

from jaxincell import Simulation
from jaxincell import diagnostics
from jax import block_until_ready
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.stats import linregress
import matplotlib

####################################################################################################

parameters = {
    "domain_parameters": {
        "length": 1e-2,
        "timestep_over_spatialstep_times_c": 0.3,
        "number_grid_points": 60,
        "total_steps": 1500,
    },
    "species_parameters": {
        "electrons": {
            "electrons0": {
                "number_pseudoparticles": 3500,
                "grid_points_per_Debye_length": 2,
                "perturbation_amplitude_x": 1e-7,
                "perturbation_wavenumber_x": 8,
                "vth_over_c_x": 0.05,
                "drift_speed_x": 1e8,
                "velocity_plus_minus_x": True,
            },
        },
        "ions": {
            "ions0": {
                "number_pseudoparticles": 3500,
                "grid_points_per_Debye_length": 2,
                "vth_over_c_x": "_electrons0",
                "vth_over_c_y": "_electrons0",
                "vth_over_c_z": "_electrons0",
                "ion_temperature_over_electron_temperature_x": 1e-2,
            },
        },
    },
    "external_field_parameters": {
        "external_electric_field_amplitude": 0,
        "external_electric_field_wavenumber": 0,
    },
    "solver_parameters": {
        "field_solver": 0,
        "print_info": False,
    },
}

increase_factor = 1.5 # Factor by which to increase the parameters
number_of_steps = 4  # Number of steps for scaling

domain_parameters = parameters["domain_parameters"]
electron_parameters = parameters["species_parameters"]["electrons"]["electrons0"]

grid_points_list = [
    int(domain_parameters["number_grid_points"] * (increase_factor ** i))
    for i in range(number_of_steps)
]
pseudoelectrons_list = [
    int(electron_parameters["number_pseudoparticles"] * (increase_factor ** i))
    for i in range(number_of_steps)
]
steps_list = [
    int(domain_parameters["total_steps"] * (increase_factor ** i))
    for i in range(number_of_steps)
]
timestep_list = [
    domain_parameters["timestep_over_spatialstep_times_c"] * (increase_factor ** i)
    for i in range(number_of_steps)
]

####################################################################################################

def run_simulation(parameter_tree):
    return block_until_ready(Simulation(parameter_tree).run())


# Function to compute the maximum relative energy error
def max_relative_energy_error(output):
    diagnostics(output)
    relative_energy_error = jnp.abs((output["total_energy"] - output["total_energy"][0]) / output["total_energy"][0])
    return jnp.max(relative_energy_error)


def set_parameter(parameter_tree, param_name, param):
    if param_name == "number_pseudoparticles":
        parameter_tree["species_parameters"]["electrons"]["electrons0"][param_name] = param
        parameter_tree["species_parameters"]["ions"]["ions0"][param_name] = param
        return

    for section_name in ("domain_parameters", "solver_parameters"):
        if param_name in parameter_tree[section_name]:
            parameter_tree[section_name][param_name] = param
            return

    raise KeyError(f"Unknown scaling parameter {param_name!r}.")


# Fuction to measure time and error
def measure_time_and_error(parameter_list, param_name):
    times = []
    max_relative_errors = []
    for j, param in enumerate(parameter_list):
        parameter_tree = deepcopy(parameters)
        set_parameter(parameter_tree, param_name, param)
        parameter_tree["species_parameters"]["electrons"]["electrons0"]["number_pseudoparticles"] += j
        parameter_tree["species_parameters"]["ions"]["ions0"]["number_pseudoparticles"] += j

        start = time.time()
        output = run_simulation(parameter_tree)
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
base_simulation = run_simulation(parameters)

print(f"\n\nRun scaling tests with {number_of_steps} steps and increase factor {increase_factor}: time vs 1) number of grid points, 2) pseudoelectrons, 3) steps, and 4) timestep factor")

print('\n\nMeasure time vs number of grid points')
times_grid_points, max_grid_points_relative_energy_error_array = measure_time_and_error(grid_points_list, 'number_grid_points')

print('\n\nMeasure time vs number of pseudoelectrons')
times_pseudoelectrons, max_pseudoelectrons_relative_energy_error_array = measure_time_and_error(pseudoelectrons_list, 'number_pseudoparticles')

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
plot_and_save_results(x_data_list, time_y_data_list, x_labels, [time_y_label] * 4, 'scaling_time.pdf')

# Plotting the relative energy error
plot_and_save_results(x_data_list, error_y_data_list, x_labels, [error_y_label] * 4, 'scaling_energy_error.pdf')

plt.show()
