## two_stream_saturation.py
# Optimize the non-linear saturation of the two_stream_instabiltity to be as small as possible
import os, sys;
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax.numpy as jnp
from jax import jit, grad
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy.optimize import least_squares
from jaxincell import simulation, diagnostics, epsilon_0, load_parameters, plot, speed_of_light
from jax.random import PRNGKey, uniform, normal
from jax.debug import print as jprint
from jaxopt import ScipyMinimize
from scipy.optimize import minimize

# Define the target phase space from .png image
rho_target = jnp.flipud(jnp.array(img.imread('target.png')[:,:,0],dtype=float))
rho_target = 1.0 + 0.02*(rho_target-0.5)
# normalize so average density is 1
rho_target /= jnp.mean(rho_target)

number_grid_points = rho_target.shape[0]

# Read from input.toml
input_parameters, solver_parameters = load_parameters('input.toml')

############### -------- #################
## Objective function to minimize

vth_electrons = input_parameters["vth_electrons_over_c"] * speed_of_light
vth_ions = jnp.sqrt(jnp.abs(input_parameters["ion_temperature_over_electron_temperature"])) * vth_electrons * jnp.sqrt(1836)
random_key = PRNGKey(42)
number_pseudoelectrons = solver_parameters["number_pseudoelectrons"]
length = input_parameters["length"]
amplitude_perturbation = 1e-3#input_parameters["amplitude_perturbation_x"]
wavenumber_electrons = input_parameters["wavenumber_electrons"]
wavenumber_ions      = input_parameters["wavenumber_ions"]
solver_parameters["total_steps"] = 200
solver_parameters["number_grid_points"] = number_grid_points
input_parameters["print_info"] = False

def initialize_particles(number_pseudoelectrons, length, amplitude_perturbation, vth, random_key, wave_number):
    xs = jnp.linspace(-length / 2, length / 2, number_pseudoelectrons)
    xs += amplitude_perturbation * jnp.sin(2 * jnp.pi * xs / length * wave_number)
    ys = uniform(random_key, shape=(number_pseudoelectrons,), minval=-length / 2, maxval=length / 2)
    zs = uniform(random_key, shape=(number_pseudoelectrons,), minval=-length / 2, maxval=length / 2)
    v_x = vth / jnp.sqrt(2) * normal(random_key, shape=(number_pseudoelectrons,))
    v_y = jnp.zeros((number_pseudoelectrons,))
    v_z = jnp.zeros((number_pseudoelectrons,))
    return jnp.stack((xs, ys, zs), axis=1), jnp.stack((v_x, v_y, v_z), axis=1)
electron_positions, electron_velocities = initialize_particles(number_pseudoelectrons, length, amplitude_perturbation, vth_electrons, random_key, wavenumber_electrons)
ion_positions,      ion_velocities      = initialize_particles(number_pseudoelectrons, length, amplitude_perturbation, vth_ions,      random_key, wavenumber_ions)
positions = jnp.concatenate((electron_positions, ion_positions))
velocities = jnp.concatenate((electron_velocities, ion_velocities))
initial_dofs = jnp.concatenate((positions[:,0], velocities[:,0]))
# print(initial_dofs.shape)
# plt.plot(electron_velocities[:,0])
# plt.plot(initial_dofs[2*number_pseudoelectrons:3*number_pseudoelectrons])
# plt.show()
# exit()

# x = initial_dofs
@jit
def loss_function(x):
    electron_positions, electron_velocities = initialize_particles(number_pseudoelectrons, length, amplitude_perturbation, vth_electrons, random_key, wavenumber_electrons)
    ion_positions,      ion_velocities      = initialize_particles(number_pseudoelectrons, length, amplitude_perturbation, vth_ions,      random_key, wavenumber_ions)
    electron_positions  = electron_positions.at[ :,0].set(x[:number_pseudoelectrons])
    ion_positions       = ion_positions.at[      :,0].set(x[1*number_pseudoelectrons:2*number_pseudoelectrons])
    electron_velocities = electron_velocities.at[:,0].set(x[2*number_pseudoelectrons:3*number_pseudoelectrons])
    ion_velocities      = ion_velocities.at[     :,0].set(x[3*number_pseudoelectrons:])
    positions                = jnp.concatenate((electron_positions, ion_positions))
    velocities               = jnp.concatenate((electron_velocities, ion_velocities))
    params = input_parameters.copy()
    output = simulation(params, **solver_parameters, positions=positions, velocities=velocities)
# plot(output)
    max_velocity_electrons = 1.2 * jnp.max(output["velocity_electrons"])
    range_bounds = jnp.array([[-length / 2, length / 2], [-max_velocity_electrons, max_velocity_electrons]])
    electron_phase_space = jnp.histogram2d(
            output["position_electrons"][-1, :, 0],
            output["velocity_electrons"][-1, :, 0],
            bins=[number_grid_points, number_grid_points],
            range=range_bounds
        )[0]
    electron_phase_space = jnp.array(electron_phase_space).T
    electron_phase_space = electron_phase_space - jnp.mean(electron_phase_space)
    electron_phase_space /= jnp.max(electron_phase_space)
    electron_phase_space = 1.0 + 0.02*(electron_phase_space)
    return jnp.mean( (electron_phase_space - rho_target)**2 )


jac = jit(grad(loss_function))
print(jnp.sum(jnp.abs(jac(initial_dofs))))
print('The gradients are all zero')
# exit()

# optimizer = ScipyMinimize(method="l-bfgs-b", fun=loss_function, tol=1e-8, options={'disp': True})
# sol = optimizer.run(initial_dofs)
# optimized_dofs = sol.params

tolerance_optimization = 1e-7
diff_step = 1e-2
maxfev = 30
def callback(xk):
    print(f'Iteration: {callback.iteration}, Loss: {loss_function(xk)}')
    callback.iteration += 1

callback.iteration = 0

sol = minimize(loss_function, x0=initial_dofs, method='Nelder-Mead',
               tol=tolerance_optimization, options={'maxfev': maxfev, 'disp': True}, callback=callback)
optimized_dofs = sol.x

print(f'Initial loss:   {loss_function(initial_dofs)}')
print(f'Optimized loss: {loss_function(optimized_dofs)}')