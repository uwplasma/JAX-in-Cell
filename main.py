import jax.numpy as jnp
from jax.random import PRNGKey, uniform
from sources import current_density, calculate_charge_density
from boundary_conditions import set_BC_positions, set_BC_particles
from particles import fields_to_particles_grid, boris_step
from fields import field_update1, field_update2, E_from_Poisson_equation
from jax import vmap
import matplotlib.pyplot as plt
from tqdm import tqdm
from jax.debug import print as jprint
from jax.random import normal
from constants import speed_of_light, epsilon_0, charge_electron, charge_proton, mass_electron, mass_proton
from jax.numpy.fft import fft, fftfreq

import jax
jax.clear_caches()

# Constants and simulation parameters
box_size = (1e-2, 1e-2, 1e-2)  # Dimensions of the simulation box (x, y, z)
number_grid_points = 50  # Number of grid points
no_pseudoelectrons = 500  # Number of pseudoelectrons
A = 0.1  # Amplitude of sinusoidal perturbation
seed = 1701  # Random seed for reproducibility
CFL_factor = 0.5  # CFL factor = dt * speed_of_light / dx
total_steps = 350  # Total number of time steps
grid_points_per_Debye_length = 9 # dx over Debye length
vth_electrons_over_c = 0.05  # Thermal velocity of electrons over speed of light

# Derived parameters
length = box_size[0]  # Length in the x-direction
dx = length / number_grid_points  # Grid spacing
grid = jnp.arange(-box_size[0] / 2 + dx / 2, box_size[0] / 2 + dx / 2, dx)  # Grid points
k = 2 * jnp.pi / length  # Wave number
dt = CFL_factor * dx / speed_of_light  # Time step
debye_length_per_dx = 1/grid_points_per_Debye_length
weight = (epsilon_0 * mass_electron * speed_of_light**2 / charge_electron**2) * number_grid_points**2 / length / 2 / no_pseudoelectrons * vth_electrons_over_c**2 / debye_length_per_dx**2

# Initial positions of pseudoelectrons and ions
key = PRNGKey(seed)
xs = jnp.array([jnp.linspace(-length / 2, length / 2, no_pseudoelectrons)])
electron_xs = xs - (A / k) * jnp.sin(k * xs)  # Perturbed electron positions
electron_ys = uniform(key, shape=(1, no_pseudoelectrons), minval=-box_size[1] / 2, maxval=box_size[1] / 2)
electron_zs = uniform(key, shape=(1, no_pseudoelectrons), minval=-box_size[2] / 2, maxval=box_size[2] / 2)
electron_positions = jnp.transpose(jnp.concatenate((electron_xs,electron_ys,electron_zs)))

ion_xs = xs
ion_ys = uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size[1]/2,maxval=box_size[1]/2)
ion_zs = uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size[2]/2,maxval=box_size[2]/2)
ion_positions = jnp.transpose(jnp.concatenate((ion_xs,ion_ys,ion_zs)))

# Combine positions for all particles
particle_positions = jnp.concatenate((electron_positions, ion_positions))

# Initial velocities
vth_electrons = vth_electrons_over_c * speed_of_light
vth_ions      = vth_electrons_over_c * speed_of_light*jnp.sqrt(mass_electron/mass_proton)
v_electrons   = vth_electrons / jnp.sqrt(2) * normal(key,shape=(no_pseudoelectrons,3))
v_ions        = vth_ions      / jnp.sqrt(2) * normal(key,shape=(no_pseudoelectrons,3))
particle_velocities = jnp.concatenate((v_electrons,v_ions))

# Particle properties
charges = jnp.concatenate((charge_electron * weight * jnp.ones((no_pseudoelectrons, 1)),
                           charge_proton   * weight * jnp.ones((no_pseudoelectrons, 1))), axis=0)
masses  = jnp.concatenate((mass_electron   * weight * jnp.ones((no_pseudoelectrons, 1)),
                           mass_proton     * weight * jnp.ones((no_pseudoelectrons, 1))), axis=0)
charge_to_mass_ratios = jnp.concatenate((charge_electron / mass_electron * jnp.ones((no_pseudoelectrons, 1)),
                                         charge_proton   / mass_proton   * jnp.ones((no_pseudoelectrons, 1))), axis=0)

# Initialize electric and magnetic fields
# E_fields = jnp.zeros((grid.size, 3))
# for i in range(grid.size):
#     E_fields = E_fields.at[i, 0].set(
#         q_electron * no_pseudoelectrons * A * jnp.sin(k * (grid[i] + dx / 2)) / (k * L * epsilon_0)
#     )
E_fields = jnp.stack((E_from_Poisson_equation(particle_positions,charges,dx,grid,0,0),
                      jnp.zeros_like(grid),
                      jnp.zeros_like(grid)), axis=1)
B_fields = jnp.zeros((grid.size, 3))
fields = (E_fields, B_fields)

# External fields
ext_E = jnp.zeros_like(E_fields)
ext_B = jnp.zeros_like(B_fields)
ext_fields = (ext_E, ext_B)

# Boundary conditions
particle_BC_left = 0  # Left boundary condition type
particle_BC_right = 0  # Right boundary condition type
field_BC_left = 0  # Left boundary condition type
field_BC_right = 0  # Right boundary condition type

# Initial boundary adjustment for particles
xs_n = particle_positions
xs_nplushalf, vs_n, qs, ms, q_ms = set_BC_particles(
    particle_positions + (dt / 2) * particle_velocities, particle_velocities, charges, masses, charge_to_mass_ratios,
    dx, grid, *box_size, particle_BC_left, particle_BC_right
)
xs_nminushalf = set_BC_positions(particle_positions - (dt / 2) * particle_velocities, charges, dx, grid, *box_size, particle_BC_left, particle_BC_right)

# Preallocate the array for storing E fields over time
E_field_over_time  = jnp.zeros((total_steps, len(grid), 3))  # Shape: (total_steps, N, 3)
E_energy_over_time = jnp.zeros((total_steps, ))
charge_density_over_time = jnp.zeros((total_steps, len(grid)))

t=0 # Time initialization (to be removed)
time_array = jnp.linspace(0, total_steps*dt, total_steps)
for i in tqdm(range(total_steps), desc="Simulation Progress"):
    # Compute current density
    J = current_density(xs_nminushalf, xs_n, xs_nplushalf, vs_n, qs, dx, dt, grid, grid[0]-dx/2, particle_BC_left, particle_BC_right)

    # Compute the electric and magnetic fields
    E_fields, B_fields = field_update1(E_fields,B_fields,dx,dt/2,J,field_BC_left,field_BC_right,t)

    total_E = E_fields+ext_E
    total_B = B_fields+ext_B

    E_fields_at_x = vmap(lambda x_n: fields_to_particles_grid(x_n,total_E,dx,grid + dx/2,grid[0]     , particle_BC_left,particle_BC_right))(xs_nplushalf)
    B_fields_at_x = vmap(lambda x_n: fields_to_particles_grid(x_n,total_B,dx,grid,       grid[0]-dx/2, particle_BC_left,particle_BC_right))(xs_nplushalf)

    # Boris step
    xs_nplus3_2,vs_nplus1 = boris_step(dt,xs_nplushalf,vs_n,q_ms,E_fields_at_x,B_fields_at_x)

    # Implement boundary conditions for particles
    xs_nplus3_2,vs_nplus1,qs,ms,q_ms = set_BC_particles(xs_nplus3_2,vs_nplus1,qs,ms,q_ms,dx,grid,*box_size,particle_BC_left,particle_BC_right)
    xs_nplus1 = set_BC_positions(xs_nplus3_2-(dt/2)*vs_nplus1,qs,dx,grid,*box_size,particle_BC_left,particle_BC_right)
    
    #find j from x_n3/2 and x_n1/2
    J = current_density(xs_nplushalf, xs_nplus1, xs_nplus3_2, vs_nplus1, qs, dx, dt, grid, grid[0]-dx/2, particle_BC_left, particle_BC_right)

    #1/2 step E&B field update
    E_fields, B_fields = field_update2(E_fields,B_fields,dx,dt/2,J,field_BC_left,field_BC_right,t)

    # Update variables for next iteration
    xs_nminushalf = xs_nplushalf
    xs_nplushalf = xs_nplus3_2
    vs_n = vs_nplus1
    xs_n = xs_nplus1
    t += dt
    
    # Store the current E field in the preallocated array
    E_field_over_time  = E_field_over_time.at[ i].set(E_fields)
    E_energy_over_time = E_energy_over_time.at[i].set(jnp.sum(0.5*epsilon_0*vmap(jnp.dot)(E_fields,E_fields)/dx))
    charge_density_over_time = charge_density_over_time.at[i].set(calculate_charge_density(xs_n,qs,dx,grid,particle_BC_left,particle_BC_right))

plt.figure()
plt.imshow(E_field_over_time[:, :, 0], aspect='auto', cmap='RdBu', origin='lower', extent=[grid[0], grid[-1], 0, total_steps * dt])
plt.colorbar(label='Electric field (V/m)')
plt.xlabel('Position (m)')
plt.ylabel('Time (s)')

plt.figure()
plt.imshow(charge_density_over_time, aspect='auto', cmap='RdBu', origin='lower', extent=[grid[0], grid[-1], 0, total_steps * dt])
plt.colorbar(label='Charge density (C/m^3)')
plt.xlabel('Position (m)')
plt.ylabel('Time (s)')

plt.figure()
plt.plot(time_array, E_energy_over_time)
plt.xlabel('Time (s)')
plt.ylabel('Electric field energy (J)')
plt.show()

# array_to_do_fft_on = charge_density_over_time[:,len(grid)//2]
array_to_do_fft_on = E_field_over_time[:,len(grid)//2,0]
array_to_do_fft_on = (array_to_do_fft_on-jnp.mean(array_to_do_fft_on))/jnp.max(array_to_do_fft_on)
plasma_frequency = jnp.sqrt(no_pseudoelectrons * weight * charge_electron**2 / (mass_electron * epsilon_0 * length))

fft_values = fft(array_to_do_fft_on)[:total_steps//2]
freqs = fftfreq(total_steps, d=dt)[:total_steps//2]*2*jnp.pi # d=dt specifies the time step
magnitude = jnp.abs(fft_values)
peak_index = jnp.argmax(magnitude)
dominant_frequency = jnp.abs(freqs[peak_index])

print(f"Dominant FFT frequency (f): {dominant_frequency} Hz")
print(f"Plasma frequency (w_p):     {plasma_frequency} Hz")
print(f"Error: {jnp.abs(dominant_frequency - plasma_frequency) / plasma_frequency * 100:.2f}%")