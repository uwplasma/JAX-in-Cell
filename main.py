import jax.numpy as jnp
from jax.random import PRNGKey, uniform
from sources import current_density
from boundary_conditions import set_BC_positions, set_BC_particles
from particles import fields_to_particles_grid, boris_step
from fields import field_update1, field_update2
from jax import vmap

# Constants and simulation parameters
box_size = (1e-2, 1e-2, 1e-2)  # Dimensions of the simulation box (x, y, z)
dx = 5e-4  # Grid spacing
grid = jnp.arange(-box_size[0] / 2 + dx / 2, box_size[0] / 2 + dx / 2, dx)  # Grid points
staggered_grid = grid + dx / 2  # For staggered grid calculations
no_pseudoelectrons = 5000  # Number of pseudoelectrons
A = 0.1  # Amplitude of sinusoidal perturbation
L = box_size[0]  # Length in the x-direction
k = 2 * jnp.pi / L  # Wave number
seed = 1701  # Random seed for reproducibility

# Initial positions of pseudoelectrons and ions
key = PRNGKey(seed)
xs = jnp.linspace(-L / 2, L / 2, no_pseudoelectrons)
electron_xs = xs - (A / k) * jnp.sin(k * xs)  # Perturbed electron positions
electron_ys = uniform(key, shape=(1, no_pseudoelectrons), minval=-box_size[1] / 2, maxval=box_size[1] / 2)
electron_zs = uniform(key, shape=(1, no_pseudoelectrons), minval=-box_size[2] / 2, maxval=box_size[2] / 2)
electron_positions = jnp.stack((electron_xs, electron_ys[0], electron_zs[0]), axis=1)

ion_positions = jnp.stack(
    (xs, uniform(key, shape=(1, no_pseudoelectrons), minval=-box_size[1] / 2, maxval=box_size[1] / 2)[0],
     uniform(key, shape=(1, no_pseudoelectrons), minval=-box_size[2] / 2, maxval=box_size[2] / 2)[0]), axis=1
)

# Combine positions for all particles
particle_positions = jnp.concatenate((electron_positions, ion_positions), axis=0)
no_pseudoparticles = particle_positions.shape[0]

# Initial velocities (all zero)
particle_velocities = jnp.zeros((no_pseudoparticles, 3))

# Particle properties
w0 = jnp.pi * 3e8 / (25 * dx)  # Angular frequency
weight = 3.15e-4 * w0**2 * L / no_pseudoelectrons  # Weight from plasma frequency

# Charges and masses
q_electron = -1.6e-19 * weight
q_ion = 1.6e-19 * weight
q_mes = -1.76e11  # Electron charge-to-mass ratio
q_mps = 9.56e7  # Ion charge-to-mass ratio

charges = jnp.concatenate((q_electron * jnp.ones((no_pseudoelectrons, 1)),
                           q_ion * jnp.ones((no_pseudoelectrons, 1))), axis=0)
masses = jnp.concatenate((9.1e-31 * weight * jnp.ones((no_pseudoelectrons, 1)),
                          1.67e-27 * weight * jnp.ones((no_pseudoelectrons, 1))), axis=0)
charge_to_mass_ratios = jnp.concatenate((q_mes * jnp.ones((no_pseudoelectrons, 1)),
                                         q_mps * jnp.ones((no_pseudoelectrons, 1))), axis=0)

# Initialize electric and magnetic fields
E_fields = jnp.zeros((grid.size, 3))
for i in range(grid.size):
    E_fields = E_fields.at[i, 0].set(
        q_electron * no_pseudoelectrons * A * jnp.sin(k * (grid[i] + dx / 2)) / (k * L * 8.85e-12)
    )
B_fields = jnp.zeros((grid.size, 3))
fields = (E_fields, B_fields)

# External fields
ext_E = jnp.zeros_like(E_fields)
ext_B = jnp.zeros_like(B_fields)
ext_fields = (ext_E, ext_B)

# Time step and simulation parameters
dt = dx / (2 * 3e8)  # Time step
total_steps = 10

# Boundary conditions
particle_BC_left = 0  # Left boundary condition type
particle_BC_right = 0  # Right boundary condition type
field_BC_left = 0  # Left boundary condition type
field_BC_right = 0  # Right boundary condition type

# time
t = 0

# Initial boundary adjustment for particles
xs_n = particle_positions
xs_nplushalf, vs_n, qs, ms, q_ms = set_BC_particles(
    particle_positions + (dt / 2) * particle_velocities, particle_velocities, charges, masses, charge_to_mass_ratios,
    dx, grid, *box_size, particle_BC_left, particle_BC_right
)
xs_nminushalf = set_BC_positions(particle_positions - (dt / 2) * particle_velocities, charges, dx, grid, *box_size, particle_BC_left, particle_BC_right)

for i in range(total_steps):
    # Compute current density
    J = current_density(xs_nminushalf, xs_n, xs_nplushalf, vs_n, qs, dx, dt, grid, grid[0]-dx/2, particle_BC_left, particle_BC_right)

    # Compute the electric and magnetic fields
    E_fields, B_fields = field_update1(E_fields,B_fields,dx,dt/2,J,field_BC_left,field_BC_right,t)

    total_E = E_fields+ext_E
    total_B = B_fields+ext_B

    E_fields_at_x = vmap(lambda x_n: fields_to_particles_grid(x_n,total_E,dx,staggered_grid,grid[0]     , particle_BC_left,particle_BC_right))(xs_nplushalf)
    B_fields_at_x = vmap(lambda x_n: fields_to_particles_grid(x_n,total_B,dx,grid,          grid[0]-dx/2, particle_BC_left,particle_BC_right))(xs_nplushalf)

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