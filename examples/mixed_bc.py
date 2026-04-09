## mixed_bc.py
# Example of mixed boundary conditions (BC=3): at each wall collision, a fraction
# of the macroparticle reflects and the rest is absorbed, controlled by mixed_BC_weight.
# mixed_BC_weight=1.0 is equivalent to fully reflective (BC=1);
# mixed_BC_weight=0.0 is equivalent to fully absorbing (BC=2).
from jaxincell import plot
from jaxincell import simulation, diagnostics
import jax.numpy as jnp
from jax import block_until_ready

input_parameters = {
    "length"                                        : 1,     # dimensions of the simulation box in (x, y, z)
    "vth_electrons_over_c_x"                        : 0.1,   # thermal velocity of electrons over speed of light
    "ion_temperature_over_electron_temperature_x"   : 1e-9,  # cold ions (fixed neutralizing background)
    "ion_mass_over_proton_mass"                     : 1e9,   # heavy ions (essentially stationary)
    "timestep_over_spatialstep_times_c"             : 0.5,   # dt * speed_of_light / dx
    "particle_BC_left"                              : 3,     # mixed BC at left wall
    "particle_BC_right"                             : 3,     # mixed BC at right wall
    "field_BC_left"                                 : 1,     # Dirichlet (E=0) at left wall
    "field_BC_right"                                : 1,     # Dirichlet (E=0) at right wall
    "mixed_BC_weight"                               : 0.5,   # fraction of each macroparticle reflected; remainder absorbed
    "print_info"                                    : True,  # print information about the simulation
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT
    "number_grid_points"     : 32,   # Number of grid points
    "number_pseudoelectrons" : 5000, # Number of pseudoelectrons
    "total_steps"            : 500,  # Total number of time steps
}

output = block_until_ready(simulation(input_parameters, **solver_parameters))

# Post-process: segregate ions/electrons, compute energies, compute FFT
diagnostics(output)

plot(output)
