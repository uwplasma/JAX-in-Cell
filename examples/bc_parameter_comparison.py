## bc_parameter_comparison.py
# Compares BC=3 (mixed, static weight) at three different weight values and BC=4
# (velocity-dependent weight). Runs all simulations and show kinetic energy
# comparison.
from jaxincell import plot, simulation, diagnostics
from jax import block_until_ready
import matplotlib.pyplot as plt
import numpy as np

####################################################################################################

input_parameters = {
    "length"                                        : 1,     # dimensions of the simulation box in (x, y, z)
    "vth_electrons_over_c_x"                        : 0.1,   # thermal velocity of electrons over speed of light
    "ion_temperature_over_electron_temperature_x"   : 1e-9,  # cold ions (fixed neutralizing background)
    "ion_mass_over_proton_mass"                     : 1e9,   # heavy ions (essentially stationary)
    "timestep_over_spatialstep_times_c"             : 0.5,   # dt * speed_of_light / dx
    "field_BC_left"                                 : 1,     # Dirichlet (E=0) at left wall
    "field_BC_right"                                : 1,     # Dirichlet (E=0) at right wall
    "print_info"                                    : True,  # print information about the simulation
}

solver_parameters = {
    "field_solver"           : 0,    # Algorithm to solve E and B fields - 0: Curl_EB, 1: Gauss_1D_FFT, 2: Gauss_1D_Cartesian, 3: Poisson_1D_FFT
    "number_grid_points"     : 32,   # Number of grid points
    "number_pseudoelectrons" : 5000, # Number of pseudoelectrons
    "total_steps"            : 500,  # Total number of time steps
}

configs = [
    {"label": "BC=3, weight=0.25", "particle_BC_left": 3, "particle_BC_right": 3, "mixed_BC_weight": 0.25},
    {"label": "BC=3, weight=0.50", "particle_BC_left": 3, "particle_BC_right": 3, "mixed_BC_weight": 0.50},
    {"label": "BC=3, weight=0.75", "particle_BC_left": 3, "particle_BC_right": 3, "mixed_BC_weight": 0.75},
    {"label": "BC=4 (vel-dep.)",   "particle_BC_left": 4, "particle_BC_right": 4},
]

####################################################################################################

outputs = []
for cfg in configs:
    label = cfg.pop("label")
    print(f"\nRunning {label}...")
    params = {**input_parameters, **cfg}
    output = block_until_ready(simulation(params, **solver_parameters))
    diagnostics(output)
    output["_label"] = label
    outputs.append(output)

####################################################################################################

# Kinetic energy comparison across all configurations
plt.figure(figsize=(8, 5))
for output in outputs:
    t  = np.asarray(output["time_array"]) * float(output["plasma_frequency"])
    ke = np.asarray(output["kinetic_energy_electrons"])
    plt.plot(t, ke / ke[0], label=output["_label"])
plt.xlabel(r"$t\,\omega_{pe}$", fontsize=13)
plt.ylabel(r"$KE_e\,/\,KE_e(0)$", fontsize=13)
plt.title("Electron kinetic energy over time", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("bc_comparison_KE.pdf", dpi=300)
plt.show()

####################################################################################################

# Full animated plot for each configuration (close each window to advance)
for output in outputs:
    print(f"\nPlotting {output['_label']} (close window to continue)...")
    plot(output, animation_interval=20)
