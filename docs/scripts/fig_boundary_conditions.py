"""Particle boundary conditions: a drifting quasi-neutral plasma with periodic,
reflective and absorbing walls, shown through the electron density n_e(x, t)."""
import numpy as np
import matplotlib.pyplot as plt
from jax import block_until_ready

from common import CMAP_DENSITY, panel_label, savefig, silence_progress_bars
from jaxincell import Simulation, speed_of_light

silence_progress_bars()


def parameters_for(bc):
    return {
        "domain_parameters": {"length": 1e-2, "timestep_over_spatialstep_times_c": 1.0,
                              "number_grid_points": 64, "total_steps": 500,
                              "particle_BC_left": bc, "particle_BC_right": bc,
                              "field_BC_left": bc, "field_BC_right": bc},
        "species_parameters": {
            "electrons": {"electrons0": {
                "number_pseudoparticles": 20000, "grid_points_per_Debye_length": 1.0,
                "perturbation_amplitude_x": 0.0, "perturbation_wavenumber_x": 0,
                "random_positions_x": True, "vth_over_c_x": 0.005, "drift_speed_x": 0.05 * speed_of_light,
                "velocity_plus_minus_x": False}},
            "ions": {"ions0": {
                "number_pseudoparticles": 20000, "grid_points_per_Debye_length": 1.0,
                "random_positions_x": True, "mass_over_proton_mass": 1.0,
                "vth_over_c_x": "_electrons0", "vth_over_c_y": "_electrons0", "vth_over_c_z": "_electrons0",
                "ion_temperature_over_electron_temperature_x": 1.0,
                "drift_speed_x": 0.05 * speed_of_light}}},
        "solver_parameters": {"field_solver": 0, "filter_passes": 0, "print_info": False},
    }


fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, (bc, name) in zip(axes, ((0, "periodic"), (1, "reflective"), (2, "absorbing"))):
    output = block_until_ready(Simulation(parameters_for(bc)).run())
    wpe = float(output["plasma_frequency"])
    t = np.asarray(output["time_array"]) * wpe
    L = float(output["length"])
    n_e = int(output["number_pseudoelectrons"])
    x = np.asarray(output["positions"][:, :n_e, 0])
    edges = np.linspace(-L / 2, L / 2, 65)
    # absorbed particles are parked outside the box with zero charge; the
    # histogram range excludes them automatically
    density = np.array([np.histogram(x[i], bins=edges)[0] for i in range(len(t))])
    centres = 0.5 * (edges[1:] + edges[:-1]) / L
    im = ax.pcolormesh(centres, t, density / density[0].mean(), cmap=CMAP_DENSITY, vmin=0, vmax=2.0,
                       rasterized=True, shading="nearest")
    ax.grid(False)
    ax.set_title(f"{name} walls")
    ax.set_xlabel("x / L")
    panel_label(ax, f"({'abc'[bc]})", x=-0.1)
axes[0].set_ylabel(r"$t\,\omega_{pe}$")
cb = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.03)
cb.set_label(r"$n_e(x,t)\,/\,\bar n_e(0)$")
savefig(fig, "boundary_conditions")
