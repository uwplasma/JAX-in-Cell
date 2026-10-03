"""Matched cold injection and fractional collection: an open particle budget."""
import matplotlib.pyplot as plt
import numpy as np
from jaxincell import Simulation, speed_of_light


def run_case():
    species = {"number_pseudoparticles": 1, "weight": 1e-20,
               "initial_positions": np.zeros((1, 3)), "initial_velocities": np.zeros((1, 3)),
               "vth_over_c_x": .01, "vth_over_c_y": 0., "vth_over_c_z": 0.}
    parameters = {
        "domain_parameters": {"length": .01, "number_grid_points": 16, "total_steps": 96,
                              "timestep_over_spatialstep_times_c": .5,
                              "particle_BC_left": 1, "particle_BC_right": 3,
                              "field_BC_left": 1, "field_BC_right": 1,
                              "mixed_BC_weight": .5, "COR_right": .8},
        "species_parameters": {"electrons": {"electrons0": species}, "ions": {"ions0": species}},
        "solver_parameters": {"field_solver": 2, "filter_passes": 0, "print_info": False},
        "source_parameters": {"source_term_active": 1, "source_species": (0, 1),
                              "how_often_source_should_produce_quasiparticles": 4,
                              "source_particles_per_second": 1e12,
                              "location_of_source": 0, "width_of_source": 1,
                              "injection_speed_x": .4 * speed_of_light,
                              "injection_speed_y": 0., "injection_speed_z": 0.},
    }
    output = Simulation(parameters).run()
    mass = np.asarray(output["masses_over_time"])
    velocity = np.asarray(output["velocities"])
    live_weight = np.asarray(output["weights_over_time"]).sum(axis=1)
    energy = np.sum(mass * velocity**2 / 2, axis=(1, 2))
    momentum = np.sum(mass * velocity, axis=1)
    np.testing.assert_allclose(live_weight + output["lost_weight"],
                               2 * species["weight"] + output["injected_weight"], rtol=1e-12)
    np.testing.assert_allclose(energy + output["wall_energy_transfer"],
                               output["injected_energy"], rtol=1e-12)
    np.testing.assert_allclose(momentum + output["wall_momentum_transfer"],
                               output["injected_momentum"], rtol=1e-12, atol=1e-30)
    assert np.isfinite(output["positions"]).all()
    return output, live_weight, energy


if __name__ == "__main__":
    output, weight, energy = run_case()
    steps = np.asarray(output["time_array"]) / float(output["dt"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for data, label in ((output["injected_weight"], "injected"), (weight, "live"),
                        (output["lost_weight"], "collected")):
        axes[0].plot(steps, data / float(output["injected_weight"][-1]), label=label)
    for data, label in ((output["injected_energy"], "injected"), (energy, "live"),
                        (output["wall_energy_transfer"], "wall transfer")):
        axes[1].plot(steps, data / float(output["injected_energy"][-1]), label=label)
    for ax, name in zip(axes, ("weight", "kinetic energy")):
        ax.set(xlabel="completed steps", ylabel=f"{name} / final injection")
        ax.legend()
    fig.tight_layout()
    fig.savefig("source_particles.png", dpi=140)
