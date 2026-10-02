"""Near-ballistic control of fractional marker return at both walls."""
import numpy as np
from jaxincell import Simulation


def run_case(return_fraction=.5, restitution=1., code=3):
    count = 32
    positions = np.zeros((count, 3))
    positions[:, 0] = np.linspace(-.004, .004, count)
    velocities = np.zeros_like(positions)
    velocities[:, 0] = 6e7*(-1.)**np.arange(count)
    species = {"number_pseudoparticles": count, "weight": 1e-20,
               "initial_positions": positions, "initial_velocities": velocities,
               "vth_over_c_x": .01, "vth_over_c_y": 0., "vth_over_c_z": 0.}
    parameters = {
        "domain_parameters": {"length": .01, "number_grid_points": 16, "total_steps": 160,
                              "timestep_over_spatialstep_times_c": .5,
                              "particle_BC_left": code, "particle_BC_right": code,
                              "field_BC_left": 1, "field_BC_right": 1,
                              "mixed_BC_weight": return_fraction,
                              "mixed_BC_velocity_scale": 1.2e8,
                              "COR_left": restitution, "COR_right": restitution},
        "species_parameters": {"electrons": {"electrons0": species},
                               "ions": {"ions0": {**species, "initial_velocities": np.zeros_like(velocities)}}},
        "solver_parameters": {"field_solver": 2, "filter_passes": 0, "print_info": False},
    }
    output = Simulation(parameters).run()
    mass = np.asarray(output["masses_over_time"])[:, :count, 0]
    speed2 = np.sum(np.asarray(output["velocities"])[:, :count]**2, axis=-1)
    return np.asarray(output["time_array"]), np.sum(mass*speed2/2, axis=1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    time, energy = run_case()
    plt.plot(time, energy/energy[0])
    plt.xlabel("time (s)")
    plt.ylabel("electron kinetic energy / first snapshot")
    plt.tight_layout()
    plt.savefig("mixed_bc.png", dpi=140)
