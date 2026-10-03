"""Compare prescribed B(x), B(x,y), B(x,z) and B(x,y,z); PIC remains 1D.

Each component varies only transverse to itself, so the imposed field is
divergence-free. Pure magnetic forces preserve speed; resolve the gyro period.
"""
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from jaxincell import Simulation

length, cells, steps = .01, 16, 256
positions = np.tile([.001, .0003, .0009], (8, 1))
velocities = np.tile([0., 1e6, 2e5], (8, 1))
species = {"number_pseudoparticles": 8, "weight": 1.,
           "initial_positions": positions, "initial_velocities": velocities,
           "vth_over_c_x": .01, "vth_over_c_y": 0., "vth_over_c_z": 0.}
parameters = {
    "domain_parameters": {"length": length, "total_steps": steps, "number_grid_points": cells,
                          "timestep_over_spatialstep_times_c": .5},
    "species_parameters": {"electrons": {"electrons0": species},
                           "ions": {"ions0": {**species, "mass_over_proton_mass": 1e10,
                                               "initial_velocities": np.zeros_like(velocities)}}},
    "solver_parameters": {"filter_passes": 0, "print_info": False},
}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for label, ny, nz in (("B(x)", 0, 0), ("B(x,y)", 8, 0), ("B(x,z)", 0, 4), ("B(x,y,z)", 8, 4)):
    p = deepcopy(parameters)
    p["domain_parameters"].update(number_grid_points_y=ny, number_grid_points_z=nz,
                                  length_y=length, length_z=length)
    counts = [cells]+([ny] if ny else [])+([nz] if nz else [])
    coordinates = [np.linspace(-length/2+length/(2*n), length/2-length/(2*n), n) for n in counts]
    mesh = np.meshgrid(*coordinates, indexing="ij")
    field = np.zeros(tuple(counts)+(3,))
    modulation = np.cos(2*np.pi*mesh[1]/length) if ny else 1.
    field[..., 2] = .2*(1+.1*np.sin(2*np.pi*mesh[0]/length)*modulation)
    if nz:
        field[..., 1] = .02*np.sin(2*np.pi*mesh[0]/length)*np.cos(2*np.pi*mesh[-1]/length)
    p["external_field_parameters"] = {"external_magnetic_field": {"B": field}}
    output = Simulation(p).run()
    orbit = np.asarray(output["positions"])[:, 0]
    speed2 = np.sum(np.asarray(output["velocities"])[:, 0]**2, axis=1)
    axes[0].plot(orbit[:, 0]*1e3, orbit[:, 1]*1e3, label=label)
    axes[1].plot(np.asarray(output["time_array"]), speed2/speed2[0]-1, label=label)
    print(label, "relative speed-squared change", np.max(np.abs(speed2/speed2[0]-1)))
axes[0].set(xlabel="x (mm)", ylabel="y (mm)")
axes[1].set(xlabel="time (s)", ylabel="relative change in speed squared")
for axis in axes:
    axis.legend()
fig.tight_layout()
path = Path("3d_external_fields.png")
fig.savefig(path, dpi=140)
print(path.resolve())
