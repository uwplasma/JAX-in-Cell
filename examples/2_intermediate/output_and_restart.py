"""Saving a run, restarting it, and reading it back with another program.

A long run is a sequence of processes, so it has to stop and start again without the
answer noticing. `Output.state` is everything a run needs to continue; `save_state`
writes it as one array per field and `load_state` reads it back, checking that the
simulation it is restored into has the same shape. The test of a restart is the run it
was cut out of: here a two-stream run of 2 x 200 steps, stopped halfway, written to disk,
read back and continued, is compared with the same 400 steps run in one go. They agree
bit for bit.

For analysis in other tools, `jaxincell.openpmd.write_openpmd` writes the fields and
particles as an openPMD series (optional dependency `openpmd-api`). The field is read
back here with openpmd-api itself, placed on the coordinates the standard's attributes
give, and compared with the array in memory.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, figure, load_state, save_run, save_state,
                       speed_of_light as c)

steps, half = 400, 200
electrons = Species.electrons(n=8000, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(5e7, 0, 0), plus_minus=True,
                              sampling="low_noise", perturbation_amplitude=5e-7, perturbation_mode=1)
ions = Species.ions(n=8000, density=4.37e17, electrons=electrons, sampling="low_noise")
simulation = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5), [electrons, ions], Solver())

whole = simulation.run(steps, seed=0, store_every=10)
first = simulation.run(half, seed=0, store_every=10)
folder = Path(tempfile.mkdtemp())
path = save_state(folder / "checkpoint", first.state, simulation)
rest = simulation.run(steps - half, seed=0, store_every=10, state=load_state(path, simulation))
identical = {name: bool(np.array_equal(np.asarray(getattr(rest, name)), np.asarray(getattr(whole, name))[half // 10:]))
             for name in ("t", "E", "B", "x", "v")}
print(f"checkpoint {path} ({os.path.getsize(path) / 1e6:.1f} MB); restarted run bit-identical: {identical}")

readback = None
try:
    import openpmd_api as io

    from jaxincell.openpmd import write_openpmd
    series = io.Series(write_openpmd(whole, folder / "run.json", particles=False), io.Access.read_only)
    last = list(series.iterations)[-1]
    mesh = series.iterations[last].meshes["E"]
    E_x = mesh["x"].load_chunk()
    series.flush()
    # x_i = (gridGlobalOffset + (i + position) gridSpacing) gridUnitSI, nothing else
    where = (mesh.grid_global_offset[0] + (np.arange(E_x.size) + mesh["x"].position[0]) * mesh.grid_spacing[0])
    readback = dict(iteration=int(last), field_difference=float(np.max(np.abs(E_x - np.asarray(whole.E[last, :, 0])))),
                    coordinate_difference=float(np.max(np.abs(where - np.asarray(whole.faces)))))
    print(f"openPMD iteration {last}: field read back to {readback['field_difference']:.1e} V/m, "
          f"coordinates to {readback['coordinate_difference']:.1e} m")
except ImportError:
    print("openpmd-api is not installed (pip install openpmd-api): the openPMD part is skipped")

t = np.asarray(whole.t) * 1e9
mode = np.abs(np.fft.rfft(np.asarray(whole.E[:, :, 0]), axis=1)[:, 1])
fig, axes = figure(2)
axes[0].semilogy(t, mode, lw=4, color="0.75", label="one run of 400 steps")
axes[0].semilogy(np.asarray(first.t) * 1e9, np.abs(np.fft.rfft(np.asarray(first.E[:, :, 0]), axis=1)[:, 1]),
                 "--", lw=2, label="first 200 steps")
axes[0].semilogy(np.asarray(rest.t) * 1e9, np.abs(np.fft.rfft(np.asarray(rest.E[:, :, 0]), axis=1)[:, 1]),
                 ":", lw=3, label="restarted from disk")
axes[0].axvline(t[half // 10], color="k", lw=1)
axes[0].set(xlabel="t (ns)", ylabel=r"$|E_{x,1}|$", title="a restart continues the run")
axes[0].legend()
axes[1].plot(np.asarray(whole.faces) * 1e3, np.asarray(whole.E[-1, :, 0]) / 1e6, lw=4, color="0.75", label="in memory")
if readback is not None:
    axes[1].plot(where * 1e3, E_x / 1e6, "--", lw=2, label="openPMD, read back")
axes[1].set(xlabel="x (mm)", ylabel=r"$E_x$ (MV/m)", title="the last field, and its openPMD copy")
axes[1].legend()
fig.tight_layout()
save_run(Path.cwd() / "output_and_restart", "output_and_restart", dict(steps=steps, restart_at=half),
         dict(bit_identical=identical, openpmd=readback), figure=fig)
plt.show()
