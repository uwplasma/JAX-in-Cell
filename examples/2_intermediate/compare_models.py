"""One problem, seven ways to run it: which switches change the answer, and which only its cost.

The problem is the warm two-stream instability: two counter-streaming electron beams on
protons, mode 1 seeded by a small displacement on a quiet start. Its growth rate is the
purely growing root of the electrostatic kinetic dispersion relation of the same
populations, `jaxincell.theory.two_stream_rate`, which knows nothing of the solver. The same
Species and Domain are then run with one switch changed at a time from the reference, the
explicit electromagnetic leapfrog with no filter:

* model="electrostatic" and field_solver="gauss": E_x from the charge rather than from Ampere's law;
* algorithm="implicit": the energy-conserving Crank-Nicolson scheme, at the same step;
* filter_passes=2: binomial smoothing of the sources, which damps short wavelengths only;
* relativistic=True: the relativistic Boris push (the beams move at 0.17 c, so the cold
  theory lowers the rate by the longitudinal mass gamma^3, a percent or two);
* Collisions(): Takizuka-Abe binary collisions, whose rate here is ten million times below
  omega_pe, so they must change nothing. They run with the electrostatic model: a collision
  turns a longitudinal beam isotropic, and transverse velocity at c dt/dx = 4.5 seeds the
  light wave the explicit electromagnetic solver cannot hold (the run blows up within sixty
  steps, and the Simulation warns when it is built).

Each line prints the fitted rate against the kinetic root, the worst relative error of the
total energy, and the wall time of the run. Run with `--quick` for fewer particles: the
same comparison, noisier.
"""

import json
import os
import sys
import time
from pathlib import Path

# Double precision: the energy errors compared below go down to 1e-13.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Collisions, Domain, Simulation, Solver, Species, diagnostics, figure, provenance,
                       speed_of_light as c)
from jaxincell.theory import two_stream_rate

# --- what to change ------------------------------------------------------------------
quick = "--quick" in sys.argv
particles = 6000 if quick else 20000  # electrons (both beams); half as many protons
length, cells, density = 0.01, 64, 4.37e17
drift, vth = 5e7, 0.05 * c            # beam speed and thermal spread, m/s
seed_ak = 1e-4                        # displacement of mode 1, times k
dt_over_dx_c = 4.5                    # omega_pe dt = 0.2; no transverse velocity, so no light-wave limit
t_end = 40.0                          # in 1/omega_pe: through saturation
variants = {                          # name: the one switch changed from the reference
    "reference": Solver(),
    "electrostatic": Solver(model="electrostatic"),
    "gauss": Solver(field_solver="gauss"),
    "implicit": Solver(algorithm="implicit"),
    "filtered": Solver(filter_passes=2),
    "relativistic": Solver(relativistic=True),
    "collisional": Solver(model="electrostatic"),   # with Collisions(); see below
}

# --- the shared setup -----------------------------------------------------------------
electrons = Species.electrons(n=particles, density=density, vth=(vth, 0, 0), drift=(drift, 0, 0),
                              plus_minus=True, sampling="quiet", perturbation_mode=1,
                              perturbation_amplitude=seed_ak * length / (2 * np.pi))
ions = Species.ions(n=particles // 2, density=density, electrons=electrons, sampling="quiet")
domain = Domain(length=length, cells=cells, dt_over_dx_c=dt_over_dx_c)


def simulation(name):
    return Simulation(domain, [electrons, ions], variants[name],
                      collisions=Collisions() if name == "collisional" else None)


reference = simulation("reference")
omega_pe = float(reference.plasma_frequency())
kinetic = two_stream_rate(reference) / omega_pe
# cold beams at fixed k: gamma^2 = -(a^2 + b - sqrt(b^2 + 4 a^2 b)), a = k v0, b = omega_b^2;
# the relativistic push divides b by gamma0^3 (the longitudinal mass)
a, b = 2 * np.pi / length * drift, omega_pe ** 2 / 2
gamma0 = 1 / np.sqrt(1 - (drift / c) ** 2)
cold = [np.sqrt(-(a ** 2 + bb - np.sqrt(bb ** 2 + 4 * a ** 2 * bb))) for bb in (b, b / gamma0 ** 3)]
references = {name: kinetic * (cold[1] / cold[0] if name == "relativistic" else 1.0) for name in variants}
store_every = 2                       # the particle history is what the energy needs, and the bulk of the memory
steps = store_every * int(round(t_end / (omega_pe * domain.dt) / store_every))
if quick:
    print("--quick is a smoke preset with 6000 particles: the same comparison, noisier; not a measurement.")


def growth(t, amplitude):
    """Slope of log|E_1| between ten times its start and a tenth of its peak."""
    peak = int(np.argmax(amplitude))
    window = (amplitude > 10 * amplitude[0]) & (amplitude < 0.1 * amplitude[peak]) & (np.arange(t.size) < peak)
    if window.sum() < 5:                                        # it never left the noise, or never grew
        return np.nan, window
    return np.polyfit(t[window], np.log(amplitude[window]), 1)[0], window


results, curves = {}, {}
for name in variants:
    sim = simulation(name)
    sim.run(2, seed=0, store_particles=False)                  # compile outside the timing
    start = time.perf_counter()
    out = sim.run(steps, seed=0, store_every=store_every)
    out.E.block_until_ready()
    seconds = time.perf_counter() - start
    t = np.asarray(out.t) * omega_pe
    amplitude = np.abs(np.fft.rfft(np.asarray(out.E[:, :, 0]), axis=1)[:, 1]) / cells
    rate, window = growth(t, amplitude)
    total = np.asarray(diagnostics(out)["total"])
    results[name] = dict(rate=float(rate), reference=float(references[name]),
                         deviation_percent=float(100 * (rate / references[name] - 1)),
                         energy_error=float(np.max(np.abs(total / total[0] - 1))), seconds=seconds)
    curves[name] = (t, amplitude, np.abs(total / total[0] - 1))
    r = results[name]
    print(f"{name:>13}: gamma/omega_pe = {rate:.4f} against {r['reference']:.4f} ({r['deviation_percent']:+.1f} %), "
          f"energy error {r['energy_error']:.1e}, {seconds:.1f} s")

# --- the figure ------------------------------------------------------------------------------------
fig, axes = figure(3)
colors = plt.cm.tab10(np.arange(len(variants)))
for (name, (t, amplitude, error)), color in zip(curves.items(), colors):
    axes[0].semilogy(t, amplitude, color=color, lw=2, label=name)
    axes[2].semilogy(t[1:], np.maximum(error[1:], 1e-16), color=color, lw=2)
t = curves["reference"][0]
axes[0].semilogy(t, curves["reference"][1][0] * np.exp(kinetic * t), "k--", lw=1.5, label="kinetic root")
axes[0].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|E_{x,1}|$ (V/m)", title="the seeded mode",
            ylim=(0.5 * curves["reference"][1].min(), 3 * max(c[1].max() for c in curves.values())))
axes[0].legend(fontsize="small", ncol=2)
names = list(variants)
deviation = [results[n]["deviation_percent"] for n in names]
axes[1].barh(names, deviation, color=colors)
axes[1].axvline(0, color="k", lw=1)
axes[1].invert_yaxis()
axes[1].set(xlabel="rate against its reference (%)", title="growth rate")
axes[2].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|\mathcal{E}/\mathcal{E}_0-1|$", title="total energy")
fig.tight_layout()

# --- the record ------------------------------------------------------------------------------------
folder = Path.cwd() / ("compare_models_quick" if quick else "compare_models")
folder.mkdir(exist_ok=True)
settings = dict(particles=particles, length=length, cells=cells, density=density, drift=drift, vth=vth,
                seed_ak=seed_ak, dt_over_dx_c=dt_over_dx_c, t_end=t_end, steps=steps,
                variants={n: repr(s) for n, s in variants.items()}, quick=quick)
(folder / "run.json").write_text(json.dumps(provenance(example="compare_models", settings=settings,
                                                       results=dict(kinetic=kinetic, runs=results)), indent=1))
np.savez(folder / "curves.npz", **{f"{n}_{q}": v for n, arrays in curves.items()
                                   for q, v in zip(("t", "E1", "energy_error"), arrays)})
fig.savefig(folder / "figure.png")
print(f"\nwrote {folder}/run.json, curves.npz and figure.png")
plt.show()
