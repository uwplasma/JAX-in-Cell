"""A plasma floats above the wall it touches, and less so when the wall reflects electrons.

The box is the edge of a large plasma. Its left wall is thermal: whatever reaches it
comes back with a fresh velocity from a Maxwellian at the starting temperature, as if
from the plasma behind it -- the source boundary of Schwager and Birdsall (Phys.
Fluids B 2, 1057, 1990). Its right wall is a floating conductor that collects what
reaches it. Electrons outrun the ions, charge the conductor negative, and are held
back by its field until no more electrons arrive than ions. Two closed-form results
follow, and this reproduces both.

* Ions enter the sheath at the Bohm speed c_s = sqrt(T_e/m_i) (Bohm 1949), and from
  there the potential falls to the wall by (Hobbs and Wesson, Plasma Phys. 9, 85, 1967)

      e dphi / T_e = (1/2) ln(m_i / 2 pi m_e) + ln(1 - R_eff),

  where R_eff is the fraction of the electron flux the wall sends back.
* R_eff is the flux average of the wall's reflection law (wall_reflection.py). A wall
  that returns half of every electron and one that returns the slow electrons with a
  Gaussian of width sigma both have R_eff = 1/2, so both hold a sheath ln 2 T_e/e
  shallower than a wall that keeps everything, although they return quite different
  electrons.

The thermal wall is what makes the comparison sharp: it keeps the electrons that
reach the conductor Maxwellian. Between two absorbing walls nothing would, and the
walls would strip the tail of the distribution the formula is derived from.

**This is a transient, and the script says so.** A thermal wall returns what reaches it
and cannot replace what the collector keeps, so the plasma between them drains: over the
one ion transit below it loses about two fifths of its ions, and the sheath is measured
while that is happening. The run finishes with a maintained reservoir in the same box,
which does not drain, so the two can be compared rather than one being assumed to stand
for the other. That is what `Source` is for, and `1_basic/sheath_unmagnetized.py` is the
example built on it.

Three things this run does not leave to assumption:

* the model is named. It used to take the default, which is electromagnetic, at a
  light-wave Courant number of 286 -- stable only because nothing here ever seeds a
  transverse field. Electrostatic gives the same sheath drop to the digit printed;
* `R_eff` is **measured**, from the weight the collector took against the weight that
  reached it, and compared with the flux average the law is supposed to have;
* the sheath edge comes from `bohm_edge`, which interpolates the crossing and says how
  many crossings there are. A bare `argmax` returns zero when the flow never reaches
  `c_s` and so invents an edge at the first bin.
"""

import json
import os
from pathlib import Path

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below to "0", for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Source, Species, bohm_edge, epsilon_0, mass_electron,
                       potential, provenance, quiet_start, elementary_charge as e_charge)

T_e, density, mass_ratio, particles, cells = 1.0, 1e16, 400.0, 30000, 120
sigma = np.sqrt(T_e * e_charge / mass_electron)                  # electron thermal spread
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
debye, c_s = sigma / omega_pe, sigma / np.sqrt(mass_ratio)
length, steps = 60 * debye, 6000                                  # about one ion transit
v_th_e, v_th_i = np.sqrt(2) * sigma, np.sqrt(2) * sigma / np.sqrt(40 * mass_ratio)   # T_i = T_e / 40

# electrostatic, so the time step follows the plasma frequency and not the speed of light
domain = Domain(length=length, cells=cells, time_step=0.2 / omega_pe,
                particle_bc=("thermal", "absorbing"), field_bc=("reflective", "absorbing"))
walls = {"absorbing": (0.0, 0.0), "returns half": (0.5, 0.5),
         "returns the slow ones": (lambda speed: jnp.exp(-speed ** 2 / (2 * sigma ** 2)), 0.5)}
x, v = quiet_start(particles, length, vth=(v_th_e, 0, 0))
x_i, v_i = quiet_start(particles, length, vth=(v_th_i, 0, 0))
ions = Species("ions", particles, 1.0, mass_ratio * mass_electron, density, (v_th_i, 0, 0)).replace(x=x_i, v=v_i)

late_step = steps // 200
late = slice(late_step, None)                                     # the second half of the stored steps
distance = (length / 2 - np.asarray(domain.faces)) / debye      # of each stored face from the conductor
centres = (length / 2 - np.asarray(domain.grid)) / debye        # where the deposited moments are
# Four binomial passes on the deposited sources. Measured on the absorbing wall below, they are
# worth 1.3 % of the sheath drop -- 3.05 T_e/e against 3.01 with none -- which is the size of the
# agreement being claimed, so it is a choice and not a detail.
solver = Solver(model="electrostatic", filter_passes=4)
results = {}
for name, (reflection, R_eff) in walls.items():
    electrons = Species.electrons(n=particles, density=density, vth=(v_th_e, 0, 0),
                                  reflection=(0.0, reflection)).replace(x=x, v=v)
    out = Simulation(domain, [electrons, ions], solver).run(steps, store_every=100, store_particles=False,
                                                            moments="flux").validate()
    phi = np.asarray(potential(out))[late] / T_e
    profile = phi.mean(axis=0) - phi[:, -1].mean()                # above the conductor
    window = np.asarray(out.moments[-1] - out.moments[late_step]) / float(out.steps[-1] - out.steps[late_step])
    n_i = window[1, 0]
    flow = np.divide(window[1, 1], n_i, out=np.zeros(cells), where=n_i > 0) / c_s
    # where the ions reach c_s, interpolated, with the number of crossings; a bare argmax returns
    # zero when there is none and puts the edge at the first bin, which is an edge that is not there
    crossing, count = bohm_edge(np.asarray(domain.grid), flow, 1.0)
    edge = float("nan") if count == 0 else (length / 2 - float(crossing)) / debye
    # R_eff as the wall actually applied it: the weight it sent back over the weight that reached it
    arrived = float(np.asarray(out.wall.arrived)[-1, 0, 1])
    measured_R = 1.0 - float(np.asarray(out.wall.collected)[-1, 0, 1]) / arrived
    # how much plasma is left: a thermal wall cannot replace what the collector keeps
    held = np.asarray(out.moments)[:, 1, 0].sum(axis=1)
    content = np.diff(held) / 100 / density
    results[name] = dict(profile=profile, flow=flow, edge=edge, crossings=int(count),
                         sheath=float(np.interp(edge, distance[::-1], profile[::-1])) if count else float("nan"),
                         theory=0.5 * np.log(mass_ratio / (2 * np.pi)) + np.log(1 - measured_R),
                         measured_R=measured_R, assumed_R=R_eff if not callable(R_eff) else None,
                         drained=float(content[-1] / content[0] - 1), content=content,
                         rho=np.asarray(out.rho)[late].mean(axis=0) / (density * e_charge))
    r = results[name]
    print(f"{name:22s} R_eff measured {measured_R:.3f}, sheath edge {edge:4.1f} lambda_D "
          f"({r['crossings']} crossing{'s' if r['crossings'] != 1 else ''}), drop {r['sheath']:.2f} T_e/e, "
          f"Hobbs-Wesson {r['theory']:.2f}, ions left {r['drained']:+.1%}")

# --- the same box, maintained ------------------------------------------------------------------
# A thermal wall returns what reaches it and cannot replace what the collector keeps, so the runs
# above drain. A reservoir supplies a flux that does not depend on what leaves. Same box, same
# resolution, same collector: what changes is the left boundary, and what it changes is whether
# the plasma the sheath sits on is still there at the end.
maintained_domain = Domain(length=length, cells=cells, time_step=0.2 / omega_pe,
                           particle_bc="absorbing", field_bc=("open", "absorbing"))
# an ion entering at c_s takes one transit to cross, which is the whole 6000 steps here, so the
# pool has to hold emit x 6000 of them: 12 a step is 72000, inside the capacity with room for the
# 30000 it starts with. The run refuses to report anything if that is wrong.
capacity, emit = 110000, 12
maintained = Simulation(
    maintained_domain,
    [Species("electrons", capacity, -1.0, mass_electron, density, (v_th_e,) * 3, active=particles,
             sampling="quiet", source=Source(density=density, vth=(v_th_e,) * 3, emit=emit)),
     Species("ions", capacity, 1.0, mass_ratio * mass_electron, density, (v_th_i,) * 3, (c_s, 0, 0),
             active=particles, sampling="quiet",
             source=Source(density=density, vth=(v_th_i,) * 3, drift=(c_s, 0, 0), emit=emit,
                           model="drifting"))],
    solver).run(steps, store_every=100, store_particles=False, moments="flux").validate()
held = np.asarray(maintained.moments)[:, 1, 0].sum(axis=1)
maintained_content = np.diff(held) / 100 / density
print(f"\nthe same box with a reservoir instead of a thermal wall: ions left "
      f"{maintained_content[-1] / maintained_content[0] - 1:+.1%}, against "
      f"{results['absorbing']['drained']:+.1%} with the thermal wall.")
print("The sheath drops above are therefore measured on a plasma that is going away, which is what\n"
      "a thermal wall between a collector and nothing does. That is the reason Source exists.")

fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for name, r in results.items():
    line, = axes[0].plot(distance, r["profile"], label=name)
    axes[0].plot(r["edge"], r["sheath"], "o", color=line.get_color())
axes[0].set(xlabel=r"distance from the wall ($\lambda_D$)", ylabel=r"$(\phi - \phi_{wall})/T_e$", xlim=(0, 30),
            title="the sheath, and where the ions reach $c_s$")
axes[0].legend(frameon=False)

axes[1].plot(centres, results["absorbing"]["flow"], label=r"ion flow $v_i/c_s$")
axes[1].plot(distance, 10 * results["absorbing"]["rho"], label=r"$10\,\rho/en_0$")
axes[1].axhline(1.0, ls="--", color="k", lw=0.8)
axes[1].set(xlabel=r"distance from the wall ($\lambda_D$)", xlim=(0, 30),
            title="the charge builds where the ions reach $c_s$")
axes[1].legend(frameon=False)

names = list(results)
axes[2].bar(range(3), [results[k]["sheath"] for k in names], color=["C0", "C1", "C2"], alpha=0.7, label="measured")
axes[2].plot(range(3), [results[k]["theory"] for k in names], "k_", ms=40, mew=2, label="Hobbs-Wesson")
axes[2].set_xticks(range(3), names, rotation=12)
axes[2].set(ylabel=r"sheath drop ($T_e/e$)", title=r"same $R_{\rm eff}$, same sheath")
axes[2].legend(frameon=False)

window_times = np.asarray(results["absorbing"]["content"]).size
elapsed = (np.arange(window_times) + 1) * 100 * float(domain.dt) * omega_pe
axes[3].plot(elapsed, results["absorbing"]["content"] / results["absorbing"]["content"][0],
             label="thermal wall")
axes[3].plot(elapsed, maintained_content / maintained_content[0], label="maintained reservoir")
axes[3].set(xlabel=r"$t\,\omega_{pe}$", ylabel="ions left, relative to the first window",
            title="a thermal wall cannot replace what a collector keeps", ylim=(0, 1.15))
axes[3].legend(frameon=False)
plt.tight_layout()

# --- the record --------------------------------------------------------------------------------
folder = Path.cwd() / "sheath_reflection"
folder.mkdir(exist_ok=True)
settings = dict(T_e=T_e, density=density, mass_ratio=mass_ratio, particles=particles, cells=cells,
                length_debye=60, steps=steps, filter_passes=4, model="electrostatic",
                temperature_ratio=1 / 40)
summary = {name: {k: r[k] for k in ("edge", "crossings", "sheath", "theory", "measured_R", "drained")}
           for name, r in results.items()}
summary["maintained_drained"] = float(maintained_content[-1] / maintained_content[0] - 1)
(folder / "run.json").write_text(json.dumps(provenance(example="sheath_reflection", settings=settings,
                                                       results=summary), indent=1))
np.savez(folder / "profiles.npz", distance=distance, centres=centres,
         **{f"{k}_{name.replace(' ', '_')}": np.asarray(r[k])
            for name, r in results.items() for k in ("profile", "flow", "rho", "content")})
fig.savefig(folder / "figure.png", dpi=150)
print(f"wrote {folder}/run.json, profiles.npz and figure.png")
plt.show()
