"""Electron/positron relaxation: full PIC and a collision-only energy control.

Run ``python twospecies_tempdiff.py --collision-only --steps 4000 --output relaxation.png``.
Full PIC is the default; the original dense benchmark under-resolves the Debye
length, so total-energy growth must be checked before interpreting its temperatures.
The collision model is nonrelativistic and weakly coupled, with small-angle/time-step
accuracy; a Coulomb-logarithm floor does not extend that physical validity range.
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax, random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from jaxincell import Simulation, load_parameters, diagnostics, boltzmann_constant, elementary_charge, speed_of_light
from jaxincell._collisions import collide, coulomb_logarithm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-only", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--store-every", type=int, default=1, help="Collision-only output interval")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parameters = load_parameters(Path(__file__).with_name("twospecies_temp.toml"))
    if args.steps:
        parameters["domain_parameters"]["total_steps"] = args.steps
    if args.particles:
        for kind in ("electrons", "ions"):
            for species in parameters["species_parameters"][kind].values():
                species["weight"] *= species["number_pseudoparticles"] / args.particles
                species["number_pseudoparticles"] = args.particles
    simulation = Simulation(parameters)
    if args.collision_only:
        # A zero-rate PIC run provides exactly the same initial realization and units.
        setup = {**parameters, "domain_parameters": {**parameters["domain_parameters"], "total_steps": 1},
                 "solver_parameters": {**parameters["solver_parameters"], "collisions": False}}
        output = Simulation(setup).run()
        x, v = output["initial_positions"], output["initial_velocities"]
        w = output["weights"].reshape(-1)
        ids = output["species_integer_index"]
        m, q = output["mass_integer_lookup"][ids], output["charge_integer_lookup"][ids]
        n = parameters["species_parameters"]["electrons"]["hot"]["number_pseudoparticles"]
        te_ev = m[0] * (0.001 * speed_of_light) ** 2 / (2 * elementary_charge)
        log = coulomb_logarithm(w[:n].sum() / output["length"], te_ev)
        def step(v, index):
            v = collide(random.fold_in(random.PRNGKey(1701), index), x, v, w, m, q,
                        ((0, n), (n, n)), ((0, 0), (0, 1), (1, 1)), log,
                        output["dt"], output["dx"], output["length"], len(output["grid"]))
            return v, v
        steps = simulation._domain_parameters["total_steps"]
        assert args.store_every > 0, "store-every must be positive"
        stored_steps = np.minimum(np.arange(args.store_every, steps + args.store_every, args.store_every), steps)
        starts = np.r_[0, stored_steps[:-1]]
        def chunk(v, bounds):
            start, end = bounds
            v = lax.fori_loop(start, end,
                             lambda index, v: step(v, index)[0], v)
            return v, v
        _, velocity = jax.jit(lambda: lax.scan(chunk, v, (jnp.asarray(starts), jnp.asarray(stored_steps))))()
        velocity = np.asarray(velocity)
    else:
        output = simulation.run()
        velocity = np.asarray(output["velocities"])
        stored_steps = np.arange(1, len(velocity) + 1)
    w, mass, q = (np.asarray(output[k]).reshape(-1) for k in ("weights", "masses", "charges"))
    initial = np.asarray(output["initial_velocities"])
    all_velocity = np.concatenate([initial[None], velocity])
    center = np.sum(mass[:, None] * initial, axis=0) / mass.sum()
    equilibrium = np.sum(mass[:, None] * (initial - center) ** 2) / (3 * boltzmann_constant * w.sum())
    temperatures = []
    for mask in (q < 0, q > 0):
        mean = np.average(all_velocity[:, mask], weights=w[mask], axis=1)
        temperatures.append(np.sum(mass[mask][None, :, None] * (all_velocity[:, mask] - mean[:, None]) ** 2,
                                   axis=(1, 2)) / (3 * boltzmann_constant * w[mask].sum()))
    time = np.r_[0, stored_steps] * float(output["dt"]) * float(output["plasma_frequency"])
    difference = temperatures[0] - temperatures[1]
    def model(time, delta, gamma):
        return delta * np.exp(-gamma * time)
    # The Maxwellian rate is approximately constant for equal masses/densities;
    # late finite-particle noise is excluded from this descriptive fit.
    fit_mask = (difference > 0.2 * difference[0]) & (time > 0)
    fit = None
    if fit_mask.sum() > 3:
        fit, _ = curve_fit(model, time[fit_mask], difference[fit_mask],
                           p0=(difference[0], 0.1), bounds=(0, np.inf))
        print(f"Fitted difference rate gamma/omega_pe={fit[1]:.6g}; nu_inter/omega_pe=3 gamma/4={0.75 * fit[1]:.6g}")
    if args.collision_only:
        energy = np.sum(mass[None, :, None] * all_velocity ** 2, axis=(1, 2)) / 2
    else:
        diagnostics(output)
        energy = np.r_[np.sum(mass[:, None] * initial ** 2) / 2, np.asarray(output["total_energy"])]
        # The plotted change uses the first stored total energy, including initialized fields.
        energy[0] = energy[1]
    print(f"Initial 3D energy equilibrium: {equilibrium:.8g} K; late temperatures: {temperatures[0][-1]:.8g}, {temperatures[1][-1]:.8g} K")
    print(f"Maximum total-energy change: {np.max(np.abs(energy / energy[0] - 1)):.6g}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(time, temperatures[0], label="Electrons")
    axes[0].plot(time, temperatures[1], label="Positrons")
    axes[0].axhline(equilibrium, color="k", linestyle=":", label="Initial 3D energy equilibrium")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(time, np.abs(difference), label="Absolute temperature difference")
    if fit is not None:
        axes[1].semilogy(time, model(time, *fit), "--", label="Early-time exponential fit")
    axes[1].set_ylabel("Temperature difference (K)")
    axes[1].legend(fontsize=8)
    axes[2].plot(time, energy / energy[0] - 1)
    axes[2].set_ylabel("Relative total-energy change")
    for ax in axes:
        ax.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
        ax.grid(alpha=0.25)
    fig.suptitle("Collision-only control" if args.collision_only else "Full PIC: under-resolved dense benchmark")
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=160)
    else:
        plt.show()


if __name__ == "__main__":
    main()
