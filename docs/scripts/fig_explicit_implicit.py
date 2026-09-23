"""Explicit against implicit: Landau damping and the two-stream instability run with the
Boris leapfrog and with the implicit Crank-Nicolson scheme, the implicit one at four times
the explicit step. The field energy against the kinetic rates, and the total-energy error."""
import numpy as np
from common import C_EXPLICIT, C_IMPLICIT, C_THEORY, COLORS, figure, maxima, panel_label, record, savefig
from dispersion import landau_root
from two_stream_setup import OMEGA_PE as OMEGA_PE_TS, build as two_stream, measure, theory, DRIFT

from jaxincell import (Domain, Simulation, Solver, Species, diagnostics, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

EPS = np.finfo(float).eps           # errors below double-precision round-off are drawn at it
STEP_RATIO = 4                       # implicit step over explicit step, in both problems
SCHEMES = (("explicit", C_EXPLICIT, Solver(filter_passes=0)),
           ("implicit", C_IMPLICIT, Solver(algorithm="implicit", filter_passes=0)))

# Landau damping at k lambda_D = 0.5, the problem of fig_landau_damping.py with a larger seed,
# so that the field energy stays above the particle noise for five of its maxima
LENGTH, CELLS, PARTICLES, SEED_AK, K_LAMBDA_D = 1.0, 64, 100000, 0.02, 0.5
K = 2 * np.pi / LENGTH
OMEGA_PE = 0.05 * c * CELLS / LENGTH                     # omega_pe dt = 0.05 at dt = dx/c
DENSITY = OMEGA_PE ** 2 * epsilon_0 * mass_electron / e_charge ** 2


def landau(solver, courant):
    electrons = Species.electrons(n=PARTICLES, density=DENSITY, sampling="quiet",
                                  vth=(K_LAMBDA_D / K * np.sqrt(2) * OMEGA_PE, 0, 0),
                                  perturbation_amplitude=SEED_AK / K, perturbation_mode=1)
    ions = Species.ions(n=PARTICLES // 8, density=DENSITY, mass_ratio=1e9, vth=(0, 0, 0), sampling="quiet")
    return Simulation(Domain(length=LENGTH, cells=CELLS, dt_over_dx_c=courant), [electrons, ions], solver)


def run(simulation, t_end, omega_pe, frames=60):
    """The fields at every step, and the relative total-energy error at about ``frames``
    stored steps of a second, identical run: the particle history of every step would
    take gigabytes."""
    steps = int(round(t_end / (omega_pe * simulation.domain.dt)))
    every = max(1, steps // frames)
    steps -= steps % every
    fields = simulation.run(steps, seed=0, store_particles=False)
    sampled = simulation.run(steps, seed=0, store_every=every)
    return {"out": fields, "t": np.asarray(fields.t) * omega_pe,
            "W": np.asarray(diagnostics(fields)["electric"]),
            "t_error": np.asarray(sampled.t) * omega_pe,
            "error": np.asarray(diagnostics(sampled)["energy_error"]),
            "omega_pe_dt": float(omega_pe * simulation.domain.dt),
            "courant": float(c * simulation.domain.dt / simulation.domain.dx)}


def decay_rate(t, W, peaks):
    """Damping rate from the maxima of the field energy, each placed by the parabola through
    ln W at it and its two neighbours, so that a coarse step does not clip the maxima; the
    energy decays at twice the rate of the field."""
    y0, y1, y2 = (np.log(W[peaks + shift]) for shift in (-1, 0, 1))
    offset = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return 0.5 * np.polyfit(t[peaks] + offset * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * offset, 1)[0]


root = landau_root(K_LAMBDA_D)
gamma_two_stream = theory(DRIFT)
landau_runs, two_stream_runs = {}, {}
for (name, _, solver), ratio in zip(SCHEMES, (1, STEP_RATIO)):
    landau_runs[name] = r = run(landau(solver, 1.0 * ratio), 25.0, OMEGA_PE)
    floor = r["W"][int(0.8 * r["W"].size):].mean()
    r["peaks"] = maxima(r["W"], above=25 * floor)
    r["gamma"] = decay_rate(r["t"], r["W"], r["peaks"])
    two_stream_runs[name] = r = run(two_stream(solver=solver, dt_over_dx_c=4.5 * ratio), 78.75, OMEGA_PE_TS)
    _, _, r["fit"] = measure(r["out"])
    r["gamma"] = r["fit"]["gamma"]
    print(f"  {name}: Landau gamma {landau_runs[name]['gamma']:.4f} (kinetic {root.imag:.4f}), "
          f"two-stream gamma {r['gamma']:.4f} (kinetic {gamma_two_stream:.4f})")

fig, axes = figure(2, 2)
(a, b), (cc, d) = axes
for name, color, _ in SCHEMES:
    r = landau_runs[name]
    a.semilogy(r["t"], r["W"], color=color,
               label=fr"{name}, $\omega_{{pe}}\Delta t={r['omega_pe_dt']:g}$: $\gamma={r['gamma']:.3f}$")
    cc.semilogy(r["t_error"], np.maximum(r["error"], EPS), color=color, label=name)
    r = two_stream_runs[name]
    b.semilogy(r["t"], r["W"], color=color,
               label=fr"{name}, $\omega_{{pe}}\Delta t={r['omega_pe_dt']:.3g}$: $\gamma={r['gamma']:.3f}$")
    d.semilogy(r["t_error"], np.maximum(r["error"], EPS), color=color, label=name)

r = landau_runs["explicit"]
start = r["peaks"][0]
span = np.linspace(r["t"][start], r["t"][r["peaks"][-1]], 2)
a.semilogy(span, r["W"][start] * np.exp(2 * root.imag * (span - span[0])), ":", color=C_THEORY,
           label=fr"kinetic: $\gamma={root.imag:.3f}$")
a.set(title=fr"Landau damping, $k\lambda_D={K_LAMBDA_D:g}$", ylim=(1e-5 * r["W"][0], 3 * r["W"][0]))

r = two_stream_runs["explicit"]
span = np.linspace(r["fit"]["t0"], r["fit"]["t1"], 2)
level = r["W"][np.searchsorted(r["t"], span[-1])]       # the field energy where the fit ends
b.semilogy(span, level * np.exp(2 * gamma_two_stream * (span - span[-1])), ":", color=C_THEORY,
           label=fr"kinetic: $\gamma={gamma_two_stream:.3f}$")
b.set(title=fr"two-stream, $kv_0/\omega_{{pe}}={2 * np.pi * DRIFT / 0.01 / OMEGA_PE_TS:.2f}$")

for ax in (a, b):
    ax.set(ylabel=r"field energy $W_E$ (J/m$^2$)")
    ax.legend(loc="lower right" if ax is b else "upper right", fontsize=15)
for ax in (cc, d):
    ax.axhspan(1e-17, EPS, color=COLORS["grey"], alpha=0.25, lw=0, label="below round-off")
    ax.set(ylabel="relative energy error", ylim=(1e-17, 1e-1))
    ax.legend(loc="center right", fontsize=15)
for ax, label in zip((a, b, cc, d), "abcd"):
    ax.set_xlabel(r"$t\,\omega_{pe}$")
    panel_label(ax, label)
fig.tight_layout()
savefig(fig, "explicit_implicit")


def deviation(gamma, reference):
    return round(float(100 * abs(gamma - reference) / abs(reference)), 1)


def largest(r):
    return f"{float(np.max(r['error'])):.1e}"


values = {"schemes_step_ratio": STEP_RATIO, "schemes_picard_iterations": SCHEMES[1][2].picard_iterations,
          "schemes_landau_particles": PARTICLES, "schemes_landau_seed_ak": SEED_AK,
          "schemes_landau_gamma_theory": round(float(root.imag), 4),
          "schemes_two_stream_gamma_theory": round(float(gamma_two_stream), 4)}
for name, _, _ in SCHEMES:
    for case, runs, reference in (("landau", landau_runs, root.imag),
                                  ("two_stream", two_stream_runs, gamma_two_stream)):
        r = runs[name]
        values[f"schemes_{case}_gamma_{name}"] = round(float(r["gamma"]), 4)
        values[f"schemes_{case}_gamma_deviation_percent_{name}"] = deviation(r["gamma"], reference)
        values[f"schemes_{case}_energy_error_{name}"] = largest(r)
        values[f"schemes_{case}_omega_pe_dt_{name}"] = round(r["omega_pe_dt"], 4)
        values[f"schemes_{case}_courant_{name}"] = round(r["courant"], 2)
    # how many cells a beam electron crosses per step; the explicit scheme wants at most about one
    values[f"schemes_two_stream_beam_cells_per_step_{name}"] = round(two_stream_runs[name]["courant"] * DRIFT / c, 2)
record(**values)
