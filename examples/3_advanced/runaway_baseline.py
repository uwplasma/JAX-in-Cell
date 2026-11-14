"""Homogeneous no-RF baseline for runaway control: imposed total DC field, evolved bulk and seed.

A gyrotropic 0D3V momentum-space model, not a PIC step with an unresolved gyro orbit. The
uniform E is parallel to B, so gyration does not change an initially gyrotropic distribution
and the magnetic Lorentz term drops out. Collisions keep all three momentum components, so
the production binary kernel is used unchanged. A coherent RF wave cannot simply be added here.

Convention. E is the TOTAL mean field: an ideal external current J_ext = -J_plasma holds it fixed
(eps0 dE/dt = -(J_plasma + J_ext) = 0), so no mean Ampere mode screens the drive. The energy the
source supplies is the DC work sum_s q E . v, which is accumulated per species. Bulk electrons,
the seed cohort and deuterons all evolve; there is no thermostat and no reservoir. Photons escape.
Energies are J/m^3: the marker weights are number densities and the 1 m reference cell converts
them to the collision kernel's per-area weights.

Radiation: the production Landau-Lifshitz kernel at E = 0 gives magnetic synchrotron emission;
the omitted electric-field emission is quadratured separately (absolute, J/m^3). Collisions are
fully ionized, leading-log, with an explicit logarithm. NOT modeled: avalanche (hard Moller),
bremsstrahlung, screening, photon reabsorption, spatial transport, collective fields. The notes
in docs/numerics/radiation.md and the L2 report bound them over the horizon.

Reference. Test-particle limit of the same model, in eta = |u|/c and pitch xi (Hirvijoki et al.
2015, eqs. 2.8-2.12 for the drive and synchrotron; collisional drag gamma^2/eta^2 and pitch
scattering (Z+1) gamma/eta^3 in units of the collision time tau_c, Connor and Hastie 1975):
    d eta/dt = a_E xi - gamma^2/(tau_c eta^2) - a_B gamma eta (1 - xi^2)
    d xi/dt  = a_E (1 - xi^2)/eta - (Z+1) xi gamma/(tau_c eta^3) + a_B xi (1 - xi^2)/gamma.
Its synchrotron-limited fixed point (gamma = 13.13 for the defaults) and its relaxation/regrowth
times are printed. The fixed point belongs to the zero-noise ODE and is not a bound on the kinetic
tail: with pitch-angle diffusion the stationary distribution peaks near gamma = 17.7 and agrees with
DREAM's kinetic solution. Avalanche growth at E/E_c = 6.5 is 0.05-0.07 /s (DREAM, CODE and
Rosenbluth and Putvinski 1997, eq. 18) and is not in the kernels. Because
tau_c is seconds at this density, the default 100 us run is a short precursor; BASELINE_TIME
and BASELINE_STEPS (tau_e = 23 us; dt = 0.2 us gave a few-percent exchange error) extend it. Requires scipy.

Run with --quick for a smoke run; BASELINE_N, BASELINE_STEPS, BASELINE_TIME, BASELINE_REPEATS,
BASELINE_RAW=0 (skip the unrestored comparison run) and BASELINE_GAMMA / BASELINE_PITCH (the seed)
override the defaults. No files are written.
"""
import os
import sys
from functools import partial

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from jaxincell import elementary_charge as e, epsilon_0, figure, mass_electron as me, speed_of_light as c
from jaxincell._collisions import collide
from jaxincell._core import landau_lifshitz

quick = "--quick" in sys.argv
ne, fraction, Te, Ti, Z = 1e18, 1e-5, 100., 100., 1               # m^-3, seed fraction, eV, eV, ion charge
B0, E_acc, logarithm = 2., .005, 15.                              # T, V/m, leading-log value
gamma_seed = float(os.environ.get("BASELINE_GAMMA", "40"))
pitch = float(os.environ.get("BASELINE_PITCH", ".1"))
count = int(os.environ.get("BASELINE_N", "128" if quick else "512"))
steps = int(os.environ.get("BASELINE_STEPS", "32" if quick else "256"))
duration = float(os.environ.get("BASELINE_TIME", "1e-4"))
repeats = int(os.environ.get("BASELINE_REPEATS", "1"))
mi = 3.3435837768e-27                                             # deuteron mass, kg
seed_count = max(8, count // 8)
tau_c = 4 * np.pi * epsilon_0 ** 2 * me ** 2 * c ** 3 / (ne * e ** 4 * logarithm)
tau_rad = e ** 2 / (6 * np.pi * epsilon_0 * me * c ** 3)
a_B, a_E = tau_rad * (e * B0 / me) ** 2, E_acc * e / (me * c)     # 1/s
E_c = ne * e ** 3 * logarithm / (4 * np.pi * epsilon_0 ** 2 * me * c ** 2)


def kinetic(u, mass):
    squared = jnp.sum(u * u, axis=-1)
    return mass * squared / (jnp.sqrt(1 + squared / c**2) + 1)


@partial(jax.jit, static_argnames=("blocks", "pairs", "steps", "restore"))
def evolve(u, density_weight, mass, charge, blocks, pairs, electric, magnetic,
           coulomb_log, dt, steps, seed=0, restore=True):
    """Strang drift/collision/drift. The drift is midpoint RK2 with the exact discrete energy
    gradient v_bar = (u_new + u_old)/(gamma_new + gamma_old), so the DC and mechanical RR work sum
    to the kinetic-energy increment. Emission quadratures the positive LL power. With ``restore``
    the collision stage returns each pair of species' weighted momentum and energy to their values
    before it; without, the raw unequal-weight defect is recorded in ``work[:, 3]``."""
    E, B = electric * jnp.array([1., 0., 0.]), magnetic * jnp.array([1., 0., 0.])
    acceleration = charge[:, None] / mass[:, None] * E
    tau = (charge / mass) * charge / (6 * jnp.pi * epsilon_0 * c**3)
    electric_power_scale = tau * (charge / mass)**2 * mass * electric**2
    zero_work = jnp.zeros((5, len(blocks)), u.dtype)  # DC, RR, magnetic emission, collision, omitted E emission

    def grouped(value):
        return jnp.stack([jnp.sum(value[start:start + count], axis=0) for start, count in blocks])

    def mean_velocity(before, after):
        def gamma(p):
            return jnp.sqrt(1 + jnp.sum((p / c)**2, axis=1, keepdims=True))
        return (before + after) / (gamma(before) + gamma(after))

    def drift(before, interval):
        first, _ = landau_lifshitz(before, jnp.zeros(3), B, charge, mass)
        middle = before + .5 * interval * (acceleration + first)
        reaction, power = landau_lifshitz(middle, jnp.zeros(3), B, charge, mass)
        after = before + interval * (acceleration + reaction)
        velocity = mean_velocity(before, after)
        dc = interval * charge * jnp.sum(E * velocity, axis=1)
        rr = interval * mass * jnp.sum(reaction * velocity, axis=1)
        electric_power = electric_power_scale * (1 + jnp.sum((middle[:, 1:] / c)**2, axis=1))
        return after, jnp.stack([grouped(density_weight * p)
                                for p in (dc, rr, interval * power, interval * electric_power)])

    def diagnostics(momentum, work, collision_momentum):
        gamma = jnp.sqrt(1 + jnp.sum((momentum / c)**2, axis=1))
        energy = kinetic(momentum, mass)
        return {"kinetic": grouped(density_weight * energy), "work": work,
                "current": grouped(density_weight * charge * momentum[:, 0] / gamma),
                "above_10MeV": grouped(density_weight * (energy > 1e7 * e)),
                "collision_momentum": collision_momentum}

    def step(carry, _):
        momentum, key, work, defect = carry
        before, first = drift(momentum, .5 * dt)
        key, draw = jax.random.split(key)
        after = collide(draw, jnp.zeros_like(before), before, density_weight * 1., mass, charge,
                        blocks, pairs, coulomb_log, dt, 1., 1., 1, relativistic=True, restore=restore)
        delta = mass * jnp.sum(mean_velocity(before, after) * (after - before), axis=1)
        defect += jnp.sum((density_weight * mass)[:, None] * (after - before), axis=0)
        momentum, second = drift(after, .5 * dt)
        drift_work = first + second
        work += jnp.concatenate((drift_work[:3], grouped(density_weight * delta)[None], drift_work[3:]), axis=0)
        return (momentum, key, work, defect), diagnostics(momentum, work, defect)

    start = diagnostics(u, zero_work, jnp.zeros(3))
    final, history = jax.lax.scan(jax.checkpoint(step),
                                  (u, jax.random.PRNGKey(seed), zero_work, jnp.zeros(3)), None, length=steps)
    history = jax.tree.map(lambda a, b: jnp.concatenate((a[None], b), axis=0), start, history)
    return final[0], history


def thermal_momentum(count, temperature_ev, mass, seed):
    """Maxwell--Juttner d^3p quadrature; antithetic pairs remove initial bulk current."""
    if count % 2:
        raise ValueError("the quiet thermal population requires an even marker count")
    theta = temperature_ev * e / (mass * c**2)
    radius = np.linspace(0., max(20 * np.sqrt(theta), 40 * theta), 20001)
    pdf = radius**2 * np.exp(-radius**2 / ((np.sqrt(1 + radius**2) + 1) * theta))
    cdf = np.r_[0., np.cumsum(pdf[1:] + pdf[:-1])]
    cdf /= cdf[-1]
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(count // 2, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    magnitude = np.interp(rng.permutation((np.arange(count // 2) + .5) / (count // 2)), cdf, radius)
    half = c * magnitude[:, None] * direction
    return np.concatenate((half, -half))


def test_particle(g0, xi0, horizon):
    """Reduced ODE solution: dense (eta(t), xi(t)) for a particle of Lorentz factor g0 and pitch cosine xi0."""
    def rhs(t, y):
        eta, xi = np.exp(y[0]), y[1]
        gamma = np.sqrt(1 + eta**2)
        d_eta = a_E * xi - gamma**2 / (tau_c * eta**2) - a_B * gamma * eta * (1 - xi**2)
        d_xi = a_E * (1 - xi**2) / eta - (Z + 1) * xi * gamma / (tau_c * eta**3) + a_B * xi * (1 - xi**2) / gamma
        return [d_eta / eta, d_xi]
    return solve_ivp(rhs, [0, horizon], [np.log(np.sqrt(g0**2 - 1)), xi0], method="LSODA", rtol=1e-10,
                     atol=1e-12, dense_output=True)


def gamma_at(solution, t):
    return np.sqrt(1 + np.exp(2 * solution.sol(t)[0]))


def reduced_scales():
    """Synchrotron-limited gamma, 1/e relaxation time from the seed and regrowth times from rest-ish starts."""
    long = test_particle(gamma_seed, np.cos(pitch), 100.)
    g_eq = gamma_at(long, 100.)
    gap = gamma_seed - g_eq
    relax = brentq(lambda t: gamma_at(long, t) - (g_eq + gap / np.e), 0, 100.)
    regrow = {}
    for g0, xi0 in ((2., .9), (5., .9), (10., .99)):
        s = test_particle(g0, xi0, 100.)
        regrow[g0] = [brentq(lambda t: gamma_at(s, t) - (g0 + f * (g_eq - g0)), 0, 100.) for f in (.5, .632, .9)]
    return g_eq, relax, regrow


blocks = ((0, count), (count, seed_count), (count + seed_count, count))
pairs = tuple((a, b) for a in range(3) for b in range(a, 3))
phase = 2 * np.pi * np.arange(seed_count) / seed_count
beam = c * np.sqrt(gamma_seed**2 - 1) * np.stack(
    (np.full(seed_count, np.cos(pitch)), np.sin(pitch) * np.cos(phase), np.sin(pitch) * np.sin(phase)), axis=1)
u0 = jnp.asarray(np.concatenate((thermal_momentum(count, Te, me, 21), beam, thermal_momentum(count, Ti, mi, 22))))
density = np.array([ne * (1 - fraction), ne * fraction, ne])
sizes = jnp.array([count, seed_count, count])
weight = jnp.repeat(jnp.asarray(density / np.array([count, seed_count, count])), sizes)
mass = jnp.repeat(jnp.array([me, me, mi]), sizes)
charge = jnp.repeat(jnp.array([-e, -e, Z * e]), sizes)


def run(restore):
    runs = jax.vmap(lambda s: evolve(u0, weight, mass, charge, blocks, pairs, -E_acc, B0, logarithm,
                                     duration / steps, steps, s, restore))(jnp.arange(repeats))
    return jax.tree.map(np.asarray, runs[1])


history = run(True)
raw = run(False) if os.environ.get("BASELINE_RAW", "1") == "1" else None
K, work = history["kinetic"], history["work"]          # (repeats, steps + 1, species[, ...])
seed_gamma = 1 + K[:, :, 1] / (density[1] * me * c**2)
t = np.arange(steps + 1) * duration / steps
reference = test_particle(gamma_seed, np.cos(pitch), max(duration, 1e-9))
g_ref = gamma_at(reference, t)
mean, error = seed_gamma.mean(axis=0), seed_gamma.std(axis=0, ddof=1) / np.sqrt(repeats) if repeats > 1 else None
g_eq, relax, regrow = reduced_scales()
assert np.isfinite(K).all() and np.isfinite(work).all()
residual = (K - K[:, :1] - work[..., 0, :] - work[..., 1, :]).sum(axis=-1)
bulk_temperature = 2 / 3 * K[:, :, 0] / (density[0] * e)
print(f"Homogeneous gyrotropic no-RF model; total E_parallel={-E_acc:g} V/m, B={B0:g} T, n={ne:g} m^-3")
print(f"E/E_c={E_acc / E_c:.3f} (E_c={E_c:.4e} V/m); tau_c={tau_c:.4f} s; tau_syn=1/a_B={1 / a_B:.4f} s; "
      f"E/(cB)={E_acc / (c * B0):.3e}")
print(f"duration={duration:g} s, dt={duration / steps:g} s, {count} bulk/{seed_count} seed/{count} ion markers, "
      f"{repeats} realization(s)")
print(f"Reduced test-particle: synchrotron-limited gamma={g_eq:.3f} ({(g_eq - 1) * me * c**2 / e / 1e6:.3f} MeV); "
      f"1/e relaxation from gamma={gamma_seed:g}: {relax:.3f} s")
print("Regrowth of the gap to gamma_eq (50%, 63.2%, 90%, s) from gamma0 = "
      + "; ".join(f"{g0:g}: " + ", ".join(f"{x:.1f}" for x in v) for g0, v in regrow.items()))
spread = f" +/- {error[-1]:.2g}" if error is not None else ""
print(f"Seed <gamma>: kinetic {mean[-1]:.6f}{spread}; reduced ODE {g_ref[-1]:.6f}; initial {mean[0]:.6f}")
print(f"Seed gamma change: kinetic {mean[-1] - mean[0]:.4e}, ODE {g_ref[-1] - g_ref[0]:.4e}")
print(f"Bulk electron T: {bulk_temperature[:, 0].mean():.4f} -> {bulk_temperature[:, -1].mean():.4f} eV")
print(f"Supplied (DC) work, bulk/seed/ion: {work[:, -1, 0].mean(axis=0)} J/m^3; "
      f"magnetic emission {work[:, -1, 2].sum(axis=-1).mean():.6g}; omitted electric emission "
      f"{work[:, -1, 4].sum(axis=-1).mean():.3g} J/m^3")
print(f"Electron density above 10 MeV: {history['above_10MeV'][:, -1, :2].sum(axis=-1).mean():.4g} m^-3")
raw_defect = raw["work"][:, :, 3].sum(axis=-1) if raw is not None else np.full_like(residual, np.nan)
print(f"Collision energy defect, raw unequal weights: {raw_defect[:, -1].mean():.3e} J/m^3 "
      f"(rms {np.sqrt((raw_defect[:, -1] ** 2).mean()):.3e}); "
      f"restored: rms {np.sqrt((work[:, -1, 3].sum(axis=-1) ** 2).mean()):.3e}; "
      f"ledger residual (K-K0-W_DC-W_RR) {np.abs(residual[:, -1]).max():.3e}")


def stat(x):
    return f"{x.mean():.4e} +/- {x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan:.1e}"


print(f"Seed collision work, J/m^3: restored {stat(work[:, -1, 3, 1])}"
      + (f"; raw {stat(raw['work'][:, -1, 3, 1])}" if raw is not None else ""))
print("Neglected: avalanche, bremsstrahlung, screening, reabsorption; see the module docstring.")

fig, axes = figure(3)
axes[0].plot(t, mean, label="kinetic")
axes[0].plot(t, g_ref, "--", label="reduced ODE")
axes[0].set(xlabel="time (s)", ylabel="seed mean gamma")
axes[0].legend()
axes[1].plot(t, bulk_temperature[:, :].mean(axis=0), label="bulk electrons")
axes[1].plot(t, 2 / 3 * K[:, :, 2].mean(axis=0) / (density[2] * e), label="ions")
axes[1].set(xlabel="time (s)", ylabel="mean kinetic energy (eV)")
axes[1].legend()
axes[2].plot(t, residual.mean(axis=0), label="K-K0-W_DC-W_RR (restored)")
axes[2].plot(t, raw_defect.mean(axis=0), "--", label="raw collision defect")
axes[2].set(xlabel="time (s)", ylabel="energy density (J/m^3)")
axes[2].legend()
plt.show()
