"""Minimum supplied RF resource to keep a runaway tail below 10 MeV: validated slow response.

Stages (command line): scenario, validate, source by default (minutes); then the long search stages
opt_single, opt_random, opt_es (240 slow-model evaluations each, about 40 min each on a loaded node), verify
(frozen controls, JIC_VERIFY=<strategy> and JIC_CANDIDATES limit it) and margin (continue the verified controls
past the hold). JIC_HOLD_EXTRA adds seconds to the hold. `--quick` shrinks sizes, a smoke run.
Optional JIC_RF_OUTPUT names a directory outside the checkout for the JSON results; nothing is written otherwise.

Model, in the order it is used.
1. Scenario. The reduced test-particle ODE of runaway_baseline.py (collisional drag, pitch scattering,
   DC drive, Landau-Lifshitz synchrotron; collision and radiation strengths are never changed). At
   B=2 T, n=1e18, E=5 mV/m the synchrotron fixed point (6.2 MeV) lies below the 10 MeV threshold, so no
   RF is needed to hold a suppressed tail: a bounded negative result. The synchrotron fixed point never
   disappears at any E; what changes is its energy. The declared numerical study choice E = 15 mV/m puts
   it at ~59 MeV, so the no-RF tail grows through 10 MeV within the horizon.
2. Slow RF response. Counter-propagating cold-plasma whistler (R wave), normal-Doppler resonance
   omega + |k| v cos(a) = Omega/gamma. Quasilinear (QL) diffusion along the wave-frame curve
   E + |v_ph| p_par = const, with D_aa = (pi/2) sum A_l^2 G(Delta_l) and Gaussian-smoothed resonance
   (width = line spacing in Delta). The same wave is simulated resolved (relativistic Boris, gyro-orbit
   resolved) on held-out windows. A resolved wave and its QL operator are never applied together.
3. Source energy. The total wave field is prescribed; the explicit drive current is
   J_drive = curl B/mu0 - eps0 dE/dt - J_plasma (cold electron+ion fluid with Spitzer friction),
   P_src = -J_drive.E, ideal nonrecovering port Q+ = int max(P,0) dt with zero-crossing quadrature
   of the piecewise-linear power; returned and net energy are reported separately.
4. Optimization of the best-found supplied energy for a fixed RF-on window plus an RF-off hold chosen
   from the measured no-RF regrowth time, DC drive active throughout.

Not modeled: avalanche, bremsstrahlung, spatial transport, antenna loading and wall-plug efficiency, finite
box and mode conversion, bulk heating by the wave. The tail carries 1e-5 of the density. Requires scipy.
"""
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist

from jaxincell import elementary_charge as e, epsilon_0 as eps, mass_electron as me, speed_of_light as c

assert jax.config.read("jax_enable_x64"), "The gates require float64."
quick = "--quick" in sys.argv
stages = [a for a in sys.argv[1:] if not a.startswith("--")] or ["scenario", "validate", "source"]
out_dir = Path(os.environ["JIC_RF_OUTPUT"]) if os.environ.get("JIC_RF_OUTPUT") else None
mi = 3.3435837768e-27                                          # deuteron mass, kg
mu0 = 1 / (eps * c**2)
ne, Z, B0, Te, logarithm = 1e18, 1, 2., 100., 15.                # m^-3, charge, T, eV, lnL
E_L2, E_study, K_thr = .005, .015, 10.                          # V/m (handoff), declared study value, MeV
gamma_thr = 1 + K_thr * 1e6 * e / (me * c**2)
tau_c = 4 * np.pi * eps**2 * me**2 * c**3 / (ne * e**4 * logarithm)
tau_rad = e**2 / (6 * np.pi * eps * me * c**3)
a_B = tau_rad * (e * B0 / me)**2
Om, Oi = e * B0 / me, e * B0 / mi
wpe2, wpi2 = ne * e**2 / (eps * me), ne * e**2 / (eps * mi)
tail_fraction = 1e-5
report = {"stages": stages, "quick": quick, "jax": jax.__version__, "gamma_threshold": gamma_thr}
t_start = time.perf_counter()


def save():
    report["elapsed_s"] = time.perf_counter() - t_start
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "runaway_rf_minimum.json").write_text(json.dumps(report, indent=1, default=float) + "\n")


def say(*args):
    print(*args, flush=True)


# ---------------------------------------------------------------- reduced ODE (numpy, scipy) --------------
def ode_rhs(E):
    a_E = E * e / (me * c)

    def f(t, y):
        eta, xi = y
        g = np.sqrt(1 + eta**2)
        return [a_E * xi - g**2 / (tau_c * eta**2) - a_B * g * eta * (1 - xi**2),
                a_E * (1 - xi**2) / eta - (Z + 1) * xi * g / (tau_c * eta**3) + a_B * xi * (1 - xi**2) / g]
    return f


def gamma_eq(E):
    """Synchrotron-limited fixed point (long integration from gamma = 40)."""
    s = solve_ivp(ode_rhs(E), [0, 600], [np.sqrt(40.**2 - 1), .99], rtol=1e-10, atol=1e-13)
    return float(np.sqrt(1 + s.y[0, -1]**2)), float(np.arccos(s.y[1, -1]))


def track(E, gamma0, pitch0, times):
    s = solve_ivp(ode_rhs(E), [0, max(times)], [np.sqrt(gamma0**2 - 1), np.cos(pitch0)], t_eval=times,
                  rtol=1e-10, atol=1e-13)
    return np.sqrt(1 + s.y[0]**2), np.arccos(s.y[1])


def crossing_time(E, gamma0, xi0, level, t_max=200.):
    def event(t, y):
        return np.sqrt(1 + y[0]**2) - level

    event.terminal, event.direction = True, 1
    s = solve_ivp(ode_rhs(E), [0, t_max], [np.sqrt(gamma0**2 - 1), xi0], events=event, rtol=1e-10, atol=1e-13)
    return float(s.t_events[0][0]) if len(s.t_events[0]) else np.inf


if "scenario" in stages:
    say("== scenario ==")
    rec = {"E_c_V_m": ne * e**3 * logarithm / (4 * np.pi * eps**2 * me * c**2), "tau_c_s": tau_c,
           "tau_sync_s": 1 / a_B, "gamma_threshold": gamma_thr}
    table = {}
    for E in (.005, .0065, .0075, .01, .015, .02, .03):
        g, a = gamma_eq(E)
        table[f"{E:g}"] = {"gamma_eq": g, "K_eq_MeV": (g - 1) * .511, "pitch_eq": a}
    rec["fixed_point_vs_E"] = table
    E_thr = brentq(lambda E: gamma_eq(E)[0] - gamma_thr, .005, .01, xtol=1e-7)
    rec["E_threshold_V_m"] = E_thr
    rec["threshold_note"] = ("The fixed point exists at every E > E_c and moves smoothly with E; no E removes it. "
                             "E_threshold is where it crosses 10 MeV.")
    # L2 scenario: bounded negative result.
    g_eq, a_eq = gamma_eq(E_L2)
    times = [0., 1., 5., 10., 20., 40.]
    negative = {"E_V_m": E_L2, "gamma_eq": g_eq, "K_eq_MeV": (g_eq - 1) * .511,
                "seed_20MeV_gamma_vs_t": dict(zip(map(str, times), track(E_L2, 40., .1, times)[0]))}
    t_fall = brentq(lambda t: track(E_L2, 40., .1, [t, t + 1e-9])[0][0] - gamma_thr, 1., 40.)
    negative["time_seed_falls_below_10MeV_s"] = t_fall
    peak = {}
    for g0, p0 in ((2., np.arccos(.9)), (3., np.arccos(.5)), (5., np.arccos(.9)), (10., np.arccos(.99)), (20., .2)):
        gg = track(E_L2, g0, p0, np.linspace(0, 60, 601))[0]
        peak[f"{g0:g}"] = float(gg.max())
    negative["max_gamma_after_suppressed_start"] = peak
    negative["minimum_RF"] = ("zero: every suppressed start (gamma <= 20) stays below the 10 MeV threshold "
                              f"(peak gamma {max(peak.values()):.2f} < {gamma_thr:.2f}) for all time; only the "
                              f"20 MeV seed itself exceeds it, for {t_fall:.2f} s, and it relaxes without RF.")
    rec["L2_scenario_negative_result"] = negative
    say("L2 scenario:", negative["minimum_RF"])
    # Declared study scenario.
    g_s, a_s = gamma_eq(E_study)
    rec["study"] = {"E_V_m": E_study, "E_over_E_c": E_study / rec["E_c_V_m"], "gamma_eq": g_s,
                    "K_eq_MeV": (g_s - 1) * .511, "margin_gamma_eq_over_threshold": g_s / gamma_thr,
                    "E_over_E_threshold": E_study / E_thr}
    regrow = crossing_time(E_study, 5., .9, gamma_thr)
    rec["study"]["regrowth_to_10MeV_from_gamma5_s"] = regrow
    rec["study"]["regrowth_to_10MeV_from_gamma3_xi0.5_s"] = crossing_time(E_study, 3., .5, gamma_thr)
    rec["study"]["no_RF_seed_gamma10_xi0.99_to_10MeV_s"] = crossing_time(E_study, 10., .99, gamma_thr)
    say("study:", rec["study"])
    report["scenario"] = rec
    save()


# ---------------------------------------------------------------- cold whistler and QL operator (JAX) ------
def refraction2(w):
    return 1 + wpe2 / (w * (Om - w)) - wpi2 / (w * (w + Oi))      # R wave, cold electrons and ions


def wavenumber(w):
    return w * jnp.sqrt(refraction2(w)) / c


slope_k = jax.grad(wavenumber)


def band_nodes(centers, widths, amplitudes, nodes=48):
    """Continuum bands: flat spectral density across centre*(1 +- width/2), rms dB/B0 = amplitude each."""
    centers, widths, amplitudes = (jnp.asarray(x, jnp.float64) for x in (centers, widths, amplitudes))
    frac = (jnp.arange(nodes) + .5) / nodes - .5
    w = (centers[:, None] * (1 + widths[:, None] * frac)).ravel()
    spacing = jnp.repeat(centers * widths / nodes, nodes)
    weight = jnp.repeat(amplitudes**2 / nodes, nodes)
    return w, jax.vmap(wavenumber)(w), jax.vmap(slope_k)(w), weight, spacing


def diffusion_tensor(X, nd):
    """QL diffusion tensor D^{ij}, X=(eta=p/mc, pitch a), summed over nodes, each along its own tangent."""
    w, k, dk, weight, spacing = nd
    eta, a = X
    g = jnp.sqrt(1 + eta**2)
    beta, xi, s = eta / g, jnp.cos(a), jnp.sin(a)
    phase = w / (k * c)
    detuning = Om / g - w - k * c * beta * xi
    width = jnp.abs(1 + c * beta * xi * dk) * spacing
    amplitude = (Om / g) * (1 + phase * xi / beta)
    d = .5 * jnp.pi * weight * amplitude**2 * jnp.exp(-.5 * (detuning / width)**2) / (width * jnp.sqrt(2 * jnp.pi))
    phi = eta * s * phase / (beta + phase * xi)
    return jnp.array([[jnp.sum(d * phi**2), jnp.sum(d * phi)], [jnp.sum(d * phi), jnp.sum(d)]])


def rf_kick(X, z, nd, dt):
    """Euler-Maruyama step of the QL Fokker-Planck equation (volume element eta^2 sin a, Ito drift)."""
    def J(y):
        return y[0]**2 * jnp.sin(y[1])

    def F(y):
        return J(y) * diffusion_tensor(y, nd)

    D, dF = diffusion_tensor(X, nd), jax.jacfwd(F)(X)
    drift = jnp.einsum("ijj->i", dF) / J(X)
    l11 = jnp.sqrt(D[0, 0] + 1e-300)
    l21 = D[0, 1] / l11
    l22 = jnp.sqrt(jnp.maximum(D[1, 1] - l21**2, 0.))
    return X + drift * dt + jnp.sqrt(2 * dt) * jnp.array([l11 * z[0], l21 * z[0] + l22 * z[1]])


def reflect(X):
    eta = jnp.maximum(X[:, 0], .05)
    a = jnp.abs(X[:, 1])
    a = jnp.where(a > jnp.pi, 2 * jnp.pi - a, a)
    return jnp.stack([eta, jnp.clip(a, 1e-3, jnp.pi - 1e-3)], axis=1)


def det_rhs(X, a_E):
    eta, a = X[:, 0], X[:, 1]
    g, xi, s = jnp.sqrt(1 + eta**2), jnp.cos(a), jnp.sin(a)
    d_eta = a_E * xi - g**2 / (tau_c * eta**2) - a_B * g * eta * s**2
    d_xi = a_E * s**2 / eta - (Z + 1) * xi * g / (tau_c * eta**3) + a_B * xi * s**2 / g
    return jnp.stack([d_eta, -d_xi / s], axis=1)


def rk4(X, a_E, dt):
    k1 = det_rhs(X, a_E)
    k2 = det_rhs(X + .5 * dt * k1, a_E)
    k3 = det_rhs(X + .5 * dt * k2, a_E)
    k4 = det_rhs(X + dt * k3, a_E)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def gamma_of(X):
    return jnp.sqrt(1 + X[:, 0]**2)


@partial(jax.jit, static_argnames=("steps", "drive"))
def ql_advance(X, key, nd, a_E, dt, steps, drive):
    """`steps` RF steps (and the DC/collision/radiation ODE if `drive`); returns X and the RF energy change."""
    def step(carry, _):
        X, key, dgam = carry
        key, sub = jax.random.split(key)
        if drive:
            X = rk4(X, a_E, dt)
        z = jax.random.normal(sub, (X.shape[0], 2))
        new = reflect(jax.vmap(lambda x, zz: rf_kick(x, zz, nd, dt))(X, z))
        return (new, key, dgam + gamma_of(new) - gamma_of(X)), None
    (X, key, dgam), _ = jax.lax.scan(step, (X, key, jnp.zeros(X.shape[0])), None, length=steps)
    return X, dgam


@partial(jax.jit, static_argnames=("steps",))
def hold_advance(X, a_E, dt, steps):
    return jax.lax.scan(lambda X, _: (rk4(X, a_E, dt), None), X, None, length=steps)[0]


# ---------------------------------------------------------------- resolved wave window (relativistic Boris) --
@partial(jax.jit, static_argnames=("steps", "record"))
def resolved_window(u, z, w, k, amp, phi, dt, steps, record):
    """u=p/m (N,3), z (N,), phi (N,M): gyro-resolved Lorentz push in B0 z + counter-propagating R-wave lines.
    Every particle sees its own random-phase realization, so ensemble statistics are wave-ensemble averages.
    Line fields: dB = amp B0 (cos b, sin b), E = -(w/k) amp B0 (sin b, -cos b), b = w t + k z + phi."""
    q_m = -e / me

    def fields(z, t):
        b = w[None] * t + k[None] * z[:, None] + phi
        Bw = B0 * jnp.sum(amp[None] * jnp.stack([jnp.cos(b), jnp.sin(b)]), axis=-1)
        Ew = -B0 * jnp.sum((amp * w / k)[None] * jnp.stack([jnp.sin(b), -jnp.cos(b)]), axis=-1)
        return Ew, Bw

    def step(carry, n):
        u, z = carry
        t = n * dt
        Ew, Bw = fields(z, t)
        E = jnp.stack([Ew[0], Ew[1], jnp.zeros_like(z)], axis=1)
        B = jnp.stack([Bw[0], Bw[1], jnp.full_like(z, B0)], axis=1)
        um = u + .5 * dt * q_m * E
        g = jnp.sqrt(1 + jnp.sum(um**2, axis=1) / c**2)
        tv = .5 * dt * q_m * B / g[:, None]
        up = um + jnp.cross(um, tv)
        uu = um + jnp.cross(up, 2 * tv / (1 + jnp.sum(tv**2, axis=1, keepdims=True)))
        u = uu + .5 * dt * q_m * E
        g = jnp.sqrt(1 + jnp.sum(u**2, axis=1) / c**2)
        z = z + dt * u[:, 2] / g
        return (u, z), None

    def block(carry, i):
        carry, _ = jax.lax.scan(step, carry, i * record + jnp.arange(record))
        u, _ = carry
        g = jnp.sqrt(1 + jnp.sum(u**2, axis=1) / c**2)
        return carry, jnp.stack([g, jnp.arccos(u[:, 2] / jnp.sqrt(jnp.sum(u**2, axis=1)))])

    (u, z), history = jax.lax.scan(block, (u, z), jnp.arange(steps // record))
    return history


# ---------------------------------------------------------------- validation of the slow response ----------
def resonance_omega(g, a):
    def f(w):
        return Om / g - w - float(wavenumber(w)) * c * np.sqrt(1 - 1 / g**2) * np.cos(a)

    return brentq(f, 1e8, .5 * Om)


def design_lines(g, a, b, M, zeta, offset=0.):
    """Line spacing from the overlap rule 2 w_b(line) = dDelta_sp * zeta (w_b: single-line bounce frequency)."""
    w0 = resonance_omega(g, a)
    beta = np.sqrt(1 - 1 / g**2)
    k, dk = float(wavenumber(w0)), float(slope_k(w0))
    A = (Om / g) * (b / np.sqrt(M)) * (1 + w0 / (k * c) * np.cos(a) / beta)
    w_b = np.sqrt(A * k * c * beta * np.sin(a))
    dw = 2 * w_b / zeta / abs(1 + c * beta * np.cos(a) * dk)
    width = M * dw / w0
    return w0 * (1 + offset * width), width, w_b, w0


def run_case(name, g0, a0, b, M, zeta, offset=0., spread=False, T=6e-7, seed=0, N=128, dt=None, jitter=.8):
    centre, width, w_b, w0 = design_lines(g0, a0, b, M, zeta, offset)
    rng = np.random.default_rng(seed)
    frac = (np.arange(M) + .5) / M - .5
    w_l = centre * (1 + width * (frac + jitter * rng.uniform(-.5, .5, M) / M))   # noise-like: no line recurrence
    k_l = np.asarray(jax.vmap(wavenumber)(jnp.asarray(w_l)))
    gam = rng.uniform(g0 - 2, g0 + 2, N) if spread else np.full(N, g0)
    alp = rng.uniform(.1, .4, N) if spread else np.full(N, a0)
    th = rng.uniform(0, 2 * np.pi, N)
    pm = np.sqrt(gam**2 - 1) * c
    u = np.stack([pm * np.sin(alp) * np.cos(th), pm * np.sin(alp) * np.sin(th), pm * np.cos(alp)], 1)
    z = rng.uniform(0, 2 * np.pi / abs(k_l.mean()), N)
    dt = dt or 2 * np.pi / (Om / gam.min()) / 60
    steps = int(round(T / dt / 10)) * 10
    hist = np.asarray(resolved_window(jnp.asarray(u), jnp.asarray(z), jnp.asarray(w_l), jnp.asarray(k_l),
                                      jnp.full(M, b / np.sqrt(M)), jnp.asarray(rng.uniform(0, 2 * np.pi, (N, M))),
                                      dt, steps, steps // 10))
    res = {key: (hist[4, i] - (gam, alp)[i], hist[-1, i] - (gam, alp)[i]) for key, i in (("gamma", 0), ("alpha", 1))}
    # QL: the same initial ensemble through the slow operator only
    nd = band_nodes([centre], [width], [b], nodes=48)
    Nq = 4000 if quick else 16000
    idx = rng.integers(0, N, Nq)
    X0 = jnp.stack([jnp.sqrt(gam[idx]**2 - 1), jnp.asarray(alp[idx])], axis=1)
    half = 200
    X1, _ = ql_advance(X0, jax.random.key(seed + 7), nd, 0., steps * dt / 2 / half, half, False)
    X2, _ = ql_advance(X1, jax.random.key(seed + 8), nd, 0., steps * dt / 2 / half, half, False)
    ql = {"gamma": tuple(np.asarray(gamma_of(X)) - gam[idx] for X in (X1, X2)),
          "alpha": tuple(np.asarray(X[:, 1]) - alp[idx] for X in (X1, X2))}
    row = {"case": name, "gamma0": g0, "pitch0": a0, "b": b, "lines": M, "zeta": zeta, "offset_over_width": offset,
           "spread": spread, "jitter": jitter, "window_s": steps * dt, "dt_s": dt, "bounce_rad_s": w_b,
           "band_rel_width": width, "resonance_omega": w0, "particles": N}
    for key in ("alpha", "gamma"):
        (r1, r2), (q1, q2) = res[key], ql[key]
        grow_r, grow_q = r2.var(ddof=1) - r1.var(ddof=1), q2.var(ddof=1) - q1.var(ddof=1)
        # standard error of the resolved variance increment (Gaussian estimate)
        se = np.sqrt(2 / (N - 1)) * np.sqrt(r1.var(ddof=1)**2 + r2.var(ddof=1)**2)
        row[key] = {"var_half_resolved": float(r1.var(ddof=1)), "var_full_resolved": float(r2.var(ddof=1)),
                    "var_half_ql": float(q1.var(ddof=1)), "var_full_ql": float(q2.var(ddof=1)),
                    "growth_resolved": float(grow_r), "growth_ql": float(grow_q), "growth_se": float(se),
                    "growth_ratio": float(grow_r / grow_q) if abs(grow_q) > 0 else None,
                    "growth_z": float((grow_r - grow_q) / se),
                    "mean_resolved": float(r2.mean()), "mean_ql": float(q2.mean()),
                    "mean_se": float(r2.std(ddof=1) / np.sqrt(N))}
    return row


if "validate" in stages:
    say("== validate ==")
    M = 16 if quick else 32
    cases = [("calibration", dict(g0=16., a0=.2, b=4e-5, M=M, zeta=3.)),
             ("amp_x0.5", dict(g0=16., a0=.2, b=2e-5, M=M, zeta=3.)),
             ("amp_x2", dict(g0=16., a0=.2, b=8e-5, M=M, zeta=3.)),
             ("phase_seed2", dict(g0=16., a0=.2, b=4e-5, M=M, zeta=3., seed=2)),
             ("phase_seed3", dict(g0=16., a0=.2, b=4e-5, M=M, zeta=3., seed=3)),
             ("detune_+0.25W", dict(g0=16., a0=.2, b=4e-5, M=M, zeta=3., offset=.25)),
             ("detune_+0.6W_outside", dict(g0=16., a0=.2, b=4e-5, M=M, zeta=3., offset=.6)),
             ("distribution_spread", dict(g0=16., a0=.25, b=4e-5, M=M, zeta=3., spread=True)),
             ("gamma10_pitch0.1", dict(g0=10., a0=.1, b=4e-5, M=M, zeta=3.)),
             ("gamma25_pitch0.35", dict(g0=25., a0=.35, b=4e-5, M=M, zeta=3.)),
             ("sparse_zeta1", dict(g0=16., a0=.2, b=4e-5, M=M, zeta=1.)),
             ("dense_zeta6", dict(g0=16., a0=.2, b=4e-5, M=M, zeta=6.)),
             ("coherent_single_line", dict(g0=16., a0=.2, b=1e-4, M=1, zeta=3.)),
             ("coherent_three_lines", dict(g0=16., a0=.2, b=1e-4, M=3, zeta=3.))]
    if quick:
        cases = cases[:2] + cases[-1:]
    rows = []
    for name, kw in cases:
        t0 = time.perf_counter()
        row = run_case(name, **kw, T=1.5e-7 if quick else 6e-7, N=32 if quick else 160)
        row["wall_s"] = time.perf_counter() - t0
        rows.append(row)
        say(f"{name:24s} alpha growth ratio {row['alpha']['growth_ratio']:.3f} (z {row['alpha']['growth_z']:+.1f})  "
            f"gamma growth ratio {row['gamma']['growth_ratio']:.3f} (z {row['gamma']['growth_z']:+.1f})  "
            f"mean gamma res {row['gamma']['mean_resolved']:+.3e} ql {row['gamma']['mean_ql']:+.3e}  "
            f"{row['wall_s']:.0f}s")
    report["validation"] = {"cases": rows}
    save()


# ---------------------------------------------------------------- explicit-current source ledger ------------
T_on = .3
tau_e = 6 * np.sqrt(2) * np.pi**1.5 * eps**2 * np.sqrt(me) * (Te * e)**1.5 / (logarithm * e**4 * ne)
nu_e = 1 / tau_e                                               # Braginskii electron collision rate (1/s)


def positive_linear(a, b, dt):
    """Positive part of the integral of a piecewise-linear power, with the zero crossing resolved."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    cross = (a * b) < 0
    den = np.where(cross, np.abs(b - a), 1.)
    area = (np.maximum(a, 0)**2 + np.maximum(b, 0)**2) / den
    return .5 * dt * np.where(cross, area, np.maximum(a + b, 0.))


def line_system(w):
    """Cold electron-ion response of a circular R wave with unit B amplitude, in the frame e^{i beta}.

    The prescribed total field comes from a vector potential, A = g(t) A0 e^{i beta}, B = curl A, E = -dA/dt,
    so Faraday's law holds during the ramps. Returns the signed k, A0 and the fluid generator."""
    k = -float(wavenumber(w))                                   # signed: counter-propagating
    Lam = np.array([[1j * (Om - w) - nu_e, nu_e], [me / mi * nu_e, -1j * (Oi + w) - me / mi * nu_e]])
    return k, 1 / k, Lam, np.array([-e / me, e / mi], complex)


def drive_power(w, g, gd, gdd, s, k, A0):
    """-J_drive.E per volume for unit B amplitude, J_drive = curl B/mu0 - eps0 dE/dt - J_plasma."""
    E = -(gd + 1j * w * g) * A0
    dE = -(gdd + 2j * w * gd - w**2 * g) * A0
    J_pl = -e * ne * s[0] + e * ne * s[1]
    J_d = k * (k * g * A0) / mu0 - eps * dE - J_pl
    return -np.real(np.conj(J_d) * E)


def ledger(w, tau_ramp=2e-5, n=4000):
    """Ramp-up (sin^2) to steady state, settle, ramp-down; per unit B^2. Returns energies and the closure."""
    k, A0, Lam, f = line_system(w)
    lam, V = np.linalg.eig(Lam)
    Vi = np.linalg.inv(V)
    settle = 6 / nu_e                                           # six electron collision times
    t = np.linspace(0, 2 * tau_ramp + settle, 2 * n + 1)
    dt = t[1] - t[0]
    up, down = np.clip(t / tau_ramp, 0, 1), np.clip((t - tau_ramp - settle) / tau_ramp, 0, 1)
    pi_t = np.pi / tau_ramp
    g = np.where(t < tau_ramp + settle, .5 * (1 - np.cos(np.pi * up)), .5 * (1 + np.cos(np.pi * down)))
    gd = np.where(t < tau_ramp + settle, .5 * pi_t * np.sin(np.pi * up), -.5 * pi_t * np.sin(np.pi * down))
    gdd = np.where(t < tau_ramp + settle, .5 * pi_t**2 * np.cos(np.pi * up), -.5 * pi_t**2 * np.cos(np.pi * down))
    # exponential integrator, forcing linear across each step (exact for piecewise-linear E): a frozen
    # forcing excites spurious gyro-frequency transients whose rectified power depends on the step.
    E = -(gd + 1j * w * g) * A0
    s = np.zeros((len(t), 2), complex)
    decay = np.exp(lam * dt)
    cf = Vi @ f
    for i in range(len(t) - 1):
        s[i + 1] = V @ (decay * (Vi @ s[i]) + cf * ((decay - 1) / lam * E[i]
                                                    + (decay - 1 - lam * dt) / lam**2 * (E[i + 1] - E[i]) / dt))
    P = np.array([drive_power(w, g[i], gd[i], gdd[i], s[i], k, A0) for i in range(len(t))])
    W_field = .5 * eps * np.abs(E)**2 + .5 * (g * k * A0)**2 / mu0
    W_kin = .5 * ne * (me * np.abs(s[:, 0])**2 + mi * np.abs(s[:, 1])**2)
    P_diss = ne * me * nu_e * np.abs(s[:, 0] - s[:, 1])**2
    supplied = positive_linear(P[:-1], P[1:], dt).sum()
    returned = positive_linear(-P[:-1], -P[1:], dt).sum()

    def cumulative(y):
        return np.concatenate(([0.], np.cumsum(.5 * (y[:-1] + y[1:]) * dt)))

    mid = len(t) // 2
    ramp_up = (cumulative(P)[mid] - (W_field + W_kin)[mid] - cumulative(P_diss)[mid])
    return {"supplied": supplied, "returned": returned, "net": cumulative(P)[-1], "dissipated": cumulative(P_diss)[-1],
            "closure_ramp_up_residual_over_stored": ramp_up / (W_field + W_kin)[mid],
            "steady_power": P[mid], "ramp_time": tau_ramp, "settle": settle,
            "stored_field_plus_kinetic": (W_field + W_kin)[mid]}


source_cache = {}


def source_coefficients(w_grid):
    """Per unit B0^2 b^2: steady power (W/m^3), ramp supplied and returned (J/m^3)."""
    out = np.zeros((len(w_grid), 3))
    for i, w in enumerate(w_grid):
        led = ledger(w, n=800)
        out[i] = led["steady_power"], led["supplied"], led["returned"]
    return out


if "source" in stages:
    say("== source ==")
    rec = {"nu_e_per_s": nu_e, "T_on_s": T_on, "convention":
           "total wave field prescribed; J_drive = curl B/mu0 - eps0 dE/dt - J_plasma; "
           "P_src = -J_drive.E averaged over a wavelength; "
           "ideal nonrecovering port Q+ = int max(P,0) dt (piecewise-linear zero crossing), returned energy Q- dumped."}
    checks = []
    for w in (3e9, 9e9, 2e10):
        led = ledger(w, n=4000)
        led2 = ledger(w, n=8000)
        checks.append({"omega": w, "n4000": led, "n8000": led2,
                       "supplied_change_on_refinement": abs(led2["supplied"] - led["supplied"]) / led2["supplied"]})
        say(f"omega {w:.1e}: steady P {led['steady_power']:.3e} W/m3 per T^2, supplied {led['supplied']:.3e}, "
            f"returned {led['returned']:.3e}, stored {led['stored_field_plus_kinetic']:.3e}, "
            f"closure {led['closure_ramp_up_residual_over_stored']:.1e}")
    rec["checks"] = checks
    # steady power against the closed form n e^2 nu |E|^2 / (m ((Om-w)^2 + nu^2)) for electrons only
    w = 9e9
    k, A0, Lam, f = line_system(w)
    approx = ne * e**2 * nu_e * abs(w * A0)**2 / (me * ((Om - w)**2 + nu_e**2))
    rec["steady_power_vs_closed_form"] = {"ledger": ledger(w, n=800)["steady_power"], "closed_form": approx}
    report["source"] = rec
    save()


# ---------------------------------------------------------------- minimum-RF optimization ------------------
a_study = E_study * e / (me * c)
dt_slow = 5e-5
step_limit = .01                                                  # D_max dt: validity of the Euler-Maruyama step
eps_c, confidence = .02, .95                                      # hard constraint: tail fraction above 10 MeV
w_grid = np.geomspace(2e9, 3.4e10, 40)
hold_extra = float(os.environ.get("JIC_HOLD_EXTRA", 0.))           # robust variant: a margin added to the hold
T_hold = round(crossing_time(E_study, 5., .9, gamma_thr), 3) + hold_extra   # no-RF regrowth gamma 5 -> 10 MeV
horizon = T_on + T_hold
NODES = 24
bounds = np.array([[9.3, 10.5], [.03, 1.6], [-7., -4.]])          # log10 omega_c, rel width, log10 b (per band)
optimize_tags = {"opt_single": "single_scan", "opt_random": "random_search", "opt_es": "multimode_es"}


def fragment(tag):
    return out_dir / f"optimize_{tag}.json" if out_dir else None


coef_grid = None


def seed_tail(n, key):
    k1, k2 = jax.random.split(key)
    g = jax.random.uniform(k1, (n,), minval=5., maxval=25.)
    a = jax.random.uniform(k2, (n,), minval=.05, maxval=.5)
    return jnp.stack([jnp.sqrt(g**2 - 1), a], axis=1)


@jax.jit
def max_diffusion(nd):
    """Largest D_aa on a (gamma, pitch) grid covering the tail: the scheme needs D_max dt << 1."""
    grid = jnp.stack(jnp.meshgrid(jnp.sqrt(jnp.geomspace(2., 50., 48)**2 - 1), jnp.array([.03, .15, .5, 1.2])), -1)
    return jnp.max(jax.vmap(jax.vmap(lambda X: diffusion_tensor(X, nd)[1, 1]))(grid))


def decode(x):
    x = np.asarray(x).reshape(-1, 3)
    return 10**x[:, 0], x[:, 1], 10**x[:, 2]


def upper_bound(k, n):
    return 1. if k >= n else float(beta_dist.ppf(confidence, k + 1, n - k))


def simulate(x, n, key, dt=dt_slow, nodes=NODES, rf=True):
    X = seed_tail(n, key)
    if rf:
        nd = band_nodes(*decode(x), nodes)
        X, dgam = ql_advance(X, jax.random.fold_in(key, 99), nd, a_study, dt, int(round(T_on / dt)), True)
    else:
        X, dgam = hold_advance(X, a_study, dt, int(round(T_on / dt))), jnp.zeros(n)
    g_on = np.asarray(gamma_of(X))
    g_end = np.asarray(gamma_of(hold_advance(X, a_study, dt, int(round(T_hold / dt)))))
    return {"above_on": int((g_on > gamma_thr).sum()), "above_end": int((g_end > gamma_thr).sum()), "n": n,
            "mean_gamma_end": float(g_end.mean()), "mean_dgamma_rf": float(np.asarray(dgam).mean()),
            "max_gamma_end": float(g_end.max())}


def cost(x, simulated=None, nodes=NODES):
    """Supplied (Q+), returned (Q-) and net energy per volume for the controls: bulk ledger plus tail exchange."""
    frac = (np.arange(nodes) + .5) / nodes - .5
    steady = plus = minus = 0.
    for c0, w0, b0 in zip(*decode(x)):
        table = np.stack([np.interp(c0 * (1 + w0 * frac), w_grid, coef_grid[:, j]) for j in range(3)])
        scale = (b0 * B0)**2 / nodes
        steady += scale * table[0].sum()
        plus += scale * table[1].sum()
        minus += scale * table[2].sum()
    tail = tail_fraction * ne * me * c**2 * simulated["mean_dgamma_rf"] / T_on if simulated else 0.
    power = steady + tail
    return {"Q_plus_J_m3": max(power, 0.) * T_on + plus, "Q_minus_J_m3": max(-power, 0.) * T_on + minus,
            "bulk_steady_W_m3": steady, "tail_W_m3": tail, "ramp_supplied_J_m3": plus,
            "net_J_m3": power * T_on + plus - minus}


def evolution_strategy(objective, budget, rng, lo, hi):
    """(mu/mu, lambda) evolution strategy over three bands, started at three wide bands."""
    x = np.concatenate([[9.6, .7, -5.], [10., .7, -5.], [10.35, .7, -5.]])
    fx, used, sigma, lam, mu = objective(x), 1, .15 * (hi - lo), 6, 2
    while used + lam + 1 <= budget:
        kids = [np.clip(x + sigma * rng.standard_normal(len(x)), lo, hi) for _ in range(lam)]
        vals = [objective(k) for k in kids]
        cand = np.mean([kids[i] for i in np.argsort(vals)[:mu]], axis=0)
        fc, used = objective(cand), used + lam + 1
        if fc < fx:
            x, fx, sigma = cand, fc, sigma * 1.2
        else:
            sigma *= .85


def run_strategy(tag, budget, n_search):
    key = jax.random.key(2024)
    history, started = [], time.perf_counter()

    def objective(x):
        step = float(max_diffusion(band_nodes(*decode(x), NODES))) * dt_slow
        sim = simulate(x, n_search, key) if step <= step_limit else {
            "above_on": n_search, "above_end": n_search, "n": n_search, "mean_gamma_end": 0., "mean_dgamma_rf": 0.,
            "max_gamma_end": 0.}
        ub, q = upper_bound(sim["above_end"], n_search), cost(x, sim)
        feasible = ub <= eps_c and step <= step_limit
        history.append({"x": np.asarray(x).tolist(), "feasible": bool(feasible), "ub": ub, "Q_plus": q["Q_plus_J_m3"],
                        "D_max_dt": step,
                        "sim": sim, "cost": q})
        if len(history) % 10 == 0:
            say(tag, len(history), "evaluations", f"{time.perf_counter() - started:.0f}s")
        return q["Q_plus_J_m3"] if feasible else 1e3 * (1 + ub)

    K = 3
    lo, hi = np.tile(bounds[:, 0], K), np.tile(bounds[:, 1], K)
    rng = np.random.default_rng({"single_scan": 1, "random_search": 2, "multimode_es": 3}[tag])
    if tag == "single_scan":                                       # grid over centre and amplitude, width 0.06
        n_w = max(budget // 30, 1)
        for width in (.06, .15, .3, .6, 1., 1.5):                  # centre x amplitude x width grid
            for wlog in np.linspace(9.5, 10.45, n_w):
                for blog in np.linspace(-6., -4.5, 5):
                    objective([wlog, width, blog])
    elif tag == "random_search":                                   # gradient-free baseline, three bands
        for _ in range(budget):
            objective(rng.uniform(lo, hi))
    else:
        evolution_strategy(objective, budget, rng, lo, hi)
    return history


def lines_required(x, zeta=3., pitch=.2):
    """Lines per band so that the overlap parameter zeta = 2 w_b / dDelta_sp holds at the resonant tail energy
    (the rule the validation windows used); a noise-like broadband drive has effectively unlimited lines."""
    counts = []
    for w0, width, b in zip(*decode(x)):
        k, dk = float(wavenumber(w0)), float(slope_k(w0))

        def mismatch(g):
            return Om / g - w0 - k * c * np.sqrt(1 - 1 / g**2) * np.cos(pitch)

        if mismatch(1.0001) * mismatch(500.) > 0:
            counts.append(None)
            continue
        g = brentq(mismatch, 1.0001, 500.)
        beta = np.sqrt(1 - 1 / g**2)
        A1 = (Om / g) * (1 + w0 / (k * c) * np.cos(pitch) / beta)
        spacing = abs(1 + c * beta * np.cos(pitch) * dk) * width * w0
        counts.append(float((spacing * zeta / 2 / np.sqrt(A1 * b * k * c * beta * np.sin(pitch)))**(4 / 3)))
    return counts


def best_of(history):
    feasible = [h for h in history if h["feasible"]]
    return min(feasible, key=lambda h: h["Q_plus"]) if feasible else None


for stage, tag in optimize_tags.items():
    if stage in stages:
        say("==", stage, "==")
        coef_grid = source_coefficients(w_grid)
        budget, n_search = (12, 256) if quick else (240, 512)
        hist = run_strategy(tag, budget, n_search)
        best = best_of(hist)
        say(tag, "evaluations", len(hist), "feasible", sum(h["feasible"] for h in hist),
            "best Q+", best["Q_plus"] if best else None)
        report[stage] = {"tag": tag, "budget_evaluations": budget, "particles_search": n_search,
                         "feasible": sum(h["feasible"] for h in hist), "history": hist,
                         "best": best, "T_on_s": T_on, "T_hold_s": T_hold}
        save()
        if out_dir is not None:
            fragment(tag).write_text(json.dumps(report[stage], indent=1, default=float) + "\n")


def check_control(xb, n_v, first_only=False):
    """Frozen-control checks: independent seeds, dt/2, 2x nodes, and the amplitude +-30% of the QL uncertainty."""
    variants = (("independent_seeds", {}), ("dt_half", {"dt": dt_slow / 2}), ("nodes_x2", {"nodes": 2 * NODES}),
                ("amplitude_x1.3", {"scale": 1.3}), ("amplitude_x0.77", {"scale": 1 / 1.3}))
    runs = []
    for label, kw in variants[:1] if first_only else variants:
        xs = np.array(xb)
        xs[2::3] += np.log10(kw.pop("scale", 1.))
        sim = simulate(xs, n_v, jax.random.key(777), **kw)
        ub = upper_bound(sim["above_end"], n_v)
        runs.append({"check": label, "above_end": sim["above_end"], "n": n_v, "fraction": sim["above_end"] / n_v,
                     "upper_bound": ub, "passes": ub <= eps_c, "mean_gamma_end": sim["mean_gamma_end"],
                     "cost": cost(xs, sim, nodes=kw.get("nodes", NODES))})
        say("   ", label, runs[-1]["fraction"], ub, runs[-1]["passes"])
    return runs


if "verify" in stages:
    say("== verify ==")
    coef_grid = source_coefficients(w_grid)
    n_v = 4096 if quick else 8192
    verification = {}
    chosen = os.environ.get("JIC_VERIFY", "").split(",")           # optional: verify only these strategies
    for tag in [t for t in optimize_tags.values() if t in chosen or chosen == [""]]:
        path = fragment(tag)
        history = json.loads(path.read_text())["history"] if path and path.exists() else []
        feasible = [h for h in history if h["feasible"]]
        verification[tag] = {"feasible_in_search": len(feasible), "rejected": [], "accepted": None}
        for cand in sorted(feasible, key=lambda h: h["Q_plus"])[:int(os.environ.get("JIC_CANDIDATES", 30))]:
            say(tag, "candidate Q+", cand["Q_plus"])                     # best first; selection used the search seeds
            runs = check_control(cand["x"], n_v, first_only=True)
            if runs[0]["passes"]:
                runs = check_control(cand["x"], n_v)
                verification[tag]["accepted"] = {
                    "search_Q_plus": cand["Q_plus"], "controls": cand["x"],
                    "bands_omega_width_b": [list(map(float, v)) for v in decode(cand["x"])],
                    "lines_required_per_band": lines_required(cand["x"]), "checks": runs,
                    "all_checks_pass": all(r["passes"] for r in runs)}
                break
            verification[tag]["rejected"].append({"search_Q_plus": cand["Q_plus"], "controls": cand["x"],
                                                  "independent_seed_fraction": runs[0]["fraction"]})
        if out_dir is not None:
            (out_dir / f"verify_{tag}.json").write_text(json.dumps(verification[tag], indent=1, default=float) + "\n")
    nr = simulate(None, n_v, jax.random.key(777), rf=False)
    verification["no_RF"] = {"above_end": nr["above_end"], "n": n_v, "fraction": nr["above_end"] / n_v,
                             "mean_gamma_end": nr["mean_gamma_end"]}
    report["verification"] = verification
    save()


if "margin" in stages:
    say("== margin ==")
    margins = {}
    for tag in optimize_tags.values():
        path = out_dir / f"verify_{tag}.json" if out_dir else None
        accepted = json.loads(path.read_text())["accepted"] if path and path.exists() else None
        if accepted is None:
            continue
        n_m = 2048 if quick else 4096
        X = seed_tail(n_m, jax.random.key(4242))
        nd = band_nodes(*decode(accepted["controls"]), NODES)
        X, _ = ql_advance(X, jax.random.key(4343), nd, a_study, dt_slow, int(round(T_on / dt_slow)), True)
        X = hold_advance(X, a_study, dt_slow, int(round(T_hold / dt_slow)))
        rows, elapsed = [], 0.
        for extra in (0., .5, 1., 2., 4.):
            X = hold_advance(X, a_study, dt_slow, int(round((extra - elapsed) / dt_slow)))
            elapsed = extra
            g = np.asarray(gamma_of(X))
            rows.append({"seconds_after_hold": extra, "fraction_above": float((g > gamma_thr).mean()),
                         "mean_gamma": float(g.mean())})
            say(tag, rows[-1])
        margins[tag] = rows
    report["margin_after_hold"] = margins
    save()
    if out_dir is not None:
        (out_dir / "margin.json").write_text(json.dumps(margins, indent=1) + "\n")
