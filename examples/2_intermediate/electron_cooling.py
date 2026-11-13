"""Classical magnetic cooling of an electron and a gyrophase-mixed beam.

Both cases use physical electron charge and mass, the actual PIC time step and
Radiation(model="ll_reduced"). The very low density suppresses collective fields:
this is an orbit/beam benchmark, not a collective-plasma cooling calculation.
The strong field makes cooling measurable within nanoseconds without scaling the
radiation force. Compare every marker with the exact uniform-B energy solution,
and check an autodiff sensitivity against its analytic derivative.

A third case is the collective ring formation of Bilbao & Silva, PRL 130, 165101 (2023): an isotropic
Maxwellian of 3D momentum spread 50 m c in a uniform field cools perpendicular momentum faster than
parallel, and its perpendicular density f(p_perp), integrated over p_parallel, develops a ring. The
self-consistent PIC fields are on; omega_p/omega_ce~3e-4, so (their Eq. 9) the ring forms long before
any maser growth. Each particle is compared with the exact uniform-B solution evolved from the same
sample, which isolates the PIC integration error; the ring radius is compared with their Eq. (7),
which assumes p_parallel << p_perp and only approximates the exact-LL ring.

A short quantum case, chi = 0.29, runs `Radiation(model="ll_quantum")`: the Gaunt factor of Niel et al.
(Plasma Phys. Control. Fusion 60, 094002, 2018) weakens the recoil, and the final energy is compared with an
independent integration of d gamma/dt = -a g(chi) (gamma^2 - 1).

Refine the macro time step to resolve gyrophase.
Runs in about 25 s; the settings marked `research` in the script give the
converged version.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar

from jaxincell import (Domain, Radiation, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0, figure, mass_electron as m, speed_of_light as c)

B0, duration, density = 1000., 1e-9, 1e-3      # T, s, m^-3; an intentionally dilute test ensemble
steps, cells, length = 4096, 8, .02   # research: 32768
frames = 128
double_precision = jax.config.read("jax_enable_x64")
gamma_tolerance = 1e-8 if double_precision else .003
energy_tolerance = 1e-10 if double_precision else .005
tau = e**2 / (6 * np.pi * epsilon_0 * m * c**3)
B_schwinger = 4.4140052e9                    # electron critical magnetic field, T


def simulation(u0, strength, count=steps, model="ll_reduced", span=duration):
    n = u0.shape[0]
    x = jnp.zeros((n, 3)).at[:, 0].set(length * ((jnp.arange(n) + .5) / n - .5))
    electrons = Species.electrons(n=n, density=density, x=x, u=u0)
    return Simulation(Domain(length=length, cells=cells, time_step=span / count),
                      (electrons,), Solver(relativistic=True),
                      external_B=jnp.broadcast_to(jnp.array([strength, 0., 0.]), (cells, 3)),
                      radiation=Radiation(model=model, species="electrons"))


def exact_gamma(u0, t, strength):
    """v_parallel is constant; the perpendicular rest-frame gamma obeys a Riccati equation."""
    eta = np.asarray(u0) / c
    g0 = np.sqrt(1 + np.sum(eta**2, axis=1))
    boost = g0 / np.sqrt(1 + np.sum(eta[:, 1:]**2, axis=1))
    transverse_gamma = g0 / boost
    a = tau * (e * strength / m)**2
    z = (transverse_gamma - 1) / (transverse_gamma + 1) * np.exp(-2 * a * np.asarray(t)[..., None] / boost)
    return boost * (1 + z) / (1 - z)


# One perpendicular electron, then a narrow beam with a spread in energy and pitch.
mono = jnp.array([[0., c * np.sqrt(1000.**2 - 1), 0.]])
g0 = np.linspace(900., 1100., 32)
pitch = np.linspace(.6, .8, 32)
phase = 2 * np.pi * np.arange(32) / 32
beam = jnp.asarray(c * np.sqrt(g0[:, None]**2 - 1) * np.column_stack(
    (np.cos(pitch), np.sin(pitch) * np.cos(phase), np.sin(pitch) * np.sin(phase))))
fig, axes = figure(3)
for label, u0 in (("electron", mono), ("beam", beam)):
    out = simulation(u0, B0).run(steps, store_every=steps // frames, moments=False).validate()
    gamma = np.sqrt(1 + np.sum((np.asarray(out.u) / c)**2, axis=-1))
    analytic = exact_gamma(u0, np.asarray(out.t), B0)
    initial = np.sqrt(1 + np.sum((np.asarray(u0) / c)**2, axis=1))
    energy0 = density * length * m * c**2 * np.mean(initial - 1)
    energy_end = float(out.work.kinetic[-1, 0])
    emitted = float(out.work.radiated_energy[-1, 0])
    reaction = float(out.work.rr_work[-1, 0])
    gamma_error = np.max(abs(gamma / analytic - 1))
    energy_residual = (energy_end + emitted - energy0) / energy0
    assert gamma_error < gamma_tolerance, f"{label}: cooling does not match the exact solution"
    assert abs(energy_residual) < energy_tolerance, f"{label}: emission does not close the magnetic energy balance"
    assert emitted > 0 and reaction < 0, f"{label}: radiation did not remove mechanical energy"
    print(f"{label}: max relative gamma error {gamma_error:.3e}; "
          f"(K_final + emitted - K_initial)/K_initial = {energy_residual:.3e}")
    print(f"  mechanical RR work {reaction:.6e} J/m^2; emitted energy {emitted:.6e} J/m^2; "
          f"initial chi <= {np.max(initial) * B0 / B_schwinger:.3e}")
    if label == "electron":
        a = tau * (e * B0 / m)**2
        s0 = np.arctanh(1 / initial[0])

        def logcosh(s):
            return np.log1p(2 * np.sinh(s / 2)**2)
        phase_exact = e * B0 / (m * a) * (logcosh(s0 + a * duration) - logcosh(s0))
        final_u = np.asarray(out.state.u[0])
        phase_error = np.angle((final_u[1] + 1j * final_u[2]) * np.exp(-1j * phase_exact))
        phase_tolerance = .05 + (0. if double_precision else .02)   # research: .001
        assert abs(phase_error) < phase_tolerance, "the Lorentz gyrophase needs a smaller macro time step"
        print(f"  final gyrophase error (modulo 2 pi): {phase_error:.3e} rad")
    line, = axes[0].plot(np.asarray(out.t) * 1e9, np.mean(gamma, axis=1), label=label)
    axes[0].plot(np.asarray(out.t) * 1e9, np.mean(analytic, axis=1), "--", color=line.get_color())
    axes[1].plot(np.asarray(out.t) * 1e9, np.mean(gamma / analytic - 1, axis=1), label=label)


# Quantum correction to the mean recoil. At chi = gamma B / B_Schwinger of order one the emitted power falls
# below the classical one by the Gaunt factor g(chi) (Niel et al., Plasma Phys. Control. Fusion 60, 094002, 2018);
# Radiation(model="ll_quantum") applies it to the force, with no stochastic emission. A perpendicular electron then
# obeys d gamma/dt = -a g(chi) (gamma^2 - 1), a = tau (eB/m)^2, which scipy integrates independently.
from scipy.integrate import solve_ivp
from jaxincell._core import gaunt

gamma_q, span_q = 1.3e6, 4e-12
u_q = jnp.array([[0., c * np.sqrt(gamma_q**2 - 1), 0.]])
a = tau * (e * B0 / m)**2
reference = solve_ivp(lambda t, g: -a * float(gaunt(g[0] * B0 / B_schwinger)) * (g**2 - 1), (0, span_q), [gamma_q],
                      rtol=1e-10, atol=1e-6, t_eval=[span_q]).y[0, -1]
final = {}
for model in ("ll_reduced", "ll_quantum"):
    q_out = simulation(u_q, B0, 512, model, span_q).run(512, store_every=512, moments=False).validate()
    final[model] = float(jnp.sqrt(1 + jnp.sum((q_out.u[-1, 0] / c)**2)))
print(f"chi = {gamma_q * B0 / B_schwinger:.2f}: final gamma {final['ll_reduced']:.4e} classical, "
      f"{final['ll_quantum']:.4e} with the Gaunt factor, {reference:.4e} from the ODE")
assert abs(final["ll_quantum"] / reference - 1) < 1e-3 and final["ll_quantum"] > final["ll_reduced"]


# Collective ring formation. Gyrophase does not matter to a homogeneous ensemble and the Boris rotation
# keeps |p_perp|, so the step follows the cooling time t_R = 3/(4 alpha chi_th), not the gyroperiod.
alpha = e**2 / (4 * np.pi * epsilon_0 * hbar * c)
B_ring, p_th, n_ring, per_ring, n_e = 1e5, 50., 20000, 400, 1e22   # research: 200000
omega_c, chi_th = e * B_ring / m, 50. * 1e5 / B_schwinger
t_ring = 3 / (4 * alpha * chi_th) / omega_c
ring_dt, ring_cells = t_ring / per_ring, 8
ring_length = ring_cells * 1.5 * c * ring_dt
eta_ring = np.random.default_rng(0).normal(0, p_th, (n_ring, 3))
ring_x = jnp.zeros((n_ring, 3)).at[:, 0].set(ring_length * ((jnp.arange(n_ring) + .5) / n_ring - .5))
ring_out = Simulation(
    Domain(length=ring_length, cells=ring_cells, time_step=ring_dt),
    (Species.electrons(n=n_ring, density=n_e, x=ring_x, u=jnp.asarray(c * eta_ring)),), Solver(relativistic=True),
    external_B=jnp.broadcast_to(jnp.array([B_ring, 0., 0.]), (ring_cells, 3)),
    radiation=Radiation(species="electrons")
).run(3 * per_ring, store_every=per_ring // 4, moments=False).validate()
ring_t, u_ring = np.asarray(ring_out.t), np.asarray(ring_out.u) / c


def ring_profile(p_perp):
    """Smoothed f(p_perp) (counts per unit perpendicular-plane area) and its peak radius."""
    counts, edges = np.histogram(p_perp, bins=np.linspace(0, np.percentile(p_perp, 99), 400))
    centre = .5 * (edges[1:] + edges[:-1])
    density = np.convolve(counts / centre, np.ones(10) / 10, "same")
    return centre, density, centre[np.argmax(density[10:-10]) + 10]


def ring_theory(t):
    """Eq. (7) of Bilbao & Silva, tau = (2/3) alpha B_0 omega_ce t with B_0 = B/B_Schwinger."""
    tau_t = 2 / 3 * alpha * B_ring / B_schwinger * omega_c * t
    return (1 + 6 * p_th**2 * tau_t**2 - np.sqrt(1 + 12 * p_th**2 * tau_t**2)) / (6 * p_th**2 * tau_t**3)


def exact_perpendicular(t):
    """|p_perp|/(m c): v_parallel is constant, so eta_parallel = gamma * eta_parallel0 / gamma0."""
    gamma_t = exact_gamma(c * eta_ring, t, B_ring)
    eta_par = gamma_t * eta_ring[:, 0] / np.sqrt(1 + np.sum(eta_ring**2, axis=1))
    return np.sqrt(np.maximum(gamma_t**2 - 1 - eta_par**2, 0.))


measured = [np.hypot(u[:, 1], u[:, 2]) for u in u_ring]
exact_ring = [exact_perpendicular(t) for t in ring_t]
radius = np.array([ring_profile(p)[2] for p in measured])
exact_radius = np.array([ring_profile(p)[2] for p in exact_ring])
errors = [abs(a / b - 1)[b > 1] for a, b in zip(measured[1:], exact_ring[1:])]
pperp_error, pperp_worst = max(np.percentile(q, 99) for q in errors), max(np.max(q) for q in errors)
late = ring_t >= .6 * t_ring
asymptotic = ring_t >= 1.5 * t_ring    # Eq. (7) needs p_parallel << p_perp, true after differential cooling
theory = ring_theory(ring_t[late])
centre, profile, peak = ring_profile(measured[-1])
inversion = np.max(profile) / np.mean(profile[centre < .3 * peak])
print(f"ring: chi_th {chi_th:.2e}, t_R {t_ring:.3e} s, {n_ring} particles, {per_ring} steps per t_R, "
      f"omega_p/omega_ce {np.sqrt(n_e * e**2 / (epsilon_0 * m)) / omega_c:.1e}")
print(f"  |p_perp|/exact - 1 for exact p_perp > m c: 99th percentile {pperp_error:.2e}, worst {pperp_worst:.2e} "
      f"; radius peaks at "
      f"{ring_t[np.argmax(radius)] / t_ring:.2f} t_R (Eq. 8: 1); f(p_R)/f(core) at 3 t_R: {inversion:.2f}")
print("  t/t_R", np.round(ring_t[late] / t_ring, 2), "\n  PIC radius", np.round(radius[late], 2),
      "\n  exact-LL ensemble", np.round(exact_radius[late], 2), "\n  Eq. (7)", np.round(theory, 2))
assert pperp_error < 3e-3, "the PIC ensemble does not follow the exact uniform-B cooling"
assert np.all(abs(radius[late] / exact_radius[late] - 1) < .06), "the PIC ring differs from its exact-LL ensemble"
assert np.all(abs(radius[asymptotic] / ring_theory(ring_t[asymptotic]) - 1) < .15), "ring radius departs from Eq. (7)"
assert .6 <= ring_t[np.argmax(radius)] / t_ring <= 1.8, "ring formation time disagrees with t_R"
assert inversion > 1.5, "no population inversion df/dp_perp > 0"


def final_gamma(strength):
    out = simulation(mono, strength).run(steps, store_every=steps, store_particles=False, moments=False)
    return jnp.sqrt(1 + jnp.sum((out.state.u[0] / c)**2))


# Forward mode stores no reverse trajectory; the control is a physical field strength.
value, derivative = jax.jit(lambda b: jax.jvp(final_gamma, (b,), (1.,)))(B0)
expected = exact_gamma(mono, duration, B0)[0]
a = tau * (e * B0 / m)**2
exact_derivative = -2 * a * duration / B0 * (expected**2 - 1)
assert abs(float(value) / expected - 1) < gamma_tolerance, "the differentiated forward result is inaccurate"
assert abs(float(derivative) / exact_derivative - 1) < (2e-7 if double_precision else .02), \
    "the field-strength derivative does not match the analytic sensitivity"
print(f"d gamma_final/d B: AD {float(derivative):.8e} /T, exact {exact_derivative:.8e} /T; "
      f"relative error {abs(float(derivative) / exact_derivative - 1):.3e}")
print("Energy accuracy alone does not establish gyrophase accuracy: refine the macro time step as well.")
axes[0].set(xlabel="time (ns)", ylabel="mean Lorentz factor", title="solid: simulation; dashed: exact")
axes[1].set(xlabel="time (ns)", ylabel="mean relative gamma error", title="uniform magnetic field")
axes[2].plot(ring_t / t_ring, radius, "o", label="PIC")
axes[2].plot(ring_t / t_ring, exact_radius, "-", label="exact-LL ensemble")
axes[2].plot(ring_t[late] / t_ring, theory, "--", label="Bilbao & Silva Eq. (7)")
axes[2].set(xlabel="t / t_R", ylabel="ring radius (m c)", title="collective ring formation")
for ax in axes:
    ax.legend()
fig.tight_layout()
plt.show()
