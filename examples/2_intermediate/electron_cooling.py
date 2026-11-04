"""Classical magnetic cooling of an electron and a gyrophase-mixed beam.

Both cases use physical electron charge and mass, the actual PIC time step and
Radiation(model="ll_reduced"). The very low density suppresses collective fields:
this is an orbit/beam benchmark, not a collective-plasma cooling calculation.
The strong field makes cooling measurable within nanoseconds without scaling the
radiation force. Compare every marker with the exact uniform-B energy solution,
and check an autodiff sensitivity against its analytic derivative.

Run with --quick for fewer steps; refine the macro time step to resolve gyrophase.
No files are written. Edit the parameters below, or save the displayed figure.
"""
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Radiation, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0, figure, mass_electron as m, speed_of_light as c)

quick = "--quick" in sys.argv
B0, duration, density = 1000., 1e-9, 1e-3      # T, s, m^-3; an intentionally dilute test ensemble
steps, cells, length = (4096 if quick else 32768), 8, .02
frames = 128
tau = e**2 / (6 * np.pi * epsilon_0 * m * c**3)
B_schwinger = 4.4140052e9                    # electron critical magnetic field, T


def simulation(u0, strength, count=steps):
    n = u0.shape[0]
    x = jnp.zeros((n, 3)).at[:, 0].set(length * ((jnp.arange(n) + .5) / n - .5))
    electrons = Species.electrons(n=n, density=density, x=x, u=u0)
    return Simulation(Domain(length=length, cells=cells, time_step=duration / count),
                      (electrons,), Solver(relativistic=True),
                      external_B=jnp.broadcast_to(jnp.array([strength, 0., 0.]), (cells, 3)),
                      radiation=Radiation(species="electrons"))


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
fig, axes = figure(2)
for label, u0 in (("electron", mono), ("beam", beam)):
    out = simulation(u0, B0).run(steps, store_every=steps // frames, moments=False)
    gamma = np.sqrt(1 + np.sum((np.asarray(out.u) / c)**2, axis=-1))
    analytic = exact_gamma(u0, np.asarray(out.t), B0)
    initial = np.sqrt(1 + np.sum((np.asarray(u0) / c)**2, axis=1))
    energy0 = density * length * m * c**2 * np.mean(initial - 1)
    energy_end = float(out.work.kinetic[-1, 0])
    emitted = float(out.work.radiated_energy[-1, 0])
    reaction = float(out.work.rr_work[-1, 0])
    print(f"{label}: max relative gamma error {np.max(abs(gamma / analytic - 1)):.3e}; "
          f"(K_final + emitted - K_initial)/K_initial = {(energy_end + emitted - energy0) / energy0:.3e}")
    print(f"  mechanical RR work {reaction:.6e} J/m^2; emitted energy {emitted:.6e} J/m^2; "
          f"initial chi <= {np.max(initial) * B0 / B_schwinger:.3e}")
    if label == "electron":
        a = tau * (e * B0 / m)**2
        s0 = np.arctanh(1 / initial[0])
        logcosh = lambda s: np.log1p(2 * np.sinh(s / 2)**2)
        phase_exact = e * B0 / (m * a) * (logcosh(s0 + a * duration) - logcosh(s0))
        final_u = np.asarray(out.state.u[0])
        phase_error = np.angle((final_u[1] + 1j * final_u[2]) * np.exp(-1j * phase_exact))
        print(f"  final gyrophase error (modulo 2 pi): {phase_error:.3e} rad")
    line, = axes[0].plot(np.asarray(out.t) * 1e9, np.mean(gamma, axis=1), label=label)
    axes[0].plot(np.asarray(out.t) * 1e9, np.mean(analytic, axis=1), "--", color=line.get_color())
    axes[1].plot(np.asarray(out.t) * 1e9, np.mean(gamma / analytic - 1, axis=1), label=label)


def final_gamma(strength):
    out = simulation(mono, strength).run(steps, store_every=steps, store_particles=False, moments=False)
    return jnp.sqrt(1 + jnp.sum((out.state.u[0] / c)**2))


# Forward mode stores no reverse trajectory; the control is a physical field strength.
value, derivative = jax.jit(lambda b: jax.jvp(final_gamma, (b,), (1.,)))(B0)
expected = exact_gamma(mono, duration, B0)[0]
a = tau * (e * B0 / m)**2
exact_derivative = -2 * a * duration / B0 * (expected**2 - 1)
print(f"d gamma_final/d B: AD {float(derivative):.8e} /T, exact {exact_derivative:.8e} /T; "
      f"relative error {abs(float(derivative) / exact_derivative - 1):.3e}")
print("Energy accuracy alone does not establish gyrophase accuracy: refine the macro time step as well.")
axes[0].set(xlabel="time (ns)", ylabel="mean Lorentz factor", title="solid: simulation; dashed: exact")
axes[1].set(xlabel="time (ns)", ylabel="mean relative gamma error", title="uniform magnetic field")
for ax in axes:
    ax.legend()
fig.tight_layout()
plt.show()
