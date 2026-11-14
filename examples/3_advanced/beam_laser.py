"""A finite-spread relativistic electron beam collides with a classical laser pulse.

The prescribed vacuum plane wave comes from a compact vector potential. A dilute
ensemble suppresses collective fields; this is not a focused-laser or dense-beam
calculation. Compare explicit PIC stepping with light-front reduced-LL equations,
an exact post-pulse energy formula, Lorentz invariants, and a pulse-strength
derivative. The full LL field-derivative term and quantum emission are absent.
Small chi is a classical-regime diagnostic, not a bound at the numerical error
level. The complete incident fluence equals the initial pulse energy per area;
the prescribed vacuum pulse is not depleted in this test-particle comparison.

For integer N>=2, a=a0*cos²(phi/(2N))*sin(phi) has a,a',a''=0 at
phi=+-pi*N and integral(a'²)dphi=a0²*pi*(3*N+1/N)/8. Writing s=gamma-eta_x,
h=1+tau*omega*s0*integral(a'²), one obtains s=s0/h and
eta_perp,f=eta_perp,0/h: the boundary term and integral a*a'² both vanish.
Thus gamma_f=(h²+s0²+eta_perp,0²)/(2*h*s0). This final-state result does not
identify the reduced-LL and full-LL trajectories inside the pulse.

Requires scipy (the development extra). No files are written; edit the settings
below or save the displayed figure. Refine the physical timestep, not only RR
substeps, to resolve the counterpropagating optical phase.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar
from scipy.integrate import solve_ivp

from jaxincell import (Domain, Radiation, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0, figure, mass_electron as m, speed_of_light as c)

wavelength, a0, cycles = .8e-6, 15., 32          # m, e*A_y/(m*c) amplitude scale, support in cycles
particles, steps, frames, density = 16, 8192, 512, 1e3
assert jax.config.read("jax_enable_x64"), "the reference accuracy gates require JAX_ENABLE_X64=1"
assert cycles >= 2 and int(cycles) == cycles, "the analytic fluence uses an integer cycle count >=2"
gamma0 = np.linspace(72., 88., particles)
pitch = np.linspace(.005, .025, particles)      # angle from the counterpropagating -x direction
phase0 = 2 * np.pi * np.arange(particles) / particles
eta0 = np.sqrt(gamma0[:, None]**2 - 1) * np.column_stack(
    (-np.cos(pitch), np.sin(pitch) * np.cos(phase0), np.sin(pitch) * np.sin(phase0)))
omega = 2 * np.pi * c / wavelength
length, duration = (2 * cycles + 16) * wavelength, np.pi * (cycles + 2) / omega
x0, cells = length / 4, 32
tau = e**2 / (6 * np.pi * epsilon_0 * m * c**3)
rho, field_scale = tau * omega, m * c * omega / e
s0 = gamma0 - eta0[:, 0]
fluence_shape = np.pi * (3 * cycles + 1 / cycles) / 8


def pulse(phi, amplitude):
    """Return a and da/dphi; a=e*A_y/(m*c), with a=da/dphi=0 at each edge."""
    window = jnp.cos(phi / (2 * cycles))**2
    a = window * jnp.sin(phi)
    derivative = window * jnp.cos(phi) - jnp.sin(phi / cycles) * jnp.sin(phi) / (2 * cycles)
    inside = jnp.abs(phi) < np.pi * cycles
    return amplitude * jnp.where(inside, a, 0.), amplitude * jnp.where(inside, derivative, 0.)


def fields(x, t, amplitude):
    phi = omega * (t - (x[:, 0] - x0) / c) - np.pi * cycles
    _, derivative = pulse(phi, amplitude)
    zero = jnp.zeros_like(phi)
    electric = -field_scale * derivative
    return jnp.stack((zero, electric, zero), axis=1), jnp.stack((zero, zero, electric / c), axis=1)


def simulate(amplitude, count=steps, radiation=True, history=False):
    x = jnp.zeros((particles, 3)).at[:, 0].set(x0)
    beam = Species.electrons(particles, density, x=x, u=jnp.asarray(c * eta0), name="beam")
    sim = Simulation(Domain(length=length, cells=cells, time_step=duration / count), (beam,),
                     Solver(relativistic=True), prescribed_fields=fields, field_parameters=amplitude,
                     radiation=Radiation(species="beam") if radiation else None)
    return sim.run(count, store_every=count // frames if history else count,
                   store_particles=history, moments=False)


def exact_final(amplitude):
    # For this odd potential, integral a*(a')² dphi vanishes over the full pulse.
    h = 1 + rho * s0 * fluence_shape * amplitude**2
    s, transverse = s0 / h, eta0[:, 1:] / h[:, None]
    return (1 + jnp.sum(transverse**2, axis=1) + s**2) / (2 * s)


def reference(phi, y):
    """Independent phase ODE: s=gamma-eta_x, r=eta_y/s, and z/s is constant."""
    s, r, time, work, emitted = y.reshape(5, particles)
    g = a0 * (.5 * np.cos(phi) + .25 * (1 + 1 / cycles) * np.cos((1 + 1 / cycles) * phi)
              + .25 * (1 - 1 / cycles) * np.cos((1 - 1 / cycles) * phi))
    gamma = .5 / s + .5 * s * (1 + r**2 + (eta0[:, 2] / s0)**2)
    return np.stack((-rho * g**2 * s**2, g / s, gamma / s, g * r, rho * g**2 * s * gamma)).ravel()


initial = np.stack((s0, eta0[:, 1] / s0, np.zeros(particles), np.zeros(particles), np.zeros(particles)))
reference_solution = solve_ivp(reference, (-np.pi * cycles, np.pi * cycles), initial.ravel(),
                               method="DOP853", rtol=2e-11, atol=2e-12, max_step=.2)
assert reference_solution.success, reference_solution.message
s_ref, r_ref, t_ref, work_ref, emitted_ref = reference_solution.y[:, -1].reshape(5, particles)
gamma_ref = .5 / s_ref + .5 * s_ref * (1 + r_ref**2 + (eta0[:, 2] / s0)**2)
analytic = np.asarray(exact_final(a0))
assert np.max(abs(gamma_ref / analytic - 1)) < 1e-9, "independent phase ODE disagrees with the exact pulse result"
assert np.max(t_ref) < omega * duration, "the simulation ends before every particle leaves the pulse"

errors = []
for count in (steps // 4, steps // 2, steps):
    out = simulate(a0, count, history=count == steps).validate()
    gamma_end = np.sqrt(1 + np.sum((np.asarray(out.state.u) / c)**2, axis=1))
    errors.append(float(np.max(abs(gamma_end / gamma_ref - 1))))
    print(f"{count:5d} steps: maximum final gamma relative error {errors[-1]:.6e}")
assert errors[-1] < 1e-6 and errors[-2] / errors[-1] > 3.5, "refine the optical timestep"

lorentz = simulate(a0, radiation=False, history=True).validate()
eta = np.asarray(lorentz.u) / c
gamma = np.sqrt(1 + np.sum(eta**2, axis=2))
phi = omega * (np.asarray(lorentz.t)[:, None] - (np.asarray(lorentz.x)[..., 0] - x0) / c) - np.pi * cycles
potential = np.asarray(pulse(jnp.asarray(phi), a0)[0])
light_front_error = np.max(abs((gamma - eta[..., 0]) / s0 - 1))
transverse_error = np.max(abs(eta[..., 1] - potential - eta0[:, 1])) / a0
assert max(light_front_error, transverse_error) < 5e-5, "the no-RR plane-wave invariants require refinement"
assert np.max(abs(np.asarray(out.x)[..., 0])) < .49 * length, "a particle reached the periodic boundary"


def final_mean_gamma(amplitude):
    eta = simulate(amplitude).state.u / c
    return jnp.mean(jnp.sqrt(1 + jnp.sum(eta**2, axis=1)))


value, derivative = jax.jit(lambda a: jax.jvp(final_mean_gamma, (a,), (1.,)))(a0)
forward = jax.jit(final_mean_gamma)
fd = float((forward(a0 + 1e-3) - forward(a0 - 1e-3)) / .002)
h = 1 + rho * s0 * fluence_shape * a0**2
exact_derivative = np.mean(rho * fluence_shape * a0 * (1 - (s0**2 + np.sum(eta0[:, 1:]**2, axis=1)) / h**2))
assert abs(float(derivative) / exact_derivative - 1) < 1e-5, "pulse-strength AD disagrees with the analytic derivative"
assert abs(float(derivative) / fd - 1) < 2e-6, "pulse-strength AD disagrees with finite differences"

chi_bound = hbar * omega / (m * c**2) * np.max(s0) * abs(a0) * (1 + 1 / (2 * cycles))
assert chi_bound < .01, "this example's classical benchmark requires a smaller quantum parameter"
energy_scale = density * length * m * c**2
work = float(out.work.prescribed_work[-1, 0])
reaction, emitted = float(out.work.rr_work[-1, 0]), float(out.work.radiated_energy[-1, 0])
delta = float(out.work.kinetic[-1, 0]) - energy_scale * np.mean(gamma0 - 1)
incident_fluence = epsilon_0 * c * field_scale**2 / omega * fluence_shape * a0**2
work_error = abs(work / (energy_scale * np.mean(work_ref)) - 1)
emission_error = abs(emitted / (energy_scale * np.mean(emitted_ref)) - 1)
residual = (delta - work - reaction) / (energy_scale * np.mean(gamma0 - 1))
assert work_error < 1e-3 and emission_error < 1e-6, "the work or emission disagrees with the phase ODE"
assert abs(residual) < 1e-9, "mechanical energy accounting failed"
print(f"mean gamma {np.mean(gamma0):.6f} -> {float(value):.6f}; chi <= {chi_bound:.4e}")
print(f"no-RR light-front / transverse invariant errors: {light_front_error:.4e} / {transverse_error:.4e}")
print(f"d mean gamma_final/d a0: AD {float(derivative):.9e}, analytic {exact_derivative:.9e}, FD {fd:.9e}")
print(f"particle work {work:.6e}, RR work {reaction:.6e}, emission {emitted:.6e}, delta K {delta:.6e} J/m^2")
print(f"particle-work / emission relative errors against ODE: {work_error:.6e} / {emission_error:.6e}")
print(f"mechanical residual / initial K: {residual:.4e}")
print(f"incident vacuum-wave fluence / initial pulse energy per area: {incident_fluence:.6e} J/m^2")
print(f"Doppler phase step <= {2 * omega * duration / steps:.4g} rad; "
      f"LL gradient scale tau*omega*s0 <= {rho * max(s0):.4e}")

fig, axes = figure(2)
for label, result in (("Lorentz force", lorentz), ("reduced LL", out)):
    gamma = np.sqrt(1 + np.sum((np.asarray(result.u) / c)**2, axis=2))
    axes[0].plot(np.asarray(result.t) * 1e15, gamma.mean(axis=1), label=label)
    axes[1].plot(gamma0, gamma[-1], "o", label=label)
axes[1].plot(gamma0, analytic, "--", label="exact post-pulse reduced LL")
axes[0].set(xlabel="time (fs)", ylabel="mean Lorentz factor", title="counterpropagating laser pulse")
axes[1].set(xlabel="initial Lorentz factor", ylabel="final Lorentz factor", title="finite energy and pitch spread")
for ax in axes:
    ax.legend()
fig.tight_layout()
plt.show()
