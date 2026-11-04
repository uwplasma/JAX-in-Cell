"""Differentiate a short prescribed-RF response and minimize its field exposure.

A dilute 20 MeV electron ensemble crosses a supplied counterpropagating helical
mode in B0=2 T. Four smooth envelope coefficients are optimized through the actual
Simulation time loop, with physical reduced Landau--Lifshitz radiation enabled.
The constraint is a finite-time rms sin(pitch), after the prescribed RF is off.
The cost is integral <E_RF^2> dt, NOT RF input energy or antenna power. Radiation
is far too slow here to establish sustained runaway suppression: collisions, the
bulk reservoir, accelerating current, avalanche, and a source-power closure would
be needed for that claim. No quasilinear diffusion is added to the resolved wave.

Requires scipy (the development extra). Run with --quick for fewer particles and
steps; it is a smoke run. No files are written. All controls and scales are below.
"""
import os
import sys
from functools import lru_cache

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from jaxincell import (Domain, Radiation, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0, figure, mass_electron as m, speed_of_light as c)

quick = "--quick" in sys.argv
B0, gamma0, pitch0 = 2., 40., .02                # T, Lorentz factor, radians
frequency, k = 476e6, -20.                      # Hz, rad/m; one fixed periodic mode
E_scale, pulse_time, hold_time = 3e5, 8e-9, 2e-9  # V/m, s, s; short orbit-response benchmark
sites, phases, cells = 8, (4 if quick else 8), 8
density, steps = 1e3, (2048 if quick else 4096)   # deliberately dilute; no thermal background
pitch_target, coefficient_max = .06, 2.5         # target rms sin(pitch), coefficient bound
design_target = 1.005 * pitch_target             # fixed 0.5% margin for independent refinement
omega, length, duration = 2 * np.pi * frequency, 2 * np.pi / abs(k), pulse_time + hold_time
v_parallel = c * np.sqrt(1 - gamma0**-2) * np.cos(pitch0)
print(f"Chosen local benchmark: B0={B0:g} T, seed energy={(gamma0 - 1) * m * c**2 / e / 1e6:.3f} MeV, "
      f"RF={frequency / 1e6:g} MHz, k_parallel={k:g} rad/m")
print(f"omega-k*v_parallel={omega - k * v_parallel:.6e} /s; Omega_e/gamma={e * B0 / (m * gamma0):.6e} /s")
print(f"Pulse {pulse_time * 1e9:g} ns, RF-off hold {hold_time * 1e9:g} ns; density {density:g} /m^3 "
      "suppresses collective response. The control target concerns pitch, not net cooling.")


def envelope(t, a):
    """sin² ramp times a cubic Bernstein polynomial; return g and dg/dt."""
    s = jnp.clip(t / pulse_time, 0., 1.)
    r = 1 - s
    shape = a[0] * r**3 + 3 * a[1] * s * r**2 + 3 * a[2] * s**2 * r + a[3] * s**3
    slope = 3 * ((a[1] - a[0]) * r**2 + 2 * (a[2] - a[1]) * s * r + (a[3] - a[2]) * s**2)
    ramp = jnp.sin(jnp.pi * s)**2
    inside = (t > 0) & (t < pulse_time)
    return (jnp.where(inside, ramp * shape, 0.),
            jnp.where(inside, (jnp.pi * jnp.sin(2 * jnp.pi * s) * shape + ramp * slope) / pulse_time, 0.))


def prescribed_fields(x, t, a):
    # A_y=(E_scale/omega) g sin(phi), A_z=(E_scale/omega) g cos(phi).
    # E=-dA/dt and B=curl(A) obey Faraday's law even during the ramps.
    g, dg = envelope(t, a)
    phi = k * x[:, 0] - omega * t
    co, si, zero = jnp.cos(phi), jnp.sin(phi), jnp.zeros_like(phi)
    E = E_scale * jnp.stack((zero, g * co - dg / omega * si, -g * si - dg / omega * co), axis=1)
    B = E_scale * k / omega * g * jnp.stack((zero, si, co), axis=1)
    return E, B


def simulate(a, count=steps, gyrophases=phases, history=False):
    n = sites * gyrophases
    x = jnp.zeros((n, 3)).at[:, 0].set(jnp.repeat(length * ((jnp.arange(sites) + .5) / sites - .5), gyrophases))
    phase = jnp.tile(2 * jnp.pi * jnp.arange(gyrophases) / gyrophases, sites)
    u = c * jnp.sqrt(gamma0**2 - 1) * jnp.stack(
        (jnp.full(n, jnp.cos(pitch0)), jnp.sin(pitch0) * jnp.cos(phase), jnp.sin(pitch0) * jnp.sin(phase)), axis=1)
    beam = Species.electrons(n=n, density=density, x=x, u=u, name="seed")
    sim = Simulation(Domain(length=length, cells=cells, time_step=duration / count), (beam,),
                     Solver(relativistic=True), radiation=Radiation(species="seed"),
                     external_B=jnp.broadcast_to(jnp.array([B0, 0., 0.]), (cells, 3)),
                     prescribed_fields=prescribed_fields, field_parameters=a)
    return sim.run(count, store_every=count // 64 if history else count, store_particles=history, moments=False)


def response(a, count=steps, gyrophases=phases):
    eta = simulate(a, count, gyrophases).state.u / c
    return jnp.mean(jnp.sum(eta[:, 1:]**2, axis=1) / jnp.sum(eta**2, axis=1))


# For this circular mode E_RF² is independent of x; these fixed Gauss nodes
# integrate the smooth exposure without sampling the fast carrier.
quad_x, quad_w = np.polynomial.legendre.leggauss(32)
quad_t = jnp.asarray((quad_x + 1) * pulse_time / 2)
quad_w = jnp.asarray(quad_w)


def budget(a):
    g, dg = envelope(quad_t, a)
    return .5 * jnp.dot(quad_w, g**2 + (dg / omega)**2)


def response_with_aux(a):
    value = response(a)
    return value, value


# Forward mode gives all four sensitivities without retaining a reverse trajectory.
response_derivative = jax.jit(jax.jacfwd(response_with_aux, has_aux=True))
budget_derivative = jax.jit(jax.value_and_grad(budget))
forward = jax.jit(response)


@lru_cache(maxsize=1)
def measured(controls):
    # SLSQP asks for the value and Jacobian separately at the same controls.
    gradient, value = response_derivative(jnp.asarray(controls))
    return float(value), np.asarray(gradient)


def cost(a):
    value, gradient = budget_derivative(jnp.asarray(a))
    return float(value), np.asarray(gradient)


start = np.full(4, 1.4)
direction = np.array([1., -.5, .25, -.75])
double_precision = jax.config.read("jax_enable_x64")
h = 1e-3 if double_precision else .03
value, gradient = measured(tuple(start))
finite_difference = float((forward(start + h * direction) - forward(start - h * direction)) / (2 * h))
derivative = float(gradient @ direction)
print(f"directional derivative: AD {derivative:.7e}, finite difference {finite_difference:.7e}; "
      f"relative mismatch {abs(derivative - finite_difference) / max(abs(derivative), 1e-15):.3e}")
gradient_tolerance = 2e-4 if double_precision else .02
assert np.isfinite(derivative) and np.isfinite(finite_difference), "nonfinite control sensitivity"
assert abs(derivative - finite_difference) <= gradient_tolerance * max(abs(derivative), pitch_target**2 / 100), \
    "the directional derivative failed its physical-response-scale check"
constraint = {"type": "ineq", "fun": lambda a: measured(tuple(a))[0] - design_target**2,
              "jac": lambda a: measured(tuple(a))[1]}
result = minimize(cost, start, jac=True, method="SLSQP", bounds=[(0., coefficient_max)] * 4,
                  constraints=constraint,
                  options={"maxiter": 25, "ftol": 1e-10 if double_precision else 1e-6, "disp": False})
controls = np.asarray(result.x)
reached = float(forward(controls))
refined = float(response(controls, count=2 * steps, gyrophases=2 * phases))
exposure = float(budget(controls)) * E_scale**2 * pulse_time
print(f"optimizer: {result.message}; coefficients {controls}")
print(f"normalized field exposure {float(budget(start)):.6g} -> {float(budget(controls)):.6g}; "
      f"physical exposure {exposure:.6g} V^2 s/m^2")
print(f"rms sin(pitch): target {pitch_target:.6f}, optimized {np.sqrt(reached):.6f}, "
      f"half-step/double-phase verification {np.sqrt(refined):.6f}")
assert result.success, result.message
assert np.all(np.isfinite(controls)) and np.isfinite(reached) and np.isfinite(refined), "nonfinite optimized response"
assert reached >= pitch_target**2 and refined >= pitch_target**2, "the optimized/refined control missed the target"
refinement_tolerance = .003 if double_precision else .01
assert abs(np.sqrt(refined) - np.sqrt(reached)) <= refinement_tolerance * pitch_target, \
    "the refined rms pitch differs by more than the declared fraction of the target"
assert float(budget(controls)) < .99 * float(budget(start)), "the optimizer did not reduce field exposure by 1%"
print("Finite-time target, derivative, and refinement gates: passed")

fig, axes = figure(2)
t_plot = np.linspace(0, duration, 400)
for label, a in (("RF off", np.zeros(4)), ("initial envelope", start), ("optimized envelope", controls)):
    out = simulate(jnp.asarray(a), history=True).validate()
    eta = np.asarray(out.u) / c
    pitch = np.sqrt(np.mean(np.sum(eta[..., 1:]**2, axis=-1) / np.sum(eta**2, axis=-1), axis=1))
    g, dg = envelope(jnp.asarray(t_plot), jnp.asarray(a))
    axes[0].plot(t_plot * 1e9, E_scale * np.sqrt(np.asarray(g**2 + (dg / omega)**2)) / 1e3, label=label)
    axes[1].plot(np.asarray(out.t) * 1e9, pitch, label=label)
    prescribed = float(out.work.prescribed_work[-1, 0])
    emitted = float(out.work.radiated_energy[-1, 0])
    reaction = float(out.work.rr_work[-1, 0])
    energy0 = density * length * (gamma0 - 1) * m * c**2
    delta = float(out.work.kinetic[-1, 0]) - energy0
    gamma_end = np.sqrt(1 + np.sum(eta[-1]**2, axis=1))
    tau = e**2 / (6 * np.pi * epsilon_0 * m * c**3)
    magnetic_rate = tau * (e * B0 / m)**2
    cooling_time = np.mean(gamma_end - 1) / (magnetic_rate * np.mean(np.sum(eta[-1, :, 1:]**2, axis=1)))
    print(f"{label}: prescribed particle work {prescribed:.6e}, RR work {reaction:.6e}, "
          f"emitted {emitted:.6e}, kinetic change {delta:.6e} J/m^2; "
          f"instantaneous magnetic cooling time {cooling_time:.3g} s")
print("This is a short, dilute orbit-ensemble control. The physical radiation force is unscaled.")
print("Field exposure is not input energy; prescribed particle work is not source power.")
print("Radiation substeps refine RR half-kicks only; refine Domain.time_step for gyromotion, RF phase and orbits.")
axes[0].set(xlabel="time (ns)", ylabel="prescribed RF |E| (kV/m)", title="fixed mode, smooth envelopes")
axes[1].set(xlabel="time (ns)", ylabel="rms sin(pitch)", title="actual particle response")
axes[1].axhline(pitch_target, color="k", ls="--", lw=1, label="target")
for ax in axes:
    ax.axvline(pulse_time * 1e9, color="0.5", ls=":", lw=1)
    ax.legend()
fig.tight_layout()
plt.show()
