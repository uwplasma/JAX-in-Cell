"""Differentiating through the whole simulation.

A Simulation is a JAX pytree whose physical parameters are leaves, so gradients
of any diagnostic with respect to them come straight from jax.grad: the time
loop, the field solve, the deposition and the particle push are all
differentiated, with no finite differences anywhere.

The demonstration is an inverse problem with a known answer. Gradient ascent on
the amplitude the seeded two-stream mode reaches by a fixed time should drive
the beam drift towards the fastest-growing wavenumber, which for cold beams is
k v_0 / omega_pe = sqrt(3/8) (Buneman, Phys. Rev. 115, 503, 1959) and moves to
about 0.70 for beams this warm.
"""

import os
from pathlib import Path

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below to "0", for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, figure, mass_electron, save_run,
                       elementary_charge as e_charge, speed_of_light as c)
from jaxincell.theory import two_stream_rate

length, cells, density, step = 0.01, 64, 4.37e17, 160
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
k = 2 * np.pi / length


def simulation(drift):
    electrons = Species.electrons(n=8000, density=density, vth=(0.05 * c, 0, 0), drift=(drift, 0, 0),
                                  plus_minus=True, sampling="quiet", perturbation_amplitude=5e-7,
                                  perturbation_mode=1)
    ions = Species.ions(n=8000, density=density, electrons=electrons, sampling="quiet")
    return Simulation(Domain(length=length, cells=cells, dt_over_dx_c=4.5), [electrons, ions],
                      Solver(filter_passes=0))


def amplification(drift):
    """ln of the seeded mode amplitude at a fixed time, which is gamma t plus a
    constant while the mode grows exponentially. Fixing the time rather than
    fitting a window keeps the objective a smooth function of the drift."""
    out = simulation(drift).run(step, seed=3, store_particles=False)
    return jnp.log(jnp.abs(jnp.fft.rfft(out.E[:, :, 0], axis=1)[-1, 1]))


value_and_grad = jax.jit(jax.value_and_grad(amplification))

# the gradient is one reverse-mode pass through the whole run; check it once
gradient = float(value_and_grad(4.0e7)[1])
h = 100.0 if jax.config.read("jax_enable_x64") else 1e4    # a step round-off does not swamp
difference = float((jax.jit(amplification)(4.0e7 + h) - jax.jit(amplification)(4.0e7 - h)) / (2 * h))
print(f"reverse mode {gradient:.6e}   central difference {difference:.6e}   "
      f"relative error {abs(difference / gradient - 1):.1e}\n")

drift, history = 2.5e7, []
for iteration in range(12):
    objective, slope = value_and_grad(drift)
    history.append((drift, float(objective)))
    print(f"{iteration:2d}  drift {drift:.4e} m/s   k v0/omega_pe {k * drift / omega_pe:.3f}   "
          f"ln|E_1| {float(objective):.4f}")
    drift = float(drift + 1.5e14 * slope)

drifts, objectives = np.array(history).T
print(f"\nascent reached k v0/omega_pe = {k * drifts[-1] / omega_pe:.3f}")
# the warm-beam optimum, from the kinetic growth rate of the same populations on a fine scan
fine = np.linspace(3.0e7, 6.0e7, 61)
optimum = fine[np.argmax([two_stream_rate(simulation(d)) for d in fine])]
print(f"kinetic optimum for these warm beams: k v0/omega_pe = {k * optimum / omega_pe:.3f} "
      f"(cold beams: sqrt(3/8) = {np.sqrt(3 / 8):.3f}); ascent is off by {100 * (drifts[-1] / optimum - 1):+.1f} %")

scan = np.linspace(2.4e7, 6.0e7, 15)
fig, ax = figure()
ax.plot(k * scan / omega_pe, [float(jax.jit(amplification)(d)) for d in scan], "-", lw=2, color="0.6",
        label="scan")
ax.plot(k * drifts / omega_pe, objectives, "o-", label="gradient ascent")
ax.axvline(k * optimum / omega_pe, color="k", label="kinetic optimum")
ax.axvline(np.sqrt(3 / 8), color="k", ls="--", label=r"$\sqrt{3/8}$ (cold beams)")
ax.set(xlabel=r"$k v_0/\omega_{pe}$", ylabel=r"$\ln|E_{k=1}|$ at a fixed time")
ax.legend(loc="lower center")
fig.tight_layout()
save_run(Path.cwd() / "optimize_two_stream", "optimize_two_stream",
         dict(length=length, cells=cells, density=density, steps=step, iterations=12),
         dict(gradient=gradient, central_difference=difference, drifts=drifts, objectives=objectives,
              kinetic_optimum_drift=optimum), figure=fig)
plt.show()
