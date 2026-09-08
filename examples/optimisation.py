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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

length, cells, density, step = 0.01, 64, 4.37e17, 160
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
k = 2 * np.pi / length


def amplification(drift):
    """ln of the seeded mode amplitude at a fixed time, which is gamma t plus a
    constant while the mode grows exponentially. Fixing the time rather than
    fitting a window keeps the objective a smooth function of the drift."""
    electrons = Species.electrons(n=8000, density=density, vth=(0.05 * c, 0, 0), drift=(drift, 0, 0),
                                  plus_minus=True, quiet=True, perturbation_amplitude=5e-7,
                                  perturbation_mode=1)
    ions = Species.ions(n=8000, density=density, electrons=electrons, quiet=True)
    out = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=4.5), [electrons, ions],
                     Solver(filter_passes=0)).run(step, seed=3, store_particles=False)
    return jnp.log(jnp.abs(jnp.fft.rfft(out.E[:, :, 0], axis=1)[-1, 1]))


value_and_grad = jax.jit(jax.value_and_grad(amplification))

# the gradient is one reverse-mode pass through the whole run; check it once
gradient = float(value_and_grad(4.0e7)[1])
h = 100.0
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
print(f"cold-beam optimum is sqrt(3/8) = {np.sqrt(3 / 8):.3f}; warm beams shift it to about 0.70")

scan = np.linspace(2.4e7, 6.0e7, 15)
plt.figure(figsize=(6, 4))
plt.plot(k * scan / omega_pe, [float(jax.jit(amplification)(d)) for d in scan], "-", color="0.6",
         label="scan")
plt.plot(k * drifts / omega_pe, objectives, "o-", label="gradient ascent")
plt.axvline(np.sqrt(3 / 8), color="k", ls="--", label=r"$\sqrt{3/8}$ (cold beams)")
plt.xlabel(r"$k v_0/\omega_{pe}$"); plt.ylabel(r"$\ln|E_{k=1}|$ at a fixed time")
plt.legend(frameon=False); plt.tight_layout(); plt.show()
