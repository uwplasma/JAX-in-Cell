# Quickstart

Five minutes, three runs. Install with `pip install jaxincell` (see
{doc}`installation`).

## A Langmuir wave

The simplest plasma experiment: displace the electrons a little and watch them
oscillate at the plasma frequency.

```python
import numpy as np
from jaxincell import (Domain, Simulation, Solver, Species, diagnostics,
                       epsilon_0, mass_electron, elementary_charge as e)

length, cells = 1.0, 32
density = 1e12
omega_pe = np.sqrt(density * e ** 2 / (epsilon_0 * mass_electron))

electrons = Species.electrons(n=20000, density=density, vth=(1e5, 0, 0), quiet=True,
                              perturbation_amplitude=1e-3 * length / (2 * np.pi),
                              perturbation_mode=1)
ions = Species.ions(n=5000, density=density, mass_ratio=1e9, vth=(0, 0, 0), quiet=True)

simulation = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=1.0),
                        [electrons, ions], Solver())
output = simulation.run(600, seed=0, store_particles=False)

d = diagnostics(output)
print(f"omega measured / omega_pe = {float(d['dominant_frequency']) / omega_pe:.3f}")
```

It should print a number close to one. The wave is slightly faster than $\omega_{pe}$
because of the thermal correction $\omega^2 = \omega_{pe}^2(1 + 3k^2\lambda_D^2)$; the
`langmuir_wave.py` example measures the whole dispersion relation.

## The two-stream instability

Two counter-streaming electron beams. The instability grows, saturates, and rolls the
phase space into a vortex.

```python
from jaxincell import plot, speed_of_light as c

electrons = Species.electrons(n=8000, density=4.37e17, vth=(0.05 * c, 0, 0),
                              drift=(6e7, 0, 0), plus_minus=True,
                              perturbation_amplitude=5e-7, perturbation_mode=1)
ions = Species.ions(n=8000, density=4.37e17, electrons=electrons)

simulation = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5),
                        [electrons, ions], Solver(filter_passes=2))
output = simulation.run(1200, seed=0, store_every=2)

energy = diagnostics(output)
print(f"energy drift {abs(float(energy['total'][-1] / energy['total'][0]) - 1):.2e}")
plot(output, omega=float(simulation.plasma_frequency()))
```

`plus_minus=True` turns one drifting population into two counter-streaming beams of
half the density each. `dt_over_dx_c=4.5` is above the light-wave Courant limit, which
is safe here because nothing excites the transverse fields — see
{doc}`../numerics/stability`.

## A gradient

The point of building this on JAX. Differentiate a diagnostic with respect to a
physical parameter, through the whole time loop:

```python
import jax, jax.numpy as jnp

def field_energy(drift):
    beams = electrons.replace(drift=(drift, 0.0, 0.0))
    out = simulation.replace(species=(beams, ions)).run(200, seed=0, store_particles=False)
    return jnp.mean(out.E[:, :, 0] ** 2)

print(jax.grad(field_energy)(6e7))
```

No adjoint, no finite differences: `jax.grad` differentiates the deposition, the field
solve, the Boris rotation and the boundary conditions.

## From a file

```bash
jaxincell examples/input.toml
```

runs the same two-stream case from a TOML description, prints the energy balance and
shows the animation.

## Next

* {doc}`first_simulation` walks through one run in detail, parameter by parameter.
* {doc}`../user_guide/index` documents every argument.
* {doc}`../numerics/index` explains what the code computes and why.
* {doc}`../examples/index` has runnable scripts for each of the standard problems.
