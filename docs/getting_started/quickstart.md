# Getting started

Install, then three short runs: a Langmuir wave, the two-stream instability, and a
gradient through the whole solver.

```{toctree}
:hidden:

first_simulation
```

## Installation

```bash
pip install jaxincell                     # or, from source:
git clone https://github.com/uwplasma/JAX-in-Cell && cd JAX-in-Cell && pip install -e .
```

| | |
|---|---|
| Python | 3.10 or newer; CI tests 3.10 to 3.13 on Linux |
| dependencies | `jax`, `matplotlib`, and `tomli` on Python 3.10, which has no `tomllib` |
| version | derived by `setuptools_scm` from the git tags, so a source install reports the last release plus the commits since |

| extra | what it adds |
|---|---|
| `openpmd` | `openpmd-api`, to write openPMD |
| `docs` | Sphinx, and scipy for the figure scripts |
| `dev` | `pytest`, `pytest-cov`, `flake8` and `openpmd-api` ({doc}`../development`) |

* **GPU**: `pip` installs the CPU build of JAX. Install the JAX wheel matching your CUDA
  or ROCm stack first, following the
  [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html) — for
  example `pip install -U "jax[cuda12]"`. Nothing in the package is device specific.
* **Precision**: importing the package enables 64-bit floating point in JAX unless
  `JAX_ENABLE_X64` is already set, which is how a run chooses single precision. The
  setting applies to the whole Python process.
* **Movies**: saving an animation to MP4 with {func}`jaxincell.plot` needs `ffmpeg` on the
  `PATH`.

```bash
python -c "import jaxincell, jax; print(jaxincell.__file__, jax.devices())"
```

## A Langmuir wave

Displace the electrons a little and watch them oscillate at the plasma frequency.

```python
import numpy as np
from jaxincell import (Domain, Simulation, Solver, Species, diagnostics,
                       epsilon_0, mass_electron, elementary_charge as e)

length, cells = 1.0, 32
density = 1e15
omega_pe = np.sqrt(density * e ** 2 / (epsilon_0 * mass_electron))

electrons = Species.electrons(n=20000, density=density, vth=(1e5, 0, 0), sampling="quiet",
                              perturbation_amplitude=1e-3 * length / (2 * np.pi),
                              perturbation_mode=1)
ions = Species.ions(n=5000, density=density, mass_ratio=1e9, vth=(0, 0, 0), sampling="quiet")

simulation = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=1.0),
                        [electrons, ions], Solver())
output = simulation.run(2000, seed=0, store_particles=False)

d = diagnostics(output)
print(f"omega measured / omega_pe = {float(d['dominant_frequency']) / omega_pe:.3f}")
```

It should print a number within a couple of per cent of one (0.997 on our machine).

* The run covers about sixty plasma periods, so the Fourier transform behind
  `dominant_frequency` resolves the frequency to 1.7 per cent. A run of only a few periods
  cannot tell $\omega_{pe}$ apart from its neighbouring frequency bins.
* At this temperature the thermal correction
  $\omega^2 = \omega_{pe}^2(1 + 3k^2\lambda_D^2)$ is negligible,
  $k\lambda_D \approx 3\times10^{-4}$. {doc}`../examples/langmuir_wave` measures the
  dispersion relation where it is not.

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

* `plus_minus=True` turns one drifting population into two counter-streaming beams of half
  the density each.
* `dt_over_dx_c=4.5` is above the light-wave Courant limit, which is safe here because
  nothing excites the transverse fields — see {doc}`../numerics/stability`.

{doc}`first_simulation` takes this run apart parameter by parameter.

## A gradient

Differentiate a diagnostic with respect to a physical parameter, through the whole time
loop:

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

Run from a clone of the repository, this runs the same two-stream case from a TOML
description, prints the energy drift and the Gauss-law residual, and shows the animation.
Without a file, `jaxincell` prints its usage.

## Next

* {doc}`first_simulation` — one run in detail, parameter by parameter.
* {doc}`../user_guide/index` — every argument.
* {doc}`../numerics/index` — what the code computes, and why.
* {doc}`../examples/index` — runnable scripts for each of the standard problems.
