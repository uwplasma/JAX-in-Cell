<p align="center">
    <img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/JAX-in-Cell_logo.png#gh-light-mode-only" width="460" alt="JAX-in-Cell">
    <img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/JAX-in-Cell_logo_dark.png#gh-dark-mode-only" width="460" alt="JAX-in-Cell">
</p>

<p align="center">
A one-dimensional, three-velocity electromagnetic particle-in-cell code written in JAX.<br>
Runs on CPU, GPU and TPU, compiles the whole time loop, and is differentiable end to end.
</p>

<p align="center">
    <a href="https://github.com/uwplasma/JAX-in-Cell/actions/workflows/build_test.yml"><img src="https://github.com/uwplasma/JAX-in-Cell/actions/workflows/build_test.yml/badge.svg" alt="Build and test"></a>
    <a href="https://github.com/uwplasma/JAX-in-Cell/actions/workflows/docs.yml"><img src="https://github.com/uwplasma/JAX-in-Cell/actions/workflows/docs.yml/badge.svg" alt="Documentation build"></a>
    <a href="https://jax-in-cell.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/jax-in-cell/badge/?version=latest" alt="Documentation"></a>
    <a href="https://codecov.io/gh/uwplasma/JAX-in-Cell"><img src="https://codecov.io/gh/uwplasma/JAX-in-Cell/branch/main/graph/badge.svg" alt="Coverage"></a>
    <a href="https://pypi.org/project/jaxincell/"><img src="https://img.shields.io/pypi/v/jaxincell" alt="PyPI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/github/license/uwplasma/JAX-in-Cell" alt="License"></a>
</p>

**Documentation: [jax-in-cell.readthedocs.io](https://jax-in-cell.readthedocs.io/)**

## What it does

JAX-in-Cell advances charged pseudo-particles in one spatial dimension and three
velocity components under the Lorentz force, and advances the electric and magnetic
fields on a staggered (Yee) grid with Maxwell's equations. It provides

* an explicit leapfrog integrator with the Boris pusher (non-relativistic or
  relativistic), a charge-conserving current deposit that keeps the discrete Gauss law
  satisfied to round-off, and a compensated digital filter;
* an implicit Crank-Nicolson integrator solved by Picard iteration, which conserves
  both energy and charge to round-off and has no time-step limit;
* binary Coulomb collisions (Takizuka-Abe), verified against the Fokker-Planck
  relaxation rates;
* periodic, reflective and absorbing boundaries, chosen separately for particles and
  fields, with a radiating condition on the fields and absorbing walls treated as
  short-circuited conductors, so that a plasma against them forms a sheath;
* any number of species, each with its own density, drift, temperature anisotropy,
  seed and, if needed, a hand-built phase space;
* gradients of any output with respect to any physical input through `jax.grad`, and
  re-execution with new inputs without recompilation.

Everything runs as one XLA program on whatever device JAX finds.

Every rate it quotes is checked against a closed-form or linear kinetic result rather than
against another simulation: see [benchmarks](#benchmarks).

## Install

```bash
pip install jaxincell
```

or from source:

```bash
git clone https://github.com/uwplasma/JAX-in-Cell
cd JAX-in-Cell
pip install -e .
```

For a GPU, install the matching JAX wheel first (for example `pip install -U "jax[cuda12]"`).

### Precision

Runs are in double precision unless `JAX_ENABLE_X64=0` is set before JAX is imported.
Every script in `examples/` sets the variable at its top, so its precision is written in
the script and can be switched from the shell:

```bash
JAX_ENABLE_X64=0 python examples/1_basic/two_stream.py
```

Single precision reproduces the growth rates, frequencies and sheath of the examples;
what it gives up is conservation to round-off. It is not automatically faster: on a CPU
the two cost about the same, and on the RTX A4000 we tested a single-precision run was
many times slower, because of how CUDA scatters in float32
([performance](https://jax-in-cell.readthedocs.io/en/latest/user_guide/performance.html)).

## Run

From the command line, with a TOML file:

```bash
jaxincell examples/input.toml
```

From Python. Four objects describe a simulation and one method runs it:

```python
from jaxincell import Domain, Simulation, Solver, Species, diagnostics, plot, speed_of_light as c

electrons = Species.electrons(n=10000, density=4.37e17, vth=(0.05 * c, 0, 0),
                              drift=(6e7, 0, 0), plus_minus=True,
                              perturbation_amplitude=5e-7, perturbation_mode=1)
ions = Species.ions(n=10000, density=4.37e17, electrons=electrons)

simulation = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5),
                        [electrons, ions], Solver(filter_passes=2))

output = simulation.run(1000, seed=0)   # compiled on the first call
diagnostics(output)                     # energies, momentum, Gauss residual, temperatures
plot(output)                            # animated fields, distributions and phase space
```

`Domain`, `Species`, `Solver` and `Simulation` are frozen dataclasses registered as
JAX pytrees. Physical quantities are leaves, so they can be changed without
recompiling and differentiated with respect to; structural settings are static.

```python
import jax, jax.numpy as jnp

gradient = jax.grad(lambda s: jnp.sum(s.run(200, seed=0).E ** 2))(simulation)
print(gradient.species[0].drift, gradient.domain.length)
```

`jax.grad` differentiates the initial sampling, the deposition, the field solve, the
Boris rotation and the boundary conditions — the whole run, with no adjoint to write
and no finite differences anywhere.

## Benchmarks

Each case is checked against a closed-form or linear kinetic result. The numbers below come
from `docs/_static/figures/measurements.json`, which `python docs/scripts/make_all.py`
regenerates with the figures. Where a TOML file is listed the run needs no Python:
`jaxincell inputs/<file>.toml`.

### 1D1V: electrostatic, one velocity component

<table>
<tr>
<td width="33%"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/landau_damping.png" alt="Landau damping"></td>
<td width="33%"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/two_stream.png" alt="Two-stream instability"></td>
<td width="33%"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/bump_on_tail.png" alt="Bump-on-tail instability"></td>
</tr>
</table>

| case | checked against | agreement | run |
|---|---|---|---|
| [Landau damping](https://jax-in-cell.readthedocs.io/en/latest/examples/landau_damping.html) | kinetic root at $k\lambda_D=0.5$ | rate 0.7 %, frequency 0.8 % | `1_basic/landau_damping.py` · `landau_damping.toml` |
| [Langmuir waves](https://jax-in-cell.readthedocs.io/en/latest/examples/langmuir_wave.html) | kinetic root, $k\lambda_D = 0.05$ to $0.5$ | 0.4 % at worst | `1_basic/langmuir_wave.py` · `langmuir_wave.toml` |
| [Two-stream](https://jax-in-cell.readthedocs.io/en/latest/examples/two_stream.html) | kinetic growth rate | 3.3 % seeded, 2.8 % mean over the unstable range | `1_basic/two_stream.py` · `two_stream.toml` |
| [Bump on tail](https://jax-in-cell.readthedocs.io/en/latest/examples/bump_on_tail.html) | kinetic growth rate, quasilinear plateau | 6.3 % | `2_intermediate/bump_on_tail.py` · `bump_on_tail.toml` |
| [Partly reflecting wall](https://jax-in-cell.readthedocs.io/en/latest/examples/wall_reflection.html) | flux average of the reflection law | 2e-4 | `2_intermediate/wall_reflection.py` |
| [Conservation](https://jax-in-cell.readthedocs.io/en/latest/examples/conservation.html) | energy, momentum and the Gauss law | implicit: energy 3e-16, Gauss law 7e-15 | `3_advanced/conservation.py` · `conservation_implicit.toml` |
| [Optimisation](https://jax-in-cell.readthedocs.io/en/latest/examples/optimize_two_stream.html) | fastest-growing drift from linear theory | 0.1 % after 12 ascent steps | `3_advanced/optimize_two_stream.py` |

<table>
<tr>
<td width="33%"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/conservation.png" alt="Energy, momentum and charge conservation"></td>
<td width="33%"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/autodiff.png" alt="Gradients through the solver"></td>
<td width="33%"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/two_stream_scan.png" alt="Growth rate across the unstable range"></td>
</tr>
</table>

### 1D2V: a magnetic field the plasma grows itself

<p align="center"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/weibel.png" width="70%" alt="Weibel instability"></p>

| case | checked against | agreement | run |
|---|---|---|---|
| [Weibel](https://jax-in-cell.readthedocs.io/en/latest/examples/weibel.html) | transverse kinetic dispersion relation, and the marginal wavenumber $k_cc=\omega_{pe}\sqrt{T_z/T_x-1}$ | 6.0 % mean, 9.2 % worst over the 5 of 7 modes that grow cleanly | `2_intermediate/weibel.py` · `weibel.toml` |

### 1D3V: sheaths, oblique fields and collisions

<table>
<tr>
<td width="33%"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/sheath_source.png" alt="A maintained sheath"></td>
<td width="33%"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/sheath_magnetized.png" alt="A sheath in an oblique magnetic field"></td>
<td width="33%"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/sheath_optimization.png" alt="A wall's reflectivity recovered from its sheath"></td>
</tr>
</table>

| case | checked against | agreement | run |
|---|---|---|---|
| [Unmagnetized sheath](https://jax-in-cell.readthedocs.io/en/latest/examples/sheath_unmagnetized.html) | kinetic sheath theory: wall potential, densities, current balance | potential 0.6 %, densities 2 %, net current 0.02 % of the ion current | `1_basic/sheath_unmagnetized.py` · `sheath_unmagnetized.toml` |
| [Oblique magnetic field](https://jax-in-cell.readthedocs.io/en/latest/examples/sheath_magnetized.html) | Chodura: a magnetic presheath, and entry along the field | impact energy within a few per cent of the field-parallel sound speed | `2_intermediate/sheath_magnetized.py` · `sheath_magnetized.toml` |
| [Grazing incidence](https://jax-in-cell.readthedocs.io/en/latest/examples/grazing_sheath.html) | GYRAZE's own entrance distribution and manifest | sampled distribution to a fraction of a per cent | `3_advanced/grazing_sheath.py` |
| [Sheath drop with reflection](https://jax-in-cell.readthedocs.io/en/latest/examples/sheath_reflection.html) | Hobbs and Wesson, with and without electron reflection | 2.0 % | `2_intermediate/sheath_reflection.py` |
| [Collisions](https://jax-in-cell.readthedocs.io/en/latest/examples/collisions.html) | Fokker-Planck relaxation rates | 2.5 % at worst of four rates | `2_intermediate/collisions.py` · `collisions.toml` |
| [Inverse problem](https://jax-in-cell.readthedocs.io/en/latest/examples/sheath_optimization.html) | a wall's reflectivity recovered from the sheath it holds | gradients agree with finite differences to 8-10 digits | `3_advanced/sheath_optimization.py` |

### Speed

<p align="center"><img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/scaling.png" width="70%" alt="Cost per particle and per cell"></p>

| what | cost |
|---|---|
| explicit step, 200 000 particles | 30 ns per particle per step |
| implicit step, 8 Picard iterations | 683 ns per particle per step |
| 1024 cells, 40 000 particles | 1.5 ms per step |

Measured on a laptop CPU with JAX 0.11; `docs/scripts/fig_scaling.py` reproduces it, and the
same code runs on a GPU or TPU without change.

Bump-on-tail instability with periodic (left) and reflective (right) walls:

<table align="center"><tr>
<td><video src="https://github.com/user-attachments/assets/5f085f92-cb65-4765-b586-19e727bd2aab" controls width="100%"></video></td>
<td><video src="https://github.com/user-attachments/assets/9f33bac8-319e-4aba-91fb-befc64bca70e" controls width="100%"></video></td>
</tr></table>

## Run one from a file

`inputs/` holds a TOML file for each of the runs above. Every table in it is a constructor
and nothing is ignored, so a misspelled key is an error in the file rather than a surprise in
the answer:

```bash
jaxincell inputs/two_stream.toml                                  # run and animate
jaxincell inputs/landau_damping.toml --no-plot                    # headless
jaxincell inputs/weibel.toml --save weibel --movie weibel.mp4     # arrays, provenance, video
```

`--save DIR` writes `fields.npz`, a `run.json` with the settings and the versions, precision,
device and commit that produced them, and a copy of the input file. `--steps` and `--seed`
override the file; `--plot`/`--no-plot` override its plotting. Scans and optimisation are not
in the file format on purpose: those are programs, and `load_toml` hands you the `Simulation`
to write them around.

## Documentation

The [documentation](https://jax-in-cell.readthedocs.io/) contains a tutorial, a
user guide covering every argument and output field, a description of the numerical
methods with their derivations (the Yee grid, shape functions, the charge-conserving
deposit, the Boris and Crank-Nicolson schemes, collisions, filtering, boundaries,
stability limits), the verification against linear kinetic theory, the examples, and
the API reference. To build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The figures and every number the text quotes are regenerated with
`python docs/scripts/make_all.py`.

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

270 tests, about eight minutes, covering every statement and branch. They
are physics tests rather than regression tests: closed-form rates and frequencies,
conservation laws, exact results for the kernels, and the behaviour of the interface.
They run on Python 3.10 to 3.13 on every pull request, together with a build of the
documentation.

## Contributing and citing

Bug reports and feature requests go to the
[issue tracker](https://github.com/uwplasma/JAX-in-Cell/issues), questions to the
[discussions](https://github.com/uwplasma/JAX-in-Cell/discussions), and code through
pull requests, as the
[development guide](https://jax-in-cell.readthedocs.io/en/latest/development.html)
describes. If you use JAX-in-Cell in your work, please cite it, together with JAX and
the papers of the methods you rely on, listed in the
[references](https://jax-in-cell.readthedocs.io/en/latest/numerics/index.html#references).
[`CITATION.cff`](CITATION.cff) is the same entry in the form GitHub's "Cite this
repository" reads:

```bibtex
@software{jaxincell,
  author = {Ma, Longyu and Jorge, Rogerio and Lu, Hongke and Tran, Aaron and Woolford, Christopher},
  title  = {{JAX-in-Cell}: a differentiable particle-in-cell code for plasma physics},
  year   = {2025},
  url    = {https://github.com/uwplasma/JAX-in-Cell}
}
```

## Acknowledgements

JAX-in-Cell was inspired by [PiC-Code-Jax](https://github.com/SeanLim2101/PiC-Code-Jax)
by Sean Lim. Development is supported by the National Science Foundation under grant
PHY-2409066 and by the [UWPlasma](https://rogerio.physics.wisc.edu/) group at the
University of Wisconsin-Madison.

## License

MIT, see [LICENSE](LICENSE).
