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
  energy to round-off and has no time-step limit;
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

Every rate quoted in the documentation is checked against the linear kinetic
dispersion relation rather than against another simulation: Landau damping to 0.4 %,
the two-stream growth rate to 2.7 % across the unstable range, the Weibel rate to
6.1 %, and the collision operator to 2.5 % of the Fokker-Planck rates. See
[verification](https://jax-in-cell.readthedocs.io/en/latest/numerics/verification.html).

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
The package enables 64-bit floating point in JAX when imported.

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

## Examples

Nine scripts in `examples/`, each reproducing a result from the literature rather
than making a picture: the two-stream instability (Buneman 1959), Landau damping
(Landau 1946), the Bohm-Gross dispersion relation, the bump-on-tail instability and
its quasilinear plateau, the Weibel instability and its marginal wavenumber
(Weibel 1959), explicit against implicit energy conservation, the collision operator
against the Fokker-Planck rates, the plasma sheath with its floating potential and Bohm
criterion, and an optimisation that recovers the fastest-growing beam by gradient
ascent through the solver. Each is described in the
[documentation](https://jax-in-cell.readthedocs.io/en/latest/examples/index.html).

<p align="center">
    <img src="https://raw.githubusercontent.com/uwplasma/JAX-in-Cell/main/docs/_static/figures/two_stream_scan.png" width="70%" alt="Two-stream growth rate against the kinetic dispersion relation">
</p>

Measured growth rate against the kinetic dispersion relation, over the whole unstable
range. Bump-on-tail instability with periodic (left) and reflective (right) walls:

<table align="center"><tr>
<td><video src="https://github.com/user-attachments/assets/5f085f92-cb65-4765-b586-19e727bd2aab" controls width="100%"></video></td>
<td><video src="https://github.com/user-attachments/assets/9f33bac8-319e-4aba-91fb-befc64bca70e" controls width="100%"></video></td>
</tr></table>

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

Thirty tests, about forty seconds. They are physics tests rather than regression
tests: closed-form rates and frequencies, conservation laws, exact results for the
kernels, and the behaviour of the interface. They run on every pull request together
with a build of the documentation.

## Contributing and citing

Bug reports and feature requests go to the
[issue tracker](https://github.com/uwplasma/JAX-in-Cell/issues), questions to the
[discussions](https://github.com/uwplasma/JAX-in-Cell/discussions), and code through
pull requests; see [CONTRIBUTING.md](CONTRIBUTING.md). If you use JAX-in-Cell in
your work, please cite it using the `CITATION.cff` file (GitHub shows it under
"Cite this repository").

## Acknowledgements

JAX-in-Cell was inspired by [PiC-Code-Jax](https://github.com/SeanLim2101/PiC-Code-Jax)
by Sean Lim. Development is supported by the National Science Foundation under grant
PHY-2409066 and by the [UWPlasma](https://rogerio.physics.wisc.edu/) group at the
University of Wisconsin-Madison.

## License

MIT, see [LICENSE](LICENSE).
