# Development

How to cite the code is in the
[README](https://github.com/uwplasma/JAX-in-Cell#contributing-and-citing).

## Contributing

Search the [issue tracker](https://github.com/uwplasma/JAX-in-Cell/issues) before
opening an issue, and give what you expected, what happened, the smallest script that
reproduces it, and the versions of Python, JAX and JAX-in-Cell. Questions go to the
[discussions](https://github.com/uwplasma/JAX-in-Cell/discussions); security problems
to rogerio.jorge@wisc.edu, not in public. Contributions are released under the MIT
licence.

1. Fork and branch, then `pip install -e ".[dev]"`.
2. Make the change with a test that exercises it, keeping the static and differentiable
   parameters consistent (see Architecture below).
3. Run `pytest` and `flake8`; CI fails on any lint violation and on coverage below
   100 per cent.
4. If a numerical method changed, rerun its figure script under `docs/scripts/` and
   update the page that describes the method.
5. Open a pull request against `main` saying what changed and why. Commit subjects are
   imperative and under about seventy characters (`Add a thermal wall`); the body says
   why.

## Architecture

The package is seven modules and about 2100 lines. Each one has a single job, and the
dependency graph is a straight line with no cycles.

```
_config.py       Domain, Species, Solver, Collisions -- frozen pytree dataclasses; constants
_core.py         the numerical kernels: shape function, deposit, gather, curls,
                 Maxwell update, Gauss solve, Boris pushers, boundaries, filter
_collisions.py   Takizuka-Abe binary collisions
_simulation.py   Simulation, Output, the time loop, TOML input, quiet_start
_diagnostics.py  energies, momentum, Gauss residual, temperatures, frequency
_plot.py         the animated overview figure and the movie writer
openpmd.py       optional openPMD export
```

`__init__.py` re-exports the public names and holds `main`, the `jaxincell` command.
`plot` is imported on first use, since matplotlib would otherwise take a third of the
import time, and `__version__` is `"unknown"` in a source tree that was never
installed, since setuptools_scm writes `jaxincell/version.py` at build time.

### Everything is a pytree

`pytree_dataclass(static=(...))` in `_config.py` is twenty lines and does the work: it
makes a frozen dataclass, registers it with `jax.tree_util`, and splits the fields into
leaves and static metadata. Leaves are traced, so they can change without
recompilation and be differentiated with respect to; static fields become part of the
treedef and therefore of the cache key. A field that holds a function, such as a
velocity-dependent reflection law, is moved into the static metadata whatever its
declaration, since a function is not an array; in a tuple such as `(law, 0.3)` only the
function moves, and the number stays a leaf.

The split is the main design decision in the package. A parameter is static if the
*shape* of the computation depends on it — particle counts, cell counts, boundary
types, the algorithm name, the number of Picard iterations — and a leaf otherwise.
Getting it wrong shows up immediately: a static physical parameter recompiles on every
change, and a leaf that controls a shape fails to trace.

JAX rebuilds an object from its leaves without calling `__init__`, so `__post_init__`
normalises and validates only what is constructed or passed to `replace`, and tree
operations can put anything in the leaves: stacked ensembles, tracers, `None`. Since
`replace` converts the stored values again, the conversions must be idempotent (the
boundary-name conversion accepts codes as well as names) and must not force a traced
value (the Courant check skips values that are not plain Python numbers). An object with
`None` where it always holds a number is a template, such as a `vmap` `in_axes`, and is
stored as given.

### The time loop

`_run` is jitted with `steps`, `store_every` and `store_particles` as static
arguments. Inside, `lax.scan` runs the chunks and an inner `lax.scan` runs the
`store_every - 1` steps that are not kept, so thinning the history costs nothing and
the whole loop is a single XLA program with no Python in it.

The step function itself is a method on `Simulation`, chosen once from
`solver.algorithm`. Because `Simulation` is a pytree and `self` is traced, the method
closes over the traced parameters without capturing them as constants.

### Adding something

**A diagnostic**: a function of `Output` in `_diagnostics.py`, added to the dictionary
that `diagnostics` returns. Nothing else has to change; it can be computed on a stored
run.

**A boundary condition**: a code in `BOUNDARIES`, a branch in `map_indices`,
`apply_particle_bc`, `_left_ghost_E`, `_right_ghost_B` and `_shift`. The branches are
resolved at trace time because the codes are static, so they cost nothing at run time.
A wall that needs random numbers, as the thermal wall does to redraw velocities, is a
position map in `apply_particle_bc` and a method on `Simulation`, which holds the key.

**A field solver or an integrator**: a branch in `Solver` and a method on
`Simulation` with the same signature as `_explicit_step`. Keep any iteration a
`lax.scan` of fixed length rather than a `lax.while_loop`, or reverse-mode
differentiation stops working.

**A species initialisation**: `Species.replace(x=..., v=...)` covers most of it from
outside the package; {func}`~jaxincell.quiet_start` exists so that a custom condition
can start from the quiet sampling.

### What the design gives up

Shapes are static, so particles cannot be created or destroyed. Absorption zeroes a
particle's weight and parks it outside the grid rather than removing it, which keeps
the arrays rectangular at the cost of memory for dead particles, and partial reflection
lowers the weight instead of splitting the particle in two. Ionisation and
injection would need the same treatment, with a pool of inactive particles.

The geometry is one-dimensional. Two and three dimensions would change `_core.py`
thoroughly, the rest much less: the configuration objects, the loop, the diagnostics
and the differentiability are not specific to one dimension.

## Tests and documentation

```bash
pip install -e ".[dev]"
pytest -q
```

The suite takes about three minutes on a laptop CPU and covers every statement and
every branch of the package; CI fails below 100 %.

That number is not the goal in itself, and the suite is not padded to reach it. It is
worth having because of what chasing it turns up: five of the defects fixed in the
rewrite were found by writing a test for a path that had never executed — the openPMD
exporter, which was broken for every input, the relativistic pusher, and three separate
ways the discrete Gauss law failed next to a wall. A line that no test reaches is a
line whose behaviour nobody has checked.

| file | what it covers |
|---|---|
| `tests/test_kernels.py` | the numerical kernels in isolation, against exact results |
| `tests/test_physics.py` | rates, frequencies and conserved quantities against the literature |
| `tests/test_boundaries_and_loop.py` | the wall closures and their mirror symmetry, the gather with images at the walls, the random keys, the carried density and $\gamma\mathbf v$, and what the time loop traces |
| `tests/test_collisions.py` | pairing inside each cell, conservation per collision, the pair density, the Coulomb logarithm and gradients through the operator |
| `tests/test_config_and_outputs.py` | configuration objects as pytrees (leaves, validation, tree operations, `vmap`), diagnostics, openPMD records and the plotting helpers |
| `tests/test_api.py` | reproducibility, gradients, `vmap`, storage options, restarts, TOML, the command line, plotting and openPMD export |

### What is tested, and how

The suite is deliberately not a set of regression tests against stored output. A
regression test tells you that something changed; it does not tell you whether the
code was ever right. Each test here compares against something known independently:

**Exact results for the kernels.** The shape-function weights sum to one and reproduce
the spline; an absorbing wall keeps exactly the part of a particle's cloud that lies
inside the box; the deposit and the gather are adjoint; the discrete curls annihilate a
constant; a vacuum light wave at Courant number one is translated by exactly one cell
per step; the Boris rotation conserves speed to round-off and turns through the
analytic angle; the boundary maps do what they claim, position by position.

**Closed-form physics.** The kinetic Langmuir frequency at two wavenumbers, one where
it coincides with Bohm-Gross and one where it is 2.9 per cent above it, with the known
frequency shift of the grid included, to half a per cent; the tabulated Landau root
$1.4157 - 0.1533\,i$ at $k\lambda_D = 0.5$; the hard-coded roots themselves, re-derived
from the Faddeeva function wherever scipy is installed; the cold two-stream rate
$\omega_{pe}/2\sqrt2$ at $kv_0/\omega_{pe} = \sqrt{3/8}$; the Weibel marginal
wavenumber $k_c c = \omega_{pe}\sqrt{T_z/T_x - 1}$, checked as a threshold on a quiet
start with every mode seeded alike — the two modes well below it grow by at least a third
of $e^{\gamma t}$, and none well above it by more than a factor of two; the NRL
relaxation rates for a fast beam; the relativistic gyrofrequency
$\Omega = qB/\gamma m$, which the relativistic pusher reproduces after a full orbit
while the non-relativistic one overshoots by $\gamma$; the flux average
$u^2/(u^2+\sigma^2)$ that a wall with a Gaussian reflection law returns, which is not
the average over the distribution; the moments of the half-Maxwellian flux a thermal
wall re-emits; and the sheath drop of Hobbs and Wesson in front of a floating wall,
with and without electron reflection.

**Conservation laws.** Charge on the grid against charge on the particles, to
round-off; momentum in a periodic box; total energy, bounded for the explicit scheme
and at round-off for the implicit one; the discrete Gauss law at every kind of wall,
including one that returns half of each electron and a thermal wall facing a floating
conductor; reflective walls holding every particle inside the box and absorbing walls
removing some but not all.

**Documented behaviour that is easy to leave untested.** `store_particles=False`
dropping exactly the diagnostics that need velocities; the openPMD switches and a run
with no particles to write; the TOML loader's fallback for Python 3.10; every example
setting `JAX_ENABLE_X64` before it imports JAX; every example the documentation names
existing under exactly that name; and the version fallback for a fresh clone that has
not been installed, since `jaxincell/version.py` is generated at build time and is not
in the repository.

**The interface.** That two runs with one seed agree bit for bit on the CPU, and to
round-off on a GPU, whose scatter kernels need not sum in the same order twice, and that
two seeds do not; that `jax.grad` matches a central difference to one part in $10^4$
through both integrators; that `vmap` over seeds gives an ensemble; that `store_every`
and a restart reproduce the full run to round-off; that changing a physical parameter
does not change the treedef, which is what guarantees no recompilation; that the
overview figure has one panel per non-zero field component and per species; that an
openPMD export reads back with the right iterations, staggering and particle records;
and that the Courant warning fires for a run that would diverge and stays quiet for the
electrostatic and implicit runs that would not.

### Writing a new one

Prefer a comparison with something that can be derived on paper. When that is not
available, a conservation law or an exact symmetry is the next best thing. Reach for a
stored reference array only when neither exists, and say in the docstring why.

Keep them fast. The physics tests use the smallest resolution that still resolves the
result — usually a few tens of thousands of particles for a few hundred steps — and
state the tolerance they need. A test that takes a minute will be skipped by someone in
a hurry, and a tolerance chosen to make today's number pass is not a test.

### Continuous integration

`build_test.yml` installs the `dev` extra and, on Python 3.10 to 3.13 for every push and
pull request, runs `flake8` over the whole repository with the configuration in
`.flake8` (120 columns, McCabe complexity 10) and then the suite under coverage. The
build fails on any lint violation and on coverage below 100 per cent of statements and
branches, the threshold set under `[tool.coverage]` in `pyproject.toml`, and it prints
the ten slowest tests. A second job runs the four quickest examples (collisions, wall
reflection, conservation and the Langmuir scan), so that a change to the
interface cannot quietly break the scripts people start from; the others take minutes
each. `docs.yml` builds the documentation with `-W`, so a broken cross-reference or a
missing substitution fails the build, and the release workflow runs the same tests
before it builds anything.

### Documentation

The documentation is MyST Markdown built with Sphinx and the `pydata-sphinx-theme`;
Read the Docs builds `latest` from `main` and `stable` from the last tag.

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

`-W` turns warnings into errors, as the CI build does, so a clean local build is the
condition for merging. The pages are grouped in the directories under `docs/`, the
bibliography is `docs/references.bib`, cited with `{cite}` roles, and the API page uses
`autodoc`, so docstrings are part of the documentation (Google-style sections, parsed
by Napoleon). Equations are LaTeX in MyST math, figures carry a caption naming the
script that made them, and parameter tables list the default and whether the parameter
is differentiable.

The figures are committed, so the site builds quickly and deterministically. The
scripts in `docs/scripts/` regenerate them, and `docs/scripts/dispersion.py` holds the
linear kinetic dispersion solvers they compare against:

```bash
python docs/scripts/make_all.py                         # all, about seven minutes on a laptop
python docs/scripts/make_all.py fig_landau_damping.py   # one script
```

Each script writes its PNG files and records the numbers it measured (growth rates,
frequencies, energy errors, timings) in `docs/_static/figures/measurements.json`, with
the commit, the library versions, the precision and the device under `_provenance`. The
documentation quotes double-precision results, so a run with `JAX_ENABLE_X64=0` refuses
to record. `conf.py` exposes the numbers as substitutions, so that a page can write
`{{ landau_gamma_measured }}` and always quote the value of the committed figure.
`fig_scaling.py` measures wall-clock time and should be run on an otherwise idle
machine.

## Releasing

Versions come from git tags through `setuptools_scm`; there is no version string to
edit. Publishing a GitHub release runs `pypi_publish.yml`, which runs the tests on the
tagged commit and only then builds and uploads to PyPI; Read the Docs rebuilds
`stable` from the tag.

```bash
git tag v0.2 && git push origin v0.2
gh release create v0.2 --generate-notes
```

## Roadmap

Planned, in no particular order; open an issue first so that the design can be discussed
before the code is written.

* A **volumetric** source: ionisation of a neutral background, which is what a discharge
  needs and which the boundary reservoir of {doc}`user_guide/sources` is not. The pool of
  inactive particles it would emit into already exists.
* A **drifting warm reservoir**. `Source` samples a Maxwellian at rest or a cold beam and
  refuses the two together, because the crossing density of a drifting Maxwellian is
  proportional to $v\exp[-(v-u)^2/2\sigma^2]$ on $v>0$ and adding a drift to a Rayleigh
  sample is not a sample of it. An inverse-CDF sampler with a derivative taken from the
  normalised distribution would lift the restriction.
* A **source with the implicit scheme**, which is refused: it would emit a new population
  inside every Picard iteration. Emitting once per step, before the iteration, and holding
  the samples fixed across it is the obvious route.
* **Sensitivities that do not follow the plasma particles** {cite}`chung2020`. The
  gradient through a wall is exact but stops estimating the physical response over a long
  window ({doc}`user_guide/differentiation`); a continuum-oriented sensitivity, carried by
  its own particles, is the published way round it and is a research project rather than a
  patch.
* A series RLC circuit between the two electrodes, so that a wall can be biased or left
  genuinely floating rather than short-circuited to its partner {cite}`verboncoeur1993`.
  The charge-evolving collector of `field_bc="open"` is the resistance-free limit of it.
* Time-dependent external fields.
* Ionisation and recombination.
* Secondary electron emission with an energy-dependent yield {cite}`furman2002`, which,
  unlike reflection, creates electrons and needs the same pool of inactive particles.
* A two-dimensional version. `_core.py` would change thoroughly; the configuration
  objects, the time loop, the diagnostics and the differentiability would not.
