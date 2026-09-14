# Tests and documentation

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

## What is tested, and how

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

## Writing a new one

Prefer a comparison with something that can be derived on paper. When that is not
available, a conservation law or an exact symmetry is the next best thing. Reach for a
stored reference array only when neither exists, and say in the docstring why.

Keep them fast. The physics tests use the smallest resolution that still resolves the
result — usually a few tens of thousands of particles for a few hundred steps — and
state the tolerance they need. A test that takes a minute will be skipped by someone in
a hurry, and a tolerance chosen to make today's number pass is not a test.

## Continuous integration

`build_test.yml` installs the `dev` extra and, on Python 3.10 to 3.13 for every push and
pull request, runs `flake8` over the whole repository with the configuration in
`.flake8` (120 columns, McCabe complexity 10) and then the suite under coverage. The
build fails on any lint violation and on coverage below 100 per cent of statements and
branches, the threshold set under `[tool.coverage]` in `pyproject.toml`, and it prints
the ten slowest tests. A second job runs the four quickest examples (collisions, wall
reflection, energy conservation and the Langmuir scan), so that a change to the
interface cannot quietly break the scripts people start from; the others take minutes
each. `docs.yml` builds the documentation with `-W`, so a broken cross-reference or a
missing substitution fails the build, and the release workflow runs the same tests
before it builds anything.

## Documentation

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
