# Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Fifty-seven tests in three files, about a minute on one CPU core, covering 99 per cent
of the package. The five statements left are the ones this environment cannot reach:
the `__main__` guard, and the two import fallbacks for an older Python and for a source
tree without a generated version file.

| file | what it covers |
|---|---|
| `tests/test_kernels.py` | the numerical kernels in isolation, against exact results |
| `tests/test_physics.py` | rates, frequencies and conserved quantities against the literature |
| `tests/test_api.py` | reproducibility, gradients, `vmap`, storage options, restarts, TOML, the command line, plotting and openPMD export |

## What is tested, and how

The suite is deliberately not a set of regression tests against stored output. A
regression test tells you that something changed; it does not tell you whether the
code was ever right. Each test here compares against something known independently:

**Exact results for the kernels.** The shape-function weights sum to one and reproduce
the spline; the deposit and the gather are adjoint; the discrete curls annihilate a
constant; a vacuum light wave at Courant number one is translated by exactly one cell
per step; the Boris rotation conserves speed to round-off and turns through the
analytic angle; the boundary maps do what they claim, position by position.

**Closed-form physics.** Bohm-Gross frequencies at two wavenumbers; the tabulated
Landau root $1.4157 - 0.1533\,i$ at $k\lambda_D = 0.5$; the cold two-stream rate
$\omega_{pe}/2\sqrt2$ at $kv_0/\omega_{pe} = \sqrt{3/8}$; the Weibel marginal
wavenumber $k_c c = \omega_{pe}\sqrt{T_z/T_x - 1}$, checked as a threshold — every
mode below it grows by more than ten, none above it by more than three; the NRL
relaxation rates for a fast beam; and the relativistic gyrofrequency
$\Omega = qB/\gamma m$, which the relativistic pusher reproduces after a full orbit
while the non-relativistic one overshoots by $\gamma$.

**Conservation laws.** Charge on the grid against charge on the particles, to
round-off; momentum in a periodic box; total energy, bounded for the explicit scheme
and at round-off for the implicit one; reflective walls holding every particle inside
the box and absorbing walls removing some but not all.

**The interface.** That two runs with one seed agree bit for bit and two seeds do not;
that `jax.grad` matches a central difference to one part in $10^4$ through both
integrators; that `vmap` over seeds gives an ensemble; that `store_every` and a restart
reproduce the full run exactly; that changing a physical parameter does not change the
treedef, which is what guarantees no recompilation; that the overview figure has one
panel per non-zero field component and per species; and that an openPMD export reads
back with the right iterations, staggering and particle records; and that the
Courant warning fires for a run that would diverge and stays quiet for the
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

The workflow in `.github/workflows/` installs the `dev` extra and runs the suite on
Python 3.10 to 3.13 on every push, and
`docs.yml` builds the documentation with `-W`, so a broken cross-reference or a
missing substitution fails the build. The figures are committed rather than rebuilt in
CI, because the full set takes a few minutes; regenerate them with
`python docs/scripts/make_all.py` when the numbers they quote would change.
