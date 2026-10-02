# Testing

The test suite lives under `tests/` and uses `pytest`. It covers the parameter
handling (defaults, validation, routing, hashing), every numerical kernel on small
inputs, the two time steps, the `Simulation` class, the command-line entry point, the
diagnostics, the plotting code and differentiability with respect to each
differentiable parameter.

```bash
pip install pytest pytest-cov
pytest
pytest --cov=jaxincell --cov-branch --cov-report=term-missing
```

Most tests run in seconds; the whole suite takes a few minutes on a laptop because
each distinct configuration compiles its own program. `tests/helpers.py` provides a
minimal parameter tree with two particles per species and one time step that most
tests start from.

## Continuous integration

`.github/workflows/build_test.yml` runs on every push and pull request to `main`, on
Python 3.9 to 3.12. It installs the package, checks that it imports from outside the
source tree, runs `flake8` for syntax errors and undefined names, runs the suite with
coverage and uploads the report to Codecov. The documentation is built by
`.github/workflows/docs.yml` with warnings treated as errors, and by Read the Docs
for the published site.

## Physics checks

The unit tests check shapes, conservation properties on small cases and consistency
between code paths. The comparisons with linear theory in {doc}`../numerics/verification`
are not part of the automated suite; they are produced by the scripts under
`docs/scripts/` and should be rerun after any change to the deposition, interpolation,
pushers or field update.
