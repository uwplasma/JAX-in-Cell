# Contributing

Contributions are welcome through pull requests on
[GitHub](https://github.com/uwplasma/JAX-in-Cell). The repository's
`CONTRIBUTING.md` describes how to report bugs and propose enhancements and links to
the code of conduct; this page covers the mechanics of a code change.

## Workflow

1. Fork the repository and clone your fork, or create a branch if you have write
   access.
2. Install in editable mode with the test dependencies:
   ```bash
   pip install -e .
   pip install pytest pytest-cov flake8
   ```
3. Make the change with a test that exercises it. Keep the static and differentiable
   parameter lists consistent (see {doc}`architecture`).
4. Run `pytest` and `flake8 . --select=E9,F63,F7,F82`.
5. If the change affects a numerical method, rerun the relevant figure script under
   `docs/scripts/` and update the documentation page that describes the method.
6. Open a pull request against `main` with a description of what changed and why.
   The continuous-integration workflow runs the tests on four Python versions and
   builds the documentation.

## Questions and discussion

Use the [discussions](https://github.com/uwplasma/JAX-in-Cell/discussions) page for
questions and the [issue tracker](https://github.com/uwplasma/JAX-in-Cell/issues) for
bugs and feature requests. Security-related reports should go by email to the
maintainers listed in `CONTRIBUTING.md`.
