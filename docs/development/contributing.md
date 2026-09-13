# Contributing

Contributions are welcome through pull requests on
[GitHub](https://github.com/uwplasma/JAX-in-Cell): bug fixes, new physics, examples and
documentation. By contributing you confirm that the work is yours to give and agree
that it is released under the project's MIT licence.

## Reporting a bug or asking for a feature

Search the [issue tracker](https://github.com/uwplasma/JAX-in-Cell/issues) first, then
open an issue with what you expected, what happened, and the smallest script that
reproduces it, together with the versions of Python, JAX and JAX-in-Cell and the
platform. Questions are better placed on the
[discussions](https://github.com/uwplasma/JAX-in-Cell/discussions) page. Do not report
a security problem in public; send it to rogerio.jorge@wisc.edu.

## Workflow

1. Fork the repository and clone your fork, or create a branch if you have write
   access.
2. Install in editable mode with the test dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Make the change with a test that exercises it. Keep the static and differentiable
   parameter lists consistent (see {doc}`architecture`).
4. Run `pytest` and `flake8`. Continuous integration fails on any lint violation under
   the configuration in `.flake8` and on coverage below 100 per cent.
5. If the change affects a numerical method, rerun the relevant figure script under
   `docs/scripts/` and update the documentation page that describes the method.
6. Open a pull request against `main` with a description of what changed and why.
   The continuous-integration workflow runs the tests on four Python versions and
   builds the documentation.

Write commit subjects in the imperative mood and under about seventy characters
(`Add a thermal wall`), and use the body for why the change was needed and anything a
reviewer would not guess from the diff.
