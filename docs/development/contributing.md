# Development

```{toctree}
:maxdepth: 1

architecture
testing
roadmap
```

{doc}`architecture` maps the modules and is the first thing to read before changing
the code, {doc}`testing` describes the test suite, the documentation build and
continuous integration, and {doc}`roadmap` lists what is done and what is planned.
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
   parameters consistent ({doc}`architecture`).
3. Run `pytest` and `flake8`; CI fails on any lint violation and on coverage below
   100 per cent.
4. If a numerical method changed, rerun its figure script under `docs/scripts/` and
   update the page that describes the method.
5. Open a pull request against `main` saying what changed and why. Commit subjects are
   imperative and under about seventy characters (`Add a thermal wall`); the body says
   why.

## Releasing

Versions come from git tags through `setuptools_scm`; there is no version string to
edit. Publishing a GitHub release runs `pypi_publish.yml`, which runs the tests on the
tagged commit and only then builds and uploads to PyPI; Read the Docs rebuilds
`stable` from the tag.

```bash
git tag v0.2 && git push origin v0.2
gh release create v0.2 --generate-notes
```
