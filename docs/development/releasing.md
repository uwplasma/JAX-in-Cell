# Releasing

Versions are derived from git tags by `setuptools_scm`; there is no version string to
edit. The `pypi_publish.yml` workflow builds a source distribution and a wheel and
uploads them to PyPI whenever a tag is pushed or a GitHub release is created.

```bash
git tag v0.2
git push origin v0.2
```

Read the Docs rebuilds the `stable` documentation from the new tag. Released versions
so far:

| tag | date |
|---|---|
| v0.1 | 2025-12-15 |
| v0.06 | 2025-06-24 |
| v0.05 | 2025-03-14 |
| v0.0.4 | 2025-03-06 |
| v0.0.3 | 2025-02-15 |
