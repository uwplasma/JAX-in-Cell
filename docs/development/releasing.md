# Releasing

Versions are derived from git tags by `setuptools_scm`; there is no version string to
edit. Publishing a GitHub release runs `pypi_publish.yml`. It runs the build-and-test
workflow on the tagged commit, builds a source distribution and a wheel from a full
clone (`setuptools_scm` needs the tags, and writes `jaxincell/version.py` during the
build), and uploads both to PyPI through trusted publishing. Nothing is uploaded if a
test fails, and pushing a tag without publishing a release uploads nothing.

```bash
git tag v0.2
git push origin v0.2
gh release create v0.2 --generate-notes
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
