# Building the documentation

The documentation is written in MyST Markdown and built with Sphinx and the
`pydata-sphinx-theme`. Read the Docs builds the `latest` version from `main` and the
`stable` version from the last tag.

## Local build

```bash
pip install -e .
pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. `-W` turns warnings into errors,
which is also what the continuous-integration build does, so a clean local build is
the condition for merging.

## Layout

| path | contents |
|---|---|
| `docs/conf.py` | Sphinx configuration; also loads `measurements.json` into MyST substitutions |
| `docs/index.md` | landing page |
| `docs/getting_started/`, `docs/user_guide/`, `docs/numerics/`, `docs/examples/`, `docs/api/`, `docs/development/` | the sections of the site |
| `docs/references.bib` | bibliography, cited with `{cite}` roles |
| `docs/_static/figures/` | figures and `measurements.json` |
| `docs/scripts/` | scripts that generate the figures |

The API pages use `autodoc` on the public names, so docstrings in the source are part
of the documentation. Google-style sections (`Args:`, `Returns:`) are parsed by
Napoleon.

## Regenerating the figures

The figures are static files committed to the repository, so that the site builds
quickly and deterministically on Read the Docs. To regenerate them after a change to
the code:

```bash
pip install scipy
python docs/scripts/make_all.py            # all figures, about ten minutes on a laptop
python docs/scripts/make_all.py fig_landau_damping.py   # one script
```

Each script writes its PNG files and records the numbers it measured (growth rates,
frequencies, energy errors, timings) in `docs/_static/figures/measurements.json`.
`conf.py` exposes those numbers as substitutions, so that a page can write
`{{ landau_gamma_measured }}` and always quote the value of the committed figure.
`fig_scaling.py` measures wall-clock time and should be run on an otherwise idle
machine.

## Style

The prose is direct and specific. Equations are written in LaTeX inside MyST math
fences, figures carry a caption that names the script that generated them, and every
parameter table lists the default and whether the parameter is differentiable.
