"""The README's gallery: one panel of a documentation figure per benchmark, at one size.

The documentation's figures have two to four panels, which read poorly as thumbnails. This
script cuts the headline panel out of each (the figures come from the ``fig_*.py`` scripts
and the examples, so nothing is run here) and writes it to ``docs/_static/readme/`` at one
width, so the README's two-column gallery lines up::

    python docs/scripts/readme_figures.py

Run it after regenerating a figure the README shows. Each entry is
``name: (figure, columns, rows, column, row)``; the panel label of the documentation figure is
kept, and the README's caption names the panel.
"""
from pathlib import Path

import numpy as np
from PIL import Image

FIGURES = Path(__file__).resolve().parents[1] / "_static" / "figures"
OUT = FIGURES.parent / "readme"
WIDTH, HEIGHT = 900, 720        # pixels; every thumbnail the same, shown at half that in the README

PANELS = {
    "landau_damping": ("landau_damping", 2, 1, 0, 0),
    "two_stream": ("two_stream_scan", 1, 1, 0, 0),
    "relativistic_two_stream": ("relativistic_two_stream", 2, 2, 0, 0),
    "bump_on_tail": ("bump_on_tail", 2, 1, 0, 0),
    "explicit_implicit": ("explicit_implicit", 2, 2, 1, 1),
    "compare_models": ("compare_models", 2, 2, 0, 0),
    "electron_field": ("electron_field", 2, 2, 0, 1),
    "parameters_and_sampling": ("parameters_and_sampling", 1, 1, 0, 0),
    "output_and_restart": ("output_and_restart", 2, 1, 0, 0),
    "conservation": ("conservation", 3, 1, 0, 0),
    "invariants": ("invariants", 3, 1, 1, 0),
    "weibel": ("weibel", 3, 1, 2, 0),
    "sheath_unmagnetized": ("sheath_source", 3, 1, 0, 0),
    "sheath_magnetized": ("sheath_magnetized", 3, 1, 1, 0),
    "sheath_reflection": ("sheath", 3, 1, 2, 0),
    "wall_reflection": ("wall_reflection", 3, 1, 0, 0),
    "grazing_sheath": ("grazing_rehearsal", 3, 1, 1, 0),
    "collisions": ("collisions", 2, 1, 0, 0),
    "external_fields_mirror": ("external_fields_3d", 3, 1, 1, 0),
    "external_fields_drift": ("external_fields_3d", 3, 1, 0, 0),
    "sheath_optimization": ("sheath_optimization", 3, 1, 1, 0),
    "optimize_two_stream": ("autodiff", 2, 1, 0, 0),
    "scaling": ("scaling", 2, 1, 0, 0),
}


def _cuts(white, parts):
    """Where to cut a figure into ``parts`` equal panels along one axis: at the middle of the
    widest all-white gap near each nominal boundary, so no tick label or axis title is split."""
    n, cuts = white.size, [0]
    for i in range(1, parts):
        lo, hi = int((i / parts - 0.12) * n), int((i / parts + 0.12) * n)
        best, start = (0, i * n // parts), None
        for j in range(lo, hi):
            if white[j]:
                start = j if start is None else start
                if j - start + 1 > best[0]:
                    best = (j - start + 1, (start + j) // 2)
            else:
                start = None
        cuts.append(best[1])
    return cuts + [n]


def panel(image, columns, rows, column, row):
    """One panel of a figure, trimmed to its content."""
    white = np.asarray(image.convert("L")) > 245
    xs, ys = _cuts(white.all(axis=0), columns), _cuts(white.all(axis=1), rows)
    cut = image.crop((xs[column], ys[row], xs[column + 1], ys[row + 1]))
    ink = ~(np.asarray(cut.convert("L")) > 245)
    rows_, cols_ = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    pad = 12
    return cut.crop((max(cols_[0] - pad, 0), max(rows_[0] - pad, 0),
                     min(cols_[-1] + pad, cut.width), min(rows_[-1] + pad, cut.height)))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, (figure, *where) in PANELS.items():
        cut = panel(Image.open(FIGURES / f"{figure}.png").convert("RGB"), *where)
        scale = min(WIDTH / cut.width, HEIGHT / cut.height)
        cut = cut.resize((round(cut.width * scale), round(cut.height * scale)), Image.LANCZOS)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
        canvas.paste(cut, ((WIDTH - cut.width) // 2, (HEIGHT - cut.height) // 2))
        cut = canvas
        cut.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE).save(OUT / f"{name}.png", optimize=True)
        print(f"{name}: {figure} -> {cut.size}")
