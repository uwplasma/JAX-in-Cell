"""The movies in the README and the documentation, one per input file in ``inputs/``.

Each is the input file's own run, with the particles kept, drawn by :func:`jaxincell.plot` and
written as H.264 with its index first: it plays in a browser as soon as it is clicked, and weighs
a few hundred kilobytes. Run from the repository root::

    python docs/scripts/movies.py [name ...]
"""
import sys
from pathlib import Path

from jaxincell import load_toml, plot

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "_static" / "movies"
STRIDE = {"two_stream": 8, "bump_on_tail": 1, "weibel": 1, "sheath_unmagnetized": 1}   # a few seconds each

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or STRIDE:
        simulation, run = load_toml(ROOT / "inputs" / f"{name}.toml")
        settings = {key: value for key, value in run.items() if key != "plot"}
        out = simulation.run(**{**settings, "store_particles": True})
        path = OUT / f"{name}.mp4"
        plot(out, save=str(path), show=False, stride=STRIDE[name], fps=20, dpi=64)
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} kB)")
