# Plotting

{func}`~jaxincell.plot` draws one animated overview of a run: space-time maps of every
non-zero field component and of the charge density, the velocity distribution of each
species, the $x$-$v$ phase space of each species, and the conservation histories.

```python
from jaxincell import plot

plot(output)                                     # animate on screen
plot(output, omega=float(simulation.plasma_frequency()))   # time axis in 1/omega_pe
plot(output, omega=Omega_i, omega_label=r"\Omega_i")       # ...or in anything else
plot(output, direction="xz", save="run.mp4")               # write a file and show it
```

Every argument, with its default, is listed in {func}`~jaxincell.plot`. The colour
limits and the axes are fixed over the whole run, so a feature that grows is visible as
growth rather than being renormalised away at every frame.

What each frame shows is that frame. A species is every slot that belongs to it, and
whether a slot holds a particle at a given step is its weight there, so a particle a
wall collected half-way through is in the early frames and not the late ones, and a slot
a source has not filled yet is in neither. Every histogram is weighted, so unequal
weights count for what they are.

Each frame is built when it is drawn, from one pass that keeps numbers rather than
arrays. On a 400-step, 256-cell, 25000-particle run that is 21 MB of working memory
against 1.1 GB for precomputing every frame first. The `diagnostics` panel costs what
{func}`~jaxincell.diagnostics` costs on top, which on that run is most of a gigabyte:
a kinetic-energy history is a pass over the whole phase space. Pass `diagnostics=False`
if that is not wanted.

Two small things a plot should not do. An empty bin on a logarithmic colour scale is
**masked** and drawn as the background, rather than given a count of one so that the
logarithm is defined — which shifted every other bin by a particle. And whatever falls
outside the velocity range is counted and said in the panel's title rather than piled
onto the end bin, where it looks like a spike in the distribution.

Fields are drawn on the coordinates they live on: $E$ and $J$ on `output.faces`, $B$ and
$\rho$ on `output.grid`, which is half a cell apart. The time axis is drawn from the
stored times themselves, so a run continued from a state, whose stored times need not be
evenly spaced, is drawn where it happened.

## Writing a movie

`save="run.mp4"` needs `ffmpeg` on the PATH; if it is missing the call warns and
carries on rather than failing. If ffmpeg is there but fails — an unwritable path, an
encoder it was built without — `plot` raises `RuntimeError` with ffmpeg's own message,
so a movie that was not written never passes for one that was. A thousand-frame movie of
25000 particles takes about eight seconds, 7.6 ms a frame including building that frame's
histograms, because the writer caches the static background once, redraws only the
artists that actually move, and pipes raw frames to ffmpeg rather than re-rendering the
whole canvas through `savefig` for every frame. Rendering everything each time, which is
what the obvious implementation does, takes about a hundred seconds for the same file.

```python
plot(output, save="run.mp4", fps=30, stride=2, show=False)
```

`stride` halves or quarters the file when the run has more stored steps than a movie
needs; `dpi` trades resolution for size. `save` and `show` are independent: ask for both
and both happen.

## Custom figures

`plot` is a convenience, not a framework. Everything it draws is available as plain
arrays, and a focused figure is usually a few lines:

```python
import numpy as np, matplotlib.pyplot as plt

amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1])
plt.semilogy(np.asarray(output.t) * omega_pe, amplitude)
plt.xlabel(r"$t\,\omega_{pe}$"); plt.ylabel(r"$|E_{k=1}|$ (V/m)")
```

The scripts in `docs/scripts/` that produce every figure in this documentation are
written that way and are worth reading as templates; `docs/scripts/common.py` holds
the shared style and the fitting helpers. Matplotlib is a required dependency, imported
when `plot` is first used rather than with the package.
