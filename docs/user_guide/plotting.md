# Plotting

{func}`~jaxincell.plot` draws one animated overview of a run: space-time maps of every
non-zero field component and of the charge density, the velocity distribution of each
species, the $x$-$v$ phase space of each species, and the conservation histories.

```python
from jaxincell import plot

plot(output)                                               # animate on screen
plot(output, omega=float(simulation.plasma_frequency()))   # time axis in 1/omega_pe
plot(output, omega=Omega_i, omega_label=r"\Omega_i")       # ...or in anything else
plot(output, direction="xz", save="run.mp4")               # write a file and show it
```

Every argument, with its default, is listed in {func}`~jaxincell.plot`.

## What a frame shows

* Colour limits and axes are fixed over the whole run, so a feature that grows is visible
  as growth rather than being renormalised away at every frame.
* A species is every slot that belongs to it, and whether a slot holds a particle at a
  given step is its weight there. A particle a wall collected half-way through is in the
  early frames and not the late ones; a slot a source has not filled yet is in neither.
* Every histogram is weighted, so unequal weights count for what they are.
* An empty bin on a logarithmic colour scale is **masked** and drawn as the background,
  rather than given a count of one so that the logarithm is defined — which would shift
  every other bin by a particle.
* Whatever falls outside the velocity range is counted and said in the panel's title,
  rather than piled onto the end bin where it looks like a spike in the distribution.
* Fields are drawn on the coordinates they live on: $E$ and $J$ on `output.faces`, $B$ and
  $\rho$ on `output.grid`, half a cell apart.
* The time axis is drawn from the stored times themselves, so a run continued from a
  state, whose stored times need not be evenly spaced, is drawn where it happened.

## Memory

Each frame is built when it is drawn, from one pass that keeps numbers rather than arrays.

| on a 400-step, 256-cell, 25 000-particle run | |
|---|---|
| building frames on demand | 21 MB |
| precomputing every frame first | 1.1 GB |
| the `diagnostics` panel, on top | most of a gigabyte |

A kinetic-energy history is a pass over the whole phase space, which is what the
diagnostics panel costs. Pass `diagnostics=False` if it is not wanted.

## Writing a movie

```python
plot(output, save="run.mp4", fps=30, stride=2, show=False)
```

| | |
|---|---|
| `save` | needs `ffmpeg` on the PATH |
| ffmpeg missing | warns and carries on rather than failing |
| ffmpeg present but failing | raises `RuntimeError` with ffmpeg's own message, so a movie that was not written never passes for one that was |
| `stride` | halves or quarters the file when the run has more stored steps than a movie needs |
| `dpi` | trades resolution for size |
| `save` and `show` | independent; ask for both and both happen |

A thousand-frame movie of 25 000 particles takes about eight seconds — 7.6 ms a frame,
including building that frame's histograms — because the writer caches the static
background once, redraws only the artists that move, and pipes raw frames to ffmpeg rather
than re-rendering the whole canvas through `savefig` every frame. The obvious
implementation, rendering everything each time, takes about a hundred seconds for the same
file.

## Custom figures

`plot` is a convenience, not a framework. Everything it draws is available as plain arrays,
and a focused figure is usually a few lines:

```python
import numpy as np, matplotlib.pyplot as plt

amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1])
plt.semilogy(np.asarray(output.t) * omega_pe, amplitude)
plt.xlabel(r"$t\,\omega_{pe}$"); plt.ylabel(r"$|E_{k=1}|$ (V/m)")
```

The scripts in `docs/scripts/` that produce every figure in this documentation are written
that way and are worth reading as templates; `docs/scripts/common.py` holds the shared
style and the fitting helpers. Matplotlib is a required dependency, imported when `plot` is
first used rather than with the package.
