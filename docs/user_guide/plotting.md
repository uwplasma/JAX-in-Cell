# Plotting

{func}`~jaxincell.plot` draws one animated overview of a run: space-time maps of every
non-zero field component and of the charge density, the velocity distribution of each
species, and the $x$-$v$ phase space of each species.

```python
from jaxincell import plot

plot(output)                                     # animate on screen
plot(output, omega=float(simulation.plasma_frequency()))   # time axis in 1/omega_pe
plot(output, direction="xz", save="run.mp4", show=False)   # write a file
```

| argument | meaning | default |
|---|---|---|
| `direction` | velocity components to show, any of `"x"`, `"y"`, `"z"`, e.g. `"xz"` | `"x"` |
| `omega` | frequency that makes the time axis dimensionless | `None` |
| `save` | file name of an MP4 to write with ffmpeg | `None` |
| `fps` | frames per second of the file | `25` |
| `stride` | keep every n-th stored step in the file | `1` |
| `dpi` | resolution of the figure | `80` |
| `interval` | delay between frames of the on-screen animation, ms | `30` |
| `show` | call `plt.show()` | `True` |
| `vbins` | velocity bins; reduced automatically if the histograms would be too large | `96` |

The colour limits and the axes are fixed over the whole run, so a feature that grows
is visible as growth rather than being renormalised away at every frame. All the
per-frame arrays — the space-time maps, the distributions, the phase-space histograms
— are computed once, up front, so both the on-screen animation and the file are one
render per frame.

## Writing a movie

`save="run.mp4"` needs `ffmpeg` on the PATH; if it is missing the call warns and
carries on rather than failing. A thousand-frame movie takes about six seconds,
because the writer caches the static background once, redraws only the artists that
actually move, and pipes raw frames to ffmpeg rather than re-rendering the whole
canvas through `savefig` for every frame. Rendering everything each time, which is
what the obvious implementation does, takes about a hundred seconds for the same file.

```python
plot(output, save="run.mp4", fps=30, stride=2, show=False)
```

`stride` halves or quarters the file when the run has more stored steps than a movie
needs; `dpi` trades resolution for size.

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
the shared style and the fitting helpers.

## Without matplotlib

Matplotlib is a dependency, so `plot` is normally there. The import is nevertheless
guarded: in an environment where matplotlib has been removed, `jaxincell` still
imports and everything except the plotting works.
