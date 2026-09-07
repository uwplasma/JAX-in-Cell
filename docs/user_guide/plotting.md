# Plotting and animation

{func}`jaxincell.plot` turns an output dictionary into an animated overview figure.
It works with the raw output of `run` and with the dictionary after
{func}`jaxincell.diagnostics`.

```python
from jaxincell import plot

plot(output)                                   # interactive window, all frames
plot(output, direction="xz")                   # also show v_z distributions and x-v_z phase space
plot(output, animation_interval=5)             # slower playback: 5 ms between frames
plot(output, save_mp4="run.mp4", fps=50, dpi=150, save_dpi=60, save_stride=5, show=False)
```

## What is shown

The figure has three groups of panels, laid out for the requested velocity components.

Heat maps of the fields as functions of $x$ (horizontal) and time (vertical)
: One panel per non-zero component of $\mathbf E$ and $\mathbf B$ (a component counts
  as non-zero when its largest absolute value exceeds `threshold`), plus one panel of
  the strongest current-density component if any magnetic field is plotted, plus the
  charge density. The colour limits are fixed over the whole run from a high
  percentile of the data, so growth and decay in time are visible. On the electric
  field panels a line shows the instantaneous profile of the current frame.

Velocity distributions $f(v)$
: One panel per requested component, for electrons and ions on separate axes, with the
  current frame drawn solid and the initial distribution dashed. The velocity axis is
  chosen per species from a robust percentile so that the ions, which move much less
  than the electrons, are not compressed into a single bin.

Phase space $x$-$v$
: One panel per requested component and species, drawn as a two-dimensional histogram
  with a logarithmic colour scale so that both the bulk and the tenuous structures
  (vortices, beams) are visible.

The animation steps through every stored time step when shown on screen. Frames are
recomputed from precomputed histograms, so playback is smooth for runs of a few
thousand steps.

## Arguments

| argument | default | meaning |
|---|---|---|
| `direction` | `"x"` | One or two of `x`, `y`, `z`, for example `"xz"`. Selects the velocity components used in the distribution and phase-space panels. The spatial axis is always $x$. |
| `threshold` | `1e-12` | Field components with $\max|F| \le$ `threshold` are not plotted. |
| `animation_interval` | `1` | Delay between frames in milliseconds when showing on screen. |
| `show` | `True` | Call `matplotlib.pyplot.show`. |
| `save_mp4` | `None` | File name; if given, the animation is written with `ffmpeg`. |
| `fps` | `30` | Frames per second of the file. |
| `dpi` | `150` | Resolution of the figure on screen. |
| `save_dpi` | `None` | Resolution of the file; `None` uses `dpi`. `60` gives small files. |
| `save_stride` | `1` | Keep every n-th frame in the file. |
| `save_crf` | `None` | Constant-rate-factor quality of the encoder (higher is smaller); `None` uses a codec default. |
| `save_codec` | `None` | Force an encoder such as `libx264`, `h264_videotoolbox` or `libx265`; `None` picks the first one that `ffmpeg` reports as available. |
| `save_preset` | `None` | Encoder preset for `libx264`/`libx265` (for example `veryfast`). |

`save_mp4` requires an `ffmpeg` executable on the `PATH`. The function queries
`ffmpeg -encoders` once and prefers hardware encoders when present.

## Species with the same charge sign

The distribution and phase-space panels separate electrons from ions by the sign of
the charge. Populations of the same sign (a bulk and a beam of electrons) are drawn
together; to look at them separately, select the particles with
`output["species_integer_index"]` and plot them yourself.

## Making your own figures

The output arrays are plain and the plotting code is not needed to work with them.
A few idioms cover most cases; the figures in this documentation are made this way
and the scripts under `docs/scripts/` can serve as templates.

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.asarray(output["time_array"]) * float(output["plasma_frequency"])
x = np.asarray(output["grid"])

# field energy on a logarithmic scale (after diagnostics)
plt.semilogy(t, output["electric_field_energy"])

# space-time diagram of one field component
Ex = np.asarray(output["electric_field"][:, :, 0])
plt.pcolormesh(x, t, Ex, cmap="RdBu_r", vmin=-abs(Ex).max(), vmax=abs(Ex).max())

# Fourier modes of E_x as functions of time
modes = np.abs(np.fft.rfft(Ex, axis=1)) / Ex.shape[1]
plt.semilogy(t, modes[:, 1:5])

# electron phase space at the last step (before diagnostics: use "positions")
electrons = output["species_integer_index"] == 0
plt.scatter(output["positions"][-1, electrons, 0], output["velocities"][-1, electrons, 0], s=1)
```
