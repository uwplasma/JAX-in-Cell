"""Animated overview figure of a simulation output."""
import subprocess
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm

__all__ = ["plot"]

_AXIS = {"x": 0, "y": 1, "z": 2}
_MAX_ELEMENTS = 2e8  # bound on the total size of the precomputed phase-space histograms


def _index(a, lo, hi, n):
    """Bin index of ``a`` on ``n`` equal bins over ``[lo, hi]``, clipped to the edge bins."""
    return np.clip(((a - lo) * (n / (hi - lo))).astype(np.int32), 0, n - 1)


def _write_movie(fig, update, animated, frames, path, fps):
    """Pipe raw frames to ffmpeg, redrawing only the artists that move.

    Caching the static background once and blitting the handful of animated
    artists onto it is what makes this quick: a full redraw of the figure costs
    about 60 ms a frame, restoring the background and blitting about 6 ms.
    """
    original, canvas = fig.canvas, FigureCanvasAgg(fig)   # blitting needs an Agg canvas
    for artist in animated:
        artist.set_animated(True)                         # keep them out of the cached background
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    width, height = canvas.get_width_height()
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(path)]
    try:
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)
    except FileNotFoundError:
        warnings.warn(f"ffmpeg was not found on PATH, so {path} was not written; install it, for "
                      "example with `conda install -c conda-forge ffmpeg`", RuntimeWarning, stacklevel=3)
    else:
        for i in frames:
            canvas.restore_region(background)
            update(i)
            for artist in animated:
                fig.draw_artist(artist)
            canvas.blit(fig.bbox)
            ffmpeg.stdin.write(canvas.buffer_rgba())
        ffmpeg.stdin.close()
        ffmpeg.wait()
    for artist in animated:
        artist.set_animated(False)
    fig.canvas = original


def _hist(codes, shape):
    """Histogram of combined integer codes, e.g. ``(frame * nv + iv) * nx + ix``, as float32."""
    return np.bincount(codes.ravel(), minlength=int(np.prod(shape))).reshape(shape).astype(np.float32)


def plot(out, direction="x", omega=None, save=None, fps=25, stride=1, dpi=80, interval=30,
         show=True, vbins=96):
    """Animate the fields, velocity distributions and phase space of a run.

    One figure, animated over the stored steps. It holds a space-time heat map
    of every non-zero component of ``E`` and ``B`` and of the charge density,
    with a line marking the current time and the instantaneous profile drawn
    over it; the velocity distribution ``f(v)`` of each species (solid: current
    step, dashed: initial step); and the ``x``-``v`` phase space of each species
    as a histogram with a logarithmic colour scale. Colour limits and axes are
    fixed over the whole run. Every per-frame array is computed once, up front,
    so both the on-screen animation and the file are one render per frame.

    Args:
        out (Output): Result of :meth:`Simulation.run`.
        direction (str): Velocity components to show, one or more of ``"x"``,
            ``"y"``, ``"z"`` (for example ``"xz"``). The spatial axis is always x.
        omega (float or None): Frequency (rad/s) that makes the time axis
            dimensionless, labelled :math:`t\\,\\omega_{pe}`; ``None`` keeps seconds.
        save (str or None): File name of the MP4 to write with ffmpeg (H.264).
        fps (int): Frames per second of the saved file.
        stride (int): Keep every n-th stored step in the saved file.
        dpi (int): Resolution of the figure, on screen and in the file.
        interval (int): Delay between frames of the on-screen animation, ms.
        show (bool): Call :func:`matplotlib.pyplot.show`; without ``save`` the
            figure animates on screen and the animation is kept in ``fig.animation``.
        vbins (int): Number of velocity bins; reduced if the histograms of all
            stored steps would exceed about ``2e8`` elements.

    Returns:
        matplotlib.figure.Figure: The figure, drawn at the last frame written
        (or at the first stored step when nothing was saved).
    """
    dirs = direction.lower()
    if not dirs or any(c not in _AXIS for c in dirs):
        raise ValueError("direction must be one or more of 'x', 'y', 'z', e.g. 'xz'")
    t, grid, rho = np.asarray(out.t, float), np.asarray(out.grid, float), np.asarray(out.rho, float)
    S, G, L, dx = len(t), len(grid), float(out.length), float(out.dx)
    tt, tlabel, tfmt = ((t * float(omega), r"$t\,\omega_{pe}$", r"$t\,\omega_{pe}$ = %.3g") if omega
                        else (t, "t (s)", "t = %.3g s"))
    h = (tt[-1] - tt[0]) / max(S - 1, 1) or 1.0
    extent = (grid[0] - dx / 2, grid[-1] + dx / 2, tt[0] - h / 2, tt[-1] + h / 2)

    fields = []  # (title, (S, G) array, unit) for every non-zero field component
    for name, F, unit in (("E", out.E, "V/m"), ("B", out.B, "T")):
        F = np.asarray(F, float)
        for c in "xyz":
            if np.abs(F[:, :, _AXIS[c]]).max() > 1e-12:
                fields.append((rf"${name}_{c}$", F[:, :, _AXIS[c]], unit))
    fields.append((r"$\rho$", rho, r"C/m$^3$"))

    # Particles: species by out.species, absorbed (charge == 0) ones excluded.
    have = out.x is not None and out.v is not None
    species, alive = np.asarray(out.species), np.asarray(out.charge) != 0
    groups = [(n, (species == k) & alive) for k, n in enumerate(out.names)] if have else []
    groups = [(n, m) for n, m in groups if m.any()]
    dirs = dirs if groups else ""
    vbins = max(8, min(int(vbins), int(_MAX_ELEMENTS // max(1, S * G * len(groups) * len(dirs)))))
    frame = np.arange(S, dtype=np.int32)[:, None]
    ix = [_index(np.asarray(out.x)[:, m, 0], -L / 2, L / 2, G) for _, m in groups]
    fv, ps, vmax, span = {}, {}, {}, {}  # keyed by direction (and species index)
    for d in dirs:
        vs = [np.asarray(out.v)[:, m, _AXIS[d]] for _, m in groups]
        vmax[d] = [1.25 * float(np.percentile(np.abs(v), 99.5)) or 1.0 for v in vs]
        span[d] = max(vmax[d])
        for k, v in enumerate(vs):
            iv = _index(v, -vmax[d][k], vmax[d][k], vbins)
            ps[k, d] = _hist((frame * vbins + iv) * G + ix[k], (S, vbins, G)) + 1.0
            f = _hist(frame * vbins + _index(v, -span[d], span[d], vbins), (S, vbins))
            fv[k, d] = f / max(float(f[0].max()), 1.0)

    n = len(fields) + len(dirs) * (1 + len(groups))
    rows = -(-n // 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, min(10.0, 3.2 * rows)), dpi=dpi, squeeze=False)
    axes = list(axes.ravel())
    images, lines = [], []  # (artist, per-frame array) pairs updated by ``update``

    for title, F, unit in fields:
        ax = axes.pop(0)
        lim = float(np.percentile(np.abs(F), 99.5)) or 1.0
        im = ax.imshow(F, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", origin="lower",
                       extent=extent, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(unit)
        lines.append((ax.axhline(tt[0], color="k", lw=0.8, ls="--"), np.stack([tt, tt], 1)))
        line, = ax.plot(grid, 0.5 + 0 * grid, "k", lw=1.2, transform=ax.get_xaxis_transform())
        lines.append((line, 0.5 + 0.475 * np.clip(F / lim, -1, 1)))
        ax.set(title=title, xlabel="x (m)", ylabel=tlabel, xlim=extent[:2], ylim=extent[2:])

    for d in dirs:
        ax = axes.pop(0)
        vc = np.linspace(-span[d], span[d], 2 * vbins + 1)[1::2]
        for k, (name, _) in enumerate(groups):
            ax.plot(vc, fv[k, d][0], "--", color=f"C{k}", lw=1)
            lines.append((ax.plot(vc, fv[k, d][0], color=f"C{k}", lw=2, label=name)[0], fv[k, d]))
        ax.legend(frameon=False, fontsize=8)
        ax.set(title=rf"$f(v_{d})$ (dashed: initial)", xlabel=rf"$v_{d}$ (m/s)",
               ylabel=r"$f/\max f_0$", xlim=(-span[d], span[d]), ylim=(0, 1.1))

    for d in dirs:
        for k, (name, _) in enumerate(groups):
            ax = axes.pop(0)
            C = ps[k, d]
            im = ax.imshow(C[0], cmap="magma", norm=LogNorm(1.0, float(C.max())), aspect="auto",
                           origin="lower", interpolation="nearest",
                           extent=(-L / 2, L / 2, -vmax[d][k], vmax[d][k]))
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("counts + 1")
            images.append((im, C))
            ax.set(title=rf"{name}: $(x, v_{d})$", xlabel="x (m)", ylabel=rf"$v_{d}$ (m/s)")

    for ax in axes:
        ax.axis("off")
    text = fig.suptitle("")

    def update(i):
        for im, data in images:
            im.set_array(data[i])
        for line, y in lines:
            line.set_ydata(y[i])
        text.set_text(tfmt % tt[i])

    update(0)
    fig.tight_layout()

    if save is not None:
        _write_movie(fig, update, [a for a, _ in images] + [a for a, _ in lines] + [text],
                     range(0, S, max(1, int(stride))), save, fps)
    elif show:
        fig.animation = FuncAnimation(fig, update, frames=S, interval=interval, blit=False)
    if show:
        plt.show()
    return fig
