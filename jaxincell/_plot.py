"""Animated overview figure of a simulation output."""
import contextlib
import subprocess
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm

__all__ = ["plot", "figure", "style"]

PANEL = (9.0, 7.0)                      # one panel of a figure, in inches

# A heavy frame, ticks turned inward on all four sides, and type large enough to read
# when the figure is scaled to the width of a page or a slide.
_STYLE = {
    "font.size": 20, "axes.titlesize": 20, "axes.labelsize": 24, "legend.fontsize": 18,
    "xtick.labelsize": 24, "ytick.labelsize": 24, "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 3.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 3.0, "ytick.major.width": 3.0, "xtick.major.size": 7.0, "ytick.major.size": 7.0,
    "xtick.minor.width": 2.0, "ytick.minor.width": 2.0, "xtick.minor.size": 5.0, "ytick.minor.size": 5.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 3.0, "lines.markersize": 9.0, "legend.frameon": False,
    "savefig.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "figure.facecolor": "white", "mathtext.fontset": "dejavusans",
}


def style():
    """Apply the package's figure style to matplotlib."""
    plt.rcParams.update(_STYLE)


def figure(ncols=1, nrows=1, aspect=None, **kwargs):
    """``ncols`` by ``nrows`` panels of :data:`PANEL` inches each, in :func:`style`;
    ``aspect`` gives a flatter panel, as a fraction of the panel width."""
    style()
    height = PANEL[0] * aspect if aspect else PANEL[1]
    return plt.subplots(nrows, ncols, figsize=(ncols * PANEL[0], nrows * height), **kwargs)


_AXIS = {"x": 0, "y": 1, "z": 2}


def _index(a, lo, hi, n):
    """Bin index of ``a`` on ``n`` equal bins over ``[lo, hi]``, and whether it was inside.

    Out-of-range values are clipped to the edge bins *and reported*, because a histogram that
    quietly piles its tails on the end bins is a histogram that says the distribution has a
    spike where the range ran out."""
    raw = np.floor((a - lo) * (n / (hi - lo))).astype(np.int64)
    return np.clip(raw, 0, n - 1), np.count_nonzero((raw < 0) | (raw >= n))


def _send_frames(ffmpeg, fig, canvas, background, update, animated, frames):
    """Draw every frame onto the cached background and write it to ffmpeg. An
    ffmpeg that exits early breaks the pipe; its exit status then says why."""
    try:
        for i in frames:
            canvas.restore_region(background)
            update(i)
            for artist in animated:
                fig.draw_artist(artist)
            canvas.blit(fig.bbox)
            ffmpeg.stdin.write(canvas.buffer_rgba())
        ffmpeg.stdin.close()
    except BrokenPipeError:
        with contextlib.suppress(BrokenPipeError):
            ffmpeg.stdin.close()


def _write_movie(fig, update, animated, frames, path, fps):
    """Pipe raw frames to ffmpeg, redrawing only the artists that move.

    Caching the static background once and blitting the handful of animated
    artists onto it is what makes this quick: a full redraw of the figure costs
    about 60 ms a frame, restoring the background and blitting about 6 ms.
    ffmpeg's messages go to a temporary file rather than a pipe, which nothing
    reads while the frames are written and which could fill and block.
    """
    original, canvas = fig.canvas, FigureCanvasAgg(fig)   # blitting needs an Agg canvas
    for artist in animated:
        artist.set_animated(True)                         # keep them out of the cached background
    try:
        canvas.draw()
        background = canvas.copy_from_bbox(fig.bbox)
        width, height = canvas.get_width_height()
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                   "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                   # flat-colour plots compress far better tuned for animation; faststart puts the index
                   # first, so a browser plays the file before it has finished downloading
                   "-preset", "slow", "-tune", "animation", "-crf", "30", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(path)]
        with tempfile.TemporaryFile() as log:
            try:
                ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            except FileNotFoundError:
                warnings.warn(f"ffmpeg was not found on PATH, so {path} was not written; install it, for "
                              "example with `conda install -c conda-forge ffmpeg`", RuntimeWarning, stacklevel=3)
                return
            _send_frames(ffmpeg, fig, canvas, background, update, animated, frames)
            if ffmpeg.wait() != 0:
                log.seek(0)
                message = log.read().decode(errors="replace").strip()
                raise RuntimeError(f"ffmpeg exited with status {ffmpeg.returncode}, so {path} was not written: "
                                   f"{message}")
    finally:
        for artist in animated:
            artist.set_animated(False)
        fig.canvas = original


def _time_axis(t, omega, label):
    """Time values, axis label and title format, in seconds or in units of ``1/omega``."""
    if omega:
        return t * float(omega), rf"$t\,{label}$", rf"$t\,{label}$ = %.3g"
    return t, "t (s)", "t = %.3g s"


def _field_maps(out):
    """``(title, (S, cells) array, unit, coordinates)`` for every non-zero field component, then
    the charge density.

    ``E`` and ``J`` live on the faces and ``B`` and ``rho`` on the centres, which is half a cell
    apart; drawing them all on the centres, as this did, puts the sharpest part of a sheath
    profile in the wrong place."""
    grid, faces = np.asarray(out.grid, float), np.asarray(out.faces, float)
    fields = []
    for name, F, unit, where in (("E", out.E, "V/m", faces), ("B", out.B, "T", grid)):
        F = np.asarray(F, float)
        fields += [(rf"${name}_{c}$", F[:, :, _AXIS[c]], unit, where)
                   for c in "xyz" if np.abs(F[:, :, _AXIS[c]]).max() > 1e-12]
    return fields + [(r"$\rho$", np.asarray(out.rho, float), r"C/m$^3$", grid)]


class _Particles:
    """The particle history as NumPy, sliced one frame at a time.

    ``np.asarray`` of a JAX array on the host shares its memory, so holding this costs nothing;
    a JAX fancy-index does not, and ``out.v[:, mask, axis]`` allocated 129 MB for 64 MB of
    data every time it was asked for. One frame of one species is a few hundred kilobytes and
    is built when it is drawn, so what a movie needs in memory is the run, not the run and a
    second copy of every frame's histogram."""

    def __init__(self, out):
        self.x = np.asarray(out.x)
        self.v = np.asarray(out.v)
        self.w = np.asarray(out.weight)
        self.cells, self.length = len(out.grid), float(out.length)
        self.frames = len(out.t)

    def live(self, mask, d, frame):
        """Position, velocity component and weight of the particles of one species that are
        alive at one stored step. Membership is a property of a slot and does not change;
        being alive is a property of a slot **at a step**, and that is the weight."""
        w = self.w[frame][mask]
        alive = w > 0
        return self.x[frame][mask][alive, 0], self.v[frame][mask][alive, _AXIS[d]], w[alive]

    def phase_space(self, mask, d, frame, vmax, vbins):
        """The weighted ``(v, x)`` histogram at one stored step, and how much fell outside the
        velocity range."""
        x, v, w = self.live(mask, d, frame)
        ix, _ = _index(x, -self.length / 2, self.length / 2, self.cells)
        iv, outside = _index(v, -vmax, vmax, vbins)
        counts = np.bincount(iv * self.cells + ix, weights=w, minlength=vbins * self.cells)
        return counts.reshape(vbins, self.cells), outside

    def distribution(self, mask, d, frame, span, vbins):
        """The weighted ``f(v)`` at one stored step, on the common range."""
        _, v, w = self.live(mask, d, frame)
        iv, _ = _index(v, -span, span, vbins)
        return np.bincount(iv, weights=w, minlength=vbins)


def _species_groups(out):
    """``(name, mask)`` of every species, whatever its particles are doing.

    Choosing the particles by the weight at the **last** step, as this did, deletes from every
    frame the ones a wall collected before the end -- so a sheath movie showed the particles
    that survived it -- and puts the ones a source has not yet emitted into the first frame,
    where they sit in a heap at their parking place."""
    if out.x is None or out.v is None or out.weight is None:
        return []
    species = np.asarray(out.species)
    return [(name, species == k) for k, name in enumerate(out.names)]


def _ranges(particles, groups, dirs, vbins):
    """What has to be the same in every frame: the velocity range of each species and direction,
    and the largest weighted count any phase-space bin reaches.

    The colour ceiling is exact, from every frame, because a histogram is a few hundred
    kilobytes and is thrown away again. The velocity range is a **presentation** choice -- where
    to cut the axis -- and is taken from up to 24 frames spread over the run rather than from
    the whole of it, because the alternative is a copy of the velocity history."""
    vmax, span, ceiling = {}, {}, {}
    sample = np.unique(np.linspace(0, particles.frames - 1, min(particles.frames, 24)).astype(int))
    for d in dirs:
        vmax[d] = []
        for _, mask in groups:
            speeds = np.concatenate([np.abs(particles.live(mask, d, i)[1]) for i in sample])
            vmax[d].append(1.25 * float(np.percentile(speeds, 99.5)) if speeds.size else 1.0)
            vmax[d][-1] = vmax[d][-1] or 1.0
        span[d] = max(vmax[d])
        for k, (_, mask) in enumerate(groups):
            top = max(float(particles.phase_space(mask, d, i, vmax[d][k], vbins)[0].max())
                      for i in range(particles.frames))
            ceiling[k, d] = top or 1.0
    return vmax, span, ceiling


def _draw_fields(fig, axes, fields, times, tlabel, lines):
    """Space-time map of each field, with the current-time marker and profile added to ``lines``.

    ``pcolormesh`` rather than ``imshow``: the stored times need not be evenly spaced -- a run
    continued from a state, or two runs joined -- and an image drawn from the first and last
    time alone puts every row somewhere it was not."""
    edges = np.concatenate([[times[0] - 0.5 * (times[1] - times[0])] if len(times) > 1 else [times[0] - 0.5],
                            0.5 * (times[1:] + times[:-1]),
                            [times[-1] + 0.5 * (times[-1] - times[-2])] if len(times) > 1 else [times[0] + 0.5]])
    for title, F, unit, where in fields:
        ax = axes.pop(0)
        lim = float(np.percentile(np.abs(F), 99.5)) or 1.0
        step = where[1] - where[0] if len(where) > 1 else 1.0
        columns = np.concatenate([where - step / 2, [where[-1] + step / 2]])
        mesh = ax.pcolormesh(columns, edges, F, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="flat")
        fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02).set_label(unit)
        lines.append((ax.axhline(times[0], color="k", lw=0.8, ls="--"), np.stack([times, times], 1)))
        line, = ax.plot(where, 0.5 + 0 * where, "k", lw=1.2, transform=ax.get_xaxis_transform())
        lines.append((line, 0.5 + 0.475 * np.clip(F / lim, -1, 1)))
        ax.set(title=title, xlabel="x (m)", ylabel=tlabel, xlim=(columns[0], columns[-1]),
               ylim=(edges[0], edges[-1]))


def _draw_distributions(axes, particles, groups, dirs, span, vbins, curves):
    """One panel per direction with the weighted ``f(v)`` of every species, at the first stored
    step (dashed) and at the current one."""
    for d in dirs:
        ax = axes.pop(0)
        edges = np.linspace(-span[d], span[d], vbins + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        first = [particles.distribution(m, d, 0, span[d], vbins) for _, m in groups]
        scale = max((f.max() for f in first), default=1.0) or 1.0
        for k, ((name, mask), f0) in enumerate(zip(groups, first)):
            ax.plot(centres, f0 / scale, "--", color=f"C{k}", lw=1)
            line, = ax.plot(centres, f0 / scale, color=f"C{k}", lw=2, label=name)
            curves.append((line, lambda i, m=mask, d=d: particles.distribution(m, d, i, span[d], vbins) / scale))
        ax.legend(frameon=False, fontsize=8)
        ax.set(title=rf"$f(v_{d})$ (dashed: first stored step)", xlabel=rf"$v_{d}$ (m/s)",
               ylabel=r"$f/\max f_0$", xlim=(-span[d], span[d]), ylim=(0, 1.1))


def _draw_phase_spaces(fig, axes, particles, groups, dirs, vmax, ceiling, vbins, length, images):
    """One logarithmic weighted ``(x, v)`` histogram per direction and species."""
    for d in dirs:
        for k, (name, mask) in enumerate(groups):
            ax = axes.pop(0)
            counts, outside = particles.phase_space(mask, d, 0, vmax[d][k], vbins)
            # zero is not a small number on a logarithmic scale: an empty bin is masked and drawn
            # as the background, where adding one to every count drew it as the bottom colour and
            # shifted every other bin by a particle
            image = ax.imshow(np.ma.masked_less_equal(counts, 0.0), cmap="magma",
                              norm=LogNorm(*_limits(ceiling[k, d])), aspect="auto", origin="lower",
                              interpolation="nearest",
                              extent=(-length / 2, length / 2, -vmax[d][k], vmax[d][k]))
            fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02).set_label("weight per bin")
            images.append((image, lambda i, m=mask, d=d, k=k: np.ma.masked_less_equal(
                particles.phase_space(m, d, i, vmax[d][k], vbins)[0], 0.0)))
            spilled = "" if not outside else f", {outside} outside the range at the first step"
            ax.set(title=rf"{name}: $(x, v_{d})${spilled}", xlabel="x (m)", ylabel=rf"$v_{d}$ (m/s)")


def _draw_diagnostics(ax, out, times, tlabel):
    """The histories a run is judged by, on one panel: what should be conserved and what only
    balances. Leaving them out of the overview leaves the one thing a glance should catch --
    a run whose energy is running away -- to a separate call nobody makes."""
    from ._diagnostics import charge_balance, energies, gauss_residual

    report = energies(out)
    curves = [("Gauss residual", np.asarray(gauss_residual(out), float)),
              ("charge balance", np.asarray(charge_balance(out), float))]
    if "energy_error" in report:
        curves.insert(0, ("energy error", np.asarray(report["energy_error"], float)))
        curves.insert(1, ("momentum error", np.asarray(report["momentum_error"], float)))
    floor = 1e-17           # below double-precision round-off; an exact zero would drag the axis to 1e-308
    for name, values in curves:
        ax.semilogy(times, np.maximum(np.abs(values), floor), lw=1.2, label=name)
    ax.set_ylim(bottom=floor / 2)
    ax.legend(frameon=False, fontsize=9)
    ax.set(xlabel=tlabel, ylabel="relative", title="conservation and residuals")


def plot(out, direction="x", omega=None, omega_label=r"\omega_{pe}", save=None, fps=25, stride=1,
         dpi=80, interval=30, show=True, vbins=96, diagnostics=True):
    """Animate the fields, velocity distributions and phase space of a run.

    One figure, animated over the stored steps. It holds a space-time map of every non-zero
    component of ``E`` and ``B`` and of the charge density, each on the coordinates it lives on,
    with a line marking the current time and the instantaneous profile drawn over it; the
    velocity distribution ``f(v)`` of each species, weighted, at the current step and at the
    first; the ``x``-``v`` phase space of each species as a weighted histogram on a logarithmic
    colour scale; and the conservation histories. Colour limits and axes are fixed over the whole
    run, from one pass that keeps numbers rather than arrays, and each frame is built when it is
    drawn.

    Args:
        out (Output): Result of :meth:`Simulation.run`.
        direction (str): Velocity components to show, one or more of ``"x"``, ``"y"``,
            ``"z"`` (for example ``"xz"``). The spatial axis is always x.
        omega (float or None): Frequency (rad/s) that makes the time axis dimensionless;
            ``None`` keeps seconds.
        omega_label (str): What to call it on the axis, as LaTeX without the dollars. The
            default is the plasma frequency, which is what ``omega`` usually is and was what
            the axis said whatever was passed.
        save (str or None): File name of the MP4 to write with ffmpeg (H.264). Without ffmpeg
            on the PATH the call warns and writes nothing. Saving and showing are independent:
            ask for both and both happen.
        fps (int): Frames per second of the saved file.
        stride (int): Keep every n-th stored step in the saved file.
        dpi (int): Resolution of the figure, on screen and in the file.
        interval (int): Delay between frames of the on-screen animation, ms.
        show (bool): Call :func:`matplotlib.pyplot.show`; the animation is kept in
            ``fig.animation``.
        vbins (int): Number of velocity bins.
        diagnostics (bool): Add the conservation and residual histories as a panel.

    Returns:
        matplotlib.figure.Figure: The figure, drawn at the last frame written (or at the first
        stored step when nothing was saved).

    Raises:
        ValueError: If ``direction`` names no velocity component.
        RuntimeError: If ffmpeg fails, with its message.
    """
    dirs = direction.lower()
    if not dirs or any(c not in _AXIS for c in dirs):
        raise ValueError("direction must be one or more of 'x', 'y', 'z', e.g. 'xz'")
    t = np.asarray(out.t, float)
    S, L = len(t), float(out.length)
    times, tlabel, tfmt = _time_axis(t, omega, omega_label)

    fields, groups = _field_maps(out), _species_groups(out)
    dirs = dirs if groups else ""
    vbins = max(8, int(vbins))
    particles = _Particles(out) if groups else None
    vmax, span, ceiling = _ranges(particles, groups, dirs, vbins) if groups else ({}, {}, {})

    n = len(fields) + len(dirs) * (1 + len(groups)) + bool(diagnostics)
    rows = -(-n // 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, min(10.0, 3.2 * rows)), dpi=dpi, squeeze=False)
    axes = list(axes.ravel())
    images, curves, lines = [], [], []      # artists and how to fill them for frame i
    _draw_fields(fig, axes, fields, times, tlabel, lines)

    _draw_distributions(axes, particles, groups, dirs, span, vbins, curves)
    _draw_phase_spaces(fig, axes, particles, groups, dirs, vmax, ceiling, vbins, L, images)

    if diagnostics:
        _draw_diagnostics(axes.pop(0), out, times, tlabel)
    for ax in axes:
        ax.axis("off")
    text = fig.suptitle("")

    def update(i):
        for image, frame in images:
            image.set_array(frame(i))
        for line, frame in curves:
            line.set_ydata(frame(i))
        for line, y in lines:
            line.set_ydata(y[i])
        text.set_text(tfmt % times[i])

    update(0)
    fig.tight_layout()

    animated = [a for a, _ in images] + [a for a, _ in curves] + [a for a, _ in lines] + [text]
    # saving and showing are independent: ask for both and both happen. They were exclusive, so
    # a caller who saved a movie and expected to see it got a still figure at the last frame
    if save is not None:
        _write_movie(fig, update, animated, range(0, S, max(1, int(stride))), save, fps)
        update(0)
    if show:
        fig.animation = FuncAnimation(fig, update, frames=S, interval=interval, blit=False)
        plt.show()
    return fig


def _limits(top):
    """Colour limits for a logarithmic scale of weighted counts, four decades below the top."""
    return max(top * 1e-4, np.finfo(float).tiny), top
