"""The movies in the README and the documentation, one per input file in ``inputs/``.

Each shows the physics of its case in two panels, in normalised units, revealed as it happens:
the phase space or the field on the left, and on the right the quantity the case is about,
drawn up to the current time. They are written as H.264 with the index first, so they play in a
browser as soon as they are clicked, at a few hundred kilobytes. Run from the repository root::

    python docs/scripts/movies.py [name ...]
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm

from jaxincell import elementary_charge as e, epsilon_0, load_toml, mass_electron, potential, style

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "_static" / "movies"
FPS, HOLD = 24, 36                                   # frames a second; the last frame is held 1.5 s
ENCODE = ["-preset", "veryslow", "-tune", "animation", "-crf", "32", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def run(name, **override):
    simulation, settings = load_toml(ROOT / "inputs" / f"{name}.toml")
    settings = {key: value for key, value in settings.items() if key not in ("plot", "moments")}
    return simulation, simulation.run(**{**settings, "store_particles": True, **override})


def canvas(title):
    style()
    plt.rcParams.update({"axes.labelsize": 18, "axes.titlesize": 18, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "legend.fontsize": 14, "lines.linewidth": 2.5})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.84, wspace=0.28)
    return fig, axes, fig.suptitle(title, fontsize=22)


def block(out, name):
    start = sum(out.counts[:out.names.index(name)])
    return slice(start, start + out.counts[out.names.index(name)])


def plasma_frequency(simulation):
    n = sum(s.density for s in simulation.species if s.charge < 0)
    return np.sqrt(n * e ** 2 / (epsilon_0 * mass_electron))


def phase_space(ax, x, v, w, extent, bins=(110, 90), depth=300):
    """A weighted log histogram of ``(x, v)``; the colour scale is set from the frame given."""
    counts = np.histogram2d(x, v, bins=bins, range=extent, weights=w)[0].T
    top = np.percentile(counts[counts > 0], 99.5)
    image = ax.imshow(np.ma.masked_less_equal(counts, 0), origin="lower", aspect="auto", cmap="magma",
                      norm=LogNorm(top / depth, top), extent=(*extent[0], *extent[1]), interpolation="bilinear")
    ax.set_facecolor("black")
    return image, lambda x, v, w: image.set_data(np.ma.masked_less_equal(
        np.histogram2d(x, v, bins=bins, range=extent, weights=w)[0].T, 0))


def trace(ax, t, y, color, label):
    ax.plot(t, y, color="0.85", lw=2)                 # the whole history, faint, so the scale is fixed
    line, = ax.plot([], [], color=color, label=label)
    dot, = ax.plot([], [], "o", color=color, ms=10)

    def update(i):
        line.set_data(t[:i + 1], y[:i + 1])
        dot.set_data([t[i]], [y[i]])
    return update


def two_stream():
    simulation, out = run("two_stream")
    w_pe, (s, sl) = plasma_frequency(simulation), (simulation.species[0], block(out, "electrons"))
    t, v0, L = np.asarray(out.t) * w_pe, s.drift[0], float(out.length)
    energy = 0.5 * epsilon_0 * np.sum(np.asarray(out.E)[:, :, 0] ** 2, axis=1) * float(out.dx)
    fig, (left, right), title = canvas("")
    x, v, w = (np.asarray(a)[:, sl] for a in (out.x[..., 0], out.v[..., 0], out.weight))
    _, show = phase_space(left, x[0] / L, v[0] / v0, w[0], [(-0.5, 0.5), (-2.2, 2.2)])
    left.set(xlabel=r"$x/L$", ylabel=r"$v_x/v_0$", title="electron phase space")
    right.set(yscale="log", xlabel=r"$t\,\omega_{pe}$", ylabel="electric field energy (J/m$^2$)",
              title="the seeded mode grows, then traps", xlim=(0, t[-1]),
              ylim=(energy[energy > 0].min() / 2, 5 * energy.max()))
    grow = trace(right, t, energy, "#0072B2", "")

    def update(i):
        show(x[i] / L, v[i] / v0, w[i])
        grow(i)
        title.set_text(rf"Two-stream instability    $t\,\omega_{{pe}} = {t[i]:.1f}$")
    return fig, update, range(0, t.size, 8)


def bump_on_tail():
    simulation, out = run("bump_on_tail")
    w_pe, vth = plasma_frequency(simulation), simulation.species[0].vth[0]
    t, L = np.asarray(out.t) * w_pe, float(out.length)
    electrons = np.r_[np.arange(out.counts[0] + out.counts[1])]
    x, v, w = (np.asarray(a)[:, electrons] for a in (out.x[..., 0], out.v[..., 0], out.weight))
    edges = np.linspace(-4, 9, 131)
    f = np.array([np.histogram(v[i] / vth, edges, weights=w[i])[0] for i in range(t.size)])
    f /= f[0].max()
    centres = 0.5 * (edges[1:] + edges[:-1])
    fig, (left, right), title = canvas("")
    _, show = phase_space(left, x[0] / L, v[0] / vth, w[0], [(-0.5, 0.5), (-4, 9)])
    left.set(xlabel=r"$x/L$", ylabel=r"$v_x/v_{th}$", title="electron phase space")
    right.semilogy(centres, f[0], "--", color="#E69F00", label="initial")
    now, = right.semilogy(centres, f[0], color="#0072B2", label="now")
    right.set(xlabel=r"$v_x/v_{th}$", ylabel=r"$f(v_x)$", title="the bump flattens into a plateau",
              xlim=(-4, 9), ylim=(1e-4, 2))
    right.legend(loc="upper right")

    def update(i):
        show(x[i] / L, v[i] / vth, w[i])
        now.set_ydata(np.maximum(f[i], 1e-6))
        title.set_text(rf"Bump-on-tail instability    $t\,\omega_{{pe}} = {t[i]:.0f}$")
    return fig, update, range(t.size)


def weibel():
    simulation, out = run("weibel", store_particles=False, steps=12000, store_every=60)
    w_pe = plasma_frequency(simulation)
    t = np.asarray(out.t) * w_pe
    B = np.asarray(out.B)[:, :, 1]
    energy = 0.5 / (4e-7 * np.pi) * np.sum(np.asarray(out.B) ** 2, axis=(1, 2)) * float(out.dx)
    fig, (left, right), title = canvas("")
    scale = np.abs(B).max()
    shown = np.full_like(B, np.nan)
    image = left.imshow(shown, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale,
                        extent=(-0.5, 0.5, t[0], t[-1]), interpolation="nearest")
    left.set(xlabel=r"$x/L$", ylabel=r"$t\,\omega_{pe}$", title=r"$B_y$: current filaments")
    right.set(yscale="log", xlabel=r"$t\,\omega_{pe}$", ylabel="magnetic energy (J/m$^2$)",
              title="grown from a temperature anisotropy", xlim=(0, t[-1]),
              ylim=(energy[energy > 0].min() / 2, 5 * energy.max()))
    grow = trace(right, t, energy, "#D55E00", "")

    def update(i):
        shown[: i + 1] = B[: i + 1]
        image.set_data(shown)
        grow(i)
        title.set_text(rf"Weibel instability    $t\,\omega_{{pe}} = {t[i]:.0f}$")
    return fig, update, range(t.size)


def sheath():
    simulation, out = run("sheath_unmagnetized", steps=1500, store_every=10)
    w_pe = plasma_frequency(simulation)
    electrons = simulation.species[0]
    T_e = mass_electron * electrons.vth[0] ** 2 / 2 / e              # eV, with v_th = sqrt(2T/m)
    debye = electrons.vth[0] / np.sqrt(2) / w_pe
    v_e = np.sqrt(T_e * e / mass_electron)
    t, L = np.asarray(out.t) * w_pe, float(out.length)
    sl = block(out, "electrons")
    x, v, w = (np.asarray(a)[:, sl] for a in (out.x[..., 0], out.v[..., 0], out.weight))
    phi = np.asarray(potential(out)) / T_e
    grid = (np.asarray(out.grid) + L / 2) / debye
    fig, (left, right), title = canvas("")
    extent = [(0, L / debye), (-4.0, 4.0)]
    _, show = phase_space(left, (x[-1] + L / 2) / debye, v[-1] / v_e, w[-1], extent, bins=(40, 48), depth=30)
    left.axhline(0.0, color="w", ls=":", lw=1)
    cut, = left.plot(grid, -np.sqrt(2 * np.clip(phi[0] - phi[0, -1], 0, None)), color="#56B4E9", ls="--", lw=2,
                     label="fastest electron the wall returns")
    left.legend(loc="lower left", fontsize=13, facecolor="black", edgecolor="w", labelcolor="w")
    left.set(xlabel=r"$x/\lambda_D$ (wall on the right)", ylabel=r"$v_{e,x}/v_{te}$",
             title="the wall keeps only the slow electrons")
    late = phi[t.size // 2:].mean(axis=0)
    right.plot(grid, late, color="0.8", lw=5, label="late-time mean")
    now, = right.plot(grid, phi[0], color="#0072B2", label="now")
    right.legend(loc="lower left")
    right.set(xlabel=r"$x/\lambda_D$", ylabel=r"$e\phi/T_e$", title="the potential drops at the wall",
              xlim=(0, L / debye), ylim=(1.4 * late.min(), 0.3))

    def update(i):
        show((x[i] + L / 2) / debye, v[i] / v_e, w[i])
        mean = phi[max(0, i - 4): i + 1].mean(axis=0)               # a short running mean against noise
        now.set_ydata(mean)
        cut.set_ydata(-np.sqrt(2 * np.clip(mean - mean[-1], 0, None)))
        title.set_text(rf"A plasma against a floating wall    $t\,\omega_{{pe}} = {t[i]:.0f}$")
    return fig, update, range(t.size)


MOVIES = {"two_stream": two_stream, "bump_on_tail": bump_on_tail, "weibel": weibel, "sheath_unmagnetized": sheath}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or MOVIES:
        fig, update, frames = MOVIES[name]()
        frames = list(frames)
        path = OUT / f"{name}.mp4"
        writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=ENCODE)
        with writer.saving(fig, str(path), dpi=100):
            for i in frames + [frames[-1]] * HOLD:
                update(i)
                writer.grab_frame()
        plt.close(fig)
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} kB, {len(frames)} frames)")
