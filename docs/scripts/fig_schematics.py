"""Schematic figures for the numerical-methods pages: the staggered grid, the
leapfrog time staggering, the particle shape functions and the response of the
digital filter. No simulation is run."""
import numpy as np
import matplotlib.pyplot as plt

from common import (C_ELECTRONS, C_IONS, COLORS, WIDE, panel_label, savefig)

# --- 1. Staggered grid -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 2.4))
ax.set_xlim(-0.8, 5.8)
ax.set_ylim(-1.0, 1.6)
ax.axis("off")
for i in range(6):
    ax.axvline(i - 0.5, color="#BBBBBB", lw=0.8, zorder=0)
ax.axhline(0, color="#444444", lw=1.0)
for i in range(6):
    ax.plot(i, 0, "o", ms=7, color=C_ELECTRONS, zorder=3)
    ax.text(i, -0.42, rf"$x_{{{i}}}$" if i < 5 else r"$x_{N_x-1}$", ha="center", va="top", fontsize=9)
    if i < 5:
        ax.plot(i + 0.5, 0, "s", ms=6, color=C_IONS, zorder=3)
ax.text(2, 0.95, r"cell centres $x_i$: $\rho$, $\mathbf{B}$", color=C_ELECTRONS,
        ha="center", fontsize=9.5)
ax.text(2, 0.55, r"cell faces $x_{i+1/2}$: $\mathbf{E}$, $\mathbf{J}$", color=C_IONS,
        ha="center", fontsize=9.5)
ax.annotate("", xy=(1.0, -0.72), xytext=(0.0, -0.72), arrowprops=dict(arrowstyle="<->", lw=0.9))
ax.text(0.5, -0.8, r"$\Delta x = L / N_x$", ha="center", va="top", fontsize=9)
ax.text(-0.42, 1.3, r"$x = -L/2$", ha="left", fontsize=9)
ax.text(5.42, 1.3, r"$x = +L/2$", ha="right", fontsize=9)
ax.axvline(-0.5, color="#444444", lw=1.4)
ax.axvline(5.5, color="#444444", lw=1.4)
savefig(fig, "staggered_grid")

# --- 2. Time staggering of the explicit scheme -----------------------------
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.set_xlim(-0.9, 2.6)
ax.set_ylim(-0.3, 3.9)
ax.axis("off")
rows = {"x": 3.0, "v": 2.0, "E, B": 1.0, "J": 0.0}
for label, y in rows.items():
    ax.text(-0.85, y, label, ha="left", va="center", fontsize=10, fontweight="bold")
    ax.axhline(y, color="#DDDDDD", lw=0.8, zorder=0)
for tt, label in ((-0.5, r"$n-1/2$"), (0, r"$n$"), (0.5, r"$n+1/2$"), (1.0, r"$n+1$"), (1.5, r"$n+3/2$")):
    ax.axvline(tt, color="#EEEEEE", lw=0.8, zorder=0)
    ax.text(tt, 3.6, label, ha="center", fontsize=9.5)
for tt in (-0.5, 0.5, 1.5):
    ax.plot(tt, rows["x"], "o", ms=7, color=C_ELECTRONS)
for tt in (0, 1.0):
    ax.plot(tt, rows["v"], "o", ms=7, color=C_ELECTRONS)
    ax.plot(tt, rows["E, B"], "s", ms=7, color=C_IONS)
    ax.plot(tt, rows["J"], "D", ms=6, color=COLORS["green"])
ax.plot(0.5, rows["E, B"], "s", ms=7, mfc="white", mec=C_IONS)
ax.annotate("", xy=(0.47, rows["x"]), xytext=(-0.47, rows["x"]), arrowprops=dict(arrowstyle="->", lw=1.1, color=C_ELECTRONS))
ax.annotate("", xy=(1.47, rows["x"]), xytext=(0.53, rows["x"]), arrowprops=dict(arrowstyle="->", lw=1.1, color=C_ELECTRONS))
ax.annotate("", xy=(0.97, rows["v"]), xytext=(0.03, rows["v"]), arrowprops=dict(arrowstyle="->", lw=1.1, color=C_ELECTRONS))
ax.text(0.5, rows["v"] + 0.18, r"Boris push with $\mathbf{E}^{n+1/2}, \mathbf{B}^{n+1/2}$ at $x^{n+1/2}$",
        ha="center", fontsize=8.5)
ax.annotate("", xy=(0.47, rows["E, B"]), xytext=(0.03, rows["E, B"]), arrowprops=dict(arrowstyle="->", lw=1.1, color=C_IONS))
ax.annotate("", xy=(0.97, rows["E, B"]), xytext=(0.53, rows["E, B"]), arrowprops=dict(arrowstyle="->", lw=1.1, color=C_IONS))
ax.text(0.25, rows["E, B"] - 0.38, r"$\Delta t/2$: E, then B", ha="center", fontsize=8)
ax.text(0.75, rows["E, B"] - 0.38, r"$\Delta t/2$: B, then E", ha="center", fontsize=8)
ax.text(0.0, rows["J"] - 0.32, r"$\mathbf{J}^{n}$ from $x^{n-1/2}\to x^{n+1/2}$", ha="center", fontsize=8)
ax.text(1.0, rows["J"] - 0.32, r"$\mathbf{J}^{n+1}$ from $x^{n+1/2}\to x^{n+3/2}$", ha="center", fontsize=8)
feed = dict(arrowstyle="->", lw=0.8, color=COLORS["green"], ls="--")
ax.annotate("", xy=(0.03, rows["E, B"] + 0.12), xytext=(0.0, rows["J"] + 0.12), arrowprops=feed)
ax.annotate("", xy=(1.0, rows["E, B"] - 0.12), xytext=(1.0, rows["J"] + 0.12), arrowprops=feed)
savefig(fig, "time_staggering")

# --- 3. Particle shape functions ------------------------------------------
xi = np.linspace(-2, 2, 801)
S0 = np.where(np.abs(xi) < 0.5, 1.0, 0.0)
S1 = np.clip(1 - np.abs(xi), 0, None)
S2 = np.where(np.abs(xi) <= 0.5, 0.75 - xi**2,
              np.where(np.abs(xi) <= 1.5, 0.5 * (1.5 - np.abs(xi))**2, 0.0))
fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.3})
ax = axes[0]
ax.plot(xi, S0, color=COLORS["grey"], lw=1.2, label=r"$S_0$: nearest grid point")
ax.plot(xi, S1, color=C_IONS, lw=1.4, label=r"$S_1$: linear (cloud-in-cell)")
ax.plot(xi, S2, color=C_ELECTRONS, lw=2.0, label=r"$S_2$: quadratic spline (used)")
ax.set_xlabel(r"$(x - x_i)/\Delta x$")
ax.set_ylabel(r"$S(x - x_i)$")
ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
ax.legend(loc="upper right", fontsize=8)
panel_label(ax, "(a)")
ax = axes[1]
xp = 0.3
for i in range(-2, 3):
    ax.axvline(i, color="#DDDDDD", lw=0.8, zorder=0)
def _s2(d):
    """The quadratic-spline weight at a distance of ``d`` cells."""
    return 0.75 - d**2 if abs(d) <= 0.5 else (0.5 * (1.5 - abs(d))**2 if abs(d) <= 1.5 else 0.0)


w = np.array([_s2(xp - i) for i in range(-2, 3)])
ax.bar(range(-2, 3), w, width=0.35, color=C_ELECTRONS, alpha=0.85)
ax.plot([xp], [0], "v", ms=9, color=C_IONS, clip_on=False, zorder=5)
ax.text(xp, -0.09, "particle", ha="center", va="top", fontsize=8.5, color=C_IONS)
for i, wi in zip(range(-2, 3), w):
    if wi > 0:
        ax.text(i, wi + 0.02, f"{wi:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(-2, 3))
ax.set_xticklabels([r"$x_{i-2}$", r"$x_{i-1}$", r"$x_i$", r"$x_{i+1}$", r"$x_{i+2}$"])
ax.set_ylim(0, 0.9)
ax.set_ylabel("fraction of the charge assigned")
ax.set_title(rf"$S_2$ weights for a particle at $x_i + {xp}\,\Delta x$", fontsize=9)
panel_label(ax, "(b)")
savefig(fig, "shape_functions")

# --- 4. Digital filter response ------------------------------------------
kdx = np.linspace(0, np.pi, 500)


def response(passes, alpha=0.5, strides=(1, 2, 4)):
    H = np.ones_like(kdx)
    if passes <= 0:
        return H
    comp_alpha = passes - alpha * (passes - 1)
    for s in strides:
        one = alpha + (1 - alpha) * np.cos(kdx * s)
        H *= one ** (passes - 1) * (comp_alpha + (1 - comp_alpha) * np.cos(kdx * s))
    return H


fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.3})
ax = axes[0]
for p, color in zip((1, 2, 3, 5, 8), (COLORS["sky"], COLORS["green"], C_IONS, C_ELECTRONS, COLORS["purple"])):
    ax.plot(kdx / np.pi, response(p), color=color, label=f"{p} pass{'es' if p > 1 else ''}")
ax.axhline(0, color="#999999", lw=0.6)
ax.set_xlabel(r"$k \Delta x / \pi$  (1 = Nyquist)")
ax.set_ylabel(r"response $H(k)$, strides (1, 2, 4), $\alpha = 0.5$")
ax.legend(loc="upper right", fontsize=8)
panel_label(ax, "(a)")
ax = axes[1]
for s, color in zip(((1,), (1, 2), (1, 2, 4)), (COLORS["sky"], C_IONS, C_ELECTRONS)):
    ax.plot(kdx / np.pi, response(5, strides=s), color=color, label=f"strides {s}")
ax.axhline(0, color="#999999", lw=0.6)
ax.set_xlabel(r"$k \Delta x / \pi$")
ax.set_ylabel(r"response $H(k)$, 5 passes, $\alpha = 0.5$")
ax.legend(loc="upper right", fontsize=8)
panel_label(ax, "(b)")
savefig(fig, "filter_response")
