"""Convergence of the explicit scheme in time step, cell size and particle count, each order
measured against the one the discretisation predicts.

(a) Time step: the frequency of a cold plasma oscillation (mode 1 of 64 cells) at four steps,
    each halving the last. Leapfrog predicts second order, (omega_pe dt)^2/24.
(b) Cell size: the same oscillation at a fixed wavelength on 32 to 256 cells, at a fixed small
    step. The centred gradients and the quadratic shape predict second order in k dx.
(c) Particle count: the thermal field energy of a randomly loaded Maxwellian plasma, which
    shot noise sets at 1/N.

The frequency is fitted, not read from peaks, so it resolves errors far below one step. The
order of (a) and (b) is the Richardson estimate log2((w1 - w2)/(w2 - w3)) of consecutive
halvings, which needs no exact reference; the errors drawn are against the extrapolated limit.
The energy error's order in the step is measured on an isolated oscillator in the collision
tests (numerics/collisions.md); in a plasma it stops falling with the step at a level the grid
sets, so it does not isolate the time integrator and is not repeated here.
"""
import numpy as np
from common import COLORS, figure, panel_label, record, savefig
from matplotlib.ticker import NullFormatter
from scipy.optimize import curve_fit

from jaxincell import Domain, Simulation, Solver, Species, diagnostics, epsilon_0, mass_electron
from jaxincell import elementary_charge as e_charge, speed_of_light as c

DENSITY, LENGTH = 1e18, 0.01
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
PERIODS = 12


def oscillation(cells, omega_dt, per_cell=32):
    """Fitted frequency, over omega_pe, of a cold oscillation in mode 1 on ``cells`` cells."""
    electrons = Species.electrons(n=cells * per_cell, density=DENSITY, sampling="lattice",
                                  perturbation_mode=1, perturbation_amplitude=1e-4 * LENGTH)
    ions = Species.ions(n=cells * per_cell, density=DENSITY, mass_ratio=1e8, vth=(0, 0, 0), sampling="lattice")
    domain = Domain(length=LENGTH, cells=cells, time_step=omega_dt / OMEGA_PE)
    steps = int(round(2 * np.pi * PERIODS / omega_dt))
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(steps, store_particles=False)
    t = np.asarray(out.t) * OMEGA_PE
    signal = np.fft.rfft(np.asarray(out.E[:, :, 0]), axis=1)[:, 1].real
    signal = signal / np.max(np.abs(signal))
    (_, w, _), _ = curve_fit(lambda t, a, w, p: a * np.cos(w * t + p), t, signal, p0=(1, 1, 0))
    return w


def richardson(values):
    """Orders of consecutive halvings, and the extrapolated limit from the last three."""
    v = np.asarray(values)
    orders = np.log2((v[:-2] - v[1:-1]) / (v[1:-1] - v[2:]))
    return orders, v[-1] + (v[-1] - v[-2]) / (2 ** orders[-1] - 1)


def noise_energy(n):
    """Mean field energy of a randomly loaded thermal plasma over t omega_pe 10-40, J/m^2."""
    electrons = Species.electrons(n=n, density=DENSITY, vth=(0.01 * c, 0, 0), sampling="random")
    ions = Species.ions(n=n, density=DENSITY, electrons=electrons, sampling="random")
    domain = Domain(length=LENGTH, cells=64, time_step=0.1 / OMEGA_PE)
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(400, seed=1, store_particles=False)
    return float(np.mean(np.asarray(diagnostics(out)["electric"])[100:]))


dts = np.array([0.4, 0.2, 0.1, 0.05])
w_dt = [oscillation(64, h) for h in dts]
orders_dt, limit_dt = richardson(w_dt)
cells = np.array([32, 64, 128, 256])
w_dx = [oscillation(n, 0.02) for n in cells]
orders_dx, limit_dx = richardson(w_dx)
counts = np.array([2000, 8000, 32000, 128000])
noise = np.array([noise_energy(n) for n in counts])
order_noise = np.polyfit(np.log(counts), np.log(noise), 1)[0]
for name, orders in (("time step", orders_dt), ("cell size", orders_dx)):
    print(f"{name}: Richardson orders " + ", ".join(f"{p:.2f}" for p in orders))
print(f"noise energy order {order_noise:.2f}")

fig, axes = figure(3)
a, b, d = axes
a.loglog(dts, np.abs(np.array(w_dt) - limit_dt), "o-", color=COLORS["blue"], label="measured")
a.loglog(dts, dts ** 2 / 24, "--", color=COLORS["black"], label=r"$(\omega_{pe}\Delta t)^2/24$")
a.set(xlabel=r"$\omega_{pe}\Delta t$", ylabel=r"$|\omega-\omega_\infty|/\omega_{pe}$", title="time step")
kdx = 2 * np.pi / cells
b.loglog(kdx, np.abs(np.array(w_dx) - limit_dx), "s-", color=COLORS["vermillion"], label="measured")
b.loglog(kdx, np.abs(w_dx[0] - limit_dx) * (kdx / kdx[0]) ** 2, "--", color=COLORS["black"], label="second order")
b.set(xlabel=r"$k\Delta x$", ylabel=r"$|\omega-\omega_\infty|/\omega_{pe}$", title="cell size")
d.loglog(counts, noise, "^-", color=COLORS["purple"], label="measured")
d.loglog(counts, noise[0] * counts[0] / counts, "--", color=COLORS["black"], label=r"$1/N$")
d.set(xlabel="electrons", ylabel=r"field energy (J/m$^2$)", title="shot noise")
for ax, label in zip(axes, "abc"):
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=14)
    panel_label(ax, label)
fig.tight_layout()
savefig(fig, "convergence")

record(convergence_dt_orders=", ".join(f"{p:.2f}" for p in orders_dt),
       convergence_dx_orders=", ".join(f"{p:.2f}" for p in orders_dx),
       convergence_dt_frequency_limit=round(float(limit_dt), 6),
       convergence_dt_error_at_0_4=float(f"{abs(w_dt[0] - limit_dt):.2e}"),
       convergence_noise_order=round(float(order_noise), 2))
