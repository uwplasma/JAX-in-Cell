"""External fields that vary across the box: the grad-B drift and a magnetic mirror.

An external field given as an array of shape (cells, ny, nz, 3) lives on an (x, y, z) grid
whose x cells are the domain's and whose ny and nz cells span the periods length_y and
length_z of the domain; it is gathered at each particle's x, y and z with the spline of the
deposit. The self-consistent fields still depend on x alone. What this opens up is single-
particle and test-particle physics in a field with structure across x: drifts, mirrors,
gradients a real device has.

Two checks against guiding-centre theory, with a single tenuous electron each:

* **grad-B drift.** B = B0 (1 + y/L) along x. The guiding centre drifts along z at
  v_d = m v_perp^2 |B x grad B| / (2 |q| B^3) = v_perp (rho/L)/2, for three gradient lengths.
  What is left over is the finite-Larmor-radius correction, second order in rho/L.
* **magnetic mirror.** B_y = B0 (1 + y^2/L^2) along y, closed by the radial field
  -(r/2) dB_y/dy that div B = 0 needs. An electron of pitch angle theta at the midplane turns
  where B/B0 = 1/sin^2(theta), at y = L cot(theta), and its magnetic moment
  mu = m v_perp^2/(2B) -- the `magnetic_moment` diagnostic -- stays constant on the way.

Run with `--quick` for one gradient length and one pitch angle.
"""

import os
import sys
from pathlib import Path

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below to "0", for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, figure, magnetic_moment, mass_electron, save_run,
                       elementary_charge as e_charge)

# --- what to change ---------------------------------------------------------------------
quick = "--quick" in sys.argv
B0 = 0.01                                            # T
speed = 1e5                                          # m/s
omega_dt = 0.1                                       # gyro-phase per step
gradient_lengths = (20,) if quick else (10, 20, 40)  # L/rho for the grad-B drift
pitch_angles = (45.0,) if quick else (30.0, 45.0, 60.0)   # degrees, at the mirror midplane
mirror_length = 50                                   # L/rho of the mirror

# --- the setup ----------------------------------------------------------------------------
omega = e_charge * B0 / mass_electron
rho = speed / omega
dt = omega_dt / omega
if quick:
    print("--quick runs one gradient length and one pitch angle; the documentation quotes the full set.\n")


def centres(n, length):
    return (np.arange(n) + 0.5) * length / n - length / 2


def track(domain, x, v, steps, B, store_every=1):
    """One electron, tenuous enough that its own field is nothing beside B."""
    electron = Species("electron", 1, -1.0, mass_electron, 1e-3, (0.0, 0.0, 0.0)).replace(
        x=jnp.asarray([x], float), v=jnp.asarray([v], float))
    sim = Simulation(domain, [electron], Solver(model="electrostatic"), external_B=B)
    return sim, sim.run(steps, store_every=store_every)


# grad-B: B along x, growing along y; the electron gyrates in the y-z plane
drift = {}
for ratio in gradient_lengths:
    L, ny, Ly, cells = ratio * rho, 64, 200 * rho, 8
    domain = Domain(length=Ly, cells=cells, time_step=dt, length_y=Ly)
    B = np.zeros((cells, ny, 1, 3))
    B[..., 0] = B0 * (1 + centres(ny, Ly) / L)[None, :, None]
    sim, out = track(domain, (0.0, 0.0, rho), (0.0, speed, 0.0), 3000, B)
    fields = np.asarray(jax.vmap(sim.external_fields_at)(out.x))[:, 0, 3:]
    v = np.asarray(out.v)[:, 0]
    # guiding centre R = x + m (v x B)/(q B^2), with q = -e
    R = np.asarray(out.x)[:, 0] - mass_electron / e_charge * np.cross(v, fields) / np.sum(fields ** 2, 1)[:, None]
    t = np.asarray(out.t)
    analytic = -speed ** 2 / (2 * omega * L)
    measured = np.polyfit(t, R[:, 2], 1)[0]
    drift[ratio] = dict(t=t * omega / (2 * np.pi), z=R[:, 2] / rho, particle_z=np.asarray(out.x)[:, 0, 2] / rho,
                        analytic_z=(R[0, 2] + analytic * t) / rho, measured=measured, analytic=analytic)
    print(f"grad-B, L/rho = {ratio:3d}: drift {measured:9.2f} m/s, analytic {analytic:9.2f} m/s, "
          f"deviation {100 * (measured / analytic - 1):+.3f} %  ((rho/L)^2 = {100 / ratio ** 2:.3f} %)")

# the mirror: along y, with its radial field in x and z
L = mirror_length * rho
cells, ny, nz = 16, 192, 16
Lx = Lz = 16 * rho
Ly = 6 * L
domain = Domain(length=Lx, cells=cells, time_step=dt, length_y=Ly, length_z=Lz)
x, y, z = np.meshgrid(centres(cells, Lx), centres(ny, Ly), centres(nz, Lz), indexing="ij")
B = np.stack([-x * y * B0 / L ** 2, B0 * (1 + y ** 2 / L ** 2), -z * y * B0 / L ** 2], axis=-1)
mirror = {}
for angle in pitch_angles:
    theta = np.radians(angle)
    v_par, v_perp = speed * np.cos(theta), speed * np.sin(theta)
    # a quarter of the small-amplitude bounce period is (pi/2) L / v_perp; run past a full one
    steps = 10 * int(1.2 * 2 * np.pi * L / v_perp / dt / 10)
    sim, out = track(domain, (v_perp / omega, 0.0, 0.0), (0.0, v_par, -v_perp), steps, B, store_every=10)
    mu = np.asarray(magnetic_moment(out, sim))[:, 0]
    strength = np.linalg.norm(np.asarray(jax.vmap(sim.external_fields_at)(out.x))[:, 0, 3:], axis=1)
    y_path = np.asarray(out.x)[:, 0, 1]
    mirror[angle] = dict(t=np.asarray(out.t) * omega / (2 * np.pi), y=y_path / L, mu=mu / mu[0], B=strength / B0,
                         turning=float(y_path.max() / L), analytic_turning=1 / np.tan(theta),
                         mu_spread=float(np.ptp(mu) / mu.mean()))
    print(f"mirror, pitch {angle:.0f} deg: turns at y/L = {mirror[angle]['turning']:.4f}, "
          f"analytic cot(theta) = {1 / np.tan(theta):.4f}; mu varies by {mirror[angle]['mu_spread']:.1e} "
          f"while B changes by a factor {1 / np.sin(theta) ** 2:.2f}")

# --- the figure -------------------------------------------------------------------------------
# (a) the drift's error against L/rho, next to (rho/L)^2; (b) the bounce, with the turning points
# L cot(theta) dashed; (c) mu through the bounce, with the field the electron sees dotted
fig, axes = figure(3)
ratios = np.array(sorted(drift))
deviation = np.array([abs(drift[r]["measured"] / drift[r]["analytic"] - 1) for r in ratios])
axes[0].loglog(ratios, deviation, "o-", label="measured")
if len(ratios) > 1:
    axes[0].loglog(ratios, 1.0 / ratios ** 2, "k--", lw=1.5, label=r"$(\rho/L)^2$")
axes[0].set(xlabel=r"$L/\rho$", ylabel=r"$|v_d/(v_\perp\rho/2L) - 1|$", title=r"grad-$B$ drift")
axes[0].set_xticks(ratios, [str(r) for r in ratios])
axes[0].minorticks_off()
axes[0].legend(frameon=False)
field_axis = axes[2].twinx()
for angle, m in mirror.items():
    line, = axes[1].plot(m["t"], m["y"], label=rf"$\theta = {angle:.0f}^\circ$")
    for sign in (1, -1):
        axes[1].axhline(sign * m["analytic_turning"], color=line.get_color(), ls="--", lw=1)
    axes[2].plot(m["t"], 1e5 * (m["mu"] - 1), color=line.get_color())
    field_axis.plot(m["t"], m["B"], color=line.get_color(), ls=":", lw=1)
axes[1].set(xlabel=r"time ($2\pi/\Omega$)", ylabel=r"$y/L$", title=r"mirror bounce; dashed: $\pm L\cot\theta$")
axes[1].legend(frameon=False, loc="lower right")
axes[2].set(xlabel=r"time ($2\pi/\Omega$)", ylabel=r"$10^5\,(\mu/\mu_0 - 1)$",
            title=r"$\mu$ (solid); $B/B_0$ at the electron (dotted)")
field_axis.set(ylabel=r"$B/B_0$", ylim=(0.9, 4.5))
field_axis.grid(False)
# the colours are those of (b)
plt.tight_layout()

folder = Path.cwd() / ("external_fields_3d_quick" if quick else "external_fields_3d")
save_run(folder, "external_fields_3d",
         dict(B0=B0, speed=speed, omega_dt=omega_dt, gradient_lengths=list(gradient_lengths),
              pitch_angles=list(pitch_angles), mirror_length=mirror_length, quick=quick),
         dict(grad_B={str(r): dict(measured=d["measured"], analytic=d["analytic"],
                                   deviation=d["measured"] / d["analytic"] - 1) for r, d in drift.items()},
              mirror={f"{a:.0f}": dict(turning=m["turning"], analytic_turning=m["analytic_turning"],
                                       mu_spread=m["mu_spread"]) for a, m in mirror.items()}),
         figure=fig,
         **{f"drift_{r}_{k}": d[k] for r, d in drift.items() for k in ("t", "z", "analytic_z")},
         **{f"mirror_{a:.0f}_{k}": m[k] for a, m in mirror.items() for k in ("t", "y", "mu", "B")})
plt.show()
