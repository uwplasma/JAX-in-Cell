"""Coulomb collisions against the Fokker-Planck relaxation rates."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from common import C_ELECTRONS, C_IONS, C_THEORY, WIDE, panel_label, record, savefig
from jax import random

from jaxincell import epsilon_0, mass_electron, elementary_charge as e_charge
from jaxincell._collisions import collide

DENSITY, COULOMB_LOG, V_BEAM, N = 1e20, 12.5, 3e7, 60000
NU_0 = e_charge ** 4 * COULOMB_LOG * DENSITY / (4 * np.pi * epsilon_0 ** 2 * mass_electron ** 2 * V_BEAM ** 3)


def beam(mass_ratio, steps=40):
    """A beam of electrons scattering off a cold background of mass
    ``mass_ratio`` electron masses. Returns t, <v_x> and <v_perp^2>."""
    rng = np.random.default_rng(1)
    initial = np.zeros((N, 3))
    initial[:, 0] = V_BEAM
    background = 3e5 / np.sqrt(mass_ratio) / np.sqrt(2) * rng.standard_normal((N, 3))
    v = jnp.asarray(np.concatenate([initial, background]))
    x, weight = jnp.zeros((2 * N, 3)), jnp.full(2 * N, DENSITY / N)
    mass = jnp.asarray(np.concatenate([np.full(N, mass_electron), np.full(N, mass_ratio * mass_electron)]))
    charge = jnp.full(2 * N, -e_charge)
    dt = 0.0005 / (2 * NU_0)                       # the beam slows by about two percent
    key, history = random.PRNGKey(0), []
    for step in range(steps + 1):
        history.append((step * dt, float(v[:N, 0].mean()), float((v[:N, 1] ** 2 + v[:N, 2] ** 2).mean())))
        key, sub = random.split(key)
        v = collide(sub, x, v, weight, mass, charge, ((0, N), (N, N)), ((0, 1),), COULOMB_LOG, dt, 1.0, 1.0, 1)
    return (np.array(column) for column in zip(*history))


fig, axes = plt.subplots(1, 2, figsize=WIDE)
rates = {}
for mass_ratio, colour, label in ((1.0, C_ELECTRONS, r"$m_b = m_a$"), (100.0, C_IONS, r"$m_b = 100\,m_a$")):
    t, v_parallel, v_perp2 = beam(mass_ratio)
    nu_slow = -np.polyfit(t, np.log(v_parallel), 1)[0]
    nu_perp = np.polyfit(t, v_perp2, 1)[0] / V_BEAM ** 2
    rates[mass_ratio] = (nu_slow / ((1 + 1 / mass_ratio) * NU_0), nu_perp / (2 * NU_0))
    print(f"  m_b/m_a {mass_ratio:5.0f}: nu_slow/theory {rates[mass_ratio][0]:.3f}, "
          f"nu_perp/theory {rates[mass_ratio][1]:.3f}")
    axes[0].plot(t * NU_0, v_parallel / V_BEAM, "o", ms=3, color=colour, label=label)
    axes[0].plot(t * NU_0, np.exp(-(1 + 1 / mass_ratio) * NU_0 * t), "-", color=C_THEORY, lw=1.0)
    axes[1].plot(t * NU_0, v_perp2 / V_BEAM ** 2, "o", ms=3, color=colour, label=label)
axes[1].plot(t * NU_0, 2 * NU_0 * t, "-", color=C_THEORY, lw=1.0, label=r"$\nu_\perp t$")
axes[0].plot([], [], "-", color=C_THEORY, lw=1.0, label=r"$e^{-\nu_s t}$")
axes[0].set(xlabel=r"$t\,\nu_0$", ylabel=r"$\langle v_x\rangle/v_{beam}$", title="slowing down")
axes[0].legend()
panel_label(axes[0], "a")
axes[1].set(xlabel=r"$t\,\nu_0$", ylabel=r"$\langle v_\perp^2\rangle/v_{beam}^2$",
            title="perpendicular diffusion")
axes[1].legend()
panel_label(axes[1], "b")
fig.tight_layout()
savefig(fig, "collisions")

record(collisions_coulomb_log=COULOMB_LOG, collisions_particles=2 * N,
       collisions_beam_over_background=round(float(V_BEAM / 3e5), 0),
       collisions_nu_slow_ratio_equal_mass=round(rates[1.0][0], 3),
       collisions_nu_perp_ratio_equal_mass=round(rates[1.0][1], 3),
       collisions_nu_slow_ratio_heavy=round(rates[100.0][0], 3),
       collisions_nu_perp_ratio_heavy=round(rates[100.0][1], 3),
       collisions_max_deviation_percent=round(float(100 * max(abs(r - 1) for pair in rates.values() for r in pair)), 1))
