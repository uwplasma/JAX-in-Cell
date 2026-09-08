"""Coulomb collisions against the Spitzer relaxation rates.

The Takizuka-Abe operator (J. Comput. Phys. 25, 205, 1977) scatters random pairs
of particles inside each cell. In the limit of a beam much faster than the
background it scatters off, the Fokker-Planck rates are closed-form (Trubnikov,
Rev. Plasma Phys. 1, 105, 1965; NRL Plasma Formulary):

    nu_0     = q_a^2 q_b^2 n_b ln(Lambda) / (4 pi eps0^2 m_a^2 v^3)
    nu_slow  = (1 + m_a/m_b) nu_0      d<v>/dt   = -nu_slow <v>
    nu_perp  = 2 nu_0                  d<v_perp^2>/dt = nu_perp v^2

which is what this measures. The operator carries no free parameter: the
variance of the scattering angle is fixed by matching nu_perp.
"""
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import random

from jaxincell import epsilon_0, mass_electron, elementary_charge as e_charge
from jaxincell._collisions import collide

density, coulomb_log, v_beam, n = 1e20, 12.5, 3e7, 60000
nu_0 = e_charge ** 4 * coulomb_log * density / (4 * np.pi * epsilon_0 ** 2 * mass_electron ** 2 * v_beam ** 3)

rng = np.random.default_rng(1)
beam = np.zeros((n, 3))
beam[:, 0] = v_beam
v = jnp.asarray(np.concatenate([beam, 3e5 / np.sqrt(2) * rng.standard_normal((n, 3))]))
x = jnp.zeros((2 * n, 3))
weight = jnp.full(2 * n, density / n)
mass, charge = jnp.full(2 * n, mass_electron), jnp.full(2 * n, -e_charge)

dt, steps = 0.0005 / (2 * nu_0), 40
key, history = random.PRNGKey(0), []
for step in range(steps + 1):
    history.append((step * dt, float(v[:n, 0].mean()), float((v[:n, 1] ** 2 + v[:n, 2] ** 2).mean())))
    key, sub = random.split(key)
    v = collide(sub, x, v, weight, mass, charge, ((0, n), (n, n)), ((0, 1),), coulomb_log, dt, 1.0, 1.0, 1)
t, v_parallel, v_perp2 = (np.array(column) for column in zip(*history))

nu_slow = -np.polyfit(t, np.log(v_parallel), 1)[0]
nu_perp = np.polyfit(t, v_perp2, 1)[0] / v_beam ** 2
print(f"nu_slow / 2 nu_0 = {nu_slow / (2 * nu_0):.3f}    nu_perp / 2 nu_0 = {nu_perp / (2 * nu_0):.3f}")

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
left.plot(t * nu_0, v_parallel / v_beam, "o", ms=3, label="JAX-in-Cell")
left.plot(t * nu_0, np.exp(-2 * nu_0 * t), "k-", label=r"$e^{-\nu_s t}$")
left.set(xlabel=r"$t\,\nu_0$", ylabel=r"$\langle v_x\rangle / v_{beam}$"); left.legend(frameon=False)
right.plot(t * nu_0, v_perp2 / v_beam ** 2, "o", ms=3, label="JAX-in-Cell")
right.plot(t * nu_0, 2 * nu_0 * t, "k-", label=r"$\nu_\perp t$")
right.set(xlabel=r"$t\,\nu_0$", ylabel=r"$\langle v_\perp^2\rangle / v_{beam}^2$"); right.legend(frameon=False)
plt.tight_layout(); plt.show()
