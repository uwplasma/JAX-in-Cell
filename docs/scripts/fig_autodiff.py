"""Automatic differentiation through a full simulation: the JAX gradient of a
time-averaged electric field with respect to the electron drift speed, compared
with one-sided finite differences. Mirrors examples/auto-differentiability.py."""
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import block_until_ready, grad

from common import (C_ELECTRONS, C_THEORY, C_IONS, EXAMPLES_DIR, WIDE, panel_label,
                    quiet_parameters, record, savefig, silence_progress_bars)
from jaxincell import Simulation, load_parameters

silence_progress_bars()
parameters = quiet_parameters(load_parameters(EXAMPLES_DIR / "input.toml"))
parameters["domain_parameters"]["total_steps"] = 400
parameters["domain_parameters"]["number_grid_points"] = 60
parameters["species_parameters"]["electrons"]["electrons0"]["number_pseudoparticles"] = 3000
parameters["species_parameters"]["ions"]["ions0"]["number_pseudoparticles"] = 3000
sim = Simulation(parameters)
total_steps = sim.domain_parameters["total_steps"]


def mean_electric_field(drift_speed):
    output = sim.run({"electrons": {"electrons0": {"drift_speed_x": drift_speed}}})
    E = jnp.mean(output["electric_field"][:, :, 0], axis=1)
    return jnp.mean(E[total_steps // 2:])


v0 = 1e8
f0 = block_until_ready(mean_electric_field(v0))
start = time.perf_counter()
dfdv_jax = float(block_until_ready(grad(mean_electric_field)(v0)))
t_grad = time.perf_counter() - start
start = time.perf_counter()
block_until_ready(grad(mean_electric_field)(v0))
t_grad_warm = time.perf_counter() - start

epsilons = np.logspace(-4, 6, 21)
fd = []
for eps in epsilons:
    fd.append((float(mean_electric_field(v0 + eps)) - float(f0)) / eps)
fd = np.array(fd)
best = np.argmin(np.abs(fd - dfdv_jax))
record(autodiff_gradient_jax=dfdv_jax, autodiff_best_finite_difference=float(fd[best]),
       autodiff_best_epsilon=float(epsilons[best]), autodiff_grad_time_first_s=t_grad,
       autodiff_grad_time_warm_s=t_grad_warm)

# A scan of the objective itself, to show the function whose slope is measured.
scan_v = np.linspace(0.8e8, 1.2e8, 9)
scan_f = np.array([float(mean_electric_field(v)) for v in scan_v])

fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.38})
ax = axes[0]
ax.plot(scan_v / 1e8, scan_f, "o-", ms=4, color=C_ELECTRONS, label=r"$\langle E_x\rangle(v_d)$")
tangent = float(f0) + dfdv_jax * (scan_v - v0)
ax.plot(scan_v / 1e8, tangent, ls="--", color=C_THEORY, label="tangent from JAX gradient")
ax.set_xlabel(r"electron drift speed $v_d$ (10$^8$ m/s)")
ax.set_ylabel(r"$\langle E_x \rangle$ (V/m)")
ax.legend(loc="best")
panel_label(ax, "(a)")

ax = axes[1]
ax.semilogx(epsilons, fd, "o-", ms=4, color=C_IONS, label="one-sided finite difference")
ax.axhline(dfdv_jax, ls="--", color=C_THEORY, label="JAX reverse-mode gradient")
ax.set_xlabel(r"finite-difference step $\epsilon$ (m/s)")
ax.set_ylabel(r"$d\langle E_x\rangle / d v_d$  (V s / m$^2$)")
ax.legend(loc="best")
panel_label(ax, "(b)")
savefig(fig, "autodiff")
