"""Two-stream instability: growth and saturation, the electron phase space, and
a scan of the growth rate against the kinetic dispersion relation."""
import numpy as np
from common import C_ELECTRONS, C_FIT, C_THEORY, figure, panel_label, phase_space_hist, record, savefig
from two_stream_setup import CELLS, DRIFT, LENGTH, OMEGA_PE, SEED_AK, build, measure, theory

from jaxincell import diagnostics

simulation = build()
output = simulation.run(900, seed=0, store_every=2)
t, amplitude, fit = measure(output)
gamma_theory = theory(DRIFT)

fig, axes = figure(2)
axes[0].semilogy(t, amplitude, color=C_ELECTRONS, label=r"$|E_{k=1}(t)|$")
span = np.linspace(fit["t0"], fit["t1"], 2)
axes[0].semilogy(span, np.exp(fit["intercept"] + fit["gamma"] * span), "--", color=C_FIT,
                 label=fr"fit: $\gamma={fit['gamma']:.3f}\,\omega_{{pe}}$")
axes[0].semilogy(span, np.exp(fit["intercept"] + gamma_theory * span), ":", color=C_THEORY,
                 label=fr"kinetic: $\gamma={gamma_theory:.3f}\,\omega_{{pe}}$")
axes[0].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|E_{k=1}|$ (V/m)", title="the seeded mode")
axes[0].legend(loc="lower right")
panel_label(axes[0], "a")

x_e, v_e = output.particles("electrons")
image = phase_space_hist(axes[1], np.asarray(x_e[-1, :, 0]), np.asarray(v_e[-1, :, 0]), LENGTH, 2.4 * DRIFT)
fig.colorbar(image, ax=axes[1], pad=0.02).set_label("pseudo-particles per bin")
axes[1].set(xlabel="$x/L$", ylabel="$v_x$ (m/s)",
            title=fr"electron phase space, $t\,\omega_{{pe}}={t[-1]:.0f}$")
panel_label(axes[1], "b")
fig.tight_layout()
savefig(fig, "two_stream")

drifts = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]) * 1e7
measured, predicted = [], []
for drift in drifts:
    _, _, result = measure(build(drift).run(900, seed=0, store_particles=False))
    measured.append(result["gamma"] if result else np.nan)
    predicted.append(theory(drift))
    print(f"  drift {drift:.2e} m/s (k v0/omega_pe {2 * np.pi / LENGTH * drift / OMEGA_PE:.3f}): "
          f"measured {measured[-1]:.4f}, kinetic {predicted[-1]:.4f}, "
          f"R2 {result['r2'] if result else float('nan'):.4f}")
measured, predicted = np.array(measured), np.array(predicted)

fine = np.linspace(2.0e7, 6.2e7, 22)
fig, ax = figure()
ax.plot(2 * np.pi / LENGTH * fine / OMEGA_PE, [theory(d) for d in fine], "-", color=C_THEORY,
        label="kinetic theory")
ax.plot(2 * np.pi / LENGTH * drifts / OMEGA_PE, measured, "o", color=C_ELECTRONS, label="JAX-in-Cell")
ax.axvline(1.0, color="0.7", lw=2)
ax.text(1.01, 0.05, "cold-beam cutoff", rotation=90, color="0.4", transform=ax.get_xaxis_transform())
ax.set(xlabel=r"$k v_0/\omega_{pe}$", ylabel=r"$\gamma/\omega_{pe}$", title="growth rate of the seeded mode")
ax.legend()
fig.tight_layout()
savefig(fig, "two_stream_scan")

deviation = 100 * np.abs(measured - predicted) / predicted
energy = np.asarray(diagnostics(output)["total"])
record(two_stream_gamma_measured=round(float(fit["gamma"]), 4),
       two_stream_gamma_theory=round(float(gamma_theory), 4),
       two_stream_gamma_deviation_percent=round(float(100 * abs(fit["gamma"] - gamma_theory) / gamma_theory), 1),
       two_stream_fit_r2=round(float(fit["r2"]), 4),
       two_stream_fit_window=f"{fit['t0']:.0f}-{fit['t1']:.0f}",
       two_stream_particles=int(2 * 20000 + 10000),
       two_stream_seed_ak=SEED_AK,
       two_stream_cells=CELLS,
       two_stream_k_v0_over_wpe=round(float(DRIFT * 2 * np.pi / LENGTH / OMEGA_PE), 3),
       two_stream_omega_pe=f"{OMEGA_PE:.3e}",
       two_stream_omega_pe_dt=round(float(OMEGA_PE * simulation.domain.dt), 4),
       two_stream_dx_over_debye=round(float(simulation.domain.dx / simulation.debye_length()), 3),
       two_stream_energy_error=f"{float(np.max(np.abs(energy / energy[0] - 1))):.1e}",
       two_stream_scan_mean_deviation_percent=round(float(np.nanmean(deviation)), 1),
       two_stream_scan_max_deviation_percent=round(float(np.nanmax(deviation)), 1))
