#!/usr/bin/env python
"""
Two-stream instability: energy-based gamma extraction and autodiff inverse demo.

This script has three main pieces:

1. energy_gamma_from_output(out):
   - Computes gamma from ln ∫|E|^2 dx in a fixed linear-phase window.

2. Benchmarks:
   (a) one detailed run at a chosen v_true, with a 3-panel figure
       (E_x, ∫|E|^2 dx, ln ∫|E|^2 dx + fit),
   (b) a scan gamma(v_d) over several drift speeds with a figure.

3. Autodiff-based inference of v_d:
   - Uses JAX grad on gamma_amp(v_d).
   - Works in log10(v_d) space, with gradient / step clipping.
   - Includes extensive debug prints per iteration.
   - Produces a publication-style figure of ln ∫|E|^2 dx for:
       * true v_d
       * initial guess
       * optimized v_d
"""

import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxincell import simulation

# ---------------------------------------------------------------------------
# 0. Global solver parameters (moderate size)
# ---------------------------------------------------------------------------
solver_parameters = {
    "field_solver": 0,                 # Curl_EB
    "number_grid_points": 80,
    "number_pseudoelectrons": 4500,
    "total_steps": 400,
    "time_evolution_algorithm": 0,     # Boris
}

# Fixed linear-phase window as FRACTIONS of total time series.
# These DO NOT depend on v_d, which is crucial for smooth autodiff.
FIT_FRAC_START = 0.30
FIT_FRAC_END   = 0.50


# ---------------------------------------------------------------------------
# 1. Utilities: build input parameters, run simulation
# ---------------------------------------------------------------------------
def make_input_parameters(electron_drift_speed_x: float,
                          print_info: bool = True):
    return {
        "length": 0.01,
        "amplitude_perturbation_x": 5.0e-7,
        "wavenumber_electrons_x": 1.0,
        "grid_points_per_Debye_length": 0.50265482457,
        "vth_electrons_over_c_x": 0.05,
        "ion_temperature_over_electron_temperature_x": 0.01,
        "ion_temperature_over_electron_temperature_y": 0.01,
        "ion_temperature_over_electron_temperature_z": 0.01,
        "timestep_over_spatialstep_times_c": 4.5,
        "electron_drift_speed_x": electron_drift_speed_x,
        "velocity_plus_minus_electrons_x": True,
        "velocity_plus_minus_electrons_y": False,
        "velocity_plus_minus_electrons_z": False,
        "print_info": print_info,
    }


def run_two_stream(v_d: float):
    """Run JAX-in-Cell two-stream simulation and return the output dict."""
    print("\n=== Running two-stream simulation ===")
    print(f"electron_drift_speed_x = {v_d:.3e} m/s")
    out = simulation(make_input_parameters(v_d), **solver_parameters)
    return out


# ---------------------------------------------------------------------------
# 2. Energy-based gamma extraction (core piece for everything)
# ---------------------------------------------------------------------------
def energy_gamma_from_output(out, *, for_jit: bool = False):
    """
    Compute gamma from ln ∫ |E|^2 dx in the quasi-linear phase.

    If for_jit=True:
        - no Python-side printing
        - no float(...) conversions
    so that the function is JIT/grad-safe.
    """

    G = out["number_grid_points"]
    L = out["length"]
    dx = L / G

    # E_x(x, t) and time array
    Ex = out["electric_field"][:, :, 0]   # shape (T, G)
    t  = out["time_array"]                # shape (T,)
    omega_p = out["plasma_frequency"]

    # ---------------------------------------------
    # 1) Integrated field energy vs time: ∫ |E|^2 dx
    # ---------------------------------------------
    energy_t = dx * jnp.sum(Ex**2, axis=1)   # shape (T,)
    energy_t = energy_t + 1e-30              # avoid log(0)
    log_energy = jnp.log(energy_t)

    # ---------------------------------------------
    # 2) Choose fit window in the linear phase
    #    (here: FIT_FRAC_START–FIT_FRAC_END of the time series)
    # ---------------------------------------------
    T = log_energy.shape[0]                  # Python int, even under jit
    i_start = int(FIT_FRAC_START * T)
    i_end   = int(FIT_FRAC_END * T)

    t_fit = t[i_start:i_end] - t[i_start]
    y_fit = log_energy[i_start:i_end]

    # ---------------------------------------------
    # 3) Linear least squares: y ≈ a + γ_E * t
    # ---------------------------------------------
    A = jnp.stack([jnp.ones_like(t_fit), t_fit], axis=1)  # (N, 2)
    ATA = A.T @ A
    ATy = A.T @ y_fit
    beta = jnp.linalg.solve(ATA, ATy)
    a_hat, gamma_energy = beta  # gamma for ln(energy)

    # Field-amplitude growth rate is γ = γ_E / 2
    gamma_amp = 0.5 * gamma_energy

    # R^2, just for diagnostics (not needed for grad)
    y_pred = a_hat + gamma_energy * t_fit
    ss_res = jnp.sum((y_fit - y_pred) ** 2)
    ss_tot = jnp.sum((y_fit - jnp.mean(y_fit)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)

    if not for_jit:
        # Only do Python printing + float conversions in non-JIT mode
        print("\n=== Gamma debug info (energy-based) ===")
        print(f"T (total steps)        = {int(T)}")
        print(f"i_start, i_end         = {int(i_start)}, {int(i_end)}")
        print(
            "t_fit[0], t_fit[-1]    = "
            f"{float(t_fit[0]):.3e}, {float(t_fit[-1]):.3e} s"
        )
        print(
            "gamma_energy           = "
            f"{float(gamma_energy):.6e} 1/s   (ln energy slope)"
        )
        print(
            "gamma_amp              = "
            f"{float(gamma_amp):.6e} 1/s   (field amplitude)"
        )
        print(
            "R^2 of ln(energy) fit  = "
            f"{float(r2):.6f}"
        )
        print(
            "plasma_frequency omega_p = "
            f"{float(omega_p):.6e} 1/s"
        )
        print("==========================================\n")

    return dict(
        gamma_energy=gamma_energy,
        gamma_amp=gamma_amp,
        energy=energy_t,
        log_energy=log_energy,
        t=t,
        t_fit=t_fit,
        y_fit=y_fit,
        y_pred=y_pred,
        omega_p=omega_p,
        i_start=i_start,
        i_end=i_end,
    )


def gamma_amp_from_vd(v_d: float):
    """
    Map drift speed -> amplitude growth rate gamma, JAX-compatible.

    This version is meant to be used inside jax.grad / jax.jit, so it:
      - disables JAX-in-Cell's internal printing (`print_info=False`)
      - calls energy_gamma_from_output(..., for_jit=True)
    """
    params = make_input_parameters(v_d, print_info=False)
    out = simulation(params, **solver_parameters)
    info = energy_gamma_from_output(out, for_jit=True)
    return info["gamma_amp"]


gamma_amp_from_vd_jit = jax.jit(gamma_amp_from_vd)

def gamma_hat_from_vd(v_d: float):
    """
    Drift speed -> dimensionless growth rate gamma_hat = gamma_amp / omega_p.

    This is what we actually optimize in the inverse problem, to keep the
    scales O(1) instead of O(1e10).
    """
    params = make_input_parameters(v_d, print_info=False)
    out = simulation(params, **solver_parameters)
    info = energy_gamma_from_output(out, for_jit=True)
    return info["gamma_amp"] / info["omega_p"]


gamma_hat_from_vd_jit = jax.jit(gamma_hat_from_vd)


# ---------------------------------------------------------------------------
# 3. Benchmark figures
# ---------------------------------------------------------------------------
def make_single_benchmark_figure(
    v_d,
    outpath="two_stream_gamma_energy_benchmark.png",
):
    """Reproduce the 3-panel debug figure for one v_d."""
    out = run_two_stream(v_d)
    info = energy_gamma_from_output(out, for_jit=False)

    t          = info["t"]
    energy     = info["energy"]
    log_energy = info["log_energy"]
    omega_p    = float(info["omega_p"])
    i_start    = info["i_start"]
    i_end      = info["i_end"]
    t_fit      = info["t_fit"]
    y_fit      = info["y_fit"]
    y_pred     = info["y_pred"]
    gamma_E    = float(info["gamma_energy"])
    gamma_amp  = float(info["gamma_amp"])

    E_mid = out["electric_field"][:, out["number_grid_points"] // 2, 0]

    tau     = t * omega_p
    tau_fit = (t_fit + t[i_start]) * omega_p

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (6.0, 4.0),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(3, 1, sharex=True)

    # Panel 1: E_x at mid cell
    axes[0].plot(tau, E_mid, linewidth=1.0)
    axes[0].axvspan(tau[i_start], tau[i_end - 1], alpha=0.15, label="fit window")
    axes[0].set_ylabel(r"$E_x(x_{\mathrm{mid}}, t)$")
    axes[0].set_title(
        rf"Two-stream gamma benchmark (energy): "
        rf"$v_d = {v_d:.2e}\,\mathrm{{m/s}}$"
    )
    axes[0].legend(frameon=False, loc="best")

    # Panel 2: ∫|E|^2 dx
    axes[1].plot(tau, energy, linewidth=1.0)
    axes[1].axvspan(tau[i_start], tau[i_end - 1], alpha=0.15)
    axes[1].set_ylabel(r"$\int |E|^2\,dx$ (arb.)")

    # Panel 3: ln ∫|E|^2 dx with fit
    axes[2].plot(tau, log_energy, linewidth=0.9,
                 label=r"$\ln \int |E|^2 dx$")
    axes[2].plot(tau_fit, y_fit, "o", markersize=2, label="fit samples")
    axes[2].plot(
        tau_fit, y_pred, "-", linewidth=1.2,
        label=(
            rf"fit: $\gamma_E = {gamma_E:.2e}\,\mathrm{{s^{{-1}}}}$" "\n"
            rf"$\gamma = \gamma_E/2 = {gamma_amp:.2e}\,\mathrm{{s^{{-1}}}}$"
        ),
    )
    axes[2].axvspan(tau[i_start], tau[i_end - 1], alpha=0.15)
    axes[2].set_xlabel(r"$t\,\omega_p$")
    axes[2].set_ylabel(r"$\ln \int |E|^2 dx$")
    axes[2].legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(outpath)
    print(f"Saved single benchmark figure to {outpath}")


def make_gamma_scan_figure(
    v_list,
    outpath="two_stream_gamma_scan.png",
):
    """
    Scan gamma_amp(v_d) over a list of drift speeds.

    - This is *not* autodiff; it's just a diagnostic to see if gamma(v_d)
      is reasonably monotone / well-behaved for inverse problems.
    """
    gamma_vals = []
    gammaE_vals = []

    plt.rcParams.update({
        "font.size": 9,
        "figure.figsize": (5.0, 3.5),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    for i, v in enumerate(v_list):
        v_f = float(v)
        print(f"\n=== Scan point {i}: v_d = {v_f:.3e} m/s ===")
        out = run_two_stream(v_f)
        info = energy_gamma_from_output(out, for_jit=False)
        gamma_vals.append(float(info["gamma_amp"]))
        gammaE_vals.append(float(info["gamma_energy"]))

    v_arr = jnp.array(v_list)
    gamma_arr = jnp.array(gamma_vals)

    fig, ax = plt.subplots()
    ax.plot(v_arr, gamma_arr, "o-", label=r"$\gamma$ from ln$\int |E|^2dx$")

    ax.set_xlabel(r"$v_d$ (m/s)")
    ax.set_ylabel(r"$\gamma$ (1/s)")
    ax.set_title("Two-stream: growth rate vs drift speed")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath)
    print(f"\nSaved gamma scan figure to {outpath}")
    print("\nScan table (for debugging):")
    for v, gE, g in zip(v_list, gammaE_vals, gamma_vals):
        print(
            f"  v_d = {float(v): .3e} m/s | "
            f"gamma_E = {gE: .3e} 1/s | gamma = {g: .3e} 1/s"
        )


# ---------------------------------------------------------------------------
# 4. Autodiff-based inverse problem with heavy debug
# ---------------------------------------------------------------------------
def autodiff_inverse_vd(
    gamma_hat_target: float,
    v_init: float,
    max_iters: int = 10,
    lr: float = 0.7,              # now acts as a Newton damping factor α
    grad_clip_abs: float = 1.0,   # clipping on Newton step, not on gradient
    step_fraction_clip: float = 0.5,
    v_min: float = 2.0e7,
    v_max: float = 1.5e8,
    omega_p_ref: float = 1.0,
    gamma_hat_tol: float = 1e-3,
):
    """
    Infer v_d such that gamma_hat_from_vd(v_d) ≈ gamma_hat_target,
    where gamma_hat = gamma_amp / omega_p is dimensionless.

    Uses a damped Newton iteration in v_d:

        v_{k+1} = v_k - α * (γ̂(v_k) - γ̂_target) / (dγ̂/dv).

    The derivative dγ̂/dv is obtained via JAX JVP, so this still
    showcases autodiff through the full PIC simulation.
    """

    # Clamp initial guess into [v_min, v_max]
    v = jnp.clip(v_init, v_min, v_max)

    print("\n=== Inference loop (Newton-style, dimensionless gamma_hat) ===")
    print(f"gamma_hat_target = {gamma_hat_target:.6e}  (gamma/omega_p)")
    print(f"initial v_d      = {v_init:.3e} m/s (clamped to {float(v):.3e})")
    print(
        f"lr (Newton damping) = {lr}, "
        f"grad_clip_abs (step clip) = {grad_clip_abs}, "
        f"step_fraction_clip = {step_fraction_clip}"
    )
    print("-------------------------------------------------------")

    history = []

    for it in range(max_iters):
        # γ̂(v) and dγ̂/dv via JVP (tangent = 1.0)
        gamma_hat, dgamma_hat_dv = jax.jvp(
            gamma_hat_from_vd_jit,
            (v,),
            (1.0,),
        )

        residual = gamma_hat - gamma_hat_target
        L = 0.5 * residual**2

        # Convert to physical γ for printing only
        gamma_phys = gamma_hat * omega_p_ref

        # Convert to Python scalars for nice prints
        v_scalar = float(v)
        gh_scalar = float(gamma_hat)
        g_scalar = float(gamma_phys)
        L_scalar = float(L)
        dgamma_dv_scalar = float(dgamma_hat_dv)
        residual_scalar = float(residual)

        print(
            f"iter {it:02d} | "
            f"v_d = {v_scalar:.6e} m/s | "
            f"gamma_hat = {gh_scalar:.4e} | "
            f"gamma = {g_scalar:.3e} 1/s | "
            f"L = {L_scalar:.3e} | "
            f"dγ̂/dv = {dgamma_dv_scalar:.3e} 1/(m/s)"
        )

        history.append((v_scalar, g_scalar, L_scalar, dgamma_dv_scalar))

        # Stopping criterion: gamma_hat close enough to target
        if abs(residual_scalar) < gamma_hat_tol:
            print(
                f"Converged: |gamma_hat - target| "
                f"< {gamma_hat_tol:.1e} ({abs(residual_scalar):.2e})."
            )
            break

        # Guard against tiny derivative: fall back to small step if needed
        denom = dgamma_hat_dv
        denom = jnp.where(
            jnp.abs(denom) < 1e-12,
            jnp.sign(denom) * 1e-12 + (denom == 0.0) * 1e-12,
            denom,
        )

        # Newton step: Δv_newton = (γ̂ - γ̂_target) / (dγ̂/dv)
        step_newton = residual / denom

        # Clip the Newton step magnitude
        step_newton = jnp.clip(
            step_newton,
            -grad_clip_abs,
            grad_clip_abs,
        )

        # Also limit relative step size to avoid wild jumps
        step_max = step_fraction_clip * jnp.abs(v)
        step_newton = jnp.clip(step_newton, -step_max, step_max)

        # Apply damped Newton update: v_{k+1} = v_k - α * Δv_newton
        v = v - lr * step_newton

        # Clamp back into [v_min, v_max]
        v = jnp.clip(v, v_min, v_max)

    v_opt = float(v)
    print("\nInference finished.")
    print(f"Optimized v_d ≈ {v_opt:.6e} m/s\n")

    return v_opt, history

# ---------------------------------------------------------------------------
# 5. Publication-style figure: ln ∫|E|^2 dx for true / init / opt
# ---------------------------------------------------------------------------
def make_two_panel_figure(
    v_true: float,
    v_init: float,
    v_opt: float,
    info_true,
    history,
    gamma_hat_target: float,
    omega_p_ref: float,
    outpath: str = "two_stream_two_panel.png",
):
    """
    Build a 2-panel, publication-ready figure:

      (left)  ln ∫|E|^2 dx vs t ω_p for true / init / optimized v_d,
              with start/end markers for each curve.

      (right) Optimization history: v_d, γ_hat, and |dγ_hat/dv_d| vs iteration.

    No titles (for LaTeX captions), clear axes and legends.
    """

    # ------------------------
    # 0) Matplotlib style
    # ------------------------
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 17,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.figsize": (9.0, 3.8),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, (ax_left, ax_right) = plt.subplots(1, 2)

    # ============================================================
    # LEFT PANEL: ln ∫|E|^2 dx vs t ω_p (true / init / opt)
    # ============================================================
    # True trace reuses existing simulation output
    t_true    = info_true["t"]
    logE_true = info_true["log_energy"]
    omega_p   = float(info_true["omega_p"])
    gamma_true = float(info_true["gamma_amp"])
    tau_true  = t_true * omega_p

    def energy_trace(vd: float):
        params = make_input_parameters(vd, print_info=False)
        out = simulation(params, **solver_parameters)
        info = energy_gamma_from_output(out, for_jit=False)
        t    = info["t"]
        tau  = t * float(info["omega_p"])
        logE = info["log_energy"]
        gamma = float(info["gamma_amp"])
        return tau, logE, gamma

    tau_init, logE_init, gamma_init = energy_trace(v_init)
    tau_opt,  logE_opt,  gamma_opt  = energy_trace(v_opt)

    # Main lines
    line_true, = ax_left.plot(
        tau_true, logE_true, "-",
        linewidth=1.6, label="true $v_d$",
    )
    line_init, = ax_left.plot(
        tau_init, logE_init, "--",
        linewidth=1.6, label="initial guess",
    )
    line_opt, = ax_left.plot(
        tau_opt, logE_opt, "-.",
        linewidth=1.6, label="optimized $v_d$",
    )

    # Start (circle) and end (square) markers so overlapping curves are visible
    def add_start_end_markers(ax, tau, logE, color):
        ax.plot(tau[0],  logE[0],  "o",  ms=4, mec=color, mfc="none")
        ax.plot(tau[-1], logE[-1], "s",  ms=4, mec=color, mfc=color)

    add_start_end_markers(ax_left, tau_true, logE_true, line_true.get_color())
    add_start_end_markers(ax_left, tau_init, logE_init, line_init.get_color())
    add_start_end_markers(ax_left, tau_opt,  logE_opt,  line_opt.get_color())

    ax_left.set_xlabel(r"$t\,\omega_p$")
    ax_left.set_ylabel(r"$\ln \int |E|^2 dx$ (arb. units)")
    ax_left.grid(True, alpha=0.3)

    ax_left.legend(frameon=False, loc="lower right")

    # ============================================================
    # RIGHT PANEL: optimization history (v_d, γ̂, |dγ̂/dv_d|)
    # ============================================================
    iters = jnp.arange(len(history))
    v_hist       = jnp.array([h[0] for h in history])  # m/s
    gamma_hist   = jnp.array([h[1] for h in history])  # physical γ
    L_hist       = jnp.array([h[2] for h in history])  # loss (unused but available)
    dgamma_dv_hat = jnp.array([h[3] for h in history])  # this is dγ̂/dv

    gamma_hat_hist = gamma_hist / omega_p_ref

    # Left y-axis: v_d and γ_hat
    line_v, = ax_right.plot(
        iters, v_hist * 1e-7, "s--", linewidth=1.6,
        label=r"$v_d/10^7$ (m/s)",
    )
    line_g, = ax_right.plot(
        iters, gamma_hat_hist, "o-",
        linewidth=1.7,
        label=r"$\hat{\gamma}$",
    )

    # Target gamma_hat as horizontal line
    ax_right.axhline(
        gamma_hat_target,
        linestyle=":",
        linewidth=1.6,
        color=line_g.get_color(),
        alpha=0.6,
        label=r"$\hat{\gamma}_\mathrm{target}$",
    )

    ax_right.set_xlabel("iteration")
    ax_right.set_ylabel(r"$v_d/10^7$ (m/s), $\hat{\gamma}$")
    ax_right.grid(True, alpha=0.3)

    # Right y-axis: |dγ̂/dv_d| in log scale
    ax2 = ax_right.twinx()
    line_grad, = ax2.semilogy(
        iters,
        jnp.abs(dgamma_dv_hat),
        "d-.",
        linewidth=1.4,
        label=r"$|d\hat{\gamma}/dv_d|$",
    )
    ax2.set_ylabel(r"$|d\hat{\gamma}/dv_d|$ (1/(m/s))")

    # Combined legend (both y-axes)
    lines = [line_v, line_g, line_grad]
    labels = [l.get_label() for l in lines]
    ax_right.legend(
        lines,
        labels,
        frameon=False,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(outpath)
    print(f"Saved 2-panel figure to {outpath}")



# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------
def main():
    # 1) Single benchmark figure at v_true
    v_true = 4.0e7
    make_single_benchmark_figure(v_true)

    # 2) Scan γ(v_d)
    vd_values = [2e7, 4e7, 6e7, 8e7, 1e8]
    make_gamma_scan_figure(vd_values)

    # 3) Compute target γ from synthetic "experiment"
    print("\n=== Computing target gamma from synthetic experiment ===")
    params_true = make_input_parameters(v_true, print_info=True)
    out_true = simulation(params_true, **solver_parameters)
    info_true = energy_gamma_from_output(out_true, for_jit=False)
    gamma_target = float(info_true["gamma_amp"])
    omega_p = float(info_true["omega_p"])
    print(f"\nTarget amplitude growth rate gamma_target = {gamma_target:.6e} 1/s")
    print(f"(In units of omega_p: gamma_target / omega_p = "
          f"{gamma_target / omega_p:.3e})")
    
    gamma_hat_target = gamma_target / omega_p

    # 4) Autodiff-based inversion (using dimensionless gamma_hat)
    v_init = 2.0e7
    v_opt, history = autodiff_inverse_vd(
        gamma_hat_target=gamma_hat_target,
        v_init=v_init,
        max_iters=8,
        lr=0.7,              # fairly aggressive Newton damping
        grad_clip_abs=2.0e7, # allow up to ~2e7 m/s Newton jumps
        step_fraction_clip=0.6,
        v_min=2e7,
        v_max=1.5e8,
        omega_p_ref=omega_p,
        gamma_hat_tol=5e-3,
    )

    # 5) Publication-ready 2-panel figure (logE traces + optimization history)
    make_two_panel_figure(
        v_true=v_true,
        v_init=v_init,
        v_opt=v_opt,
        info_true=info_true,
        history=history,
        gamma_hat_target=gamma_hat_target,
        omega_p_ref=omega_p,
        outpath="two_stream_two_panel.png",
    )


if __name__ == "__main__":
    main()
