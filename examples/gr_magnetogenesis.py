import os, time, copy
from jax import block_until_ready
from jaxincell import (simulation, plot,
    wave_spectrum_movie, phase_space_movie,
    particle_box_movie, vv_phase_grid_movie
)

# --------- global knobs (easy to see / edit) ---------
SAVE_MOVIES    = True
STEPS_TO_PLOT  = 450
MAKE_ZERO_DECAY_VARIANTS = True
SUMMARY_DIRECTION = "xz"
MOVIE_DIRECTION   = "xz"

# --------- base inputs (merge with input.toml for convenience) ---------
# input_parameters, solver_parameters = load_parameters("input.toml")

SOLVER = {
    "time_evolution_algorithm": 0,
    "field_solver": 0,
    "number_grid_points": 251,
    "number_pseudoelectrons": 13000,
    "total_steps": 8000,
    # **solver_parameters,
}

# --------- scenarios (simple, readable dicts) ---------
CASES = [
    # Bianchi 8: linear + cosine pump
    # metric equation: a_i(t) = a0i [1 + A_i t (1 + B_i cos)]
    {
        "name": "maxwellian_bianchi8_epsdecay{tau_z}_Bz{Bz}_omegaz{Omegaz}",
        "metric": {"kind": 8, "params": dict(
            a0x=1.0, a0y=1.0, a0z=1.0,
            Ax=0.0, Ay=0.0, Az=0.3,
            Bx=0.0, By=0.0, Bz=0.5,
            Omegax=0.0, Omegay=0.0, Omegaz=0.5,
            phix=0.0, phiy=0.0, phiz=0.0,
            tau_x=0.0, tau_y=0.0, tau_z=30.0,
        )},
    },
    
    {
        "name": "maxwellian_bianchi8_epsdecay{tau_z}_Bz{Bz}_omegaz{Omegaz}",
        "metric": {"kind": 8, "params": dict(
            a0x=1.0, a0y=1.0, a0z=1.0,
            Ax=0.0, Ay=0.0, Az=0.3,
            Bx=0.0, By=0.0, Bz=0.0,
            Omegax=0.0, Omegay=0.0, Omegaz=0.0,
            phix=0.0, phiy=0.0, phiz=0.0,
            tau_x=0.0, tau_y=0.0, tau_z=30.0,
        )},
    },

    # Flat spacetime
    {
        "name": "maxwellian_flatspace",
        "metric": {"kind": 0},  # no params
    },

    # Bianchi 9: volume-preserving cosine pump
    {
        "name": "maxwellian_bianchi9_eps{eps}_epsdecay{tau_eps}",
        "metric": {"kind": 9, "params": dict(
            a0x=1.0, a0y=1.0, a0z=1.0,
            eps=0.3, Omega=2.0, phi=0.0, tau_eps=30.0,
        )},
    },
    
    {
        "name": "maxwellian_bianchi9_eps{eps}_epsdecay{tau_eps}",
        "metric": {"kind": 9, "params": dict(
            a0x=1.0, a0y=1.0, a0z=1.0,
            eps=0.5, Omega=2.0, phi=0.0, tau_eps=30.0,
        )},
    },
]

BASE_INPUT = {
    # **input_parameters,
    "length": 0.03,
    "amplitude_perturbation_x": 0,
    "wavenumber_electrons_x": 0,
    "wavenumber_ions_x": 0,
    "grid_points_per_Debye_length": 0.7,
    "vth_electrons_over_c_x": 0.1,
    "vth_electrons_over_c_y": 0.1,
    "vth_electrons_over_c_z": 0.1,
    "electron_drift_speed_x": 0.0,
    "electron_drift_speed_y": 0.0,
    "electron_drift_speed_z": 0.0,
    "velocity_plus_minus_electrons_x": False,
    "velocity_plus_minus_electrons_y": False,
    "velocity_plus_minus_electrons_z": False,
    "random_positions_x": True,
    "random_positions_y": True,
    "random_positions_z": True,
    "ion_temperature_over_electron_temperature_x": 1.0,
    "timestep_over_spatialstep_times_c": 0.4,
    "relativistic": True,
    "filter_passes": 5,
    "filter_alpha": 0.5,
    "filter_strides": [1, 2, 4],
}

# Small inline helper to zero any known decay knobs (used only when requested)
def no_decay(params: dict) -> dict:
    if not params:  # flat space case
        return params
    params = params.copy()
    for k in ("tau_x", "tau_y", "tau_z", "tau_eps"):
        if k in params:
            params[k] = 0.0
    return params

# Single helper to avoid repeating the save/plot boilerplate
def run_case(run_name: str, params: dict):
    out_dir = os.path.join("outputs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    output = block_until_ready(simulation(params, **SOLVER))
    print(f"[{run_name}] Wall clock: {time.time()-t0:.2f}s")

    # Static summary (last frame)
    plot(
        output, direction=SUMMARY_DIRECTION,
        save_path=os.path.join(out_dir, f"{run_name}_summary.png"),
        bins_x=150, fps=30, dpi=60, crf=23,
        steps_to_plot=STEPS_TO_PLOT, bins_v=150, points_per_species=250
    )

    if not SAVE_MOVIES:
        return

    # Movies (common settings)
    fps, dpi, crf = 30, 60, 23
    plot(output, direction=MOVIE_DIRECTION,
         save_path=os.path.join(out_dir, f"{run_name}_summary.mp4"),
         bins_x=100, fps=fps, dpi=dpi, crf=crf,
         steps_to_plot=STEPS_TO_PLOT, bins_v=100, points_per_species=150)

    wave_spectrum_movie(
        output, direction="x", show_B=True, steps_to_plot=STEPS_TO_PLOT,
        save_path=os.path.join(out_dir, f"{run_name}_waves.mp4"), fps=fps, crf=crf, dpi=dpi
    )

    for d in ("x", "y", "z"):
        phase_space_movie(
            output, direction=d, species="both", points_per_species=250, dpi=dpi,
            steps_to_plot=STEPS_TO_PLOT, save_path=os.path.join(out_dir, f"{run_name}_phase_space_{d}.mp4"),
            fps=fps, interval_ms=33
        )
        particle_box_movie(
            output, direction=d, trail_len=20, n_electrons=120, n_ions=120,
            show_field=True, field_alpha=0.32, field_cmap="coolwarm", dpi=dpi,
            steps_to_plot=STEPS_TO_PLOT, save_path=os.path.join(out_dir, f"{run_name}_particles_{d}.mp4"),
            fps=fps
        )

    vv_phase_grid_movie(
        output, fps=fps, steps_to_plot=STEPS_TO_PLOT, interval_ms=33,
        save_path=os.path.join(out_dir, f"{run_name}_vv_phase_grid.mp4"),
        bins_v=120, points_per_species=250, seed=7, cmap="twilight", dpi=dpi, crf=crf
    )

# --------- main (simple loop; optional zero-decay) ---------
if __name__ == "__main__":
    for case in CASES:
        # assemble inputs for this case
        params = copy.deepcopy(BASE_INPUT)
        params["metric"] = case["metric"]

        # render name (fills {tau_*} if present; ignored if not)
        md = case["metric"]
        md_params = md.get("params", {})
        run_name = case["name"].format(
            tau_x=md_params.get("tau_x", 0.0),
            tau_y=md_params.get("tau_y", 0.0),
            tau_z=md_params.get("tau_z", 0.0),
            tau_eps=md_params.get("tau_eps", 0.0),
            Bz=md_params.get("Bz", 0.0),
            Omegaz=md_params.get("Omegaz", 0.0),
            eps=md_params.get("eps", 0.0),
        )
        run_case(run_name, params)

        # optional: zero-decay variant
        if MAKE_ZERO_DECAY_VARIANTS and md["kind"] != 0:
            params0 = copy.deepcopy(params)
            params0["metric"]["params"] = no_decay(md_params)
            run_case(run_name + "_nodecay", params0)
