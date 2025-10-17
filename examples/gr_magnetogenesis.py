# Example script to run the simulation and plot the results
import os
import time
import numpy as np
from jax import block_until_ready
from jaxincell import plot, simulation, load_parameters
from jaxincell import wave_spectrum_movie, phase_space_movie, particle_box_movie

# Read from input.toml
input_parameters, solver_parameters = load_parameters('input.toml')

input_parameters = {
 "length": 1.0,                     # dimensions of the simulation box in (x, y, z)
 "amplitude_perturbation_x": 0,
 "wavenumber_electrons_x": 0,
 "wavenumber_ions_x": 0,
 "grid_points_per_Debye_length": 0.4,
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
 "ion_temperature_over_electron_temperature_x": 0.01,
 "timestep_over_spatialstep_times_c": 0.4,
 "relativistic": True,
 "filter_passes": 5,
 "filter_alpha":  0.5,
 "filter_strides": [1, 2, 4],
 "metric": { "kind": 6, "params": { "a0x": 1.0, "a0y": 1.0, "a0z": 1.0, "Hx": 0e0, "Hy": 0e0, "Hz": 5e0 } }
}

solver_parameters = {
 "time_evolution_algorithm": 0,
 "field_solver": 0,
 "number_grid_points": 280,
 "number_pseudoelectrons": 4000,
 "total_steps": 1300,
}

save_movies = True

# Run a simulation with GR metric kind=6 (anisotropic expanding universe)
start = time.time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Wall clock time: {time.time()-start}s")
# --------- run naming ---------
RUN_NAME   = "maxwellian_gr"
OUTPUT_DIR = os.path.join("outputs", RUN_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------- save static summary plot ---------
# Animated MP4
# plot(output, direction="xz", save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_summary.mp4"), fps=30, dpi=130, crf=23)
# Animated GIF
# plot(output, direction="xz", save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_summary.gif"), fps=20, dpi=110)
# Static PNG of the LAST frame
plot(output, direction="xz", save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_summary.png"), dpi=150)
if save_movies:
    # --------- 1) Waves + spectra + energy (E & B) ---------
    wave_spectrum_movie( output, direction="x", show_B=True, 
        save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_waves.mp4"), fps=30, crf=23, dpi=110)
    # --------- 2) Phase space (both species) ---------
    phase_space_movie(output, direction="x", species="both", points_per_species=250,
        save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_phase_space_x.mp4"), fps=30, interval_ms=33)
    phase_space_movie(output, direction="y", species="both", points_per_species=250,
        save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_phase_space_y.mp4"), fps=30, interval_ms=33)
    phase_space_movie(output, direction="z", species="both", points_per_species=250,
        save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_phase_space_z.mp4"), fps=30, interval_ms=33)
    # --------- 3) Particle box (with field overlay) ---------
    particle_box_movie(output, direction="x", trail_len=20, n_electrons=120, n_ions=120,
                    show_field=True, field_alpha=0.32, field_cmap="coolwarm",
                    save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_particles_x.mp4"), fps=30,)
    particle_box_movie(output, direction="y", trail_len=20, n_electrons=120, n_ions=120,
                    show_field=True, field_alpha=0.32, field_cmap="coolwarm",
                    save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_particles_y.mp4"), fps=30,)
    particle_box_movie(output, direction="z", trail_len=20, n_electrons=120, n_ions=120,
                    show_field=True, field_alpha=0.32, field_cmap="coolwarm",
                    save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_particles_z.mp4"), fps=30,)
    # --------- Save the raw output ---------
    # np.savez(os.path.join(OUTPUT_DIR, f"{RUN_NAME}_output.npz"), **output)
    print(f"Saved results under: {OUTPUT_DIR}")

# Change metric kind to 0 (flat spacetime) and rerun the simulation for comparison
input_parameters["metric"]["kind"] = 0
start = time.time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Wall clock time: {time.time()-start}s")
# --------- run naming ---------
RUN_NAME   = "maxwellian_flatspace"
OUTPUT_DIR = os.path.join("outputs", RUN_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------- save static summary plot ---------
# Animated MP4
# plot(output, direction="xz", save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_summary.mp4"), fps=30, dpi=130, crf=23)
# Animated GIF
# plot(output, direction="xz", save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_summary.gif"), fps=20, dpi=110)
# Static PNG of the LAST frame
plot(output, direction="xz", save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_summary.png"), dpi=150)
if save_movies:
    # --------- 1) Waves + spectra + energy (E & B) ---------
    wave_spectrum_movie( output, direction="x", show_B=True, 
        save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_waves.mp4"), fps=30, crf=23, dpi=110)
    # --------- 2) Phase space (both species) ---------
    phase_space_movie(output, direction="x", species="both", points_per_species=250,
        save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_phase_space_x.mp4"), fps=30, interval_ms=33)
    phase_space_movie(output, direction="y", species="both", points_per_species=250,
        save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_phase_space_y.mp4"), fps=30, interval_ms=33)
    phase_space_movie(output, direction="z", species="both", points_per_species=250,
        save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_phase_space_z.mp4"), fps=30, interval_ms=33)
    # --------- 3) Particle box (with field overlay) ---------
    particle_box_movie(output, direction="x", trail_len=20, n_electrons=120, n_ions=120,
                    show_field=True, field_alpha=0.32, field_cmap="coolwarm",
                    save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_particles_x.mp4"), fps=30,)
    particle_box_movie(output, direction="y", trail_len=20, n_electrons=120, n_ions=120,
                    show_field=True, field_alpha=0.32, field_cmap="coolwarm",
                    save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_particles_y.mp4"), fps=30,)
    particle_box_movie(output, direction="z", trail_len=20, n_electrons=120, n_ions=120,
                    show_field=True, field_alpha=0.32, field_cmap="coolwarm",
                    save_path=os.path.join(OUTPUT_DIR, f"{RUN_NAME}_particles_z.mp4"), fps=30,)
    # --------- Save the raw output ---------
    # np.savez(os.path.join(OUTPUT_DIR, f"{RUN_NAME}_output.npz"), **output)
    print(f"Saved results under: {OUTPUT_DIR}")

