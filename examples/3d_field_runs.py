## 3d_field_runs.py
# Compare 1D, 2D, and 3D uniform external magnetic fields.
#
# This is intentionally an exploratory script rather than a strict test. The
# physical external field arrays have different dimensionality, but the field is
# uniform, so the particle trajectories should line up across the three cases.

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from jax import block_until_ready

from jaxincell import Simulation, diagnostics, plot


# Grid controls
NUMBER_GRID_POINTS_X = 32
NUMBER_GRID_POINTS_Y = 16
NUMBER_GRID_POINTS_Z = 8

# Run controls
TOTAL_STEPS = 120
NUMBER_PSEUDOPARTICLES = 100
DOMAIN_LENGTH_X = 1.0e-2
DOMAIN_LENGTH_Y = 1.0e-2
DOMAIN_LENGTH_Z = 1.0e-2

# Uniform external magnetic field, in Tesla. B = (0, 0, 1) is deliberately
# simple so dimensional interpolation should not change the value seen by a
# particle.
MAGNETIC_FIELD_VECTOR = np.array([0.0, 0.0, 1.0], dtype=np.float32)
LINEAR_FIELD_EDGE_FRACTION = 0.25
SINUSOIDAL_FIELD_AMPLITUDE_FRACTION = 0.25
SINUSOIDAL_FIELD_WAVENUMBER = 3

# Identical initial phase space for every run.
INITIAL_X_SPAN = 0.25 * DOMAIN_LENGTH_X
INITIAL_VELOCITY = np.array([0.0, 1.0e7, 0.0])

# Deterministic initial-condition noise. Position noise is uniform, like
# JAX-in-Cell's random position initialization; velocity noise is Gaussian,
# like its thermal velocity initialization.
INITIAL_NOISE_SEED = 2026
POSITION_NOISE_FRACTION = 0.01
VELOCITY_NOISE_FRACTION = 0.01

SAVE_FIGURES = False
RUN_REGULAR_DIAGNOSTICS = True
RUN_REGULAR_PLOTS = True
REGULAR_PLOT_DIRECTION = "xy"
REGULAR_PLOT_ANIMATION_INTERVAL = 20

FIELD_PROFILES = (
    ("constant uniform B", "constant"),
    ("linear B(x)", "linear_x"),
    ("sinusoidal B(x)", "sinusoidal_x"),
)


def initial_phase_space():
    rng = np.random.default_rng(INITIAL_NOISE_SEED)
    x = np.linspace(
        -INITIAL_X_SPAN,
        INITIAL_X_SPAN,
        NUMBER_PSEUDOPARTICLES,
        dtype=float,
    )
    positions = np.stack(
        (
            x,
            np.zeros_like(x),
            np.zeros_like(x),
        ),
        axis=1,
    )
    velocities = np.repeat(
        INITIAL_VELOCITY[np.newaxis, :],
        NUMBER_PSEUDOPARTICLES,
        axis=0,
    )
    position_noise_scale = POSITION_NOISE_FRACTION * np.array(
        [DOMAIN_LENGTH_X, DOMAIN_LENGTH_Y, DOMAIN_LENGTH_Z],
        dtype=float,
    )
    positions += rng.uniform(
        low=-position_noise_scale,
        high=position_noise_scale,
        size=positions.shape,
    )

    velocity_noise_scale = VELOCITY_NOISE_FRACTION * max(
        np.linalg.norm(INITIAL_VELOCITY),
        1.0,
    )
    velocities += rng.normal(
        loc=0.0,
        scale=velocity_noise_scale,
        size=velocities.shape,
    )
    return positions, velocities


def x_grid_centers():
    dx = DOMAIN_LENGTH_X / NUMBER_GRID_POINTS_X
    return np.linspace(
        -DOMAIN_LENGTH_X / 2 + dx / 2,
        DOMAIN_LENGTH_X / 2 - dx / 2,
        NUMBER_GRID_POINTS_X,
        dtype=float,
    )


def magnetic_field_scale(profile_name):
    x = x_grid_centers()
    if profile_name == "constant":
        return np.ones_like(x)
    if profile_name == "linear_x":
        return 1.0 + LINEAR_FIELD_EDGE_FRACTION * x / (DOMAIN_LENGTH_X / 2)
    if profile_name == "sinusoidal_x":
        return (
            1.0
            + SINUSOIDAL_FIELD_AMPLITUDE_FRACTION
            * np.sin(2.0 * np.pi * SINUSOIDAL_FIELD_WAVENUMBER * x / DOMAIN_LENGTH_X)
        )
    raise ValueError(f"Unknown magnetic field profile {profile_name!r}.")


def external_fields(number_dimensions, profile_name):
    shape = [NUMBER_GRID_POINTS_X]
    if number_dimensions >= 2:
        shape.append(NUMBER_GRID_POINTS_Y)
    if number_dimensions >= 3:
        shape.append(NUMBER_GRID_POINTS_Z)

    magnetic_field_x = (
        magnetic_field_scale(profile_name)[:, np.newaxis]
        * MAGNETIC_FIELD_VECTOR[np.newaxis, :]
    ).astype(np.float32)
    if number_dimensions == 1:
        magnetic_field = magnetic_field_x
    elif number_dimensions == 2:
        magnetic_field = np.broadcast_to(
            magnetic_field_x[:, np.newaxis, :],
            (*shape, 3),
        ).copy()
    elif number_dimensions == 3:
        magnetic_field = np.broadcast_to(
            magnetic_field_x[:, np.newaxis, np.newaxis, :],
            (*shape, 3),
        ).copy()
    else:
        raise ValueError("number_dimensions must be 1, 2, or 3.")

    electric_field = np.zeros_like(magnetic_field)
    return electric_field, magnetic_field


def base_parameters():
    positions, velocities = initial_phase_space()
    base_species = {
        "number_pseudoparticles": NUMBER_PSEUDOPARTICLES,
        "grid_points_per_Debye_length": 1.0,
        "weight": 1.0,
        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
        "perturbation_wavenumber_x": 0.0,
        "perturbation_wavenumber_y": 0.0,
        "perturbation_wavenumber_z": 0.0,
        "random_positions_x": False,
        "random_positions_y": False,
        "random_positions_z": False,
        "vth_over_c_x": 0.0,
        "vth_over_c_y": 0.0,
        "vth_over_c_z": 0.0,
        "drift_speed_x": 0.0,
        "drift_speed_y": 0.0,
        "drift_speed_z": 0.0,
        "velocity_plus_minus_x": False,
        "velocity_plus_minus_y": False,
        "velocity_plus_minus_z": False,
        "initial_positions": positions,
        "initial_velocities": velocities,
    }

    return {
        "domain_parameters": {
            "length": DOMAIN_LENGTH_X,
            "length_y": DOMAIN_LENGTH_Y,
            "length_z": DOMAIN_LENGTH_Z,
            "timestep_over_spatialstep_times_c": 0.5,
            "number_grid_points": NUMBER_GRID_POINTS_X,
            "number_grid_points_y": 0,
            "number_grid_points_z": 0,
            "total_steps": TOTAL_STEPS,
        },
        "species_parameters": {
            "electrons": {
                "electrons0": {
                    **base_species,
                    "charge_over_elementary_charge": -1.0,
                },
            },
            "ions": {
                "ions0": {
                    **base_species,
                    "charge_over_elementary_charge": 1.0,
                    "mass_over_proton_mass": 1.0,
                    "ion_temperature_over_electron_temperature_x": 1.0,
                    "ion_temperature_over_electron_temperature_y": 1.0,
                    "ion_temperature_over_electron_temperature_z": 1.0,
                },
            },
        },
        "solver_parameters": {
            "field_solver": 0,
            "time_evolution_algorithm": 0,
            "relativistic": False,
            "filter_passes": 0,
            "filter_alpha": 0.5,
            "filter_strides": (1, 2, 4),
            "print_info": False,
        },
    }


def parameters_for_dimensions(number_dimensions, profile_name):
    parameters = deepcopy(base_parameters())
    if number_dimensions >= 2:
        parameters["domain_parameters"]["number_grid_points_y"] = NUMBER_GRID_POINTS_Y
    if number_dimensions >= 3:
        parameters["domain_parameters"]["number_grid_points_z"] = NUMBER_GRID_POINTS_Z

    external_electric_field, external_magnetic_field = external_fields(
        number_dimensions,
        profile_name,
    )
    parameters["external_field_parameters"] = {
        "external_electric_field": {"E": external_electric_field},
        "external_magnetic_field": {"B": external_magnetic_field},
    }
    return parameters


def run_case(label, number_dimensions, profile_name, profile_label):
    print(f"Running {label} external field for {profile_label}")
    sim = Simulation(parameters_for_dimensions(number_dimensions, profile_name))
    output = block_until_ready(sim.run())
    if RUN_REGULAR_DIAGNOSTICS:
        diagnostics(output)
    print(
        f"  B shape: {output['external_magnetic_field'].shape}; "
        f"padded B shape: {output['padded_external_magnetic_field'].shape}"
    )
    if RUN_REGULAR_PLOTS:
        plot(
            output,
            direction=REGULAR_PLOT_DIRECTION,
            animation_interval=REGULAR_PLOT_ANIMATION_INTERVAL,
        )
    return output


def electron_positions(output):
    if "position_electrons" not in output:
        return np.asarray(output["positions"])
    return np.asarray(output["position_electrons"])


def combined_positions(output):
    if "position_electrons" not in output:
        return np.asarray(output["positions"])
    return np.concatenate(
        (
            np.asarray(output["position_electrons"]),
            np.asarray(output["position_ions"]),
        ),
        axis=1,
    )


def combined_velocities(output):
    if "velocity_electrons" not in output:
        return np.asarray(output["velocities"])
    return np.concatenate(
        (
            np.asarray(output["velocity_electrons"]),
            np.asarray(output["velocity_ions"]),
        ),
        axis=1,
    )


def plot_outputs(outputs, profile_name, profile_label):
    labels = list(outputs.keys())
    reference = outputs[labels[0]]
    time = np.asarray(reference["time_array"])
    electron_index = 0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(profile_label)
    trajectory_ax = axes[0, 0]
    x_ax = axes[0, 1]
    y_ax = axes[1, 0]
    diff_ax = axes[1, 1]

    reference_positions = electron_positions(reference)[:, electron_index, :]

    for label, output in outputs.items():
        positions = electron_positions(output)[:, electron_index, :]
        trajectory_ax.plot(positions[:, 0], positions[:, 1], label=label)
        x_ax.plot(time, positions[:, 0], label=label)
        y_ax.plot(time, positions[:, 1], label=label)

        delta = np.linalg.norm(positions - reference_positions, axis=1)
        diff_ax.semilogy(time, np.maximum(delta, 1e-30), label=label)

    trajectory_ax.set_title("Electron 0 trajectory")
    trajectory_ax.set_xlabel("x")
    trajectory_ax.set_ylabel("y")
    trajectory_ax.axis("equal")

    x_ax.set_title("Electron 0 x(t)")
    x_ax.set_xlabel("time")
    x_ax.set_ylabel("x")

    y_ax.set_title("Electron 0 y(t)")
    y_ax.set_xlabel("time")
    y_ax.set_ylabel("y")

    diff_ax.set_title("Position difference from 1D run")
    diff_ax.set_xlabel("time")
    diff_ax.set_ylabel("|x_case - x_1d|")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.legend()

    if SAVE_FIGURES:
        fig.savefig(f"3d_field_runs_{profile_name}.png", dpi=150)


def print_summary(outputs, profile_label):
    labels = list(outputs.keys())
    reference_positions = combined_positions(outputs[labels[0]])
    reference_velocities = combined_velocities(outputs[labels[0]])

    print(f"\nMaximum absolute differences from the 1D run for {profile_label}:")
    for label in labels[1:]:
        positions = combined_positions(outputs[label])
        velocities = combined_velocities(outputs[label])
        max_position_delta = np.max(np.abs(positions - reference_positions))
        max_velocity_delta = np.max(np.abs(velocities - reference_velocities))
        print(
            f"  {label}: max |dx| = {max_position_delta:.3e}, "
            f"max |dv| = {max_velocity_delta:.3e}"
        )


def run_profile_suite(profile_label, profile_name):
    outputs = {
        "1D B(x)": run_case("1D", 1, profile_name, profile_label),
        "2D B(x,y)": run_case("2D", 2, profile_name, profile_label),
        "3D B(x,y,z)": run_case("3D", 3, profile_name, profile_label),
    }
    print_summary(outputs, profile_label)
    plot_outputs(outputs, profile_name, profile_label)


def main():
    for profile_label, profile_name in FIELD_PROFILES:
        run_profile_suite(profile_label, profile_name)
    plt.show()


if __name__ == "__main__":
    main()
