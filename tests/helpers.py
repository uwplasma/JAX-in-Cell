import numpy as np


def scalar(value):
    return float(np.asarray(value))


def base_simulation_parameters():
    base_species = {
        "number_pseudoparticles": 2,
        "grid_points_per_Debye_length": 1.0,
        "weight": 1.0,
        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
        "perturbation_wavenumber_x": 1.0,
        "perturbation_wavenumber_y": 1.0,
        "perturbation_wavenumber_z": 1.0,
        "random_positions_x": False,
        "random_positions_y": False,
        "random_positions_z": False,
        "vth_over_c_x": 0.01,
        "vth_over_c_y": 0.01,
        "vth_over_c_z": 0.01,
        "drift_speed_x": 1.0,
        "drift_speed_y": 0.0,
        "drift_speed_z": 0.0,
        "velocity_plus_minus_x": False,
        "velocity_plus_minus_y": False,
        "velocity_plus_minus_z": False,
    }
    return {
        "domain_parameters": {
            "total_steps": 1,
            "number_grid_points": 4,
            "number_grid_points_y": 3,
            "number_grid_points_z": 3,
            "length": 0.01,
            "length_y": 0.01,
            "length_z": 0.01,
        },
        "species_parameters": {
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
            "electrons": {
                "electrons0": {
                    **base_species,
                    "charge_over_elementary_charge": -1.0,
                },
            },
        },
        "solver_parameters": {
            "filter_passes": 0,
            "filter_alpha": 0.5,
            "print_info": False,
        },
    }
