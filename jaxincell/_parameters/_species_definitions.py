__all__ = [
    "ALL_ION_PARAMETERS",
    "ALL_ELECTRON_PARAMETERS",
    "ALL_SPECIES_PARAMETERS",
    "SPECIES_TYPES",
    "SPECIES_AXES",
    "SPECIES_DEFAULT_LABELS",
    "SPECIES_CANONICAL_PREFIXES",
    "SPECIES_PARAMETER_KEYS",
    "SPECIES_DIFFERENTIABLE_KEYS",
    "DIFFERENTIABLE_ION_PARAMETERS",
    "DIFFERENTIABLE_ELECTRON_PARAMETERS",
    "DIFFERENTIABLE_SPECIES_PARAMETERS",
    "SPECIES_DEFAULT_PARAMETERS",
    "SPECIES_INITIAL_DEFAULT_PARAMETERS",
    "DEFAULT_INITIAL_ION_PARAMETERS",
    "DEFAULT_INITIAL_ELECTRON_PARAMETERS",
    "DEFAULT_ION_PARAMETERS",
    "DEFAULT_ELECTRON_PARAMETERS",
]

SPECIES_TYPES = ("electrons", "ions")
SPECIES_AXES = ("x", "y", "z")
SPECIES_DEFAULT_LABELS = {
    "electrons": "electrons0",
    "ions": "ions0",
}
SPECIES_CANONICAL_PREFIXES = {
    "electrons": "electrons",
    "ions": "ions",
}

DEFAULT_CHARGED_SPECIES_PARAMETERS = {
    "number_pseudoparticles": 500,
    "grid_points_per_Debye_length": 2,
    "weight": 0,
    "perturbation_amplitude_x": 0.0,
    "perturbation_amplitude_y": 0.0,
    "perturbation_amplitude_z": 0.0,
    "perturbation_wavenumber_x": 0,
    "perturbation_wavenumber_y": 0,
    "perturbation_wavenumber_z": 0,
    "random_positions_x": False,
    "random_positions_y": True,
    "random_positions_z": True,
    "vth_over_c_x": 0,
    "vth_over_c_y": 0,
    "vth_over_c_z": 0,
    "drift_speed_x": 0,
    "drift_speed_y": 0,
    "drift_speed_z": 0,
    "velocity_plus_minus_x": False,
    "velocity_plus_minus_y": False,
    "velocity_plus_minus_z": False,
    "seed_position_override": False,
    "seed_position": None,
}

DEFAULT_ELECTRON_PARAMETERS = {
    **DEFAULT_CHARGED_SPECIES_PARAMETERS,
    "charge_over_elementary_charge": -1,
}

ALL_ELECTRON_PARAMETERS = list(DEFAULT_ELECTRON_PARAMETERS.keys())

DEFAULT_ION_PARAMETERS = {
    **DEFAULT_CHARGED_SPECIES_PARAMETERS,
    "charge_over_elementary_charge": 1,
    "mass_over_proton_mass": 1,
    "ion_temperature_over_electron_temperature_x": 1,
    "ion_temperature_over_electron_temperature_y": 1,
    "ion_temperature_over_electron_temperature_z": 1,
}

ALL_ION_PARAMETERS = list(DEFAULT_ION_PARAMETERS.keys())

ALL_SPECIES_PARAMETERS = list(dict.fromkeys(ALL_ION_PARAMETERS + ALL_ELECTRON_PARAMETERS))

SPECIES_PARAMETER_KEYS = {
    "ions": ALL_ION_PARAMETERS,
    "electrons": ALL_ELECTRON_PARAMETERS,
}

DEFAULT_INITIAL_ELECTRON_PARAMETERS = {
    **DEFAULT_ELECTRON_PARAMETERS,
    "perturbation_amplitude_x": 1e-7,
    "perturbation_wavenumber_x": 8,
    "vth_over_c_x": 0.05,
    "drift_speed_x": 1e8,
    "velocity_plus_minus_x": True,
}

DEFAULT_INITIAL_ION_PARAMETERS = {
    **DEFAULT_ION_PARAMETERS,
    "perturbation_amplitude_x": 1e-7,
    "vth_over_c_x": "_electrons0",
    "vth_over_c_y": "_electrons0",
    "vth_over_c_z": "_electrons0",
}

SPECIES_DEFAULT_PARAMETERS = {
    "electrons": DEFAULT_ELECTRON_PARAMETERS,
    "ions": DEFAULT_ION_PARAMETERS,
}

SPECIES_INITIAL_DEFAULT_PARAMETERS = {
    "electrons": DEFAULT_INITIAL_ELECTRON_PARAMETERS,
    "ions": DEFAULT_INITIAL_ION_PARAMETERS,
}

DIFFERENTIABLE_CHARGED_SPECIES_PARAMETERS = [
    "grid_points_per_Debye_length",
    "weight",
    "charge_over_elementary_charge",
    "perturbation_amplitude_x",
    "perturbation_amplitude_y",
    "perturbation_amplitude_z",
    "perturbation_wavenumber_x",
    "perturbation_wavenumber_y",
    "perturbation_wavenumber_z",
    "vth_over_c_x",
    "vth_over_c_y",
    "vth_over_c_z",
    "drift_speed_x",
    "drift_speed_y",
    "drift_speed_z",
]

DIFFERENTIABLE_ELECTRON_PARAMETERS = [
    *DIFFERENTIABLE_CHARGED_SPECIES_PARAMETERS,
]

DIFFERENTIABLE_ION_PARAMETERS = [
    *DIFFERENTIABLE_CHARGED_SPECIES_PARAMETERS,
    "mass_over_proton_mass",
    "ion_temperature_over_electron_temperature_x",
    "ion_temperature_over_electron_temperature_y",
    "ion_temperature_over_electron_temperature_z",
]

DIFFERENTIABLE_SPECIES_PARAMETERS = list(dict.fromkeys(
    DIFFERENTIABLE_ION_PARAMETERS + DIFFERENTIABLE_ELECTRON_PARAMETERS
))

SPECIES_DIFFERENTIABLE_KEYS = {
    "ions": DIFFERENTIABLE_ION_PARAMETERS,
    "electrons": DIFFERENTIABLE_ELECTRON_PARAMETERS,
}
