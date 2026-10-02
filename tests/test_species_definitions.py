from jaxincell._parameters._species_definitions import (
    ALL_ELECTRON_PARAMETERS,
    ALL_ION_PARAMETERS,
    ALL_SPECIES_PARAMETERS,
    DEFAULT_ELECTRON_PARAMETERS,
    DEFAULT_INITIAL_ELECTRON_PARAMETERS,
    DEFAULT_INITIAL_ION_PARAMETERS,
    DEFAULT_ION_PARAMETERS,
    DIFFERENTIABLE_ELECTRON_PARAMETERS,
    DIFFERENTIABLE_ION_PARAMETERS,
    DIFFERENTIABLE_SPECIES_PARAMETERS,
    SPECIES_AXES,
    SPECIES_CANONICAL_PREFIXES,
    SPECIES_DEFAULT_LABELS,
    SPECIES_DEFAULT_PARAMETERS,
    SPECIES_DIFFERENTIABLE_KEYS,
    SPECIES_INITIAL_DEFAULT_PARAMETERS,
    SPECIES_PARAMETER_KEYS,
    SPECIES_TYPES,
)


EXPECTED_SHARED_SPECIES_KEYS = [
    "number_pseudoparticles",
    "grid_points_per_Debye_length",
    "weight",
    "charge_over_elementary_charge",
    "seed_position_override",
    "seed_position",
    "initial_positions",
    "initial_velocities",
]
EXPECTED_SHARED_AXIS_PREFIXES = [
    "perturbation_amplitude",
    "perturbation_wavenumber",
    "random_positions",
    "vth_over_c",
    "drift_speed",
    "velocity_plus_minus",
]
ION_ONLY_KEYS = [
    "mass_over_proton_mass",
    "ion_temperature_over_electron_temperature_x",
    "ion_temperature_over_electron_temperature_y",
    "ion_temperature_over_electron_temperature_z",
]
EXPECTED_SHARED_DIFFERENTIABLE_KEYS = [
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
    "initial_positions",
    "initial_velocities",
]
EXPECTED_DIFFERENTIABLE_ELECTRON_KEYS = [
    *EXPECTED_SHARED_DIFFERENTIABLE_KEYS,
]
EXPECTED_DIFFERENTIABLE_ION_KEYS = [
    *EXPECTED_SHARED_DIFFERENTIABLE_KEYS,
    "mass_over_proton_mass",
    "ion_temperature_over_electron_temperature_x",
    "ion_temperature_over_electron_temperature_y",
    "ion_temperature_over_electron_temperature_z",
]
EXPECTED_DIFFERENTIABLE_SPECIES_KEYS = list(
    dict.fromkeys(EXPECTED_DIFFERENTIABLE_ION_KEYS + EXPECTED_DIFFERENTIABLE_ELECTRON_KEYS)
)
NONDIFFERENTIABLE_SPECIES_KEYS = [
    "number_pseudoparticles",
    "random_positions_x",
    "random_positions_y",
    "random_positions_z",
    "velocity_plus_minus_x",
    "velocity_plus_minus_y",
    "velocity_plus_minus_z",
    "seed_position_override",
    "seed_position",
]


def axis_keys(prefix):
    return [f"{prefix}_{axis}" for axis in SPECIES_AXES]


def test_species_default_maps_are_consistent_with_species_types():
    """Test jaxincell._parameters._species_definitions default maps.

    Cases covered:
    - SPECIES_TYPES keeps electrons before ions so electron reference initialization is available.
    - every species type has default and initial-default entries.
    - charged species shared defaults are present for ions and electrons.
    - ion-only and electron-only defaults are confined to the intended species type.
    """
    assert SPECIES_TYPES == ("electrons", "ions")
    assert SPECIES_AXES == ("x", "y", "z")
    assert SPECIES_DEFAULT_LABELS == {
        "electrons": "electrons0",
        "ions": "ions0",
    }
    assert SPECIES_CANONICAL_PREFIXES == {
        "electrons": "electrons",
        "ions": "ions",
    }
    assert SPECIES_TYPES.index("electrons") < SPECIES_TYPES.index("ions")

    for mapping in [
        SPECIES_DEFAULT_LABELS,
        SPECIES_CANONICAL_PREFIXES,
        SPECIES_DEFAULT_PARAMETERS,
        SPECIES_INITIAL_DEFAULT_PARAMETERS,
        SPECIES_PARAMETER_KEYS,
        SPECIES_DIFFERENTIABLE_KEYS,
    ]:
        assert set(mapping) == set(SPECIES_TYPES)

    assert SPECIES_DEFAULT_PARAMETERS["electrons"] is DEFAULT_ELECTRON_PARAMETERS
    assert SPECIES_DEFAULT_PARAMETERS["ions"] is DEFAULT_ION_PARAMETERS
    assert SPECIES_INITIAL_DEFAULT_PARAMETERS["electrons"] is DEFAULT_INITIAL_ELECTRON_PARAMETERS
    assert SPECIES_INITIAL_DEFAULT_PARAMETERS["ions"] is DEFAULT_INITIAL_ION_PARAMETERS

    for key in EXPECTED_SHARED_SPECIES_KEYS:
        assert key in DEFAULT_ELECTRON_PARAMETERS
        assert key in DEFAULT_ION_PARAMETERS
    for prefix in EXPECTED_SHARED_AXIS_PREFIXES:
        for key in axis_keys(prefix):
            assert key in DEFAULT_ELECTRON_PARAMETERS
            assert key in DEFAULT_ION_PARAMETERS

    for key in ION_ONLY_KEYS:
        assert key in DEFAULT_ION_PARAMETERS
        assert key not in DEFAULT_ELECTRON_PARAMETERS


def test_species_parameter_key_lists_match_default_dictionaries():
    """Test species parameter key lists.

    Cases covered:
    - ALL_ION_PARAMETERS matches ion defaults.
    - ALL_ELECTRON_PARAMETERS matches electron defaults.
    - shared, ion-only, and electron-only groups compose without accidental omissions.
    - SPECIES_PARAMETER_KEYS points each species type to the matching all-key list.
    """
    assert set(ALL_ION_PARAMETERS) == set(DEFAULT_ION_PARAMETERS)
    assert set(ALL_ELECTRON_PARAMETERS) == set(DEFAULT_ELECTRON_PARAMETERS)
    assert set(ALL_SPECIES_PARAMETERS) == set(ALL_ION_PARAMETERS) | set(ALL_ELECTRON_PARAMETERS)

    assert SPECIES_PARAMETER_KEYS["ions"] is ALL_ION_PARAMETERS
    assert SPECIES_PARAMETER_KEYS["electrons"] is ALL_ELECTRON_PARAMETERS

    for key in EXPECTED_SHARED_SPECIES_KEYS:
        assert key in ALL_SPECIES_PARAMETERS
    for prefix in EXPECTED_SHARED_AXIS_PREFIXES:
        for key in axis_keys(prefix):
            assert key in ALL_SPECIES_PARAMETERS
    for key in ION_ONLY_KEYS:
        assert key in ALL_ION_PARAMETERS
        assert key in ALL_SPECIES_PARAMETERS
        assert key not in ALL_ELECTRON_PARAMETERS


def test_species_differentiable_keys_exist_in_species_defaults():
    """Test species differentiable key definitions.

    Cases covered:
    - every differentiable ion key exists in ion defaults.
    - every differentiable electron key exists in electron defaults.
    - differentiable shared charged-species keys are included for both species types.
    - ion-only differentiable keys are included only for ions.
    - structural species keys are not exposed as differentiable inputs.
    """
    assert DIFFERENTIABLE_ELECTRON_PARAMETERS == EXPECTED_DIFFERENTIABLE_ELECTRON_KEYS
    assert DIFFERENTIABLE_ION_PARAMETERS == EXPECTED_DIFFERENTIABLE_ION_KEYS
    assert DIFFERENTIABLE_SPECIES_PARAMETERS == EXPECTED_DIFFERENTIABLE_SPECIES_KEYS

    assert SPECIES_DIFFERENTIABLE_KEYS == {
        "ions": DIFFERENTIABLE_ION_PARAMETERS,
        "electrons": DIFFERENTIABLE_ELECTRON_PARAMETERS,
    }

    for key in DIFFERENTIABLE_ION_PARAMETERS:
        assert key in DEFAULT_ION_PARAMETERS
    for key in DIFFERENTIABLE_ELECTRON_PARAMETERS:
        assert key in DEFAULT_ELECTRON_PARAMETERS

    for key in EXPECTED_SHARED_DIFFERENTIABLE_KEYS:
        assert key in DIFFERENTIABLE_ION_PARAMETERS
        assert key in DIFFERENTIABLE_ELECTRON_PARAMETERS

    for key in ION_ONLY_KEYS:
        assert key in DIFFERENTIABLE_ION_PARAMETERS
        assert key not in DIFFERENTIABLE_ELECTRON_PARAMETERS

    for key in NONDIFFERENTIABLE_SPECIES_KEYS:
        assert key not in DIFFERENTIABLE_ION_PARAMETERS
        assert key not in DIFFERENTIABLE_ELECTRON_PARAMETERS
        assert key not in DIFFERENTIABLE_SPECIES_PARAMETERS


def test_initial_species_defaults_preserve_required_two_stream_overrides():
    """Test initial species defaults.

    Cases covered:
    - initial ions and electrons preserve the intended two-stream defaults.
    - initial defaults contain exactly the same keys as the plain defaults.
    - species default maps keep plain and initial defaults separate.
    """
    assert set(DEFAULT_INITIAL_ELECTRON_PARAMETERS) == set(DEFAULT_ELECTRON_PARAMETERS)
    assert set(DEFAULT_INITIAL_ION_PARAMETERS) == set(DEFAULT_ION_PARAMETERS)

    electron_overrides = {
        "perturbation_amplitude_x": 1e-7,
        "perturbation_wavenumber_x": 8,
        "vth_over_c_x": 0.05,
        "drift_speed_x": 1e8,
        "velocity_plus_minus_x": True,
    }
    for key, value in electron_overrides.items():
        assert DEFAULT_INITIAL_ELECTRON_PARAMETERS[key] == value
        assert DEFAULT_INITIAL_ELECTRON_PARAMETERS[key] != DEFAULT_ELECTRON_PARAMETERS[key]
    for key in set(DEFAULT_ELECTRON_PARAMETERS) - set(electron_overrides):
        assert DEFAULT_INITIAL_ELECTRON_PARAMETERS[key] == DEFAULT_ELECTRON_PARAMETERS[key]

    ion_overrides = {
        "perturbation_amplitude_x": 1e-7,
        "vth_over_c_x": "_electrons0",
        "vth_over_c_y": "_electrons0",
        "vth_over_c_z": "_electrons0",
    }
    for key, value in ion_overrides.items():
        assert DEFAULT_INITIAL_ION_PARAMETERS[key] == value
        assert DEFAULT_INITIAL_ION_PARAMETERS[key] != DEFAULT_ION_PARAMETERS[key]
    for key in set(DEFAULT_ION_PARAMETERS) - set(ion_overrides):
        assert DEFAULT_INITIAL_ION_PARAMETERS[key] == DEFAULT_ION_PARAMETERS[key]

    assert SPECIES_DEFAULT_PARAMETERS["electrons"] is not SPECIES_INITIAL_DEFAULT_PARAMETERS["electrons"]
    assert SPECIES_DEFAULT_PARAMETERS["ions"] is not SPECIES_INITIAL_DEFAULT_PARAMETERS["ions"]
