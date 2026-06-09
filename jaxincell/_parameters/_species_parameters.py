import jax.numpy as jnp
from .._constants import mass_proton, mass_electron
from .._utils import as_float_parameter
from ._species_definitions import (
    ALL_ELECTRON_PARAMETERS,
    ALL_ION_PARAMETERS,
    ALL_LEGACY_SPECIES_PARAMETERS,
    ALL_SPECIES_PARAMETERS,
    DEFAULT_ELECTRON_PARAMETERS,
    DEFAULT_ION_PARAMETERS,
    DEFAULT_LEGACY_SPECIES_INPUT_PARAMETERS,
    DIFFERENTIABLE_ELECTRON_PARAMETERS,
    DIFFERENTIABLE_ION_PARAMETERS,
    DIFFERENTIABLE_LEGACY_SPECIES_PARAMETERS,
    DIFFERENTIABLE_SPECIES_PARAMETERS,
    LEGACY_SPECIES_PARAMETER_ROUTES,
    SPECIES_AXES,
    SPECIES_DIFFERENTIABLE_KEYS,
    SPECIES_PARAMETER_KEYS,
    SPECIES_TYPES,
)
from ._species_definitions import __all__ as _species_definition_exports

__all__ = [
    name for name in _species_definition_exports if not name.startswith("DEFAULT_")
] + [
    "clean_and_initialize_species_parameters",
    "build_species_hash",
]

def merge_species_parameter_dicts(species_parameters, input_parameters):
    species_parameters = {**species_parameters}
    for key, value in input_parameters.items():
        if isinstance(value, dict) and isinstance(species_parameters.get(key), dict):
            species_parameters[key] = {**species_parameters[key], **value}
        else:
            species_parameters[key] = value
    return species_parameters

def warn_legacy_species_input():
    print(
        "Warning: You are using legacy input methods for species parameters. "
        "Please update your input file to use the new nested structure for species parameters. "
        "See the documentation for more details."
    )

def extract_legacy_species_input(species_parameters, input_parameters):
    legacy_parameters = {**DEFAULT_LEGACY_SPECIES_INPUT_PARAMETERS}
    legacy_was_used = False

    for key in legacy_parameters.keys():
        if key in species_parameters.keys():
            legacy_parameters[key] = species_parameters.pop(key)
            legacy_was_used = True
        if key in input_parameters.keys():
            legacy_parameters[key] = input_parameters.pop(key)
            legacy_was_used = True

    if legacy_was_used:
        warn_legacy_species_input()

    return legacy_parameters, legacy_was_used

def build_legacy_ion_parameters(legacy_parameters):
    return {
        "number_pseudoparticles": legacy_parameters["number_pseudoelectrons"],
        "grid_points_per_Debye_length": legacy_parameters["grid_points_per_Debye_length"],
        "weight": legacy_parameters["weight"],
        "charge_over_elementary_charge": legacy_parameters["ion_charge_over_elementary_charge"],
        "mass_over_proton_mass": legacy_parameters["ion_mass_over_proton_mass"],
        "perturbation_amplitude_x": legacy_parameters["amplitude_perturbation_x"],
        "perturbation_amplitude_y": legacy_parameters["amplitude_perturbation_y"],
        "perturbation_amplitude_z": legacy_parameters["amplitude_perturbation_z"],
        "perturbation_wavenumber_x": legacy_parameters["wavenumber_ions_x"],
        "perturbation_wavenumber_y": legacy_parameters["wavenumber_ions_y"],
        "perturbation_wavenumber_z": legacy_parameters["wavenumber_ions_z"],
        "random_positions_x": legacy_parameters["random_positions_x"],
        "random_positions_y": legacy_parameters["random_positions_y"],
        "random_positions_z": legacy_parameters["random_positions_z"],
        "vth_over_c_x": '_electrons0',
        "vth_over_c_y": '_electrons0',
        "vth_over_c_z": '_electrons0',
        "ion_temperature_over_electron_temperature_x": legacy_parameters["ion_temperature_over_electron_temperature_x"],
        "ion_temperature_over_electron_temperature_y": legacy_parameters["ion_temperature_over_electron_temperature_y"],
        "ion_temperature_over_electron_temperature_z": legacy_parameters["ion_temperature_over_electron_temperature_z"],
        "drift_speed_x": legacy_parameters["ion_drift_speed_x"],
        "drift_speed_y": legacy_parameters["ion_drift_speed_y"],
        "drift_speed_z": legacy_parameters["ion_drift_speed_z"],
        "velocity_plus_minus_x": legacy_parameters["velocity_plus_minus_ions_x"],
        "velocity_plus_minus_y": legacy_parameters["velocity_plus_minus_ions_y"],
        "velocity_plus_minus_z": legacy_parameters["velocity_plus_minus_ions_z"],
        "seed_position_override": False,
        "seed_position": None,
    }

def build_legacy_electron_parameters(legacy_parameters):
    return {
        "number_pseudoparticles": legacy_parameters["number_pseudoelectrons"],
        "grid_points_per_Debye_length": legacy_parameters["grid_points_per_Debye_length"],
        "weight": legacy_parameters["weight"],
        "charge_over_elementary_charge": legacy_parameters["electron_charge_over_elementary_charge"],
        "perturbation_amplitude_x": legacy_parameters["amplitude_perturbation_x"],
        "perturbation_amplitude_y": legacy_parameters["amplitude_perturbation_y"],
        "perturbation_amplitude_z": legacy_parameters["amplitude_perturbation_z"],
        "perturbation_wavenumber_x": legacy_parameters["wavenumber_electrons_x"],
        "perturbation_wavenumber_y": legacy_parameters["wavenumber_electrons_y"],
        "perturbation_wavenumber_z": legacy_parameters["wavenumber_electrons_z"],
        "random_positions_x": legacy_parameters["random_positions_x"],
        "random_positions_y": legacy_parameters["random_positions_y"],
        "random_positions_z": legacy_parameters["random_positions_z"],
        "vth_over_c_x": legacy_parameters["vth_electrons_over_c_x"],
        "vth_over_c_y": legacy_parameters["vth_electrons_over_c_y"],
        "vth_over_c_z": legacy_parameters["vth_electrons_over_c_z"],
        "drift_speed_x": legacy_parameters["electron_drift_speed_x"],
        "drift_speed_y": legacy_parameters["electron_drift_speed_y"],
        "drift_speed_z": legacy_parameters["electron_drift_speed_z"],
        "velocity_plus_minus_x": legacy_parameters["velocity_plus_minus_electrons_x"],
        "velocity_plus_minus_y": legacy_parameters["velocity_plus_minus_electrons_y"],
        "velocity_plus_minus_z": legacy_parameters["velocity_plus_minus_electrons_z"],
        "seed_position_override": False,
        "seed_position": None,
    }

def merge_nested_species_inputs(species_parameters, input_parameters, species_type):
    return merge_species_parameter_dicts(
        species_parameters.pop(species_type, {}),
        input_parameters.pop(species_type, {}),
    )

def pack_loose_species_parameters(species_parameters, default_species_label):
    has_nested_species = any(isinstance(item, dict) for item in species_parameters.values())
    if not has_nested_species and len(species_parameters) > 0:
        return {default_species_label: {**species_parameters}}
    return species_parameters

def merge_legacy_species(species_parameters, legacy_parameters, legacy_was_used, legacy_species_label):
    if not legacy_was_used:
        if len(species_parameters) > 0:
            first_key = next(iter(species_parameters.keys()))
            species_parameters[first_key] = {**legacy_parameters, **species_parameters[first_key]}
        else:
            species_parameters = {legacy_species_label: legacy_parameters}
    else:
        species_parameters[legacy_species_label] = legacy_parameters
    return species_parameters

def apply_species_defaults(species_parameters, default_parameters):
    for key in species_parameters.keys():
        species_parameters[key] = {**default_parameters, **species_parameters[key]}
    return species_parameters

def canonicalize_species_keys(species_parameters, species_prefix, legacy_species_label):
    species_parameters_out = {}
    if legacy_species_label in species_parameters.keys():
        species_parameters_out[f'_{species_prefix}0'] = species_parameters.pop(legacy_species_label)
        species_parameters_out[f'_{species_prefix}0']['user_label'] = legacy_species_label
        ind = 1
    else:
        ind = 0

    for key in species_parameters.keys():
        species_parameters_out[f'_{species_prefix}{ind}'] = species_parameters[key]
        species_parameters_out[f'_{species_prefix}{ind}']['user_label'] = key
        ind += 1
    return species_parameters_out

def clean_and_initialize_species_parameters(species_parameters, input_parameters={}):
    (
        legacy_parameters,
        legacy_was_used,
    ) = extract_legacy_species_input(species_parameters, input_parameters)

    legacy_ions = build_legacy_ion_parameters(legacy_parameters)
    legacy_electrons = build_legacy_electron_parameters(legacy_parameters)

    ions_dict = merge_nested_species_inputs(species_parameters, input_parameters, "ions")
    electrons_dict = merge_nested_species_inputs(species_parameters, input_parameters, "electrons")

    ions_dict = pack_loose_species_parameters(ions_dict, "ions0")
    electrons_dict = pack_loose_species_parameters(electrons_dict, "electrons0")

    ions_dict = merge_legacy_species(
        ions_dict,
        legacy_ions,
        legacy_was_used,
        "_legacy_ions",
    )
    electrons_dict = merge_legacy_species(
        electrons_dict,
        legacy_electrons,
        legacy_was_used,
        "_legacy_electrons",
    )

    ions_dict = apply_species_defaults(ions_dict, DEFAULT_ION_PARAMETERS)
    electrons_dict = apply_species_defaults(electrons_dict, DEFAULT_ELECTRON_PARAMETERS)

    species_parameters["ions"] = canonicalize_species_keys(
        ions_dict,
        "ions",
        "_legacy_ions",
    )
    species_parameters["electrons"] = canonicalize_species_keys(
        electrons_dict,
        "electrons",
        "_legacy_electrons",
    )

    # Now fill in all the cross referenced information
    all_ion_species = []
    all_electron_species = []
    for species in species_parameters['ions'].keys():
        all_ion_species.append(species)
    for species in species_parameters['electrons'].keys():
        all_electron_species.append(species)
    for type_species in ['ions', 'electrons']:
        for species in species_parameters[type_species].keys():
            sp = species_parameters[type_species][species]
            for key, value in sp.items():
                if value in all_ion_species and not key == 'user_label':
                    assert type(species_parameters['ions'][value][key]) != str, f"Cross referenced species values cannot reference another reference. For {type_species} {species} referencing {value}, got {species_parameters['ions'][key]}."
                    if key == "vth_over_c_x":
                        sp[key] = jnp.sqrt(1 / sp['ion_temperature_over_electron_temperature_x']) * species_parameters['ions'][value][key] * jnp.sqrt(species_parameters['ions'][value]['mass_over_proton_mass'] * mass_proton / mass_electron)
                    elif key == "vth_over_c_y":
                        sp[key] = jnp.sqrt(1 / sp['ion_temperature_over_electron_temperature_y']) * species_parameters['ions'][value][key] * jnp.sqrt(species_parameters['ions'][value]['mass_over_proton_mass'] * mass_proton / mass_electron)
                    elif key == "vth_over_c_z":
                        sp[key] = jnp.sqrt(1 / sp['ion_temperature_over_electron_temperature_z']) * species_parameters['ions'][value][key] * jnp.sqrt(species_parameters['ions'][value]['mass_over_proton_mass'] * mass_proton / mass_electron)
                    else:
                        sp[key] = species_parameters['ions'][value][key]
                elif value in all_electron_species and not key == 'user_label':
                    assert type(species_parameters['electrons'][value][key]) != str, f"Cross referenced species values cannot reference another reference. For {type_species} {species} referencing {value}, got {species_parameters['electrons'][key]}."
                    if key == "vth_over_c_x":
                        sp[key] = jnp.sqrt(sp['ion_temperature_over_electron_temperature_x']) * species_parameters['electrons'][value][key] * jnp.sqrt(mass_electron / (sp['mass_over_proton_mass'] * mass_proton))
                    elif key == "vth_over_c_y":
                        sp[key] = jnp.sqrt(sp['ion_temperature_over_electron_temperature_y']) * species_parameters['electrons'][value][key] * jnp.sqrt(mass_electron / (sp['mass_over_proton_mass'] * mass_proton))
                    elif key == "vth_over_c_z":
                        sp[key] = jnp.sqrt(sp['ion_temperature_over_electron_temperature_z']) * species_parameters['electrons'][value][key] * jnp.sqrt(mass_electron / (sp['mass_over_proton_mass'] * mass_proton))
                    else:
                        sp[key] = species_parameters['ions'][value][key]
                elif type(value) == str and not key == 'user_label':
                    raise ValueError(f'Cross referenced species value did not reference another species. {type_species} {species} referenced {value} for {key}. Valid references are {all_ion_species} referencing ions or {all_electron_species} referencing electrons.')

    # Now do all the assert statements to check that the parameters are valid
    for type_species in ['ions', 'electrons']:
        for species in species_parameters[type_species].keys():
            sp = species_parameters[type_species][species]
            assert type(sp["number_pseudoparticles"]) == int and sp["number_pseudoparticles"] > 0, f"Number of pseudoparticles for {species} must be a positive integer. Got {sp['number_pseudoparticles']}."
            assert sp["grid_points_per_Debye_length"] > 0, f"Grid points per Debye length must be positive. Got {sp['grid_points_per_Debye_length']}."
            sp["weight"] = as_float_parameter(sp["weight"])
            assert sp["weight"] >= 0, f"Weight must be non-negative. Got {sp['weight']}."
            sp["charge_over_elementary_charge"] = as_float_parameter(sp["charge_over_elementary_charge"])
            #assert sp["charge_over_elementary_charge"] != 0, f"Charge over elementary charge must be non-zero. Got {sp['charge_over_elementary_charge']}."
            if type_species == 'ions':
                sp["mass_over_proton_mass"] = as_float_parameter(sp["mass_over_proton_mass"])
                assert sp["mass_over_proton_mass"] > 0, f"Mass over proton mass must be positive. Got {sp['mass_over_proton_mass']}."
                sp["ion_temperature_over_electron_temperature_x"] = as_float_parameter(sp["ion_temperature_over_electron_temperature_x"])
                assert sp["ion_temperature_over_electron_temperature_x"] >=0, f"Ion temperature over electron temperature x must be positive. Got {sp["ion_temperature_over_electron_temperature_x"]}."
                sp["ion_temperature_over_electron_temperature_y"] = as_float_parameter(sp["ion_temperature_over_electron_temperature_y"])
                assert sp["ion_temperature_over_electron_temperature_y"] >=0, f"Ion temperature over electron temperature y must be positive. Got {sp["ion_temperature_over_electron_temperature_y"]}."
                sp["ion_temperature_over_electron_temperature_z"] = as_float_parameter(sp["ion_temperature_over_electron_temperature_z"])
                assert sp["ion_temperature_over_electron_temperature_z"] >=0, f"Ion temperature over electron temperature z must be positive. Got {sp["ion_temperature_over_electron_temperature_z"]}."
            sp["perturbation_amplitude_x"] = as_float_parameter(sp["perturbation_amplitude_x"])
            sp["perturbation_amplitude_y"] = as_float_parameter(sp["perturbation_amplitude_y"])
            sp["perturbation_amplitude_z"] = as_float_parameter(sp["perturbation_amplitude_z"])
            sp["perturbation_wavenumber_x"] = as_float_parameter(sp["perturbation_wavenumber_x"])
            sp["perturbation_wavenumber_y"] = as_float_parameter(sp["perturbation_wavenumber_y"])
            sp["perturbation_wavenumber_z"] = as_float_parameter(sp["perturbation_wavenumber_z"])
            assert type(sp["random_positions_x"]) == bool, f"Random position x must be a boolean. Got {sp["random_positions_x"]}."
            assert type(sp["random_positions_y"]) == bool, f"Random position y must be a boolean. Got {sp["random_positions_y"]}."
            assert type(sp["random_positions_z"]) == bool, f"Random position z must be a boolean. Got {sp["random_positions_z"]}."
            sp["vth_over_c_x"] = as_float_parameter(sp["vth_over_c_x"])
            # Assert positive or less than the speed of light?
            sp["vth_over_c_y"] = as_float_parameter(sp["vth_over_c_y"])
            #
            sp["vth_over_c_z"] = as_float_parameter(sp["vth_over_c_z"])
            #
            sp["drift_speed_x"] = as_float_parameter(sp["drift_speed_x"])
            # Assert less than the speed of light?
            sp["drift_speed_y"] = as_float_parameter(sp["drift_speed_y"])
            # Assert less than the speed of light?
            sp["drift_speed_z"] = as_float_parameter(sp["drift_speed_z"])
            # Assert less than the speed of light?
            assert type(sp["velocity_plus_minus_x"]) == bool, f"Velocity plus minus x must be a boolean. Got {sp["velocity_plus_minus_x"]}."
            assert type(sp["velocity_plus_minus_y"]) == bool, f"Velocity plus minus y must be a boolean. Got {sp["velocity_plus_minus_y"]}."
            assert type(sp["velocity_plus_minus_z"]) == bool, f"Velocity plus minus z must be a boolean. Got {sp["velocity_plus_minus_z"]}."
            assert type(sp["seed_position_override"]) == bool, f"Seed position override must be a boolean. Got {sp["seed_position_override"]}."
            if sp["seed_position_override"]:
                assert type(sp["seed_position"]) == int and sp["seed_position"] > 0, f"Seed position must be a positive integer. Got {sp["seed_position"]}."
    
    return species_parameters

def build_species_hash(species_parameters):
    hash_list_pre_species = ['nonspeciesdata']
    for key, value in species_parameters.items():
        if key not in ['ions', 'electrons']:
            hash_list_pre_species.append(str(key))
            hash_list_pre_species.append(str(value))
    hash_pre_species = "".join(hash_list_pre_species)

    ion_hash_list_list = ['ions']
    for key in species_parameters['ions']:
        ion_hash_list = [str(key)]
        sp = species_parameters['ions'][key]
        for key, value in sp.items():
            ion_hash_list.append(str(key))
            ion_hash_list.append(str(value))
        ion_hash = "".join(ion_hash_list)
        ion_hash_list_list.append(ion_hash)
    ion_hash = str(tuple(ion_hash_list_list))
    
    electron_hash_list_list = ['electrons']
    for key in species_parameters['electrons']:
        electron_hash_list = [str(key)]
        sp = species_parameters['electrons'][key]
        for key, value in sp.items():
            electron_hash_list.append(str(key))
            electron_hash_list.append(str(value))
        electron_hash = "".join(electron_hash_list)
        electron_hash_list_list.append(electron_hash)
    electron_hash = str(tuple(electron_hash_list_list))

    species_hash = str((hash_pre_species, ion_hash, electron_hash))
    return species_hash
