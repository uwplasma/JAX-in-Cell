import jax.numpy as jnp
from .._constants import mass_proton, mass_electron
from ._species_definitions import (
    SPECIES_AXES,
    SPECIES_CANONICAL_PREFIXES,
    SPECIES_DEFAULT_LABELS,
    SPECIES_DEFAULT_PARAMETERS,
    SPECIES_INITIAL_DEFAULT_PARAMETERS,
    SPECIES_TYPES,
)

__all__ = [
    "clean_and_initialize_species_parameters",
    "resolve_species_references",
    "build_species_hash",
]

REFERENCE_PREFIX = "_reference_"
COMMON_SPECIES_FLOAT_PARAMETERS = (
    "weight",
    "charge_over_elementary_charge",
    *(f"perturbation_amplitude_{axis}" for axis in SPECIES_AXES),
    *(f"perturbation_wavenumber_{axis}" for axis in SPECIES_AXES),
    *(f"vth_over_c_{axis}" for axis in SPECIES_AXES),
    *(f"drift_speed_{axis}" for axis in SPECIES_AXES),
)
ION_SPECIES_FLOAT_PARAMETERS = (
    "mass_over_proton_mass",
    "ion_temperature_over_electron_temperature_x",
    "ion_temperature_over_electron_temperature_y",
    "ion_temperature_over_electron_temperature_z",
)
COMMON_SPECIES_BOOLEAN_PARAMETERS = (
    *(f"random_positions_{axis}" for axis in SPECIES_AXES),
    *(f"velocity_plus_minus_{axis}" for axis in SPECIES_AXES),
    "seed_position_override",
)

def normalize_species_inputs(species_parameters, input_parameters, species_type, default_species_label):
    species_values = {**species_parameters.pop(species_type, {})}
    for key, value in input_parameters.pop(species_type, {}).items():
        if isinstance(value, dict) and isinstance(species_values.get(key), dict):
            species_values[key] = {**species_values[key], **value}
        else:
            species_values[key] = value

    if len(species_values) == 0:
        return {default_species_label: {}}
    if not any(isinstance(item, dict) for item in species_values.values()):
        return {default_species_label: {**species_values}}
    return species_values

def validate_species_parameters(species_type, species_label, species):
    assert type(species["number_pseudoparticles"]) == int and species["number_pseudoparticles"] > 0, (
        f"Number of pseudoparticles for {species_label} must be a positive integer. "
        f"Got {species['number_pseudoparticles']}."
    )
    assert species["grid_points_per_Debye_length"] > 0, (
        f"Grid points per Debye length must be positive. Got {species['grid_points_per_Debye_length']}."
    )

    for key in COMMON_SPECIES_FLOAT_PARAMETERS:
        species[key] = jnp.asarray(species[key], dtype=float)
    assert species["weight"] >= 0, f"Weight must be non-negative. Got {species['weight']}."
    #assert species["charge_over_elementary_charge"] != 0, f"Charge over elementary charge must be non-zero. Got {species['charge_over_elementary_charge']}."

    if species_type == "ions":
        for key in ION_SPECIES_FLOAT_PARAMETERS:
            species[key] = jnp.asarray(species[key], dtype=float)
        assert species["mass_over_proton_mass"] > 0, (
            f"Mass over proton mass must be positive. Got {species['mass_over_proton_mass']}."
        )
        for axis in SPECIES_AXES:
            temperature_key = f"ion_temperature_over_electron_temperature_{axis}"
            assert species[temperature_key] >= 0, (
                f"Ion temperature over electron temperature {axis} must be positive. "
                f"Got {species[temperature_key]}."
            )

    for key in COMMON_SPECIES_BOOLEAN_PARAMETERS:
        assert type(species[key]) == bool, f"{key} must be a boolean. Got {species[key]}."
    if species["seed_position_override"]:
        assert type(species["seed_position"]) == int and species["seed_position"] > 0, (
            f"Seed position must be a positive integer. Got {species['seed_position']}."
        )

def resolve_reference_value(species_parameters, species_type, species, key, reference_type, reference_species):
    sp = species_parameters[species_type][species]
    referenced_sp = species_parameters[reference_type][reference_species]
    assert type(referenced_sp[key]) != str, (
        "Cross referenced species values cannot reference another reference. "
        f"For {species_type} {species} referencing {reference_species}, got {referenced_sp[key]}."
    )

    if key.startswith("vth_over_c_"):
        axis = key.removeprefix("vth_over_c_")
        temperature_key = f"ion_temperature_over_electron_temperature_{axis}"

        if species_type == "ions" and reference_type == "electrons":
            return (
                jnp.sqrt(sp[temperature_key])
                * referenced_sp[key]
                * jnp.sqrt(mass_electron / (sp["mass_over_proton_mass"] * mass_proton))
            )

        if species_type == "electrons" and reference_type == "ions":
            return (
                jnp.sqrt(1 / referenced_sp[temperature_key])
                * referenced_sp[key]
                * jnp.sqrt(referenced_sp["mass_over_proton_mass"] * mass_proton / mass_electron)
            )

    return referenced_sp[key]

def resolve_species_references(species_parameters):
    all_ion_species = list(species_parameters["ions"].keys())
    all_electron_species = list(species_parameters["electrons"].keys())

    for species_type in SPECIES_TYPES:
        for species in species_parameters[species_type].keys():
            sp = species_parameters[species_type][species]
            for key, value in list(sp.items()):
                if key == "user_label" or key.startswith(REFERENCE_PREFIX):
                    continue
                if value in all_ion_species:
                    sp[f"{REFERENCE_PREFIX}{key}"] = ("ions", value)
                elif value in all_electron_species:
                    sp[f"{REFERENCE_PREFIX}{key}"] = ("electrons", value)
                elif type(value) == str:
                    raise ValueError(
                        f"Cross referenced species value did not reference another species. "
                        f"{species_type} {species} referenced {value} for {key}. "
                        f"Valid references are {all_ion_species} referencing ions or "
                        f"{all_electron_species} referencing electrons."
                    )

    for species_type in SPECIES_TYPES:
        for species in species_parameters[species_type].keys():
            sp = species_parameters[species_type][species]
            for metadata_key, metadata_value in list(sp.items()):
                if not metadata_key.startswith(REFERENCE_PREFIX):
                    continue
                key = metadata_key.removeprefix(REFERENCE_PREFIX)
                reference_type, reference_species = metadata_value
                sp[key] = resolve_reference_value(
                    species_parameters,
                    species_type,
                    species,
                    key,
                    reference_type,
                    reference_species,
                )
    return species_parameters

def clean_and_initialize_species_parameters(species_parameters, input_parameters={}):
    for species_type in SPECIES_TYPES:
        species_values = normalize_species_inputs(
            species_parameters,
            input_parameters,
            species_type,
            SPECIES_DEFAULT_LABELS[species_type],
        )

        canonical_species = {}
        for index, (user_label, values) in enumerate(species_values.items()):
            defaults = (
                SPECIES_INITIAL_DEFAULT_PARAMETERS[species_type]
                if index == 0
                else SPECIES_DEFAULT_PARAMETERS[species_type]
            )
            canonical_label = f"_{SPECIES_CANONICAL_PREFIXES[species_type]}{index}"
            canonical_species[canonical_label] = {
                **defaults,
                **values,
                "user_label": user_label,
            }
        species_parameters[species_type] = canonical_species

    resolve_species_references(species_parameters)

    # Now do all the assert statements to check that the parameters are valid
    for species_type in SPECIES_TYPES:
        for species in species_parameters[species_type].keys():
            validate_species_parameters(
                species_type,
                species,
                species_parameters[species_type][species],
            )
    
    return species_parameters

def build_species_hash(species_parameters):
    hash_list_pre_species = ['nonspeciesdata']
    for key, value in species_parameters.items():
        if key not in SPECIES_TYPES:
            hash_list_pre_species.append(str(key))
            hash_list_pre_species.append(str(value))
    hash_pre_species = "".join(hash_list_pre_species)

    species_hashes = []
    for species_type in SPECIES_TYPES:
        species_hash_list = [species_type]
        for species_label, sp in species_parameters[species_type].items():
            species_hash = [str(species_label)]
            for key, value in sp.items():
                species_hash.append(str(key))
                species_hash.append(str(value))
            species_hash_list.append("".join(species_hash))
        species_hashes.append(str(tuple(species_hash_list)))

    return str((hash_pre_species, *species_hashes))
