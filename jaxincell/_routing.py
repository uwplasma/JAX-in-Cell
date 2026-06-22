import jax.numpy as jnp
from copy import deepcopy

from ._parameters._sections import (
    FLAT_DIFFERENTIABLE_INPUT_PARAMETERS,
    PARAMETER_SECTION_KEYS,
    PARAMETER_SECTIONS,
)
from ._parameters._species_definitions import (
    SPECIES_DIFFERENTIABLE_KEYS,
    SPECIES_PARAMETER_KEYS,
    SPECIES_TYPES,
)
__all__ = [
    "build_runtime_flat_parameter_routes",
    "build_runtime_parameter_sections",
    "build_runtime_species_label_routes",
    "clean_runtime_input_parameters",
    "iter_species_parameter_groups",
    "merge_parameter_trees",
    "put_parameter_path",
    "route_flat_initial_parameters",
    "route_nested_initial_species_parameters",
]

def merge_parameter_trees(base_parameters, override_parameters):
    '''
    Recursively merge two parameter trees, with override_parameters taking precedence over base_parameters.
    '''
    merged_parameters = deepcopy(base_parameters)
    for key, value in override_parameters.items():
        if isinstance(value, dict) and isinstance(merged_parameters.get(key), dict):
            merged_parameters[key] = merge_parameter_trees(merged_parameters[key], value)
        else:
            merged_parameters[key] = deepcopy(value)
    return merged_parameters

def put_parameter_path(container, path, value):
    for key in path[:-1]:
        container = container.setdefault(key, {})
    container[path[-1]] = value

def _get_initial_species_target_labels(parameters, species_type):
    species_values = parameters.get("species_parameters", {}).get(species_type, {})
    if not isinstance(species_values, dict):
        return (None,)
    if not any(isinstance(value, dict) for value in species_values.values()):
        return (None,)
    return tuple(
        species_label
        for species_label, species_parameters in species_values.items()
        if isinstance(species_parameters, dict)
    )

def iter_species_parameter_groups(species_type, species_values, strict=False):
    '''
    Iterate over nested or loose species parameter groups.
    '''
    if not isinstance(species_values, dict):
        if strict:
            raise TypeError(f"Runtime input parameter '{species_type}' must be a dictionary.")
        return []
    if not any(isinstance(value, dict) for value in species_values.values()):
        return [(None, species_values)]
    if not strict:
        return species_values.items()

    species_groups = []
    for species_label, grouped_species_values in species_values.items():
        if not isinstance(grouped_species_values, dict):
            raise TypeError(
                f"Runtime input parameter '{species_type}.{species_label}' must be a dictionary."
            )
        species_groups.append((species_label, grouped_species_values))
    return species_groups

def route_nested_initial_species_parameters(input_parameters, parameters, differentiable_parameters, cleaner_input_parameters):
    for species_type in SPECIES_TYPES:
        for species_label, species_values in iter_species_parameter_groups(species_type, input_parameters.get(species_type, {})):
            target_species_labels = (
                (species_label,)
                if species_label is not None
                else _get_initial_species_target_labels(parameters, species_type)
            )
            for key, value in species_values.items():
                input_path = (
                    (species_type, key)
                    if species_label is None
                    else (species_type, species_label, key)
                )
                target_paths = tuple(
                    (species_type, key)
                    if target_species_label is None
                    else (species_type, target_species_label, key)
                    for target_species_label in target_species_labels
                )
                if key in SPECIES_DIFFERENTIABLE_KEYS.get(species_type, ()):
                    put_parameter_path(differentiable_parameters, input_path, jnp.asarray(value, dtype=float))
                    for target_path in target_paths:
                        put_parameter_path(cleaner_input_parameters, target_path, value)
                elif key in SPECIES_PARAMETER_KEYS.get(species_type, ()):
                    species_parameters = parameters.setdefault("species_parameters", {})
                    for target_path in target_paths:
                        put_parameter_path(species_parameters, target_path, value)
                        put_parameter_path(cleaner_input_parameters, target_path, value)
                else:
                    for target_path in target_paths:
                        put_parameter_path(cleaner_input_parameters, target_path, value)

def route_flat_initial_parameters(input_parameters, parameters, differentiable_parameters, cleaner_input_parameters):
    unrouted_input_parameters = {}

    for key, value in input_parameters.items():
        if key in SPECIES_TYPES:
            continue
        if key in FLAT_DIFFERENTIABLE_INPUT_PARAMETERS:
            differentiable_parameters[key] = jnp.asarray(value, dtype=float)
            cleaner_input_parameters[key] = value
            continue

        routed_parameter = False
        for section_name, section_keys in PARAMETER_SECTION_KEYS.items():
            if key in section_keys:
                parameters.setdefault(section_name, {})[key] = value
                routed_parameter = True
                break

        if not routed_parameter:
            cleaner_input_parameters[key] = value
            unrouted_input_parameters[key] = value

    return unrouted_input_parameters

def build_runtime_flat_parameter_routes():
    flat_routes = {}

    for section_name, section_metadata in PARAMETER_SECTIONS.items():
        for key in section_metadata["differentiable"]:
            route = (section_name, key)
            routes = flat_routes.setdefault(key, [])
            if route not in routes:
                routes.append(route)

    return flat_routes

def build_runtime_species_label_routes(species_parameters):
    species_label_routes = {}
    for species_type in SPECIES_TYPES:
        species_label_routes[species_type] = {}
        for canonical_label, species_values in species_parameters[species_type].items():
            labels = [canonical_label, species_values.get("user_label")]
            for label in labels:
                if label is None:
                    continue
                routes = species_label_routes[species_type].setdefault(label, [])
                if canonical_label not in routes:
                    routes.append(canonical_label)
    return species_label_routes

def clean_runtime_input_parameters(
    input_parameters,
    runtime_flat_parameter_routes,
    runtime_species_label_routes,
    species_parameters,
):
    if input_parameters is None:
        input_parameters = {}
    if not isinstance(input_parameters, dict):
        raise TypeError("Runtime input_parameters must be a dictionary.")

    cleaned_input_parameters = {section_name: {} for section_name in PARAMETER_SECTIONS}
    invalid_parameter_paths = []
    unrouted_parameters = []

    for key, value in input_parameters.items():
        if key in SPECIES_TYPES:
            for species_label, species_values in iter_species_parameter_groups(key, value, strict=True):
                for species_key, species_value in species_values.items():
                    if species_key not in SPECIES_DIFFERENTIABLE_KEYS.get(key, ()):
                        if species_label is None:
                            invalid_parameter_paths.append(f"{key}.{species_key}")
                        else:
                            invalid_parameter_paths.append(f"{key}.{species_label}.{species_key}")
                        continue

                    if species_label is None:
                        canonical_labels = tuple(species_parameters[key].keys())
                    elif species_label in runtime_species_label_routes[key]:
                        canonical_labels = tuple(runtime_species_label_routes[key][species_label])
                    else:
                        raise ValueError(f"Could not find {key} species {species_label!r}.")

                    for canonical_label in canonical_labels:
                        put_parameter_path(
                            cleaned_input_parameters,
                            ("species_parameters", key, canonical_label, species_key),
                            species_value,
                        )
            continue

        if key not in FLAT_DIFFERENTIABLE_INPUT_PARAMETERS:
            invalid_parameter_paths.append(key)
            continue

        routes = runtime_flat_parameter_routes.get(key, ())
        if not routes:
            unrouted_parameters.append(key)
            continue

        for route in routes:
            put_parameter_path(cleaned_input_parameters, route, value)

    if invalid_parameter_paths:
        invalid_parameter_paths = ", ".join(invalid_parameter_paths)
        raise ValueError(
            "Runtime input_parameters can only contain differentiable parameters. "
            f"Invalid parameter(s): {invalid_parameter_paths}"
        )
    if unrouted_parameters:
        unrouted_parameters = ", ".join(unrouted_parameters)
        raise ValueError(
            "Runtime input_parameters were validated but could not be routed. "
            f"Unrouted parameter(s): {unrouted_parameters}"
        )

    return cleaned_input_parameters

def build_runtime_parameter_sections(base_parameter_sections, input_parameters):
    return {
        section_name: merge_parameter_trees(
            base_parameters,
            input_parameters.get(section_name, {}),
        )
        for section_name, base_parameters in base_parameter_sections.items()
    }
