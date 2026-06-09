from copy import deepcopy

from ._parameters._sections import (
    FLAT_DIFFERENTIABLE_INPUT_PARAMETERS,
    PARAMETER_SECTIONS,
)
from ._parameters._species_definitions import (
    LEGACY_SPECIES_PARAMETER_ROUTES,
    SPECIES_DIFFERENTIABLE_KEYS,
    SPECIES_PARAMETER_KEYS,
    SPECIES_TYPES,
)
from ._utils import make_differentiable_type

__all__ = [
    "build_runtime_flat_parameter_routes",
    "build_runtime_parameter_sections",
    "build_runtime_species_label_routes",
    "clean_runtime_input_parameters",
    "iter_species_parameter_groups",
    "merge_parameter_trees",
    "put_parameter_path",
    "put_species_parameter",
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
            merged_parameters[key] = value
    return merged_parameters

def put_parameter_path(container, path, value):
    for key in path[:-1]:
        container = container.setdefault(key, {})
    container[path[-1]] = value

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

def put_species_parameter(container, species_type, species_label, key, value):
    species_container = container.setdefault(species_type, {})
    if species_label is None:
        species_container[key] = value
    else:
        species_container.setdefault(species_label, {})[key] = value

def route_nested_initial_species_parameters(input_parameters, parameters, differentiable_parameters, cleaner_input_parameters):
    for species_type in SPECIES_TYPES:
        for species_label, species_values in iter_species_parameter_groups(species_type, input_parameters.get(species_type, {})):
            for key, value in species_values.items():
                if key in SPECIES_DIFFERENTIABLE_KEYS.get(species_type, ()):
                    put_species_parameter(differentiable_parameters, species_type, species_label, key, make_differentiable_type(value))
                    put_species_parameter(cleaner_input_parameters, species_type, species_label, key, value)
                elif key in SPECIES_PARAMETER_KEYS.get(species_type, ()):
                    species_parameters = parameters.setdefault("species_parameters", {})
                    put_species_parameter(species_parameters, species_type, species_label, key, value)
                    put_species_parameter(cleaner_input_parameters, species_type, species_label, key, value)
                else:
                    put_species_parameter(cleaner_input_parameters, species_type, species_label, key, value)

def build_runtime_flat_parameter_routes(species_parameters):
    flat_routes = {}

    def add_route(key, *path):
        routes = flat_routes.setdefault(key, [])
        route = tuple(path)
        if route not in routes:
            routes.append(route)

    for section_name, section_metadata in PARAMETER_SECTIONS.items():
        for key in section_metadata["section_differentiable"]:
            add_route(key, section_name, key)

    for key, species_routes in LEGACY_SPECIES_PARAMETER_ROUTES.items():
        for species_type, species_key in species_routes:
            for species_label in species_parameters[species_type]:
                add_route(
                    key,
                    "species_parameters",
                    species_type,
                    species_label,
                    species_key,
                )

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

def canonical_runtime_species_labels(species_parameters, species_label_routes, species_type, species_label=None):
    if species_label is None:
        return tuple(species_parameters[species_type].keys())
    if species_label in species_label_routes[species_type]:
        return tuple(species_label_routes[species_type][species_label])
    raise ValueError(f"Could not find {species_type} species {species_label!r}.")

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

                    for canonical_label in canonical_runtime_species_labels(
                        species_parameters,
                        runtime_species_label_routes,
                        key,
                        species_label,
                    ):
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
