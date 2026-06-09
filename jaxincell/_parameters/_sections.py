from ._domain_parameters import (
    ALL_DOMAIN_PARAMETERS,
    DIFFERENTIABLE_DOMAIN_PARAMETERS,
    build_domain_hash,
    clean_and_initialize_domain_parameters,
)
from ._external_field_parameters import (
    ALL_EXTERNAL_FIELD_PARAMETERS,
    DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS,
    build_external_field_hash,
    clean_and_initialize_external_field_parameters,
)
from ._solver_parameters import (
    ALL_SOLVER_PARAMETERS,
    DIFFERENTIABLE_SOLVER_PARAMETERS,
    build_solver_hash,
    clean_and_initialize_solver_parameters,
)
from ._source_parameters import (
    ALL_SOURCE_PARAMETERS,
    DIFFERENTIABLE_SOURCE_PARAMETERS,
    build_source_hash,
    clean_and_initialize_source_parameters,
)
from ._species_definitions import (
    ALL_LEGACY_SPECIES_PARAMETERS,
    DIFFERENTIABLE_LEGACY_SPECIES_PARAMETERS,
    DIFFERENTIABLE_SPECIES_PARAMETERS,
)
from ._species_parameters import (
    build_species_hash,
    clean_and_initialize_species_parameters,
)

__all__ = [
    "PARAMETER_SECTIONS",
    "PARAMETER_SECTION_KEYS",
    "FLAT_DIFFERENTIABLE_INPUT_PARAMETERS",
    "DIFFERENTIABLE_INPUT_PARAMETERS",
]

PARAMETER_SECTIONS = {
    "domain_parameters": {
        "all": ALL_DOMAIN_PARAMETERS,
        "flat_differentiable": DIFFERENTIABLE_DOMAIN_PARAMETERS,
        "section_differentiable": DIFFERENTIABLE_DOMAIN_PARAMETERS,
        "cleaner": clean_and_initialize_domain_parameters,
        "hasher": build_domain_hash,
        "attribute": "_domain_parameters",
        "hash_attribute": "domain_hash",
    },
    "species_parameters": {
        "all": ALL_LEGACY_SPECIES_PARAMETERS,
        "flat_differentiable": DIFFERENTIABLE_LEGACY_SPECIES_PARAMETERS,
        "section_differentiable": [],
        "cleaner": clean_and_initialize_species_parameters,
        "hasher": build_species_hash,
        "attribute": "_species_parameters",
        "hash_attribute": "species_hash",
    },
    "external_field_parameters": {
        "all": ALL_EXTERNAL_FIELD_PARAMETERS,
        "flat_differentiable": DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS,
        "section_differentiable": DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS,
        "cleaner": clean_and_initialize_external_field_parameters,
        "hasher": build_external_field_hash,
        "attribute": "_external_field_parameters",
        "hash_attribute": "external_field_hash",
    },
    "source_parameters": {
        "all": ALL_SOURCE_PARAMETERS,
        "flat_differentiable": DIFFERENTIABLE_SOURCE_PARAMETERS,
        "section_differentiable": DIFFERENTIABLE_SOURCE_PARAMETERS,
        "cleaner": clean_and_initialize_source_parameters,
        "hasher": build_source_hash,
        "attribute": "_source_parameters",
        "hash_attribute": "source_hash",
    },
    "solver_parameters": {
        "all": ALL_SOLVER_PARAMETERS,
        "flat_differentiable": DIFFERENTIABLE_SOLVER_PARAMETERS,
        "section_differentiable": DIFFERENTIABLE_SOLVER_PARAMETERS,
        "cleaner": clean_and_initialize_solver_parameters,
        "hasher": build_solver_hash,
        "attribute": "_solver_parameters",
        "hash_attribute": "solver_hash",
    },
}

PARAMETER_SECTION_KEYS = {
    section_name: section_metadata["all"]
    for section_name, section_metadata in PARAMETER_SECTIONS.items()
}

FLAT_DIFFERENTIABLE_INPUT_PARAMETERS = list(dict.fromkeys(
    parameter
    for section_metadata in PARAMETER_SECTIONS.values()
    for parameter in section_metadata["flat_differentiable"]
))

DIFFERENTIABLE_INPUT_PARAMETERS = list(dict.fromkeys(
    FLAT_DIFFERENTIABLE_INPUT_PARAMETERS + DIFFERENTIABLE_SPECIES_PARAMETERS
))
