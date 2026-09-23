from jaxincell._parameters._domain_parameters import (
    ALL_DOMAIN_PARAMETERS,
    DIFFERENTIABLE_DOMAIN_PARAMETERS,
    build_domain_hash,
    clean_and_initialize_domain_parameters,
)
from jaxincell._parameters._external_field_parameters import (
    ALL_EXTERNAL_FIELD_PARAMETERS,
    DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS,
    build_external_field_hash,
    clean_and_initialize_external_field_parameters,
)
from jaxincell._parameters._export_parameters import (
    ALL_EXPORT_PARAMETERS,
    DIFFERENTIABLE_EXPORT_PARAMETERS,
    build_export_hash,
    clean_and_initialize_export_parameters,
)
from jaxincell._parameters._sections import (
    DIFFERENTIABLE_INPUT_PARAMETERS,
    FLAT_DIFFERENTIABLE_INPUT_PARAMETERS,
    PARAMETER_SECTION_KEYS,
    PARAMETER_SECTIONS,
)
from jaxincell._parameters._solver_parameters import (
    ALL_SOLVER_PARAMETERS,
    DIFFERENTIABLE_SOLVER_PARAMETERS,
    build_solver_hash,
    clean_and_initialize_solver_parameters,
)
from jaxincell._parameters._source_parameters import (
    ALL_SOURCE_PARAMETERS,
    DIFFERENTIABLE_SOURCE_PARAMETERS,
    build_source_hash,
    clean_and_initialize_source_parameters,
)
from jaxincell._parameters._species_definitions import DIFFERENTIABLE_SPECIES_PARAMETERS
from jaxincell._parameters._species_parameters import (
    build_species_hash,
    clean_and_initialize_species_parameters,
)

REQUIRED_SECTION_METADATA_KEYS = {
    "all",
    "differentiable",
    "cleaner",
    "hasher",
    "attribute",
    "hash_attribute",
}

EXPECTED_SECTION_METADATA = {
    "domain_parameters": {
        "all": ALL_DOMAIN_PARAMETERS,
        "differentiable": DIFFERENTIABLE_DOMAIN_PARAMETERS,
        "cleaner": clean_and_initialize_domain_parameters,
        "hasher": build_domain_hash,
        "attribute": "_domain_parameters",
        "hash_attribute": "domain_hash",
    },
    "species_parameters": {
        "all": [],
        "differentiable": [],
        "cleaner": clean_and_initialize_species_parameters,
        "hasher": build_species_hash,
        "attribute": "_species_parameters",
        "hash_attribute": "species_hash",
    },
    "external_field_parameters": {
        "all": ALL_EXTERNAL_FIELD_PARAMETERS,
        "differentiable": DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS,
        "cleaner": clean_and_initialize_external_field_parameters,
        "hasher": build_external_field_hash,
        "attribute": "_external_field_parameters",
        "hash_attribute": "external_field_hash",
    },
    "source_parameters": {
        "all": ALL_SOURCE_PARAMETERS,
        "differentiable": DIFFERENTIABLE_SOURCE_PARAMETERS,
        "cleaner": clean_and_initialize_source_parameters,
        "hasher": build_source_hash,
        "attribute": "_source_parameters",
        "hash_attribute": "source_hash",
    },
    "solver_parameters": {
        "all": ALL_SOLVER_PARAMETERS,
        "differentiable": DIFFERENTIABLE_SOLVER_PARAMETERS,
        "cleaner": clean_and_initialize_solver_parameters,
        "hasher": build_solver_hash,
        "attribute": "_solver_parameters",
        "hash_attribute": "solver_hash",
    },
    "export_parameters": {
        "all": ALL_EXPORT_PARAMETERS,
        "differentiable": DIFFERENTIABLE_EXPORT_PARAMETERS,
        "cleaner": clean_and_initialize_export_parameters,
        "hasher": build_export_hash,
        "attribute": "_export_parameters",
        "hash_attribute": "export_hash",
    },
}


def test_parameter_sections_include_all_registered_sections():
    """Test jaxincell._parameters._sections.PARAMETER_SECTIONS.

    Cases covered:
    - every public parameter section is represented exactly once.
    - each section has all required metadata keys.
    - section cleaners and hash builders are callable.
    - each section is wired to the expected cleaner and hash builder.
    - species parameters remain nested-only in this registry.
    """
    assert set(PARAMETER_SECTIONS) == set(EXPECTED_SECTION_METADATA)

    for section_name, section_metadata in PARAMETER_SECTIONS.items():
        expected_metadata = EXPECTED_SECTION_METADATA[section_name]

        assert set(section_metadata) == REQUIRED_SECTION_METADATA_KEYS
        assert callable(section_metadata["cleaner"])
        assert callable(section_metadata["hasher"])
        assert section_metadata["cleaner"] is expected_metadata["cleaner"]
        assert section_metadata["hasher"] is expected_metadata["hasher"]
        assert isinstance(section_metadata["all"], list)
        assert isinstance(section_metadata["differentiable"], list)
        assert set(section_metadata["all"]) == set(expected_metadata["all"])
        assert set(section_metadata["differentiable"]) == set(
            expected_metadata["differentiable"]
        )
        assert section_metadata["attribute"] == expected_metadata["attribute"]
        assert section_metadata["hash_attribute"] == expected_metadata["hash_attribute"]

        if section_name != "species_parameters":
            assert set(section_metadata["differentiable"]) <= set(section_metadata["all"])
        else:
            assert section_metadata["all"] == []
            assert section_metadata["differentiable"] == []


def test_parameter_section_keys_match_section_metadata():
    """Test jaxincell._parameters._sections.PARAMETER_SECTION_KEYS.

    Cases covered:
    - each entry matches the matching PARAMETER_SECTIONS all-keys list.
    - the generated keys cover the same sections as the metadata registry.
    - each generated entry still maps to the section-specific source key list.
    - species parameters remain nested-only in this registry.
    """
    assert set(PARAMETER_SECTION_KEYS) == set(PARAMETER_SECTIONS)

    for section_name, section_keys in PARAMETER_SECTION_KEYS.items():
        expected_metadata = EXPECTED_SECTION_METADATA[section_name]

        assert section_keys == PARAMETER_SECTIONS[section_name]["all"]
        assert set(section_keys) == set(expected_metadata["all"])

    assert PARAMETER_SECTION_KEYS["species_parameters"] == []


def test_flat_differentiable_input_parameters_are_deduplicated_and_routed():
    """Test flat differentiable input parameter aggregation.

    Cases covered:
    - flat differentiable keys are built from section metadata.
    - duplicate flat keys across sections are deduplicated.
    - species-only nested keys are not exposed as flat keys.
    """
    expected_flat_differentiable_parameters = {
        parameter
        for section_name, section_metadata in EXPECTED_SECTION_METADATA.items()
        if section_name != "species_parameters"
        for parameter in section_metadata["differentiable"]
    }

    assert (
        set(FLAT_DIFFERENTIABLE_INPUT_PARAMETERS)
        == expected_flat_differentiable_parameters
    )
    assert len(FLAT_DIFFERENTIABLE_INPUT_PARAMETERS) == len(
        set(FLAT_DIFFERENTIABLE_INPUT_PARAMETERS)
    )
    assert set(DIFFERENTIABLE_SPECIES_PARAMETERS).isdisjoint(
        FLAT_DIFFERENTIABLE_INPUT_PARAMETERS
    )


def test_differentiable_input_parameters_include_flat_and_species_keys():
    """Test full differentiable input parameter aggregation.

    Cases covered:
    - every flat differentiable key is included.
    - every nested differentiable species key is included.
    - the combined list remains complete and deduplicated.
    """
    expected_differentiable_parameters = set(FLAT_DIFFERENTIABLE_INPUT_PARAMETERS) | set(
        DIFFERENTIABLE_SPECIES_PARAMETERS
    )

    assert set(DIFFERENTIABLE_INPUT_PARAMETERS) == expected_differentiable_parameters
    assert set(FLAT_DIFFERENTIABLE_INPUT_PARAMETERS) <= set(DIFFERENTIABLE_INPUT_PARAMETERS)
    assert set(DIFFERENTIABLE_SPECIES_PARAMETERS) <= set(DIFFERENTIABLE_INPUT_PARAMETERS)
    assert len(DIFFERENTIABLE_INPUT_PARAMETERS) == len(set(DIFFERENTIABLE_INPUT_PARAMETERS))
