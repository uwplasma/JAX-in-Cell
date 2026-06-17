import pytest

pytestmark = pytest.mark.skip(reason="scaffold only")


def test_parameter_sections_include_all_registered_sections():
    """Test jaxincell._parameters._sections.PARAMETER_SECTIONS.

    Cases to implement:
    - every public parameter section is represented exactly once.
    - each section has all required metadata keys.
    - section cleaners and hash builders are callable.
    """


def test_parameter_section_keys_match_section_metadata():
    """Test jaxincell._parameters._sections.PARAMETER_SECTION_KEYS.

    Cases to implement:
    - each entry matches the matching PARAMETER_SECTIONS all-keys list.
    - section key collections do not alias mutable defaults unexpectedly.
    - adding a new section requires only one metadata update.
    """


def test_flat_differentiable_input_parameters_are_deduplicated_and_routed():
    """Test flat differentiable input parameter aggregation.

    Cases to implement:
    - flat differentiable keys are built from section metadata.
    - duplicate flat keys across sections are deduplicated in first-seen order.
    - species-only nested keys are not exposed as flat keys.
    """


def test_differentiable_input_parameters_include_flat_and_species_keys():
    """Test full differentiable input parameter aggregation.

    Cases to implement:
    - every flat differentiable key is included.
    - every nested differentiable species key is included.
    - the combined list remains stable and deduplicated.
    """
