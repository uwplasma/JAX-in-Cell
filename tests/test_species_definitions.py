import pytest

pytestmark = pytest.mark.skip(reason="scaffold only")


def test_species_default_maps_are_consistent_with_species_types():
    """Test jaxincell._parameters._species_definitions default maps.

    Cases to implement:
    - SPECIES_TYPES keeps electrons before ions so electron reference initialization is available.
    - every species type has default and initial-default entries.
    - charged species shared defaults are present for ions and electrons.
    - ion-only and electron-only defaults are confined to the intended species type.
    """


def test_species_parameter_key_lists_match_default_dictionaries():
    """Test species parameter key lists.

    Cases to implement:
    - ALL_ION_PARAMETERS matches ion defaults.
    - ALL_ELECTRON_PARAMETERS matches electron defaults.
    - shared, ion-only, and electron-only groups compose without accidental omissions.
    """


def test_species_differentiable_keys_exist_in_species_defaults():
    """Test species differentiable key definitions.

    Cases to implement:
    - every differentiable ion key exists in ion defaults.
    - every differentiable electron key exists in electron defaults.
    - differentiable shared charged-species keys are included for both species types.
    """


def test_initial_species_defaults_preserve_required_two_stream_overrides():
    """Test initial species defaults.

    Cases to implement:
    - initial ions and electrons preserve the intended two-stream defaults.
    - user values merged into the first species override initial defaults.
    - additional species use plain defaults instead of initial two-stream defaults.
    """
