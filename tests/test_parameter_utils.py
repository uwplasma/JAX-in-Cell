import pytest

pytestmark = pytest.mark.skip(reason="scaffold only")


def test_overlay_parameter_defaults_precedence_and_no_input_case():
    """Test jaxincell._parameters._utils.overlay_parameter_defaults.

    Cases to implement:
    - default parameters are used when explicit parameters omit a key.
    - explicit parameters override defaults.
    - input_parameters override both defaults and explicit parameters for matching keys.
    - input_parameters keys not present in the default/explicit parameter set are ignored.
    """


def test_build_parameter_hash_is_stable_and_order_documented():
    """Test jaxincell._parameters._utils.build_parameter_hash.

    Cases to implement:
    - identical dictionaries with the same insertion order produce identical hashes.
    - changing a key or value changes the hash.
    - document whether insertion-order differences should or should not produce different hashes.
    """
