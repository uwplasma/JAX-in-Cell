from jaxincell._parameters._utils import build_parameter_hash, overlay_parameter_defaults


def test_overlay_parameter_defaults_precedence_and_no_input_case():
    """Test jaxincell._parameters._utils.overlay_parameter_defaults.

    Cases covered:
    - default parameters are used when explicit parameters omit a key.
    - explicit parameters override defaults.
    - input_parameters override both defaults and explicit parameters for matching keys.
    - input_parameters keys not present in the default/explicit parameter set are ignored.
    """
    default_parameters = {
        "default_only": 1.0,
        "explicit_override": 2.0,
        "input_override": 3.0,
    }
    parameters = {
        "explicit_override": 20.0,
        "input_override": 30.0,
    }
    input_parameters = {
        "input_override": 300.0,
        "unrouted_input": 400.0,
    }

    overlaid_parameters = overlay_parameter_defaults(
        default_parameters,
        parameters,
        input_parameters=input_parameters,
    )

    assert overlaid_parameters == {
        "default_only": 1.0,
        "explicit_override": 20.0,
        "input_override": 300.0,
    }
    assert "unrouted_input" not in overlaid_parameters
    assert overlaid_parameters is not default_parameters
    assert overlaid_parameters is not parameters

    without_input_parameters = overlay_parameter_defaults(default_parameters, parameters)
    with_empty_input_parameters = overlay_parameter_defaults(
        default_parameters,
        parameters,
        input_parameters={},
    )

    assert without_input_parameters == {
        "default_only": 1.0,
        "explicit_override": 20.0,
        "input_override": 30.0,
    }
    assert with_empty_input_parameters == without_input_parameters


def test_build_parameter_hash_is_stable_and_order_documented():
    """Test jaxincell._parameters._utils.build_parameter_hash.

    Cases covered:
    - identical dictionaries with the same insertion order produce identical hashes.
    - changing a key or value changes the hash.
    - insertion-order differences currently produce different hashes.
    """
    parameters = {
        "alpha": 1.0,
        "beta": (2.0, 3.0),
    }
    matching_parameters = {
        "alpha": 1.0,
        "beta": (2.0, 3.0),
    }
    changed_value_parameters = {
        "alpha": 1.0,
        "beta": (2.0, 4.0),
    }
    changed_key_parameters = {
        "alpha": 1.0,
        "gamma": (2.0, 3.0),
    }
    reordered_parameters = {
        "beta": (2.0, 3.0),
        "alpha": 1.0,
    }

    assert build_parameter_hash(parameters) == build_parameter_hash(matching_parameters)
    assert build_parameter_hash(parameters) != build_parameter_hash(changed_value_parameters)
    assert build_parameter_hash(parameters) != build_parameter_hash(changed_key_parameters)
    assert build_parameter_hash(parameters) != build_parameter_hash(reordered_parameters)
    assert build_parameter_hash({"alpha": 1.0}) == "alpha1.0"
