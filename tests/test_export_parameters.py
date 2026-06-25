import pytest

from jaxincell._parameters._export_parameters import (
    DEFAULT_EXPORT_PARAMETERS,
    clean_and_initialize_export_parameters,
    build_export_hash,
)


def test_clean_and_initialize_export_parameters_defaults_and_overrides():
    """Test jaxincell._parameters._export_parameters.clean_and_initialize_export_parameters.

    Cases:
    - defaults are populated when no export parameters are supplied.
    - input_parameters override explicit export_parameters for matching keys.
    - openPMD output defaults are disabled and groupBased.
    """
    default_applied = clean_and_initialize_export_parameters({})
    assert default_applied == DEFAULT_EXPORT_PARAMETERS
    assert default_applied["openpmd_output"] is False
    assert default_applied["openpmd_filename"] == "jaxincell_openpmd.h5"
    assert default_applied["openpmd_iteration_encoding"] == "groupBased"
    assert default_applied["openpmd_iteration_stride"] == 1

    input_parameters = {
        "openpmd_filename": "from_input.h5",
        "openpmd_iteration_stride": 2,
    }
    overridden = clean_and_initialize_export_parameters(
        {"openpmd_filename": "explicit.h5"},
        input_parameters,
    )
    assert overridden["openpmd_filename"] == "from_input.h5"
    assert overridden["openpmd_iteration_stride"] == 2


def test_clean_and_initialize_export_parameters_rejects_invalid_values():
    """Test openPMD export option validation."""
    input_parameters = {"openpmd_output": "yes"}
    with pytest.raises(AssertionError, match="openpmd_output must be a boolean."):
        clean_and_initialize_export_parameters({}, input_parameters)

    input_parameters = {"openpmd_filename": ""}
    with pytest.raises(AssertionError, match="openpmd_filename must be a non-empty string."):
        clean_and_initialize_export_parameters({}, input_parameters)

    input_parameters = {"openpmd_meshes_path": ""}
    with pytest.raises(AssertionError, match="openpmd_meshes_path must be a non-empty string."):
        clean_and_initialize_export_parameters({}, input_parameters)

    input_parameters = {"openpmd_particles_path": ""}
    with pytest.raises(AssertionError, match="openpmd_particles_path must be a non-empty string."):
        clean_and_initialize_export_parameters({}, input_parameters)

    input_parameters = {"openpmd_iteration_encoding": "variableBased"}
    with pytest.raises(AssertionError, match="openpmd_iteration_encoding must be 'groupBased' or 'fileBased'."):
        clean_and_initialize_export_parameters({}, input_parameters)

    file_based = clean_and_initialize_export_parameters(
        {},
        {"openpmd_iteration_encoding": "fileBased"},
    )
    assert file_based["openpmd_iteration_encoding"] == "fileBased"

    input_parameters = {"openpmd_overwrite": "no"}
    with pytest.raises(AssertionError, match="openpmd_overwrite must be a boolean."):
        clean_and_initialize_export_parameters({}, input_parameters)

    input_parameters = {"openpmd_iteration_stride": 0}
    with pytest.raises(AssertionError, match="openpmd_iteration_stride must be a positive integer."):
        clean_and_initialize_export_parameters({}, input_parameters)

    input_parameters = {"openpmd_iteration_stride": 1.5}
    with pytest.raises(AssertionError, match="openpmd_iteration_stride must be a positive integer."):
        clean_and_initialize_export_parameters({}, input_parameters)

    input_parameters = {"openpmd_separate_particles_and_meshes": "false"}
    with pytest.raises(AssertionError, match="openpmd_separate_particles_and_meshes must be a boolean."):
        clean_and_initialize_export_parameters({}, input_parameters)

    input_parameters = {"openpmd_write_pmd_sidecar": "true"}
    with pytest.raises(AssertionError, match="openpmd_write_pmd_sidecar must be a boolean."):
        clean_and_initialize_export_parameters({}, input_parameters)


def test_build_export_hash_is_static_for_output_only_settings():
    """Test jaxincell._parameters._export_parameters.build_export_hash.

    Cases:
    - changing the destination filename does not change the current static hash.
    - changing openPMD cadence does not change the current static hash.
    """
    default_parameters = clean_and_initialize_export_parameters({})
    default_hash = build_export_hash(default_parameters)

    filename_changed_parameters = clean_and_initialize_export_parameters(
        {"openpmd_filename": "different.h5"}
    )
    stride_changed_parameters = clean_and_initialize_export_parameters(
        {"openpmd_iteration_stride": 3}
    )

    assert default_hash == "export_hash_static"
    assert build_export_hash(filename_changed_parameters) == default_hash
    assert build_export_hash(stride_changed_parameters) == default_hash
