import warnings

from ._utils import build_parameter_hash, overlay_parameter_defaults

__all__ = [
    "ALL_EXTERNAL_FIELD_PARAMETERS",
    "warn_if_external_field_request_is_ignored",
    "DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS",
    "clean_and_initialize_external_field_parameters",
    "build_external_field_hash",
]

DEFAULT_EXTERNAL_FIELD_PARAMETERS = {
        "external_electric_field_amplitude":  0.,   # Amplitude of sinusoidal (cos) perturbation in x
        "external_electric_field_wavenumber": 0.,   # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
        "external_magnetic_field_amplitude":  0.,   # Amplitude of sinusoidal (cos) perturbation in x
        "external_magnetic_field_wavenumber": 0.,   # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
        "external_electric_field_function": None,   # Function of (x, y, z, t) that returns the external electric field vector at a given position and time.
        "external_magnetic_field_function": None,   # Function of (x, y, z, t) that returns the external magnetic field vector at a given position and time.
    }

DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS = []

ALL_EXTERNAL_FIELD_PARAMETERS = list(DEFAULT_EXTERNAL_FIELD_PARAMETERS.keys())

def clean_and_initialize_external_field_parameters(external_field_parameters, input_parameters=None):
    if input_parameters is None:
        input_parameters = {}
    external_field_parameters = overlay_parameter_defaults(
        DEFAULT_EXTERNAL_FIELD_PARAMETERS,
        external_field_parameters,
        input_parameters,
    )

    external_field_parameters["external_electric_field_amplitude"] = float(external_field_parameters["external_electric_field_amplitude"])
    external_field_parameters["external_electric_field_wavenumber"] = float(external_field_parameters["external_electric_field_wavenumber"])
    external_field_parameters["external_magnetic_field_amplitude"] = float(external_field_parameters["external_magnetic_field_amplitude"])
    external_field_parameters["external_magnetic_field_wavenumber"] = float(external_field_parameters["external_magnetic_field_wavenumber"])

    warn_if_external_field_request_is_ignored(external_field_parameters)

    return external_field_parameters

def warn_if_external_field_request_is_ignored(external_field_parameters):
    """Warn about unsupported analytic field options; supplied grid arrays still apply."""
    ignored = [
        name for name, requested in (
            ("external_electric_field_amplitude",
             external_field_parameters["external_electric_field_amplitude"] != 0.0),
            ("external_magnetic_field_amplitude",
             external_field_parameters["external_magnetic_field_amplitude"] != 0.0),
            ("external_electric_field_function",
             external_field_parameters["external_electric_field_function"] is not None),
            ("external_magnetic_field_function",
             external_field_parameters["external_magnetic_field_function"] is not None),
        ) if requested
    ]
    if not ignored:
        return
    warnings.warn(
        f"{', '.join(ignored)} set but not applied: this release does not build an "
        "external field from an amplitude, a wavenumber or a function. "
        "Supplied grid arrays still apply, for example "
        "external_field_parameters={'external_magnetic_field': "
        "{'B': array of shape (number_grid_points, 3)}}.",
        UserWarning,
        stacklevel=3,
    )

def build_external_field_hash(external_field_parameters):
    return build_parameter_hash(external_field_parameters)
