__all__ = [
    "ALL_EXTERNAL_FIELD_PARAMETERS",
    "DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS",
    "clean_and_initialize_external_field_parameters",
    "build_external_field_hash",
]

ALL_EXTERNAL_FIELD_PARAMETERS = [
    "external_electric_field_amplitude",
    "external_electric_field_wavenumber",
    "external_magnetic_field_amplitude",
    "external_magnetic_field_wavenumber",
    "external_electric_field_function",
    "external_magnetic_field_function",
    "external_electric_field",
    "external_magnetic_field",
]

DIFFERENTIABLE_EXTERNAL_FIELD_PARAMETERS = []

def clean_and_initialize_external_field_parameters(external_field_parameters, input_parameters={}):
    default_external_field_parameters = {
        "external_electric_field_amplitude":  0.,   # Amplitude of sinusoidal (cos) perturbation in x
        "external_electric_field_wavenumber": 0.,   # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
        "external_magnetic_field_amplitude":  0.,   # Amplitude of sinusoidal (cos) perturbation in x
        "external_magnetic_field_wavenumber": 0.,   # Wavenumber of sinusoidal (cos) perturbation in x (factor of 2pi/length)
        "external_electric_field_function": None,   # Function of (x, y, z, t) that returns the external electric field vector at a given position and time.
        "external_magnetic_field_function": None,   # Function of (x, y, z, t) that returns the external magnetic field vector at a given position and time.
    }
    external_field_parameters = {**default_external_field_parameters, **external_field_parameters}
    for key in external_field_parameters.keys():
        if key in input_parameters.keys():
            external_field_parameters[key] = input_parameters[key]

    external_field_parameters["external_electric_field_amplitude"] = float(external_field_parameters["external_electric_field_amplitude"])
    external_field_parameters["external_electric_field_wavenumber"] = float(external_field_parameters["external_electric_field_wavenumber"])
    external_field_parameters["external_magnetic_field_amplitude"] = float(external_field_parameters["external_magnetic_field_amplitude"])
    external_field_parameters["external_magnetic_field_wavenumber"] = float(external_field_parameters["external_magnetic_field_wavenumber"])

    return external_field_parameters

def build_external_field_hash(external_field_parameters):
    hash_list = []
    for key, value in external_field_parameters.items():
        hash_list.append(str(key))
        hash_list.append(str(value))
    external_field_hash = "".join(hash_list)
    return external_field_hash
