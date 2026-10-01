from hashlib import sha256
import numpy as np

__all__ = ["overlay_parameter_defaults", "build_parameter_hash"]


def overlay_parameter_defaults(default_parameters, parameters, input_parameters=None):
    parameters = {**default_parameters, **parameters}
    if input_parameters is None:
        return parameters

    for key in parameters:
        if key in input_parameters:
            parameters[key] = input_parameters[key]
    return parameters


def build_parameter_hash(parameters):
    hash_list = []
    for key, value in parameters.items():
        if isinstance(value, dict):
            value = build_parameter_hash(value)
        elif hasattr(value, "dtype"):
            array = np.asarray(value)
            value = (array.dtype.str, array.shape, sha256(array.tobytes()).hexdigest())
        hash_list.append(str(key))
        hash_list.append(str(value))
    return "".join(hash_list)
