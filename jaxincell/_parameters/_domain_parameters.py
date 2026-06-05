from .._utils import as_float_parameter

__all__ = [
    "ALL_DOMAIN_PARAMETERS",
    "DIFFERENTIABLE_DOMAIN_PARAMETERS",
    "clean_and_initialize_domain_parameters",
    "build_domain_hash",
]

ALL_DOMAIN_PARAMETERS = [
    "total_steps",
    "timestep_over_spatialstep_times_c",
    "number_grid_points",
    "number_grid_points_y",
    "number_grid_points_z",
    "length",
    "length_y",
    "length_z",
    "grid_points_per_Debye_length",
    "particle_BC_left",
    "particle_BC_right",
    "field_BC_left",
    "field_BC_right",
]

DIFFERENTIABLE_DOMAIN_PARAMETERS = [
    "timestep_over_spatialstep_times_c",
    "length",
    "length_y",
    "length_z",
]

def clean_and_initialize_domain_parameters(domain_parameters, input_parameters={}):
    default_domain_parameters = {
        "total_steps": 350,                       # Total number of time steps to evolve the simulation
        "timestep_over_spatialstep_times_c": 1.0,   # dt * speed_of_light / dx
        "number_grid_points": 50,                       # Number of grid points in the simulation box
        "number_grid_points_y": 0,                       # Number of grid points in the y direction (if None, same as number_grid_points)
        "number_grid_points_z": 0,                       # Number of grid points in the z direction (if None, same as number_grid_points)
        "length": 1e-2,                           # Dimensions of the simulation box
        "length_y": 0,                           # Dimensions of the simulation box in y
        "length_z": 0,                           # Dimensions of the simulation box in z
        "grid_points_per_Debye_length": 2,        # dx over Debye length
        "particle_BC_left": 0,                   # Left boundary condition for particles
        "particle_BC_right": 0,                   # Right boundary condition for particles
        "field_BC_left": 0,                     # Left boundary condition for fields
        "field_BC_right": 0,                    # Right boundary condition for fields
    }
    domain_parameters = {**default_domain_parameters, **domain_parameters}
    for key in domain_parameters.keys():
        if key in input_parameters.keys():
            domain_parameters[key] = input_parameters[key]

    domain_parameters["length"] = as_float_parameter(domain_parameters["length"])
    domain_parameters["length_y"] = as_float_parameter(domain_parameters["length_y"])
    domain_parameters["length_z"] = as_float_parameter(domain_parameters["length_z"])
    domain_parameters["grid_points_per_Debye_length"] = as_float_parameter(domain_parameters["grid_points_per_Debye_length"])

    assert type(domain_parameters["total_steps"]) == int and domain_parameters["total_steps"] > 0, "Total number of time steps must be an integer."
    assert domain_parameters["length"] > 0, "Length of the simulation box must be positive."
    assert domain_parameters["length_y"] >= 0, "Length of the simulation box in y must be positive."
    assert domain_parameters["length_z"] >= 0, "Length of the simulation box in z must be positive."
    assert domain_parameters["grid_points_per_Debye_length"] > 0, "Grid points per Debye length must be positive."
    assert domain_parameters["particle_BC_left"] in [0, 1, 2], "Invalid particle boundary condition for left boundary. Must be 0 (periodic), 1 (reflecting), or 2 (absorbing)."
    assert domain_parameters["particle_BC_right"] in [0, 1, 2], "Invalid particle boundary condition for right boundary. Must be 0 (periodic), 1 (reflecting), or 2 (absorbing)."
    assert domain_parameters["field_BC_left"] in [0, 1, 2], "Invalid field boundary condition for left boundary. Must be 0 (periodic), 1 (reflecting), or 2 (absorbing)."
    assert domain_parameters["field_BC_right"] in [0, 1, 2], "Invalid field boundary condition for right boundary. Must be 0 (periodic), 1 (reflecting), or 2 (absorbing)."

    return domain_parameters

def build_domain_hash(domain_parameters):
    hash_list = []
    for key, value in domain_parameters.items():
        hash_list.append(str(key))
        hash_list.append(str(value))
    domain_hash = "".join(hash_list)
    return domain_hash
