import jax.numpy as jnp

from ._utils import build_parameter_hash, overlay_parameter_defaults

__all__ = [
    "ALL_DOMAIN_PARAMETERS",
    "DIFFERENTIABLE_DOMAIN_PARAMETERS",
    "clean_and_initialize_domain_parameters",
    "build_domain_hash",
]

DEFAULT_DOMAIN_PARAMETERS = {
        "total_steps": 350,                       # Total number of time steps to evolve the simulation
        "timestep_over_spatialstep_times_c": 1.0,   # dt * speed_of_light / dx
        "number_grid_points": 50,                       # Number of grid points in the simulation box
        "number_grid_points_y": 0,                       # Number of grid points in the y direction (if None, same as number_grid_points)
        "number_grid_points_z": 0,                       # Number of grid points in the z direction (if None, same as number_grid_points)
        "length": 1e-2,                           # Dimensions of the simulation box
        "length_y": 0,                           # Dimensions of the simulation box in y
        "length_z": 0,                           # Dimensions of the simulation box in z
        "particle_BC_left": 0,                   # Left boundary condition for particles
        "particle_BC_right": 0,                   # Right boundary condition for particles
        "field_BC_left": 0,                     # Left boundary condition for fields
        "field_BC_right": 0,                    # Right boundary condition for fields
    }

DIFFERENTIABLE_DOMAIN_PARAMETERS = [
    "timestep_over_spatialstep_times_c",
    "length",
    "length_y",
    "length_z",
]

ALL_DOMAIN_PARAMETERS = list(DEFAULT_DOMAIN_PARAMETERS.keys())

def clean_and_initialize_domain_parameters(domain_parameters, input_parameters={}):
    domain_parameters = overlay_parameter_defaults(
        DEFAULT_DOMAIN_PARAMETERS,
        domain_parameters,
        input_parameters,
    )

    domain_parameters["length"] = jnp.asarray(domain_parameters["length"], dtype=float)
    domain_parameters["length_y"] = jnp.asarray(domain_parameters["length_y"], dtype=float)
    domain_parameters["length_z"] = jnp.asarray(domain_parameters["length_z"], dtype=float)

    assert type(domain_parameters["total_steps"]) == int and domain_parameters["total_steps"] > 0, "Total number of time steps must be an integer."
    assert domain_parameters["length"] > 0, "Length of the simulation box must be positive."
    assert domain_parameters["length_y"] >= 0, "Length of the simulation box in y must be positive."
    assert domain_parameters["length_z"] >= 0, "Length of the simulation box in z must be positive."
    assert domain_parameters["particle_BC_left"] in [0, 1, 2], "Invalid particle boundary condition for left boundary. Must be 0 (periodic), 1 (reflecting), or 2 (absorbing)."
    assert domain_parameters["particle_BC_right"] in [0, 1, 2], "Invalid particle boundary condition for right boundary. Must be 0 (periodic), 1 (reflecting), or 2 (absorbing)."
    assert domain_parameters["field_BC_left"] in [0, 1, 2], "Invalid field boundary condition for left boundary. Must be 0 (periodic), 1 (reflecting), or 2 (absorbing)."
    assert domain_parameters["field_BC_right"] in [0, 1, 2], "Invalid field boundary condition for right boundary. Must be 0 (periodic), 1 (reflecting), or 2 (absorbing)."

    return domain_parameters

def build_domain_hash(domain_parameters):
    return build_parameter_hash(domain_parameters)
