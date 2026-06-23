from ._utils import build_parameter_hash, overlay_parameter_defaults

__all__ = [
    "ALL_SOLVER_PARAMETERS",
    "DIFFERENTIABLE_SOLVER_PARAMETERS",
    "clean_and_initialize_solver_parameters",
    "build_solver_hash",
]

DEFAULT_SOLVER_PARAMETERS = {
        "print_info": True,                       # Print information about the simulation
        "field_solver": 0,                       # Algorithm for solving fields - 0: Gauss's law, 1: FDTD
        "relativistic": False,                    # Whether to use the relativistic version of the Boris push (only relevant if time_evolution_algorithm is 0 (Boris))
        "time_evolution_algorithm": 0,             # Algorithm to evolve particles in time - 0: Boris, 1: Implicit_Crank Nicholson
        "max_number_of_Picard_iterations_implicit_CN": 20, # Maximum number of Picard iterations for implicit Crank-Nicholson method
        "number_of_particle_substeps_implicit_CN": 2, # Number of particle substeps per field step for implicit Crank-Nicholson method
        "tolerance_Picard_iterations_implicit_CN": 1e-6, # Tolerance for Picard iterations in implicit Crank-Nicholson method
        "filter_passes": 5,       # number of passes of the digital filter applied to ρ and J
        "filter_alpha": 0.5,      # filter strength (0 < alpha < 1)
        "filter_strides": (1, 2, 4),  # multi-scale strides for filtering
        "seed": 1701,                        # Random seed for reproducibility (if None, will be random every time)
        "openpmd_output": False,              # Write assembled output using the openPMD standard
        "openpmd_filename": "jaxincell_openpmd.h5",  # Destination file or series name for openPMD output
        "openpmd_meshes_path": "meshes/",      # openPMD meshesPath attribute
        "openpmd_particles_path": "particles/", # openPMD particlesPath attribute
        "openpmd_iteration_encoding": "groupBased", # openPMD iteration encoding; fileBased is reserved for later
        "openpmd_overwrite": False,            # Overwrite an existing openPMD destination
        "openpmd_iteration_stride": 1,         # Write every Nth output iteration to openPMD
        "openpmd_separate_particles_and_meshes": False, # Write particles and meshes to separate openPMD series
        "openpmd_write_pmd_sidecar": True,     # Write a .pmd sidecar listing generated openPMD series files
    }

DIFFERENTIABLE_SOLVER_PARAMETERS = [
    "filter_alpha",
]

ALL_SOLVER_PARAMETERS = list(DEFAULT_SOLVER_PARAMETERS.keys())

def clean_and_initialize_solver_parameters(solver_parameters, input_parameters=None):
    if input_parameters is None:
        input_parameters = {}
    solver_parameters = overlay_parameter_defaults(
        DEFAULT_SOLVER_PARAMETERS,
        solver_parameters,
        input_parameters,
    )

    solver_parameters["tolerance_Picard_iterations_implicit_CN"] = float(solver_parameters["tolerance_Picard_iterations_implicit_CN"])
    if type(solver_parameters["filter_strides"]) != tuple:
        solver_parameters["filter_strides"] = tuple(solver_parameters["filter_strides"])

    assert solver_parameters["field_solver"] in [0, 1], "Invalid field solver. Must be 0 (Gauss's law) or 1 (FDTD)."
    assert solver_parameters["time_evolution_algorithm"] in [0, 1], "Invalid time evolution algorithm. Must be 0 (Boris) or 1 (Implicit Crank-Nicholson)."
    assert type(solver_parameters["max_number_of_Picard_iterations_implicit_CN"]) == int and solver_parameters["max_number_of_Picard_iterations_implicit_CN"] > 0, "Maximum number of Picard iterations for implicit Crank-Nicholson method must be a positive integer."
    assert type(solver_parameters["number_of_particle_substeps_implicit_CN"]) == int and solver_parameters["number_of_particle_substeps_implicit_CN"] > 0, "Number of particle substeps per field step for implicit Crank-Nicholson method must be a positive integer."
    assert solver_parameters["tolerance_Picard_iterations_implicit_CN"] > 0, "Tolerance for Picard iterations in implicit Crank-Nicholson method must be positive."
    assert type(solver_parameters["filter_passes"]) == int and solver_parameters["filter_passes"] >= 0, "Number of passes of the digital filter must be a non-negative integer."
    assert 0 < solver_parameters["filter_alpha"] < 1, "Filter strength must be a float between 0 and 1."
    assert all(type(s) == int and s > 0 for s in solver_parameters["filter_strides"]), "Filter strides must be a tuple of positive integers."
    assert type(solver_parameters["openpmd_output"]) == bool, "openpmd_output must be a boolean."
    assert type(solver_parameters["openpmd_filename"]) == str and solver_parameters["openpmd_filename"], "openpmd_filename must be a non-empty string."
    assert type(solver_parameters["openpmd_meshes_path"]) == str and solver_parameters["openpmd_meshes_path"], "openpmd_meshes_path must be a non-empty string."
    assert type(solver_parameters["openpmd_particles_path"]) == str and solver_parameters["openpmd_particles_path"], "openpmd_particles_path must be a non-empty string."
    assert solver_parameters["openpmd_iteration_encoding"] in ["groupBased", "fileBased"], "openpmd_iteration_encoding must be 'groupBased' or 'fileBased'."
    if solver_parameters["openpmd_iteration_encoding"] == "fileBased":
        raise NotImplementedError("openPMD fileBased iteration encoding is not implemented yet.")
    assert type(solver_parameters["openpmd_overwrite"]) == bool, "openpmd_overwrite must be a boolean."
    assert type(solver_parameters["openpmd_iteration_stride"]) == int and solver_parameters["openpmd_iteration_stride"] > 0, "openpmd_iteration_stride must be a positive integer."
    assert type(solver_parameters["openpmd_separate_particles_and_meshes"]) == bool, "openpmd_separate_particles_and_meshes must be a boolean."
    assert type(solver_parameters["openpmd_write_pmd_sidecar"]) == bool, "openpmd_write_pmd_sidecar must be a boolean."

    return solver_parameters

def build_solver_hash(solver_parameters):
    return build_parameter_hash(solver_parameters)
