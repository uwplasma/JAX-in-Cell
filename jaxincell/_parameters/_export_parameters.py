from ._utils import build_parameter_hash, overlay_parameter_defaults

__all__ = [
    "ALL_EXPORT_PARAMETERS",
    "DIFFERENTIABLE_EXPORT_PARAMETERS",
    "clean_and_initialize_export_parameters",
    "build_export_hash",
]

DEFAULT_EXPORT_PARAMETERS = {
        "openpmd_output": False,              # Write assembled output using the openPMD standard
        "openpmd_filename": "jaxincell_openpmd.h5",  # Destination file or series name for openPMD output
        "openpmd_meshes_path": "meshes/",      # openPMD meshesPath attribute
        "openpmd_particles_path": "particles/", # openPMD particlesPath attribute
        "openpmd_iteration_encoding": "groupBased", # openPMD iteration encoding
        "openpmd_overwrite": False,            # Overwrite an existing openPMD destination
        "openpmd_iteration_stride": 1,         # Write every Nth output iteration to openPMD
        "openpmd_separate_particles_and_meshes": False, # Write particles and meshes to separate openPMD series
        "openpmd_write_pmd_sidecar": True,     # Write a .pmd sidecar listing generated openPMD series files
    }

DIFFERENTIABLE_EXPORT_PARAMETERS = []

ALL_EXPORT_PARAMETERS = list(DEFAULT_EXPORT_PARAMETERS.keys())

def clean_and_initialize_export_parameters(export_parameters, input_parameters=None):
    if input_parameters is None:
        input_parameters = {}
    export_parameters = overlay_parameter_defaults(
        DEFAULT_EXPORT_PARAMETERS,
        export_parameters,
        input_parameters,
    )

    assert type(export_parameters["openpmd_output"]) == bool, "openpmd_output must be a boolean."
    assert type(export_parameters["openpmd_filename"]) == str and export_parameters["openpmd_filename"], "openpmd_filename must be a non-empty string."
    assert type(export_parameters["openpmd_meshes_path"]) == str and export_parameters["openpmd_meshes_path"], "openpmd_meshes_path must be a non-empty string."
    assert type(export_parameters["openpmd_particles_path"]) == str and export_parameters["openpmd_particles_path"], "openpmd_particles_path must be a non-empty string."
    assert export_parameters["openpmd_iteration_encoding"] in ["groupBased", "fileBased"], "openpmd_iteration_encoding must be 'groupBased' or 'fileBased'."
    assert type(export_parameters["openpmd_overwrite"]) == bool, "openpmd_overwrite must be a boolean."
    assert type(export_parameters["openpmd_iteration_stride"]) == int and export_parameters["openpmd_iteration_stride"] > 0, "openpmd_iteration_stride must be a positive integer."
    assert type(export_parameters["openpmd_separate_particles_and_meshes"]) == bool, "openpmd_separate_particles_and_meshes must be a boolean."
    assert type(export_parameters["openpmd_write_pmd_sidecar"]) == bool, "openpmd_write_pmd_sidecar must be a boolean."

    return export_parameters

def build_export_hash(export_parameters):
    # Use the real hash once export parameters affect the jitted simulation path.
    # return build_parameter_hash(export_parameters)
    return "export_hash_static"
