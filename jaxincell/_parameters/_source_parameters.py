from ._utils import build_parameter_hash, overlay_parameter_defaults

__all__ = [
    "ALL_SOURCE_PARAMETERS",
    "DIFFERENTIABLE_SOURCE_PARAMETERS",
    "clean_and_initialize_source_parameters",
    "build_source_hash",
]

DEFAULT_SOURCE_PARAMETERS = {
        "source_term_active": 0,                  # Whether the source term is active or not
        # 0 for electrons, 1 for ions, 2+ for extra species (sequential)
        "source_species": 1,                 # Which species should have sources (all species must be defined and named in the input file previously)
        "how_often_source_should_produce_quasiparticles": 20, # How many timesteps between each new quasiparticle produced by the source
        "source_particles_per_second": 1e16,       # (tuple for multiple sources)
        "location_of_source": 0,                  # Where the source term is (0 is center, 1 is left, 2 is right, 3 is whole domain) (tuple for multiple sources)
        "width_of_source":    1,                  # How many grid points there should be sources on (will round to .5's if parity doesn't match with grid size on center location) (tuple for multiple sources)
        "injection_speed_x": 1e7,                 # (tuple for multiple sources)
        "injection_speed_y": 0,                   # (tuple for multiple sources)
        "injection_speed_z": 0,                   # (tuple for multiple sources)
        # Should add temps for maxwellian injection profiles, but will be constant for now
    }

DIFFERENTIABLE_SOURCE_PARAMETERS = []

ALL_SOURCE_PARAMETERS = list(DEFAULT_SOURCE_PARAMETERS.keys())
SOURCE_SCALAR_PARAMETERS = ("source_term_active", "source_species")
SOURCE_FLOAT_TUPLE_PARAMETERS = (
    "source_particles_per_second",
    "injection_speed_x",
    "injection_speed_y",
    "injection_speed_z",
)
SOURCE_INJECTION_SPEED_PARAMETERS = (
    "injection_speed_x",
    "injection_speed_y",
    "injection_speed_z",
)

def make_tuple(thing_to_make_a_tuple):
    if not isinstance(thing_to_make_a_tuple, (list, tuple)):
        thing_to_make_a_tuple = (thing_to_make_a_tuple,)
    else:
        thing_to_make_a_tuple = tuple(thing_to_make_a_tuple)
    return thing_to_make_a_tuple

def make_tuple_values_floats(thing_to_make_tuple_values_floats):
    thing_to_make_tuple_values_floats = list(thing_to_make_tuple_values_floats)
    for i in range(len(thing_to_make_tuple_values_floats)):
        thing_to_make_tuple_values_floats[i] = float(thing_to_make_tuple_values_floats[i])
    return tuple(thing_to_make_tuple_values_floats)

def clean_and_initialize_source_parameters(source_parameters, input_parameters=None):
    if input_parameters is None:
        input_parameters = {}
    source_parameters = overlay_parameter_defaults(
        DEFAULT_SOURCE_PARAMETERS,
        source_parameters,
        input_parameters,
    )

    assert source_parameters["source_term_active"] in [0, 1], f"Source term active must be 0 (inactive) or 1 (active). Got {source_parameters['source_term_active']}."
    
    source_parameters["source_species"] = make_tuple(source_parameters["source_species"])
    assert all(isinstance(s, int) for s in source_parameters["source_species"]), f"All source species must be specified as integers corresponding to the order of species defined in the input file. Got {source_parameters['source_species']}."

    for key in source_parameters.keys():
        if key in SOURCE_SCALAR_PARAMETERS:
            continue

        source_parameters[key] = make_tuple(source_parameters[key])
        if source_parameters["source_term_active"]:
            if len(source_parameters[key]) != len(source_parameters["source_species"]):
                source_parameters[key] = source_parameters[key] * len(source_parameters["source_species"])
            assert len(source_parameters[key]) == len(source_parameters["source_species"]), f"Length of {key} must match length of 'source_species' or be a single value. Got {len(source_parameters[key])} and {len(source_parameters['source_species'])}."
    
    assert all(isinstance(hosspq, int) and hosspq > 0 for hosspq in source_parameters["how_often_source_should_produce_quasiparticles"]), f"All values in 'how_often_source_should_produce_quasiparticles' must be positive integers. Got {source_parameters['how_often_source_should_produce_quasiparticles']}."

    for key in SOURCE_FLOAT_TUPLE_PARAMETERS:
        source_parameters[key] = make_tuple_values_floats(source_parameters[key])

    assert all(spps > 0 for spps in source_parameters["source_particles_per_second"]), f"All values in 'source_particles_per_second' must be positive. Got {source_parameters['source_particles_per_second']}."

    assert all(los in [0, 1, 2, 3] for los in source_parameters["location_of_source"]), f"All values in 'location_of_source' must be 0 (center), 1 (left), 2 (right), or 3 (whole domain). Got {source_parameters['location_of_source']}."

    assert all(isinstance(wos, int) and wos > 0 for wos in source_parameters["width_of_source"]), f"All values in 'width_of_source' must be positive integers. Got {source_parameters['width_of_source']}."

    for key in SOURCE_INJECTION_SPEED_PARAMETERS:
        assert all(isinstance(value, float) for value in source_parameters[key]), f"All values in '{key}' must be floats. Got {source_parameters[key]}."

    return source_parameters

def build_source_hash(source_parameters):
    return build_parameter_hash(source_parameters)
