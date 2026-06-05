import jax.numpy as jnp
from .._constants import mass_proton, mass_electron
from .._utils import as_float_parameter

__all__ = [
    "ALL_ION_PARAMETERS",
    "ALL_ELECTRON_PARAMETERS",
    "ALL_SPECIES_PARAMETERS",
    "ALL_LEGACY_SPECIES_PARAMETERS",
    "DIFFERENTIABLE_ION_PARAMETERS",
    "DIFFERENTIABLE_ELECTRON_PARAMETERS",
    "DIFFERENTIABLE_SPECIES_PARAMETERS",
    "DIFFERENTIABLE_LEGACY_SPECIES_PARAMETERS",
    "clean_and_initialize_species_parameters",
    "build_species_hash",
]

ALL_ION_PARAMETERS = [
    "number_pseudoparticles",
    "grid_points_per_Debye_length",
    "weight",
    "charge_over_elementary_charge",
    "mass_over_proton_mass",
    "perturbation_amplitude_x",
    "perturbation_amplitude_y",
    "perturbation_amplitude_z",
    "perturbation_wavenumber_x",
    "perturbation_wavenumber_y",
    "perturbation_wavenumber_z",
    "random_positions_x",
    "random_positions_y",
    "random_positions_z",
    "vth_over_c_x",
    "vth_over_c_y",
    "vth_over_c_z",
    "ion_temperature_over_electron_temperature_x",
    "ion_temperature_over_electron_temperature_y",
    "ion_temperature_over_electron_temperature_z",
    "drift_speed_x",
    "drift_speed_y",
    "drift_speed_z",
    "velocity_plus_minus_x",
    "velocity_plus_minus_y",
    "velocity_plus_minus_z",
    "seed_position_override",
    "seed_position",
]

ALL_ELECTRON_PARAMETERS = [
    "number_pseudoparticles",
    "grid_points_per_Debye_length",
    "weight",
    "charge_over_elementary_charge",
    "perturbation_amplitude_x",
    "perturbation_amplitude_y",
    "perturbation_amplitude_z",
    "perturbation_wavenumber_x",
    "perturbation_wavenumber_y",
    "perturbation_wavenumber_z",
    "random_positions_x",
    "random_positions_y",
    "random_positions_z",
    "vth_over_c_x",
    "vth_over_c_y",
    "vth_over_c_z",
    "drift_speed_x",
    "drift_speed_y",
    "drift_speed_z",
    "velocity_plus_minus_x",
    "velocity_plus_minus_y",
    "velocity_plus_minus_z",
    "seed_position_override",
    "seed_position",
]

ALL_SPECIES_PARAMETERS = list(dict.fromkeys(ALL_ION_PARAMETERS + ALL_ELECTRON_PARAMETERS))

ALL_LEGACY_SPECIES_PARAMETERS = [
    "number_pseudoelectrons",
    "grid_points_per_Debye_length",
    "weight",
    "electron_charge_over_elementary_charge",
    "ion_charge_over_elementary_charge",
    "ion_mass_over_proton_mass",
    "random_positions_x",
    "random_positions_y",
    "random_positions_z",
    "amplitude_perturbation_x",
    "amplitude_perturbation_y",
    "amplitude_perturbation_z",
    "wavenumber_electrons_x",
    "wavenumber_electrons_y",
    "wavenumber_electrons_z",
    "wavenumber_ions_x",
    "wavenumber_ions_y",
    "wavenumber_ions_z",
    "vth_electrons_over_c_x",
    "vth_electrons_over_c_y",
    "vth_electrons_over_c_z",
    "ion_temperature_over_electron_temperature_x",
    "ion_temperature_over_electron_temperature_y",
    "ion_temperature_over_electron_temperature_z",
    "electron_drift_speed_x",
    "electron_drift_speed_y",
    "electron_drift_speed_z",
    "ion_drift_speed_x",
    "ion_drift_speed_y",
    "ion_drift_speed_z",
    "velocity_plus_minus_electrons_x",
    "velocity_plus_minus_electrons_y",
    "velocity_plus_minus_electrons_z",
    "velocity_plus_minus_ions_x",
    "velocity_plus_minus_ions_y",
    "velocity_plus_minus_ions_z",
]

DIFFERENTIABLE_ION_PARAMETERS = [
    "grid_points_per_Debye_length",
    "weight",
    "charge_over_elementary_charge",
    "mass_over_proton_mass",
    "perturbation_amplitude_x",
    "perturbation_amplitude_y",
    "perturbation_amplitude_z",
    "perturbation_wavenumber_x",
    "perturbation_wavenumber_y",
    "perturbation_wavenumber_z",
    "vth_over_c_x",
    "vth_over_c_y",
    "vth_over_c_z",
    "ion_temperature_over_electron_temperature_x",
    "ion_temperature_over_electron_temperature_y",
    "ion_temperature_over_electron_temperature_z",
    "drift_speed_x",
    "drift_speed_y",
    "drift_speed_z",
]

DIFFERENTIABLE_ELECTRON_PARAMETERS = [
    "grid_points_per_Debye_length",
    "weight",
    "charge_over_elementary_charge",
    "perturbation_amplitude_x",
    "perturbation_amplitude_y",
    "perturbation_amplitude_z",
    "perturbation_wavenumber_x",
    "perturbation_wavenumber_y",
    "perturbation_wavenumber_z",
    "vth_over_c_x",
    "vth_over_c_y",
    "vth_over_c_z",
    "drift_speed_x",
    "drift_speed_y",
    "drift_speed_z",
]

DIFFERENTIABLE_SPECIES_PARAMETERS = list(dict.fromkeys(
    DIFFERENTIABLE_ION_PARAMETERS + DIFFERENTIABLE_ELECTRON_PARAMETERS
))

DIFFERENTIABLE_LEGACY_SPECIES_PARAMETERS = [
    "grid_points_per_Debye_length",
    "weight",
    "electron_charge_over_elementary_charge",
    "ion_charge_over_elementary_charge",
    "ion_mass_over_proton_mass",
    "amplitude_perturbation_x",
    "amplitude_perturbation_y",
    "amplitude_perturbation_z",
    "wavenumber_electrons_x",
    "wavenumber_electrons_y",
    "wavenumber_electrons_z",
    "wavenumber_ions_x",
    "wavenumber_ions_y",
    "wavenumber_ions_z",
    "vth_electrons_over_c_x",
    "vth_electrons_over_c_y",
    "vth_electrons_over_c_z",
    "ion_temperature_over_electron_temperature_x",
    "ion_temperature_over_electron_temperature_y",
    "ion_temperature_over_electron_temperature_z",
    "electron_drift_speed_x",
    "electron_drift_speed_y",
    "electron_drift_speed_z",
    "ion_drift_speed_x",
    "ion_drift_speed_y",
    "ion_drift_speed_z",
]

def merge_species_parameter_dicts(species_parameters, input_parameters):
    species_parameters = {**species_parameters}
    for key, value in input_parameters.items():
        if isinstance(value, dict) and isinstance(species_parameters.get(key), dict):
            species_parameters[key] = {**species_parameters[key], **value}
        else:
            species_parameters[key] = value
    return species_parameters

def clean_and_initialize_species_parameters(species_parameters, input_parameters={}):

    ##########   Do a 1-D to catch everything from input parameters
    ##########   Then do the existing structure to set stuff up in nested dicts using the 1d as the defauls
    ##########   Then fill in with species parameters and input parameters dicts to catch everyhitng else I guess?

    default_input_parameters = {
        "number_pseudoelectrons": 500,   # Number of pseudoelectrons
        "grid_points_per_Debye_length": 2,    # Specify Debye length to get a weight
        "weight": 0,                          # Specify weight directly (overrides grid_points_per_Debye_length if both specified)

        "electron_charge_over_elementary_charge": -1,   # Electron charge in units of the elementary charge
        "ion_charge_over_elementary_charge": 1,   # Ion charge in units of the elementary charge
        "ion_mass_over_proton_mass": 1,           # Ion mass in units of the proton mass

        "random_positions_x": False,          # Whether to use random positions in x for this species (overrides linspace if False)
        "random_positions_y": True,          # Whether to use random positions in y for this species (overrides linspace if False)
        "random_positions_z": True,          # Whether to use random positions in z for this species (overrides linspace if False)
        "amplitude_perturbation_x": 1e-7,      # Amplitude of sinusoidal (sin) perturbation in x position for this species
        "amplitude_perturbation_y": 0.0,       # Amplitude of sinusoidal (sin) perturbation in y position for this species
        "amplitude_perturbation_z": 0.0,       # Amplitude of sinusoidal (sin) perturbation in z position for this species
        "wavenumber_electrons_x": 8,        # Wavenumber of sinusoidal electron density perturbation in x (factor of 2pi/length)
        "wavenumber_electrons_y": 0,        # Wavenumber of sinusoidal electron density perturbation in y (factor of 2pi/length)
        "wavenumber_electrons_z": 0,        # Wavenumber of sinusoidal electron density perturbation in z (factor of 2pi/length)
        "wavenumber_ions_x": 0,         # Wavenumber of sinusoidal ion density perturbation in x (factor of 2pi/length)
        "wavenumber_ions_y": 0,         # Wavenumber of sinusoidal ion density perturbation in y (factor of 2pi/length)
        "wavenumber_ions_z": 0,         # Wavenumber of sinusoidal ion density perturbation in z (factor of 2pi/length)

        "vth_electrons_over_c_x": 0.05,           # Thermal velocity of electrons over speed of light in the x direction
        "vth_electrons_over_c_y": 0.0,            # Thermal velocity of electrons over speed of light in the y direction
        "vth_electrons_over_c_z": 0.0,            # Thermal velocity of electrons over speed of light in the z direction
        "ion_temperature_over_electron_temperature_x": 1, # Temperature ratio of ions to electrons in the x direction
        "ion_temperature_over_electron_temperature_y": 1, # Temperature ratio of ions to electrons in the y direction
        "ion_temperature_over_electron_temperature_z": 1, # Temperature ratio of ions to electrons in the z direction

        "electron_drift_speed_x": 1e8,              # Drift speed of electrons in the x direction
        "electron_drift_speed_y": 0,              # Drift speed of electrons in the y direction
        "electron_drift_speed_z": 0,              # Drift speed of electrons in the z direction
        "ion_drift_speed_x": 0,                   # Drift speed of ions in the x direction
        "ion_drift_speed_y": 0,                   # Drift speed of ions in the y direction
        "ion_drift_speed_z": 0,                   # Drift speed of ions in the z direction

        "velocity_plus_minus_electrons_x": True,       # Whether to create two groups of electrons moving in opposite directions in the x direction
        "velocity_plus_minus_electrons_y": False,       # Whether to create two groups of electrons moving in opposite directions in the y direction
        "velocity_plus_minus_electrons_z": False,       # Whether to create two groups of electrons moving in opposite directions in the z direction
        "velocity_plus_minus_ions_x": False,       # Whether to create two groups of ions moving in opposite directions in the x direction
        "velocity_plus_minus_ions_y": False,       # Whether to create two groups of ions moving in opposite directions in the y direction
        "velocity_plus_minus_ions_z": False,       # Whether to create two groups of ions moving in opposite directions in the z direction
    }
    # Tracking whether or not legacy data input was used for ions and electrons and adding it to the legacy input dict if so
    ions_default_changed = False
    for key in default_input_parameters.keys():
        if key in species_parameters.keys():
            default_input_parameters[key] = species_parameters.pop(key)
            if not ions_default_changed:
                ions_default_changed = True
                print("Warning: You are using legacy input methods for ion parameters. Please update your input file to use the new nested structure for species parameters. See the documentation for more details.")
        if key in input_parameters.keys():
            default_input_parameters[key] = input_parameters.pop(key)
            if not ions_default_changed:
                ions_default_changed = True
                print("Warning: You are using legacy input methods for ion parameters. Please update your input file to use the new nested structure for species parameters. See the documentation for more details.")
    electrons_default_changed = False
    for key in default_input_parameters.keys():
        if key in species_parameters.keys():
            default_input_parameters[key] = species_parameters[key]
            if not electrons_default_changed:
                electrons_default_changed = True
                print("Warning: You are using legacy input methods for electron parameters. Please update your input file to use the new nested structure for species parameters. See the documentation for more details.")
        if key in input_parameters.keys():
            default_input_parameters[key] = input_parameters[key]
            if not electrons_default_changed:
                electrons_default_changed = True
                print("Warning: You are using legacy input methods for electron parameters. Please update your input file to use the new nested structure for species parameters. See the documentation for more details.")

    # Put together the two stream default dicts using legacy info if it exists
    _legacy_ions = {
        "number_pseudoparticles": default_input_parameters["number_pseudoelectrons"],
        "grid_points_per_Debye_length": default_input_parameters["grid_points_per_Debye_length"],
        "weight": default_input_parameters["weight"],

        "charge_over_elementary_charge": default_input_parameters["ion_charge_over_elementary_charge"],
        "mass_over_proton_mass": default_input_parameters["ion_mass_over_proton_mass"],

        "perturbation_amplitude_x": default_input_parameters["amplitude_perturbation_x"],
        "perturbation_amplitude_y": default_input_parameters["amplitude_perturbation_y"],
        "perturbation_amplitude_z": default_input_parameters["amplitude_perturbation_z"],
        "perturbation_wavenumber_x": default_input_parameters["wavenumber_ions_x"],
        "perturbation_wavenumber_y": default_input_parameters["wavenumber_ions_y"],
        "perturbation_wavenumber_z": default_input_parameters["wavenumber_ions_z"],

        "random_positions_x": default_input_parameters["random_positions_x"],
        "random_positions_y": default_input_parameters["random_positions_y"],
        "random_positions_z": default_input_parameters["random_positions_z"],

        "vth_over_c_x": '_electrons0',
        "vth_over_c_y": '_electrons0',
        "vth_over_c_z": '_electrons0',
        "ion_temperature_over_electron_temperature_x": default_input_parameters["ion_temperature_over_electron_temperature_x"],
        "ion_temperature_over_electron_temperature_y": default_input_parameters["ion_temperature_over_electron_temperature_y"],
        "ion_temperature_over_electron_temperature_z": default_input_parameters["ion_temperature_over_electron_temperature_z"],

        "drift_speed_x": default_input_parameters["ion_drift_speed_x"],
        "drift_speed_y": default_input_parameters["ion_drift_speed_y"],
        "drift_speed_z": default_input_parameters["ion_drift_speed_z"],

        "velocity_plus_minus_x": default_input_parameters["velocity_plus_minus_ions_x"],
        "velocity_plus_minus_y": default_input_parameters["velocity_plus_minus_ions_y"],
        "velocity_plus_minus_z": default_input_parameters["velocity_plus_minus_ions_z"],

        "seed_position_override": False,
        "seed_position": None,
    }

    _legacy_electrons = {
        "number_pseudoparticles": default_input_parameters["number_pseudoelectrons"],
        "grid_points_per_Debye_length": default_input_parameters["grid_points_per_Debye_length"],
        "weight": default_input_parameters["weight"],

        "charge_over_elementary_charge": default_input_parameters["electron_charge_over_elementary_charge"],

        "perturbation_amplitude_x": default_input_parameters["amplitude_perturbation_x"],
        "perturbation_amplitude_y": default_input_parameters["amplitude_perturbation_y"],
        "perturbation_amplitude_z": default_input_parameters["amplitude_perturbation_z"],
        "perturbation_wavenumber_x": default_input_parameters["wavenumber_electrons_x"],
        "perturbation_wavenumber_y": default_input_parameters["wavenumber_electrons_y"],
        "perturbation_wavenumber_z": default_input_parameters["wavenumber_electrons_z"],

        "random_positions_x": default_input_parameters["random_positions_x"],
        "random_positions_y": default_input_parameters["random_positions_y"],
        "random_positions_z": default_input_parameters["random_positions_z"],

        "vth_over_c_x": default_input_parameters["vth_electrons_over_c_x"],
        "vth_over_c_y": default_input_parameters["vth_electrons_over_c_y"],
        "vth_over_c_z": default_input_parameters["vth_electrons_over_c_z"],

        "drift_speed_x": default_input_parameters["electron_drift_speed_x"],
        "drift_speed_y": default_input_parameters["electron_drift_speed_y"],
        "drift_speed_z": default_input_parameters["electron_drift_speed_z"],

        "velocity_plus_minus_x": default_input_parameters["velocity_plus_minus_electrons_x"],
        "velocity_plus_minus_y": default_input_parameters["velocity_plus_minus_electrons_y"],
        "velocity_plus_minus_z": default_input_parameters["velocity_plus_minus_electrons_z"],

        "seed_position_override": False,
        "seed_position": None,
    }

    # Get the non-legacy parameters from the species_parameters and input_parameters dicts and put them in new dicts for ions and electrons
    ions_dict = merge_species_parameter_dicts(species_parameters.pop("ions", {}), input_parameters.pop("ions", {}))
    electrons_dict = merge_species_parameter_dicts(species_parameters.pop("electrons", {}), input_parameters.pop("electrons", {}))

    # Pack the ion data into a dict if it wasn't already in one
    ions_new_data = False
    for item in ions_dict.values():
        if isinstance(item, dict):
            if not ions_new_data:
                ions_new_data = True

    # Check for loose data in the dict
    if not ions_new_data and len(ions_dict) > 0:
        ions_dict = {"ions0": {**ions_dict}}
        ions_new_data = True

    # Pack the electron data into a dict if it wasn't already in one
    electrons_new_data = False
    for item in electrons_dict.values():
        if isinstance(item, dict):
            if not electrons_new_data:
                electrons_new_data = True
    
    # Check for loose data in the dict
    if not electrons_new_data and len(electrons_dict) > 0:
        electrons_dict = {"electrons0": {**electrons_dict}}
        electrons_new_data = True
    
    # If there wasn't any legacy data, merge the non-legacy data into the two stream default
    if not ions_default_changed:
        if len(ions_dict) > 0:
            first_key = next(iter(ions_dict.keys()))
            ions_dict[first_key] = {**_legacy_ions, **ions_dict[first_key]}
        else:
            ions_dict = {"_legacy_ions": _legacy_ions}
    # Otherwise add the legacy data merged into the two stream default as its own species
    # This assumes that legacy input and non-legacy input are separate and that legacy input comes first
    else:
        ions_dict["_legacy_ions"] = _legacy_ions
    
    # Do the same for electrons
    if not electrons_default_changed:
        if len(electrons_dict) > 0:
            first_key = next(iter(electrons_dict.keys()))
            electrons_dict[first_key] = {**_legacy_electrons, **electrons_dict[first_key]}
        else:
            electrons_dict = {"_legacy_electrons": _legacy_electrons}
    else:
        electrons_dict["_legacy_electrons"] = _legacy_electrons

    # Set the ion and electron default values
    _default_ion_dict = {
        "number_pseudoparticles": 500,
        "grid_points_per_Debye_length": 2,
        "weight": 0,

        "charge_over_elementary_charge": 1,
        "mass_over_proton_mass": 1,

        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
        "perturbation_wavenumber_x": 0,
        "perturbation_wavenumber_y": 0,
        "perturbation_wavenumber_z": 0,
        "random_positions_x": False,
        "random_positions_y": True,
        "random_positions_z": True,

        "vth_over_c_x": 0,
        "vth_over_c_y": 0,
        "vth_over_c_z": 0,
        "ion_temperature_over_electron_temperature_x": 1,
        "ion_temperature_over_electron_temperature_y": 1,
        "ion_temperature_over_electron_temperature_z": 1,

        "drift_speed_x": 0,
        "drift_speed_y": 0,
        "drift_speed_z": 0,

        "velocity_plus_minus_x": False,
        "velocity_plus_minus_y": False,
        "velocity_plus_minus_z": False,

        "seed_position_override": False,
        "seed_position": None
    }

    _default_electron_dict = {
        "number_pseudoparticles": 500,
        "grid_points_per_Debye_length": 2,
        "weight": 0,

        "charge_over_elementary_charge": -1,

        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
        "perturbation_wavenumber_x": 0,
        "perturbation_wavenumber_y": 0,
        "perturbation_wavenumber_z": 0,
        "random_positions_x": False,
        "random_positions_y": True,
        "random_positions_z": True,

        "vth_over_c_x": 0,
        "vth_over_c_y": 0,
        "vth_over_c_z": 0,

        "drift_speed_x": 0,
        "drift_speed_y": 0,
        "drift_speed_z": 0,

        "velocity_plus_minus_x": False,
        "velocity_plus_minus_y": False,
        "velocity_plus_minus_z": False,

        "seed_position_override": False,
        "seed_position": None
    }

    # Merge the default ion dict into each ion dict and the default electron dict into each electron dict
    for key in ions_dict.keys():
        ions_dict[key] = {**_default_ion_dict, **ions_dict[key]}
    for key in electrons_dict.keys():
        electrons_dict[key] = {**_default_electron_dict, **electrons_dict[key]}
    
    ions_dict_out = {}
    electrons_dict_out = {}
    if '_legacy_ions' in ions_dict.keys():
        ions_dict_out['_ions0'] = ions_dict.pop('_legacy_ions')
        ions_dict_out['_ions0']['user_label'] = '_legacy_ions'
        ind = 1
    else:
        ind = 0
    for key in ions_dict.keys():
        ions_dict_out[f'_ions{ind}'] = ions_dict[key]
        ions_dict_out[f'_ions{ind}']['user_label'] = key
        ind += 1
    if '_legacy_electrons' in electrons_dict.keys():
        electrons_dict_out['_electrons0'] = electrons_dict.pop('_legacy_electrons')
        electrons_dict_out['_electrons0']['user_label'] = '_legacy_electrons'
        ind = 1
    else:
        ind = 0
    for key in electrons_dict.keys():
        electrons_dict_out[f'_electrons{ind}'] = electrons_dict[key]
        electrons_dict_out[f'_electrons{ind}']['user_label'] = key
        ind += 1
    
    species_parameters["ions"] = ions_dict_out
    species_parameters["electrons"] = electrons_dict_out

    # Now fill in all the cross referenced information
    all_ion_species = []
    all_electron_species = []
    for species in species_parameters['ions'].keys():
        all_ion_species.append(species)
    for species in species_parameters['electrons'].keys():
        all_electron_species.append(species)
    for type_species in ['ions', 'electrons']:
        for species in species_parameters[type_species].keys():
            sp = species_parameters[type_species][species]
            for key, value in sp.items():
                if value in all_ion_species and not key == 'user_label':
                    assert type(species_parameters['ions'][value][key]) != str, f"Cross referenced species values cannot reference another reference. For {type_species} {species} referencing {value}, got {species_parameters['ions'][key]}."
                    if key == "vth_over_c_x":
                        sp[key] = jnp.sqrt(1 / sp['ion_temperature_over_electron_temperature_x']) * species_parameters['ions'][value][key] * jnp.sqrt(species_parameters['ions'][value]['mass_over_proton_mass'] * mass_proton / mass_electron)
                    elif key == "vth_over_c_y":
                        sp[key] = jnp.sqrt(1 / sp['ion_temperature_over_electron_temperature_y']) * species_parameters['ions'][value][key] * jnp.sqrt(species_parameters['ions'][value]['mass_over_proton_mass'] * mass_proton / mass_electron)
                    elif key == "vth_over_c_z":
                        sp[key] = jnp.sqrt(1 / sp['ion_temperature_over_electron_temperature_z']) * species_parameters['ions'][value][key] * jnp.sqrt(species_parameters['ions'][value]['mass_over_proton_mass'] * mass_proton / mass_electron)
                    else:
                        sp[key] = species_parameters['ions'][value][key]
                elif value in all_electron_species and not key == 'user_label':
                    assert type(species_parameters['electrons'][value][key]) != str, f"Cross referenced species values cannot reference another reference. For {type_species} {species} referencing {value}, got {species_parameters['electrons'][key]}."
                    if key == "vth_over_c_x":
                        sp[key] = jnp.sqrt(sp['ion_temperature_over_electron_temperature_x']) * species_parameters['electrons'][value][key] * jnp.sqrt(mass_electron / (sp['mass_over_proton_mass'] * mass_proton))
                    elif key == "vth_over_c_y":
                        sp[key] = jnp.sqrt(sp['ion_temperature_over_electron_temperature_y']) * species_parameters['electrons'][value][key] * jnp.sqrt(mass_electron / (sp['mass_over_proton_mass'] * mass_proton))
                    elif key == "vth_over_c_z":
                        sp[key] = jnp.sqrt(sp['ion_temperature_over_electron_temperature_z']) * species_parameters['electrons'][value][key] * jnp.sqrt(mass_electron / (sp['mass_over_proton_mass'] * mass_proton))
                    else:
                        sp[key] = species_parameters['ions'][value][key]
                elif type(value) == str and not key == 'user_label':
                    raise ValueError(f'Cross referenced species value did not reference another species. {type_species} {species} referenced {value} for {key}. Valid references are {all_ion_species} referencing ions or {all_electron_species} referencing electrons.')

    # Now do all the assert statements to check that the parameters are valid
    for type_species in ['ions', 'electrons']:
        for species in species_parameters[type_species].keys():
            sp = species_parameters[type_species][species]
            assert type(sp["number_pseudoparticles"]) == int and sp["number_pseudoparticles"] > 0, f"Number of pseudoparticles for {species} must be a positive integer. Got {sp['number_pseudoparticles']}."
            assert sp["grid_points_per_Debye_length"] > 0, f"Grid points per Debye length must be positive. Got {sp['grid_points_per_Debye_length']}."
            sp["weight"] = as_float_parameter(sp["weight"])
            assert sp["weight"] >= 0, f"Weight must be non-negative. Got {sp['weight']}."
            sp["charge_over_elementary_charge"] = as_float_parameter(sp["charge_over_elementary_charge"])
            #assert sp["charge_over_elementary_charge"] != 0, f"Charge over elementary charge must be non-zero. Got {sp['charge_over_elementary_charge']}."
            if type_species == 'ions':
                sp["mass_over_proton_mass"] = as_float_parameter(sp["mass_over_proton_mass"])
                assert sp["mass_over_proton_mass"] > 0, f"Mass over proton mass must be positive. Got {sp['mass_over_proton_mass']}."
                sp["ion_temperature_over_electron_temperature_x"] = as_float_parameter(sp["ion_temperature_over_electron_temperature_x"])
                assert sp["ion_temperature_over_electron_temperature_x"] >=0, f"Ion temperature over electron temperature x must be positive. Got {sp["ion_temperature_over_electron_temperature_x"]}."
                sp["ion_temperature_over_electron_temperature_y"] = as_float_parameter(sp["ion_temperature_over_electron_temperature_y"])
                assert sp["ion_temperature_over_electron_temperature_y"] >=0, f"Ion temperature over electron temperature y must be positive. Got {sp["ion_temperature_over_electron_temperature_y"]}."
                sp["ion_temperature_over_electron_temperature_z"] = as_float_parameter(sp["ion_temperature_over_electron_temperature_z"])
                assert sp["ion_temperature_over_electron_temperature_z"] >=0, f"Ion temperature over electron temperature z must be positive. Got {sp["ion_temperature_over_electron_temperature_z"]}."
            sp["perturbation_amplitude_x"] = as_float_parameter(sp["perturbation_amplitude_x"])
            sp["perturbation_amplitude_y"] = as_float_parameter(sp["perturbation_amplitude_y"])
            sp["perturbation_amplitude_z"] = as_float_parameter(sp["perturbation_amplitude_z"])
            sp["perturbation_wavenumber_x"] = as_float_parameter(sp["perturbation_wavenumber_x"])
            sp["perturbation_wavenumber_y"] = as_float_parameter(sp["perturbation_wavenumber_y"])
            sp["perturbation_wavenumber_z"] = as_float_parameter(sp["perturbation_wavenumber_z"])
            assert type(sp["random_positions_x"]) == bool, f"Random position x must be a boolean. Got {sp["random_positions_x"]}."
            assert type(sp["random_positions_y"]) == bool, f"Random position y must be a boolean. Got {sp["random_positions_y"]}."
            assert type(sp["random_positions_z"]) == bool, f"Random position z must be a boolean. Got {sp["random_positions_z"]}."
            sp["vth_over_c_x"] = as_float_parameter(sp["vth_over_c_x"])
            # Assert positive or less than the speed of light?
            sp["vth_over_c_y"] = as_float_parameter(sp["vth_over_c_y"])
            #
            sp["vth_over_c_z"] = as_float_parameter(sp["vth_over_c_z"])
            #
            sp["drift_speed_x"] = as_float_parameter(sp["drift_speed_x"])
            # Assert less than the speed of light?
            sp["drift_speed_y"] = as_float_parameter(sp["drift_speed_y"])
            # Assert less than the speed of light?
            sp["drift_speed_z"] = as_float_parameter(sp["drift_speed_z"])
            # Assert less than the speed of light?
            assert type(sp["velocity_plus_minus_x"]) == bool, f"Velocity plus minus x must be a boolean. Got {sp["velocity_plus_minus_x"]}."
            assert type(sp["velocity_plus_minus_y"]) == bool, f"Velocity plus minus y must be a boolean. Got {sp["velocity_plus_minus_y"]}."
            assert type(sp["velocity_plus_minus_z"]) == bool, f"Velocity plus minus z must be a boolean. Got {sp["velocity_plus_minus_z"]}."
            assert type(sp["seed_position_override"]) == bool, f"Seed position override must be a boolean. Got {sp["seed_position_override"]}."
            if sp["seed_position_override"]:
                assert type(sp["seed_position"]) == int and sp["seed_position"] > 0, f"Seed position must be a positive integer. Got {sp["seed_position"]}."
    
    return species_parameters

def build_species_hash(species_parameters):
    hash_list_pre_species = ['nonspeciesdata']
    for key, value in species_parameters.items():
        if key not in ['ions', 'electrons']:
            hash_list_pre_species.append(str(key))
            hash_list_pre_species.append(str(value))
    hash_pre_species = "".join(hash_list_pre_species)

    ion_hash_list_list = ['ions']
    for key in species_parameters['ions']:
        ion_hash_list = [str(key)]
        sp = species_parameters['ions'][key]
        for key, value in sp.items():
            ion_hash_list.append(str(key))
            ion_hash_list.append(str(value))
        ion_hash = "".join(ion_hash_list)
        ion_hash_list_list.append(ion_hash)
    ion_hash = str(tuple(ion_hash_list_list))
    
    electron_hash_list_list = ['electrons']
    for key in species_parameters['electrons']:
        electron_hash_list = [str(key)]
        sp = species_parameters['electrons'][key]
        for key, value in sp.items():
            electron_hash_list.append(str(key))
            electron_hash_list.append(str(value))
        electron_hash = "".join(electron_hash_list)
        electron_hash_list_list.append(electron_hash)
    electron_hash = str(tuple(electron_hash_list_list))

    species_hash = str((hash_pre_species, ion_hash, electron_hash))
    return species_hash
