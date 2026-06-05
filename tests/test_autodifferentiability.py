from copy import deepcopy
from dataclasses import dataclass

import jax.numpy as jnp
import pytest
from jax import block_until_ready, grad

from jaxincell import Simulation
from jaxincell._parameters import (
    DIFFERENTIABLE_DOMAIN_PARAMETERS,
    DIFFERENTIABLE_ELECTRON_PARAMETERS,
    DIFFERENTIABLE_ION_PARAMETERS,
    DIFFERENTIABLE_LEGACY_SPECIES_PARAMETERS,
    DIFFERENTIABLE_SOLVER_PARAMETERS,
)


@dataclass(frozen=True)
class DifferentiableParameterCase:
    section: str
    key: str
    value: float

    @property
    def id(self):
        return f"{self.section}:{self.key}"


PARAMETER_VALUES = {
    "amplitude_perturbation_x": 1e-7,
    "amplitude_perturbation_y": 1e-7,
    "amplitude_perturbation_z": 1e-7,
    "charge_over_elementary_charge": 1.0,
    "drift_speed_x": 1.0,
    "drift_speed_y": 1.0,
    "drift_speed_z": 1.0,
    "electron_charge_over_elementary_charge": -1.0,
    "electron_drift_speed_x": 1.0,
    "electron_drift_speed_y": 1.0,
    "electron_drift_speed_z": 1.0,
    "filter_alpha": 0.5,
    "grid_points_per_Debye_length": 1.0,
    "ion_charge_over_elementary_charge": 1.0,
    "ion_drift_speed_x": 1.0,
    "ion_drift_speed_y": 1.0,
    "ion_drift_speed_z": 1.0,
    "ion_mass_over_proton_mass": 1.0,
    "ion_temperature_over_electron_temperature_x": 1.0,
    "ion_temperature_over_electron_temperature_y": 1.0,
    "ion_temperature_over_electron_temperature_z": 1.0,
    "length": 0.01,
    "length_y": 0.01,
    "length_z": 0.01,
    "mass_over_proton_mass": 1.0,
    "perturbation_amplitude_x": 1e-7,
    "perturbation_amplitude_y": 1e-7,
    "perturbation_amplitude_z": 1e-7,
    "perturbation_wavenumber_x": 1.0,
    "perturbation_wavenumber_y": 1.0,
    "perturbation_wavenumber_z": 1.0,
    "timestep_over_spatialstep_times_c": 0.5,
    "vth_electrons_over_c_x": 0.01,
    "vth_electrons_over_c_y": 0.01,
    "vth_electrons_over_c_z": 0.01,
    "vth_over_c_x": 0.01,
    "vth_over_c_y": 0.01,
    "vth_over_c_z": 0.01,
    "wavenumber_electrons_x": 1.0,
    "wavenumber_electrons_y": 1.0,
    "wavenumber_electrons_z": 1.0,
    "wavenumber_ions_x": 1.0,
    "wavenumber_ions_y": 1.0,
    "wavenumber_ions_z": 1.0,
    "weight": 1.0,
}


def parameter_value(key):
    try:
        return PARAMETER_VALUES[key]
    except KeyError:
        raise KeyError(f"No autodifferentiability test value configured for {key!r}.")


def differentiable_parameter_cases():
    cases = []
    for key in DIFFERENTIABLE_DOMAIN_PARAMETERS:
        cases.append(DifferentiableParameterCase("domain", key, parameter_value(key)))
    for key in DIFFERENTIABLE_SOLVER_PARAMETERS:
        cases.append(DifferentiableParameterCase("solver", key, parameter_value(key)))
    for key in DIFFERENTIABLE_LEGACY_SPECIES_PARAMETERS:
        cases.append(DifferentiableParameterCase("legacy_species", key, parameter_value(key)))
    for key in DIFFERENTIABLE_ION_PARAMETERS:
        cases.append(DifferentiableParameterCase("ions", key, parameter_value(key)))
    for key in DIFFERENTIABLE_ELECTRON_PARAMETERS:
        cases.append(DifferentiableParameterCase("electrons", key, parameter_value(key)))
    return cases


def base_simulation_parameters():
    base_species = {
        "number_pseudoparticles": 2,
        "grid_points_per_Debye_length": 1.0,
        "weight": 1.0,
        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
        "perturbation_wavenumber_x": 1.0,
        "perturbation_wavenumber_y": 1.0,
        "perturbation_wavenumber_z": 1.0,
        "random_positions_x": False,
        "random_positions_y": False,
        "random_positions_z": False,
        "vth_over_c_x": 0.01,
        "vth_over_c_y": 0.01,
        "vth_over_c_z": 0.01,
        "drift_speed_x": 1.0,
        "drift_speed_y": 0.0,
        "drift_speed_z": 0.0,
        "velocity_plus_minus_x": False,
        "velocity_plus_minus_y": False,
        "velocity_plus_minus_z": False,
    }
    return {
        "domain_parameters": {
            "total_steps": 1,
            "number_grid_points": 4,
            "number_grid_points_y": 3,
            "number_grid_points_z": 3,
            "length": 0.01,
            "length_y": 0.01,
            "length_z": 0.01,
        },
        "species_parameters": {
            "ions": {
                "ions0": {
                    **base_species,
                    "charge_over_elementary_charge": 1.0,
                    "mass_over_proton_mass": 1.0,
                    "ion_temperature_over_electron_temperature_x": 1.0,
                    "ion_temperature_over_electron_temperature_y": 1.0,
                    "ion_temperature_over_electron_temperature_z": 1.0,
                },
            },
            "electrons": {
                "electrons0": {
                    **base_species,
                    "charge_over_elementary_charge": -1.0,
                },
            },
        },
        "solver_parameters": {
            "filter_passes": 0,
            "filter_alpha": 0.5,
            "print_info": False,
        },
    }


def set_runtime_parameter(runtime_input_parameters, parameter_case, value):
    if parameter_case.section in ("domain", "solver", "legacy_species"):
        runtime_input_parameters[parameter_case.key] = value
    elif parameter_case.section == "ions":
        runtime_input_parameters.setdefault("ions", {}).setdefault("ions0", {})[parameter_case.key] = value
    elif parameter_case.section == "electrons":
        runtime_input_parameters.setdefault("electrons", {}).setdefault("electrons0", {})[parameter_case.key] = value
    else:
        raise ValueError(f"Unknown differentiable parameter section {parameter_case.section!r}.")


def objective_from_output(output):
    return (
        jnp.mean(output["electric_field"])
        + 1e-20 * jnp.mean(output["positions"])
        + 1e-20 * jnp.mean(output["velocities"])
        + 1e-30 * jnp.mean(output["charges"])
        + 1e-30 * jnp.mean(output["masses"])
    )


@pytest.mark.parametrize(
    "parameter_case",
    differentiable_parameter_cases(),
    ids=[case.id for case in differentiable_parameter_cases()],
)
def test_differentiable_parameter_forward_and_grad_pass(parameter_case):
    sim = Simulation(base_simulation_parameters())
    base_input_parameters = deepcopy(sim.input_parameters)

    def objective(value):
        runtime_input_parameters = deepcopy(base_input_parameters)
        set_runtime_parameter(runtime_input_parameters, parameter_case, value)
        output = sim.run(runtime_input_parameters)
        return objective_from_output(output)

    value = jnp.asarray(parameter_case.value, dtype=float)
    forward_value = block_until_ready(objective(value))
    gradient_value = block_until_ready(grad(objective)(value))

    assert jnp.all(jnp.isfinite(forward_value))
    assert jnp.all(jnp.isfinite(gradient_value))
