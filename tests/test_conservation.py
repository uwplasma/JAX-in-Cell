import numpy as np
import pytest

from jaxincell._constants import epsilon_0
from jaxincell._diagnostics import diagnostics
from jaxincell._simulation import Simulation


def two_stream_parameters(boundary=0, filter_passes=0, total_steps=80):
    electrons = {
        "number_pseudoparticles": 400,
        "grid_points_per_Debye_length": 0.5,
        "charge_over_elementary_charge": -1,
        "perturbation_amplitude_x": 1e-4,
        "perturbation_wavenumber_x": 1,
        "vth_over_c_x": 0.05,
        "drift_speed_x": 6e7,
        "velocity_plus_minus_x": True,
    }
    ions = {
        "number_pseudoparticles": 400,
        "grid_points_per_Debye_length": 0.5,
        "charge_over_elementary_charge": 1,
        "mass_over_proton_mass": 1,
        "vth_over_c_x": 0.05,
    }
    return {
        "domain_parameters": {
            "length": 0.01,
            "number_grid_points": 32,
            "total_steps": total_steps,
            "timestep_over_spatialstep_times_c": 0.5,
            "particle_BC_left": boundary,
            "particle_BC_right": boundary,
            "field_BC_left": boundary,
            "field_BC_right": boundary,
        },
        "species_parameters": {
            "electrons": {"electrons0": electrons},
            "ions": {"ions0": ions},
        },
        "solver_parameters": {"filter_passes": filter_passes, "print_info": False},
    }


def gauss_law_residual(output, periodic):
    """max_x |dE_x/dx - rho/epsilon_0| / max_x |rho/epsilon_0| at every stored step."""
    Ex = np.asarray(output["electric_field"][..., 0])
    rho = np.asarray(output["charge_density"]) / epsilon_0
    if periodic:
        Ex_left = np.roll(Ex, 1, axis=1)
    else:
        Ex_left = np.concatenate([np.zeros_like(Ex[:, :1]), Ex[:, :-1]], axis=1)
    residual = (Ex - Ex_left) / float(output["dx"]) - rho
    return np.max(np.abs(residual), axis=1) / np.max(np.abs(rho), axis=1)


@pytest.mark.parametrize(
    "boundary, filter_passes",
    [(0, 0), (0, 5), (1, 0)],
    ids=["periodic", "periodic-filtered", "reflective"],
)
def test_explicit_scheme_keeps_gauss_law_to_round_off(boundary, filter_passes):
    # The explicit step deposits a charge-conserving current, so the discrete
    # continuity equation holds and Gauss's law, satisfied at t = 0, stays satisfied
    # by the stored E and rho at every step.
    output = Simulation(two_stream_parameters(boundary, filter_passes)).run()
    residual = gauss_law_residual(output, periodic=(boundary == 0))
    assert residual.max() < 1e-9
    diagnostics(output)
    assert np.max(np.asarray(output["gauss_error_Linf_rel"])) < 1e-9
