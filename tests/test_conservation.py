import numpy as np
import pytest

from jaxincell._constants import elementary_charge, epsilon_0, mass_electron, speed_of_light
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


def test_electrostatic_boris_field_follows_the_updated_particles():
    # With field_solver=1 the step replaces E by Gauss's law for x^{n+1}. Solving it
    # for x^n instead lags the field by one step, an O(dt) energy error; here that
    # was 2.9e-5 over 200 steps, against 1.0e-6 without the lag.
    parameters = two_stream_parameters(total_steps=200)
    parameters["solver_parameters"]["field_solver"] = 1
    output = Simulation(parameters).run()
    diagnostics(output)
    energy = np.asarray(output["total_energy"])
    assert np.max(np.abs(energy - energy[0])) / energy[0] < 1e-5




def test_electrostatic_boris_conserves_the_momentum_of_a_drifting_plasma():
    # A cold beam streaming through a uniform ion background carries a net current. The Ampere
    # half steps grew a uniform E_x from it, which the Gauss solve then removed, but the push had
    # already felt it: the beam slowed by ~ (omega_pe dt)^2 / 2 per step. Without the mean current
    # the scheme is Vlasov-Poisson and the beam keeps its momentum.
    length, grid_points, particles = 0.01, 32, 3200
    dt_over_dx_c, omega_dt = 0.5, 0.2
    dt = dt_over_dx_c * length / grid_points / speed_of_light
    weight = (omega_dt / dt) ** 2 * epsilon_0 * mass_electron / elementary_charge**2 * length / particles
    species = {"number_pseudoparticles": particles, "weight": weight, "perturbation_amplitude_x": 0.0,
               "vth_over_c_x": 0.0, "vth_over_c_y": 0.0, "vth_over_c_z": 0.0}
    parameters = {
        "domain_parameters": {"length": length, "number_grid_points": grid_points, "total_steps": 50,
                              "timestep_over_spatialstep_times_c": dt_over_dx_c},
        "species_parameters": {
            "electrons": {"electrons0": {**species, "charge_over_elementary_charge": -1, "drift_speed_x": 1e6,
                                         "velocity_plus_minus_x": False}},
            "ions": {"ions0": {**species, "charge_over_elementary_charge": 1, "mass_over_proton_mass": 1e9}},
        },
        "solver_parameters": {"field_solver": 1, "filter_passes": 0, "print_info": False},
    }
    output = Simulation(parameters).run()
    diagnostics(output)
    # Electron momentum: a uniform E_x exerts no net force on the neutral plasma, so the total
    # (with the ions) would hide the effect.
    beam_velocity = np.mean(np.asarray(output["velocity_electrons"])[:, :, 0], axis=1)
    assert np.max(np.abs(beam_velocity - 1e6)) < 1e-3 * 1e6
