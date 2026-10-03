"""Linear Landau damping set-up shared by the figure scripts.

A Maxwellian electron plasma on a fixed ion background, loaded with a quiet
start: equally spaced positions with a small sinusoidal displacement, and
Maxwellian velocities taken at the quantiles of a bit-reversed (van der Corput)
sequence so that neighbouring particles have very different velocities. The
quiet start lowers the noise floor by orders of magnitude, which leaves room for
the wave to decay through several e-foldings at the linear rate.
"""
import numpy as np
from scipy.special import erfinv

from dispersion import landau_root
from jaxincell import speed_of_light

LENGTH = 1.0
GRID = 32
MODE = 1.02
GRID_POINTS_PER_DEBYE = 0.4
VTH_OVER_C = 0.35
k = 2 * np.pi * MODE / LENGTH
lambda_D = LENGTH / GRID / GRID_POINTS_PER_DEBYE
k_lambda_D = k * lambda_D
root = landau_root(k_lambda_D)
gamma_theory, omega_theory = root.imag, root.real


def base_parameters(steps, electrons):
    """Parameters of the Landau-damping box; ``electrons`` overrides the electron
    population (particle number, loading, perturbation)."""
    return {
        "domain_parameters": {"length": LENGTH, "timestep_over_spatialstep_times_c": 1.0,
                              "number_grid_points": GRID, "total_steps": steps},
        "species_parameters": {
            "electrons": {"electrons0": {"dx_over_Debye_length": GRID_POINTS_PER_DEBYE,
                                         "vth_over_c_x": VTH_OVER_C, **electrons}},
            "ions": {"ions0": {"number_pseudoparticles": 40000,
                               "dx_over_Debye_length": GRID_POINTS_PER_DEBYE,
                               "mass_over_proton_mass": 1e9, "vth_over_c_x": "_electrons0",
                               "vth_over_c_y": "_electrons0", "vth_over_c_z": "_electrons0",
                               "ion_temperature_over_electron_temperature_x": 1e-9}}},
        "solver_parameters": {"field_solver": 0, "time_evolution_algorithm": 0, "print_info": False,
                              "filter_passes": 0},
    }


def van_der_corput(n, base=2):
    """First n terms of the base-b van der Corput sequence, in (0, 1)."""
    q = np.zeros(n)
    denominator = np.ones(n)
    i = np.arange(1, n + 1)
    while i.any():
        denominator *= base
        q += (i % base) / denominator
        i //= base
    return q


def quiet_start(n, a_k):
    """Positions displaced by a sinusoid of relative amplitude a k, and Maxwellian
    velocities at the quantiles of a bit-reversed sequence."""
    a = a_k / k
    x0 = np.linspace(-LENGTH / 2, LENGTH / 2, n, endpoint=False) + LENGTH / (2 * n)
    x = x0 + a * np.sin(k * x0)
    sigma = VTH_OVER_C * speed_of_light / np.sqrt(2)
    v = sigma * np.sqrt(2) * erfinv(2 * van_der_corput(n) - 1)
    positions = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    velocities = np.stack([v, np.zeros(n), np.zeros(n)], axis=1)
    return positions, velocities


def quiet_parameters(n, a_k, steps):
    """The Landau-damping box with ``n`` quiet-start electrons."""
    positions, velocities = quiet_start(n, a_k)
    return base_parameters(steps, {"number_pseudoparticles": n, "initial_positions": positions,
                                   "initial_velocities": velocities})


def mode_one(output):
    """Complex amplitude of the first Fourier mode of E_x, per step."""
    Ex = np.asarray(output["electric_field"][:, :, 0])
    return np.fft.rfft(Ex, axis=1)[:, 1] / Ex.shape[1]


def maxima_above_floor(amplitude, factor=10.0, floor_samples=100):
    """Indices of the local maxima of |E_1| that stand ``factor`` above the noise
    floor, estimated as the mean over the last ``floor_samples`` steps."""
    floor = amplitude[-floor_samples:].mean()
    peaks = [i for i in range(1, len(amplitude) - 1)
             if amplitude[i] > amplitude[i - 1] and amplitude[i] > amplitude[i + 1]]
    return [i for i in peaks if amplitude[i] > factor * floor], floor
