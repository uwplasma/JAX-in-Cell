"""The quiet two-stream problem the figure scripts share: the setup, its kinetic growth
rate, and the fit of the seeded mode."""
import numpy as np
from common import maxwellian_populations
from dispersion import electrostatic_epsilon, purely_growing_roots

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

LENGTH, CELLS, DENSITY, DRIFT, VTH = 0.01, 64, 4.37e17, 5e7, 0.05 * c
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
SEED_AK = 1e-4


def build(drift=DRIFT, n=20000, solver=None, dt_over_dx_c=4.5):
    """Two counter-streaming quiet beams of ``n`` electrons on ``n/2`` mobile protons, with
    mode 1 seeded by a displacement of amplitude a k = 1e-4."""
    electrons = Species.electrons(n=n, density=DENSITY, vth=(VTH, 0, 0), drift=(drift, 0, 0),
                                  plus_minus=True, sampling="quiet", perturbation_mode=1,
                                  perturbation_amplitude=SEED_AK * LENGTH / (2 * np.pi))
    ions = Species.ions(n=n // 2, density=DENSITY, electrons=electrons, sampling="quiet")
    return Simulation(Domain(length=LENGTH, cells=CELLS, dt_over_dx_c=dt_over_dx_c), [electrons, ions],
                      solver or Solver(filter_passes=0))


def theory(drift):
    """Growth rate of mode 1 from the kinetic dispersion relation, in units of
    omega_pe. NaN when the mode is stable."""
    populations = maxwellian_populations(build(drift))
    roots = purely_growing_roots(lambda w: electrostatic_epsilon(w, 2 * np.pi / LENGTH, populations), OMEGA_PE)
    return max(roots) / OMEGA_PE if roots else np.nan


def measure(out):
    """Rate of the seeded mode, fitted between ten times the seed amplitude and a
    tenth of saturation. The window is set by amplitude rather than by time so
    that it follows the same part of the growth at every drift and step."""
    t = np.asarray(out.t) * OMEGA_PE
    amplitude = np.abs(np.fft.rfft(np.asarray(out.E[:, :, 0]), axis=1)[:, 1]) / CELLS
    peak = int(np.argmax(amplitude))
    window = ((amplitude > 10 * amplitude[0]) & (amplitude < 0.1 * amplitude[peak])
              & (np.arange(t.size) < peak))
    if window.sum() < 10:
        return t, amplitude, None
    slope, intercept = np.polyfit(t[window], np.log(amplitude[window]), 1)
    fit = np.polyval((slope, intercept), t[window])
    r2 = 1 - np.var(np.log(amplitude[window]) - fit) / np.var(np.log(amplitude[window]))
    return t, amplitude, {"gamma": slope, "intercept": intercept, "r2": r2,
                          "t0": t[window][0], "t1": t[window][-1]}
