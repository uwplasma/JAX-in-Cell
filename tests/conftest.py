"""Small configurations and the measurements the physics tests make, and the guard that
keeps a comparison from being made against nothing."""
import numpy as np
import pytest

from jaxincell import Domain, Simulation, Solver, Species, epsilon_0, mass_electron
from jaxincell import elementary_charge as e_charge
from jaxincell import speed_of_light as c

_UNSET = object()
_approx, _allclose = pytest.approx, np.allclose


def _check(expected, floor, call):
    """Refuse a comparison whose expected value is below the absolute tolerance it keeps.

    ``pytest.approx`` keeps ``abs=1e-12`` and ``np.allclose`` an ``atol`` of ``1e-8``
    *whatever relative tolerance is asked for*. A wall energy of 5e-20 J, a time of 2e-9 s or
    a field of 1e-23 V/m is below both, so the comparison passes for the right answer, for
    zero, for the wrong sign and for anything else that small: it is not a test. Three of them
    were in this suite, one of them standing in for an analytic control of an impact that
    never happened.

    A comparison of an SI-scale quantity therefore has to say what absolute tolerance it
    means -- ``abs=0`` or ``atol=0`` for a purely relative one, or a number taken from a
    physical scale. Passing one explicitly is taken at face value; it is the silent default
    that is refused.
    """
    try:
        size = float(np.max(np.abs(np.asarray(expected, dtype=float))))
    except (TypeError, ValueError):
        return
    if size < 10 * floor:
        raise AssertionError(
            f"{call} compares against {size:.3e}, which the default absolute tolerance of "
            f"{floor:.0e} swallows: zero and the wrong sign would pass too. Give abs=0 or "
            "atol=0 for a purely relative comparison, or an absolute tolerance from a "
            "physical scale.")


def approx(expected, rel=None, abs=None, nan_ok=False):
    if abs is None:
        _check(expected, 1e-12, "pytest.approx")
    return _approx(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def allclose(a, b, rtol=1e-05, atol=_UNSET, **kwargs):
    if atol is _UNSET:
        _check(b, 1e-08, "np.allclose")
    return _allclose(a, b, rtol=rtol, atol=1e-08 if atol is _UNSET else atol, **kwargs)


pytest.approx = approx
np.allclose = allclose


def density_for(omega_pe):
    """Electron density that gives the plasma frequency ``omega_pe``."""
    return omega_pe ** 2 * epsilon_0 * mass_electron / e_charge ** 2


def electron_plasma(n, length, cells, omega_pe_dt, vth_over_c, mode=1.0, amplitude_k=0.0,
                    drift=0.0, plus_minus=False):
    """Quiet electrons on a neutralising background of heavy ions, sized so that
    the plasma frequency and the cell size give ``omega_pe * dt`` with
    ``dt = dx / c``. Returns the simulation and its plasma frequency."""
    omega_pe = omega_pe_dt * c * cells / length
    n_e = density_for(omega_pe)
    k = 2 * np.pi * mode / length
    electrons = Species.electrons(n=n, density=n_e, vth=(vth_over_c * c, 0.0, 0.0), drift=(drift, 0.0, 0.0),
                                  perturbation_amplitude=amplitude_k / k, perturbation_mode=mode,
                                  plus_minus=plus_minus, quiet=True)
    ions = Species.ions(n=n // 4, density=n_e, mass_ratio=1e9, vth=(0.0, 0.0, 0.0), quiet=True)
    return Simulation(Domain(length=length, cells=cells, dt_over_dx_c=1.0), [electrons, ions],
                      Solver(filter_passes=0)), omega_pe


def mode_amplitude(out, mode):
    """Complex Fourier amplitude of one mode of E_x at every stored step."""
    return np.fft.rfft(np.asarray(out.E[:, :, 0]), axis=1)[:, mode] / out.E.shape[1]


def maxima(amplitude, above=0.0):
    """Indices of the local maxima of a sampled envelope that stand above ``above``."""
    i = np.arange(1, amplitude.size - 1)
    return i[(amplitude[1:-1] > amplitude[:-2]) & (amplitude[1:-1] > amplitude[2:]) & (amplitude[1:-1] > above)]


def rate_and_frequency(t, amplitude, above=0.0):
    """Growth rate and angular frequency of a damped or growing oscillation, from
    the maxima of its modulus: the slope of ln|E_k| through them, and pi over
    their mean spacing, since successive maxima of a modulus are half a period
    apart. Robust when only two or three oscillations rise above the noise."""
    i = maxima(amplitude, above)
    assert i.size >= 3, f"only {i.size} maxima above the floor"
    return np.polyfit(t[i], np.log(amplitude[i]), 1)[0], np.pi / np.mean(np.diff(t[i]))


def growth_rate(t, amplitude, window):
    """Slope of ln(amplitude) over a boolean window."""
    return np.polyfit(t[window], np.log(amplitude[window]), 1)[0]
