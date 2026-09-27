"""The linear kinetic theory the examples and figures compare with (jaxincell.theory)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from jaxincell import Domain, Simulation, Species, elementary_charge, mass_electron, speed_of_light as c  # noqa: E402
from jaxincell import theory  # noqa: E402


def test_the_landau_root_is_the_classical_one():
    """omega/omega_pe = 1.4157 - 0.1533 i at k lambda_D = 0.5 (Canosa, J. Plasma Phys. 8, 187, 1972)."""
    assert abs(theory.landau_root(0.5) - (1.4157 - 0.1533j)) < 2e-4
    # the backward-wave twin -conj(omega) is as damped; the forward one is returned at every k
    assert all(theory.landau_root(k).real > 0.9 for k in (0.25, 0.31))   # where it once was not


def test_the_derivatives_of_Z_are_those_of_its_differential_equation():
    xi, h = 0.3 + 0.2j, 1e-6
    assert abs((theory.Z(xi + h) - theory.Z(xi - h)) / (2 * h) - theory.Zprime(xi)) < 1e-6
    assert abs((theory.Zprime(xi + h) - theory.Zprime(xi - h)) / (2 * h) - theory.Zsecond(xi)) < 1e-6


def test_the_weibel_rate_vanishes_at_the_marginal_wavenumber():
    """Growth below k_c c = omega_pe sqrt(A - 1), none above; the rates tests/test_physics.py hard-codes."""
    wp = 1e9
    k_c = np.sqrt(24.0) * wp / c
    rates = [theory.weibel_rate(f * k_c, wp, 0.02 * c, 25.0) / wp for f in (0.25, 0.5, 1.05)]
    assert rates[0] == pytest.approx(0.0473, abs=1e-4) and rates[1] == pytest.approx(0.0428, abs=1e-4)
    assert rates[2] == 0.0
    # the derivative the Newton iterations use is the derivative of the function
    pops = [{"wp": wp, "vthx": 0.02 * c, "A": 25.0}]
    w, h = (0.3 + 0.04j) * wp, 1e-4 * wp
    D = [theory.weibel_dispersion(w + s * h, k_c / 2, pops)[0] for s in (1, -1)]
    derivative = theory.weibel_dispersion(w, k_c / 2, pops)[1]
    assert abs((D[0] - D[1]) / (2 * h) - derivative) < 1e-6 * abs(derivative)


def test_the_two_stream_rate_reads_the_simulation_populations():
    """A plus_minus species is two beams; an immobile species with vth = 0 is still a population.
    Cold beams: gamma = omega_b / 2 at the fastest k, with omega_b the plasma frequency of one
    beam, and no growth at k v0 > sqrt 2 omega_b."""
    density, drift = 1e16, 1e6
    electrons = Species.electrons(n=100, density=density, vth=(1e3, 0, 0), drift=(drift, 0, 0), plus_minus=True)
    ions = Species.ions(n=100, density=density, electrons=electrons, vth=(0, 0, 0))
    wb = theory.plasma_frequency(density / 2, elementary_charge, mass_electron)
    fastest = 2 * np.pi * drift / (np.sqrt(3) / 2 * wb)

    def box(length):
        return Simulation(Domain(length=length, cells=16), [electrons, ions])

    pops = theory.populations(box(fastest))
    assert len(pops) == 3 and pops[0]["u"] == -pops[1]["u"] and pops[2]["vth"] == 1.0
    assert theory.two_stream_rate(box(fastest)) / wb == pytest.approx(0.5, rel=2e-2)
    assert theory.two_stream_rate(box(fastest / 4)) == 0.0


def test_newton_and_the_root_search_reject_what_is_not_a_root():
    assert theory.newton(lambda w: (np.nan, 1.0), 1.0) is None                     # not finite
    assert theory.newton(lambda w: (1.0, 0.0), 1.0) is None                        # flat
    assert theory.newton(lambda w: (w * w + 1, 2 * w), 1.0, max_iter=2) is None    # not converged
    # a step below tolerance where the value is not zero is not a root
    assert theory.most_unstable_root(lambda w: (1.0, 1e30), (0, 1), (0, 1), n_real=2, n_imag=2) is None
    # no start converges: nothing is returned
    assert theory.most_unstable_root(lambda w: (np.nan, 1.0), (0, 1), (0, 1), n_real=2, n_imag=2) is None
    # every start converges to the one root, which is kept once
    assert theory.most_unstable_root(lambda w: (w - 1j, 1.0), (0, 1), (0, 1), n_real=2, n_imag=2) == 1j


def test_the_damped_mode_fit_measures_its_floor_where_the_decay_stops():
    """A damped cosine that sinks into a constant floor: the fit recovers the rate and the
    frequency however long the tail, and refuses a run that never reached the floor."""
    t = np.linspace(0, 60, 6001)
    # the noise adds to the mode with an unrelated phase, so in quadrature
    wave = np.cos(1.4 * t) * 2000 * np.exp(-0.15 * t)
    amplitude = np.sqrt(wave ** 2 + (40 * (1 + 0.3 * np.sin(7.3 * t) ** 2)) ** 2)
    for end in (3001, 6001):
        rate, frequency, used, floor = theory.damped_mode(t[:end], amplitude[:end])
        assert rate == pytest.approx(-0.15, rel=0.03) and frequency == pytest.approx(1.4, rel=0.01)
        assert 40 < floor < 60 and used.size >= 3
    with pytest.raises(ValueError, match="run longer"):
        theory.damped_mode(t[:1000], np.abs(np.cos(1.4 * t[:1000])) * np.exp(-0.15 * t[:1000]))
    with pytest.raises(ValueError, match="only"):
        theory.damped_mode(t, amplitude, above=100.0)
