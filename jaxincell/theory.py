"""Linear kinetic theory of drifting (bi-)Maxwellians, as an independent reference.

The roots here are what the examples, the documentation figures and the tests compare
measured growth and damping rates with. Like :mod:`jaxincell.sheath`, it is plain NumPy
and shares nothing with the deposit, the gather, the field solver or the push, so a
simulation compared with it is compared with mathematics and not with itself. Evaluate it
before or after a run and pass the numbers around; it has no place inside a trace.

The plasma dispersion function needs the Faddeeva function, which is SciPy's
(:func:`scipy.special.wofz`). SciPy is imported on first use, so it stays a dependency of
the examples and the validation and not of the package.

Thermal speeds follow the JAX-in-Cell convention :math:`v_{th} = \\sqrt{2T/m}`, so the
one-dimensional Maxwellian is :math:`f(v) = e^{-(v-u)^2/v_{th}^2}/(\\sqrt\\pi v_{th})` and
:math:`Z(\\xi) = i\\sqrt\\pi\\,w(\\xi)`. A population is a dict: ``wp`` its plasma frequency,
``u`` its drift along :math:`x`, ``vth`` its thermal speed along :math:`x` (``vthx`` for the
transverse relation) and ``A`` its anisotropy :math:`T_z/T_x`.
"""
import numpy as np

from ._config import epsilon_0, speed_of_light


def Z(xi):
    """The plasma dispersion function :math:`Z(\\xi) = i\\sqrt\\pi\\,w(\\xi)`."""
    from scipy.special import wofz
    with np.errstate(all="ignore"):
        return 1j * np.sqrt(np.pi) * wofz(xi)


def Zprime(xi):
    """:math:`Z'(\\xi) = -2[1 + \\xi Z(\\xi)]`."""
    with np.errstate(all="ignore"):
        return -2.0 * (1.0 + xi * Z(xi))


def Zsecond(xi):
    """:math:`Z''(\\xi) = -2[Z(\\xi) + \\xi Z'(\\xi)]`."""
    with np.errstate(all="ignore"):
        return -2.0 * (Z(xi) + xi * Zprime(xi))


def plasma_frequency(density, charge, mass):
    """:math:`\\omega_p = \\sqrt{n q^2/(\\epsilon_0 m)}`, in rad/s."""
    return np.sqrt(density * charge**2 / (epsilon_0 * mass))


def electrostatic_epsilon(omega, k, species):
    """Longitudinal dielectric function :math:`1 + \\sum_s \\chi_s` and its derivative in omega."""
    eps = 1.0 + 0j
    deps = 0j
    for s in species:
        xi = (omega - k * s["u"]) / (k * s["vth"])
        pref = -(s["wp"] ** 2) / (k**2 * s["vth"] ** 2)
        eps += pref * Zprime(xi)
        deps += pref * Zsecond(xi) / (k * s["vth"])
    return eps, deps


def weibel_dispersion(omega, k, species):
    """Transverse dispersion function for :math:`k` along :math:`x` and :math:`E` along :math:`z`,
    and its derivative in omega, made dimensionless by the largest plasma frequency:

    :math:`D = \\omega^2 - k^2c^2 - \\sum_s \\omega_{ps}^2[1 - A_s(1 + \\xi_s Z(\\xi_s))]`,
    with :math:`\\xi_s = \\omega/(k v_{th,x})`.
    """
    scale = max(s["wp"] for s in species) ** 2
    D = (omega**2 - k**2 * speed_of_light**2) / scale
    dD = 2 * omega / scale
    for s in species:
        xi = omega / (k * s["vthx"])
        A = s["A"]
        D -= s["wp"] ** 2 / scale * (1.0 - A * (1.0 + xi * Z(xi)))
        dD += s["wp"] ** 2 / scale * A * (Z(xi) + xi * Zprime(xi)) / (k * s["vthx"])
    return D, dD


def newton(func, guess, tol=1e-12, max_iter=200):
    """Newton's method on ``func(omega) -> (value, derivative)``; ``None`` if it does not converge."""
    omega = complex(guess)
    for _ in range(max_iter):
        with np.errstate(all="ignore"):
            value, derivative = func(omega)
        if not (np.isfinite(value) and np.isfinite(derivative)) or derivative == 0:
            return None
        step = value / derivative
        omega -= step
        if abs(step) < tol * max(1.0, abs(omega)):
            return omega
    return None


def most_unstable_root(func, real_range, imag_range, n_real=40, n_imag=25, scale=1.0):
    """Newton from a grid of starting points; the root with the largest imaginary part, or ``None``."""
    best = None
    for re in np.linspace(*real_range, n_real):
        for im in np.linspace(*imag_range, n_imag):
            root = newton(func, (re + 1j * im) * scale)
            if root is None:
                continue
            with np.errstate(all="ignore"):
                value, _ = func(root)
            if not np.isfinite(value) or abs(value) > 1e-6:
                continue
            if best is None or root.imag > best.imag + 1e-9 * scale:
                best = root
    return best


def landau_root(k_lambda_D, wp=1.0):
    """Least damped electrostatic root of one Maxwellian, in units of ``wp``.

    At :math:`k\\lambda_D = 0.5` it is :math:`\\omega/\\omega_{pe} = 1.4156 - 0.1533i`.
    """
    species = [{"wp": wp, "u": 0.0, "vth": np.sqrt(2.0)}]   # lambda_D = 1 at wp = 1
    root = most_unstable_root(lambda w: electrostatic_epsilon(w, k_lambda_D, species), (0.5, 3.0), (-1.5, 0.05))
    # roots come in pairs omega and -conj(omega), equally damped; Newton may land on either
    return complex(abs(root.real), root.imag)


def purely_growing_roots(func, scale, gamma_max=1.0, samples=2000):
    """Growth rates of the roots on the imaginary axis, by bracketing sign changes, ascending.

    Symmetric counter-streaming beams and the Weibel instability grow without
    oscillating; on the imaginary axis their dispersion function is real, so the roots
    can be bracketed exactly rather than found by Newton iterations, which can land on
    another Riemann sheet of ``Z`` and report a growth rate for a stable mode. The list is
    empty when there is none below ``gamma_max * scale``.
    """
    def real(g):
        return func(1j * g)[0].real

    grid = np.linspace(1e-4, gamma_max, samples) * scale
    values = np.array([real(g) for g in grid])
    roots = []
    for i in np.nonzero(values[:-1] * values[1:] < 0)[0]:
        low, high = grid[i], grid[i + 1]
        for _ in range(80):
            mid = 0.5 * (low + high)
            low, high = (low, mid) if real(low) * real(mid) <= 0 else (mid, high)
        roots.append(0.5 * (low + high))
    return roots


def populations(simulation):
    """The drifting-Maxwellian populations of a :class:`~jaxincell.Simulation`, in the form the
    functions above take. A species built with ``plus_minus`` is two beams of half the density."""
    result = []
    for s in simulation.species:
        common = {"name": s.name, "vthx": s.vth[0] or 1.0, "vth": s.vth[0] or 1.0,
                  "A": (s.vth[2] / s.vth[0]) ** 2 if s.vth[0] else 1.0}
        if s.plus_minus:
            wp = plasma_frequency(s.density / 2, s.charge_si, s.mass)
            result += [{**common, "wp": wp, "u": sign * s.drift[0], "density": s.density / 2} for sign in (+1, -1)]
        else:
            result.append({**common, "wp": plasma_frequency(s.density, s.charge_si, s.mass),
                           "u": s.drift[0], "density": s.density})
    return result


def two_stream_rate(simulation, mode=1):
    """Growth rate of ``mode`` from the electrostatic kinetic relation of ``simulation``'s own
    populations, in rad/s; 0 when it is stable."""
    k = 2 * np.pi * mode / simulation.domain.length
    pops = populations(simulation)
    scale = max(p["wp"] for p in pops)
    roots = purely_growing_roots(lambda w: electrostatic_epsilon(w, k, pops), scale)
    return max(roots, default=0.0)


def weibel_rate(k, wp, vthx, anisotropy):
    """Growth rate of the Weibel mode at wavenumber ``k`` of one bi-Maxwellian population, in
    rad/s; 0 at and above the marginal wavenumber :math:`k_c c = \\omega_p\\sqrt{A-1}`."""
    pops = [{"wp": wp, "vthx": vthx, "A": anisotropy}]
    return max(purely_growing_roots(lambda w: weibel_dispersion(w, k, pops), wp, gamma_max=0.5), default=0.0)


def damped_mode(t, amplitude, above=5.0):
    """Rate and frequency of a damped oscillation that decays into a noise floor, from the maxima
    of its modulus ``amplitude(t)``: the slope of its logarithm through them, and pi over their
    mean spacing, since successive maxima of a modulus are half a period apart.

    The floor is measured where the amplitude stops falling -- the median of the maxima from
    the first one that is not below its predecessor -- rather than over a fixed fraction of the
    run, so the answer does not depend on how long the run went on after the wave died. Only
    the maxima before that point and above ``above`` times the floor are fitted.

    Returns:
        tuple: ``(rate, frequency, indices, floor)``; ``indices`` are the maxima used.

    Raises:
        ValueError: when the maxima never stop falling (the run ended before the floor, which
            is then unknown) or fewer than three stand above it.
    """
    t, amplitude = np.asarray(t), np.asarray(amplitude)
    i = np.arange(1, amplitude.size - 1)
    peaks = i[(amplitude[1:-1] > amplitude[:-2]) & (amplitude[1:-1] > amplitude[2:])]
    tops = amplitude[peaks]
    rises = np.nonzero(tops[1:] >= tops[:-1])[0]
    if rises.size == 0:
        raise ValueError("the maxima never stop falling: run longer, until the mode reaches its noise floor")
    end = rises[0] + 1
    floor = float(np.median(tops[end:]))
    used = peaks[:end][tops[:end] > above * floor]
    if used.size < 3:
        raise ValueError(f"only {used.size} maxima stand above {above:g} times the noise floor")
    rate = np.polyfit(t[used], np.log(amplitude[used]), 1)[0]
    return float(rate), float(np.pi / np.mean(np.diff(t[used]))), used, floor


def field_epsilon(omega, k, kappa, drift=0.0):
    """The dielectric function of Maxwellian electrons in a uniform field, and its derivative in omega,
    from Fried's characteristics with the zero-order field kept (Beving, Hopkins and Baalrud, Phys.
    Plasmas 30, 112105, 2023, eq. 7):

    :math:`\\epsilon = 1 - Z'(\\zeta)/(k^2 s^2)`, :math:`s^2 = 2(1 + i\\kappa/k)`,
    :math:`\\zeta = (\\omega - k u)/(k s)`.

    Dimensionless, in that paper's units: ``k`` in :math:`1/\\lambda_{De}`, ``omega`` in
    :math:`\\omega_{pe}`, ``drift`` :math:`u` in :math:`v_{Te} = \\sqrt{T_e/m_e}` (not this module's
    :math:`\\sqrt{2T/m}`), and ``kappa`` is :math:`\\kappa_e\\lambda_{De} = eE_0\\lambda_{De}/T_e`
    with the field antiparallel to ``k``. At ``kappa = 0`` it is :func:`electrostatic_epsilon` of one
    Maxwellian. The drift enters only through :math:`\\omega - ku`, so it Doppler-shifts every root
    and leaves every growth rate unchanged.

    The relation treats the accelerating distribution as a steady state. For collisionless
    Vlasov-Poisson on an exactly uniform immobile background, the change of variables
    :math:`x' = x - at^2/2`, :math:`v' = v - at` removes the field altogether, so its growth is a
    prediction to test and not a consequence of the equations.
    """
    s = np.sqrt(2.0 * (1.0 + 1j * kappa / k))
    xi = (omega - k * drift) / (k * s)
    return 1.0 - Zprime(xi) / (k * s) ** 2, -Zsecond(xi) / (k * s) ** 3


def field_rate(k, kappa, drift=0.0):
    """The small-field, large-phase-velocity growth rate of the same paper (eq. 10), in
    :math:`\\omega_{pe}`, with :func:`field_epsilon`'s units: :math:`\\Theta(\\omega_r)/2\\,[3k\\kappa -
    \\sqrt{\\pi/8}\\,e^{-3/2 - 1/(2k^2)}/|k|^3]`, where :math:`\\omega_r = \\sqrt{1 + 3k^2} - ku` is
    the paper's eq. 8. That frequency vanishes where the wave stands still in the frame of the ions,
    :math:`k = 1/u` for :math:`u \\gg 1` (eq. 11), the paper's cutoff."""
    k = np.asarray(k, float)
    frequency = np.sqrt(1 + 3 * k ** 2) - k * drift
    with np.errstate(over="ignore", divide="ignore"):
        landau = np.sqrt(np.pi / 8) * np.exp(-1.5 - 0.5 / k ** 2) / np.abs(k) ** 3
    return 0.5 * (frequency > 0) * (3 * k * kappa - landau)


def field_root(k, kappa, drift=0.0):
    """The growing electron plasma wave of :func:`field_epsilon`, by Newton from the Bohm-Gross
    frequency Doppler-shifted by the drift; ``None`` if Newton does not converge."""
    guess = np.sqrt(1 + 3 * k ** 2) + k * drift + 0.5j * field_rate(k, kappa)
    return newton(lambda w: field_epsilon(w, k, kappa, drift), guess)
