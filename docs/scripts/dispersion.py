"""Linear kinetic theory used to overlay reference rates on the figures.

All species are drifting (bi-)Maxwellians. Thermal speeds follow the JAX-in-Cell
convention v_th = sqrt(2 T / m), so the one-dimensional Maxwellian is
f(v) = exp(-(v - u)^2 / v_th^2) / (sqrt(pi) v_th) and the plasma dispersion
function is Z(xi) = i sqrt(pi) w(xi), with w the Faddeeva function.
"""
import numpy as np
from scipy.special import wofz

from jaxincell import epsilon_0, speed_of_light


def Z(xi):
    with np.errstate(all="ignore"):
        return 1j * np.sqrt(np.pi) * wofz(xi)


def Zprime(xi):
    with np.errstate(all="ignore"):
        return -2.0 * (1.0 + xi * Z(xi))


def Zsecond(xi):
    with np.errstate(all="ignore"):
        return -2.0 * (Z(xi) + xi * Zprime(xi))


def plasma_frequency(density, charge, mass):
    return np.sqrt(density * charge**2 / (epsilon_0 * mass))


def electrostatic_epsilon(omega, k, species):
    """Longitudinal dielectric function 1 + sum_s chi_s and its omega derivative.

    ``species`` is a list of dicts with keys wp (plasma frequency), u (drift
    along x) and vth (thermal speed along x).
    """
    eps = 1.0 + 0j
    deps = 0j
    for s in species:
        xi = (omega - k * s["u"]) / (k * s["vth"])
        pref = -(s["wp"] ** 2) / (k**2 * s["vth"] ** 2)
        eps += pref * Zprime(xi)
        deps += pref * Zsecond(xi) / (k * s["vth"])
    return eps, deps


def weibel_dispersion(omega, k, species):
    """Transverse dispersion function for k along x and E along z.

    D = omega^2 - k^2 c^2 - sum_s wp_s^2 [1 - A_s (1 + xi_s Z(xi_s))], with
    A_s = T_z / T_x the temperature anisotropy and xi_s = omega / (k v_thx).
    """
    c = speed_of_light
    scale = max(s["wp"] for s in species) ** 2  # makes D dimensionless
    D = (omega**2 - k**2 * c**2) / scale
    dD = 2 * omega / scale
    for s in species:
        xi = omega / (k * s["vthx"])
        A = s["A"]
        D -= s["wp"] ** 2 / scale * (1.0 - A * (1.0 + xi * Z(xi)))
        # d/d omega of -wp^2 [1 - A(1 + xi Z)] = wp^2 A (Z + xi Z') / (k vthx)
        dD += s["wp"] ** 2 / scale * A * (Z(xi) + xi * Zprime(xi)) / (k * s["vthx"])
    return D, dD


def newton(func, guess, tol=1e-12, max_iter=200):
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
    """Newton iterations from a grid of starting points; return the root with the
    largest imaginary part (the fastest growing, or least damped, mode)."""
    best = None
    for re in np.linspace(*real_range, n_real):
        for im in np.linspace(*imag_range, n_imag):
            root = newton(func, (re + 1j * im) * scale, tol=1e-12)
            if root is None or not np.isfinite(root):
                continue
            with np.errstate(all="ignore"):
                value, _ = func(root)
            if not np.isfinite(value) or abs(value) > 1e-6:
                continue
            if best is None or root.imag > best.imag + 1e-9 * scale:
                best = root
    return best


def landau_root(k_lambda_D, wp=1.0):
    """Least damped electrostatic root for a single Maxwellian, in units of wp.

    For k lambda_D = 0.5 the classical value is omega/wp = 1.4156 - 0.1533 i.
    """
    vth = np.sqrt(2.0)  # lambda_D = vth / (sqrt(2) wp) = 1 with wp = 1
    k = k_lambda_D
    species = [{"wp": wp, "u": 0.0, "vth": vth}]
    func = lambda w: electrostatic_epsilon(w, k, species)
    return most_unstable_root(func, (0.5, 3.0), (-1.5, 0.05))
