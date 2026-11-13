"""Kinetic theory of the collisionless planar sheath, as an independent reference.

Nothing here touches the deposit, the gather, the field solver or the particle
push: these are closed-form solutions of the same Vlasov-Poisson problem the code
integrates, so a simulation compared with them is compared with mathematics and
not with itself. The examples, the documentation figures and the tests all use
this module, which is why it is public.

It is plain NumPy and returns plain arrays. That is deliberate: a reference has no
business inside a differentiated run, and keeping it off JAX makes the separation
one the interpreter enforces. Evaluate it before the simulation and pass the
numbers in.

Everything is in the normalisation of the sheath literature: the potential in
:math:`T_e/e`, speeds in the electron spread :math:`\\sigma_e=\\sqrt{T_e/m_e}`,
densities in the upstream plasma density and lengths in the Debye length. Turn a
measured potential into these units by dividing by :math:`T_e/e`.

**The source-to-collector problem.** A reservoir at :math:`x=0` sends in electrons
with the velocity density :math:`(n_{e0}/\\sqrt{2\\pi})\\exp(-v^2/2)` for :math:`v>0`
and a cold ion beam :math:`n_{i0}\\delta(v-v_0)`; what comes back leaves freely. At
:math:`x=L` a conductor collects everything and floats. Both species conserve energy,
so the electrons that reach the wall are those launched above the cutoff
:math:`v_{\\rm cut}=\\sqrt{-2\\phi_w}`, and equal particle currents at a floating
collector give the wall potential in closed form.
"""
import math

import numpy as np

__all__ = ["floating_potential", "source_density", "densities", "hobbs_wesson"]

_erfn = np.vectorize(math.erf)      # the host's own erf: this module is never traced


def floating_potential(beam_speed, iterations=80):
    """Wall potential :math:`\\phi_w` in :math:`T_e/e` of a floating collector fed by a
    cold ion beam of speed ``beam_speed`` in :math:`\\sigma_e`.

    The electron flux that clears the cutoff is
    :math:`(n_{e0}/\\sqrt{2\\pi})\\exp(\\phi_w)` and the ion flux is :math:`n_{i0}v_0`;
    with the source amplitude of :func:`source_density`, which makes the source plane
    neutral, equal currents give

    .. math:: \\frac{\\exp\\phi_w}{1+\\operatorname{erf}\\sqrt{-\\phi_w}} = \\sqrt{\\pi/2}\\,v_0.

    The left side falls monotonically from :math:`1` at :math:`\\phi_w=0` to zero, so a root
    exists exactly when :math:`0 < \\sqrt{\\pi/2}\\,v_0 < 1`, that is
    :math:`0 < v_0 < \\sqrt{2/\\pi} = 0.7979`, and a bisection converges for any of them.

    That a root **exists** is not that the sheath is admissible. The beam has to be fast
    enough to reach the wall at all, :math:`v_0 \\ge \\sqrt{1/(m_i/m_e)}`, and a beam entering
    far above the sound speed has no presheath to speak of and no Bohm point, so the picture
    behind the closed form is not the one a slow entrance gives. The bound below is the
    algebraic one; admissibility is the caller's to check, and :func:`~jaxincell.bohm_edge`
    is what says whether a run has an edge.
    """
    target = np.sqrt(np.pi / 2) * np.asarray(beam_speed, float)
    if np.any(target <= 0) or np.any(target >= 1.0):
        raise ValueError("a floating collector needs 0 < beam_speed < sqrt(2/pi) = 0.7979 in units of "
                         "sigma_e, which is where exp(phi)/[1 + erf(sqrt(-phi))] = sqrt(pi/2) v_0 has a root")
    low, high = np.full_like(target, -60.0), np.zeros_like(target)
    for _ in range(iterations):                       # 80 halvings take 60 to below 1e-16
        mid = 0.5 * (low + high)
        too_big = np.exp(mid) / (1 + _erfn(np.sqrt(-mid))) > target
        low, high = np.where(too_big, low, mid), np.where(too_big, mid, high)
    return 0.5 * (low + high)


def source_density(phi_wall):
    """Amplitude :math:`n_{e0}` of the incoming electron distribution that makes the
    source plane neutral, :math:`2/[1+\\operatorname{erf}\\sqrt{-\\phi_w}]`.

    It is not the electron density there: the plane also holds the electrons the sheath
    has turned back, and the two together make :math:`n_e(0)=n_{i0}`.
    """
    return 2.0 / (1 + _erfn(np.sqrt(-np.asarray(phi_wall, float))))


def densities(phi, phi_wall, beam_speed, mass_ratio, amplitude=None):
    """Electron and ion densities where the potential is ``phi``, in units of the
    upstream density. Both follow from energy conservation alone, so this is a local
    relation a simulation can be tested against point by point, without knowing where
    in the box the potential took that value.

    Electrons launched above :math:`\\sqrt{-2\\phi}` are present moving towards the wall;
    those below :math:`\\sqrt{2(\\phi-\\phi_w)}` in local speed have turned back and are
    counted twice:

    .. math:: n_e = \\tfrac12 n_{e0}e^{\\phi}\\,[1+\\operatorname{erf}\\sqrt{\\phi-\\phi_w}].

    The beam simply speeds up at fixed flux, :math:`n_i = v_0/\\sqrt{v_0^2-2\\phi/(m_i/m_e)}`.

    ``phi_wall`` enters twice, and in a measurement the two are different numbers. The
    **cutoff** :math:`\\sqrt{\\phi-\\phi_w}` is the wall potential the run actually reached,
    because that is what turns an electron back. The **amplitude** :math:`n_{e0}` is the
    reservoir the run was configured with, which was chosen once from a wall potential
    predicted in advance and does not follow the run. Give the configured amplitude in
    ``amplitude`` and the measured wall potential in ``phi_wall``; without it both are taken
    from ``phi_wall``, which is the self-consistent solution rather than a measurement.

    Domain: the electron relation needs :math:`\\phi \\ge \\phi_w` and the ion one
    :math:`\\phi < \\tfrac12 v_0^2 (m_i/m_e)`, which at the parameters of the sheath examples
    is :math:`\\phi < 36`. A presheath sitting a few hundredths above the source plane is well
    inside that: **positive potentials are in the domain and are not to be clipped**, and
    :class:`ValueError` is raised for the ones that are not.
    """
    phi, phi_wall = np.asarray(phi, float), np.asarray(phi_wall, float)
    ceiling = 0.5 * beam_speed ** 2 * mass_ratio
    if np.any(phi < phi_wall) or np.any(phi >= ceiling):
        raise ValueError(f"the kinetic relation holds for {float(np.min(phi_wall)):.4f} <= phi < "
                         f"{ceiling:.4f} in T_e/e, and it was asked for phi from "
                         f"{float(np.min(phi)):.4f} to {float(np.max(phi)):.4f}. Outside it an electron "
                         "has no turning point or the beam has stopped; quantify the excursion rather "
                         "than clipping it back in.")
    n_e0 = source_density(phi_wall) if amplitude is None else np.asarray(amplitude, float)
    n_e = 0.5 * n_e0 * np.exp(phi) * (1 + _erfn(np.sqrt(phi - phi_wall)))
    return n_e, beam_speed / np.sqrt(beam_speed ** 2 - 2 * phi / mass_ratio)


def hobbs_wesson(mass_ratio, reflected=0.0):
    """Potential drop from the sheath edge to a floating wall in :math:`T_e/e`,
    :math:`\\tfrac12\\ln(m_i/2\\pi m_e) + \\ln(1-R)`, for ions entering at the Bohm speed
    and a wall that returns the fraction ``R`` of the electron flux (Hobbs and Wesson,
    Plasma Phys. 9, 85, 1967). It assumes a Maxwellian electron flux at the edge, which
    a wall facing an unmaintained plasma does not see for long.
    """
    return 0.5 * np.log(mass_ratio / (2 * np.pi)) + np.log1p(-np.asarray(reflected, float))
