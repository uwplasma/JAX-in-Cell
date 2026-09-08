"""Configuration objects: the domain, the particle species, the solver and the
collision model.

Each is a frozen dataclass registered as a JAX pytree. Physical quantities
(lengths, temperatures, drifts, densities, the filter weight, the coefficient of
restitution) are pytree leaves and are therefore traced: they can be changed
without recompiling and differentiated with respect to. Structural settings
(particle counts, cell counts, boundary types, algorithm switches) are static
and become part of the compiled program.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields

import jax
import numpy as np

from ._constants import elementary_charge, mass_electron, mass_proton, speed_of_light

__all__ = ["Domain", "Species", "Solver", "Collisions", "BOUNDARIES"]

BOUNDARIES = {"periodic": 0, "reflective": 1, "absorbing": 2}


def pytree_dataclass(static=()):
    """Turn a class into a frozen dataclass that JAX treats as a pytree, with
    the named fields kept static (they must be hashable)."""
    static = tuple(static)

    def wrap(cls):
        cls = dataclass(frozen=True)(cls)
        leaves = tuple(f.name for f in fields(cls) if f.name not in static)

        def flatten(obj):
            return ([getattr(obj, n) for n in leaves], tuple(getattr(obj, n) for n in static))

        def unflatten(aux, values):
            return cls(**dict(zip(leaves, values)), **dict(zip(static, aux)))

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
        cls.replace = dataclasses.replace
        return cls

    return wrap


def _float(value):
    """Plain Python numbers become floats so that they are differentiable leaves;
    arrays and tracers pass through untouched."""
    if isinstance(value, (bool, np.bool_)):
        return value
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return value


def _floats(values):
    return tuple(_float(v) for v in values)


def _boundary_codes(value):
    """Wall names to codes; already-converted codes pass through, so the
    conversion is safe to repeat when JAX rebuilds the object."""
    if isinstance(value, str):
        value = (value, value)
    return tuple(BOUNDARIES[v] if isinstance(v, str) else int(v) for v in value)


@pytree_dataclass(static=("cells", "particle_bc", "field_bc"))
class Domain:
    """The simulation box.

    Args:
        length: Box length :math:`L` in metres; the box spans :math:`[-L/2, L/2]`.
        cells: Number of cells :math:`N_x`.
        dt_over_dx_c: Time step as :math:`c\\,\\Delta t/\\Delta x`.
        particle_bc: ``"periodic"``, ``"reflective"`` or ``"absorbing"``, or a
            ``(left, right)`` pair.
        field_bc: Same choices for the fields.
        restitution: Coefficient of restitution of reflective walls; the normal
            velocity is multiplied by ``-restitution`` on reflection.
        length_y, length_z: Periods of the ignorable coordinates.
    """
    length: float = 1e-2
    cells: int = 64
    dt_over_dx_c: float = 1.0
    particle_bc: object = "periodic"
    field_bc: object = "periodic"
    restitution: float = 1.0
    length_y: float = 1e-2
    length_z: float = 1e-2

    def __post_init__(self):
        for name in ("length", "dt_over_dx_c", "restitution", "length_y", "length_z"):
            object.__setattr__(self, name, _float(getattr(self, name)))
        object.__setattr__(self, "particle_bc", _boundary_codes(self.particle_bc))
        object.__setattr__(self, "field_bc", _boundary_codes(self.field_bc))
        assert self.cells >= 4, "need at least four cells"
        for bc in (self.particle_bc, self.field_bc):
            assert (0 in bc) == (bc == (0, 0)), "a periodic wall needs a periodic partner"

    @property
    def dx(self):
        return self.length / self.cells

    @property
    def dt(self):
        return self.dt_over_dx_c * self.dx / speed_of_light

    @property
    def grid(self):
        """Cell centres :math:`x_i = -L/2 + (i + 1/2)\\Delta x`."""
        import jax.numpy as jnp
        return -self.length / 2 + (jnp.arange(self.cells) + 0.5) * self.dx


@pytree_dataclass(static=("name", "n", "plus_minus", "quiet", "random_positions"))
class Species:
    """One population of pseudo-particles.

    Velocities are Maxwellian with thermal speed ``vth`` per component, defined
    by :math:`f(v) \\propto \\exp(-v^2/v_{th}^2)`, that is
    :math:`v_{th} = \\sqrt{2 k_B T/m}`. The pseudo-particle weight is
    ``density * length / n``.

    Args:
        name: Label used in the output.
        n: Number of pseudo-particles.
        charge: Charge in units of the elementary charge.
        mass: Mass in kilograms.
        density: Number density in :math:`\\mathrm{m^{-3}}`.
        vth: Thermal speed per component, m/s.
        drift: Drift velocity per component, m/s.
        perturbation_amplitude: Amplitude :math:`a` (metres) of the displacement
            :math:`x \\to x + a \\sin(2\\pi m x / L)`.
        perturbation_mode: Mode number :math:`m` of that displacement.
        plus_minus: Negate the x velocity of every second particle, which turns
            one drifting population into two counter-streaming beams.
        quiet: Quiet start: equally spaced positions and velocities at the
            quantiles of the Maxwellian, ordered by a bit-reversed sequence.
        random_positions: Uniformly random positions instead of equally spaced.
        x, v: Optional arrays of shape ``(n, 3)`` that replace the generated
            phase space.
    """
    name: str
    n: int
    charge: float
    mass: float
    density: float
    vth: tuple = (0.0, 0.0, 0.0)
    drift: tuple = (0.0, 0.0, 0.0)
    perturbation_amplitude: float = 0.0
    perturbation_mode: float = 0.0
    plus_minus: bool = False
    quiet: bool = False
    random_positions: bool = False
    x: object = None
    v: object = None

    def __post_init__(self):
        assert self.n > 0, "a species needs at least one particle"
        for name in ("charge", "mass", "density", "perturbation_amplitude", "perturbation_mode"):
            object.__setattr__(self, name, _float(getattr(self, name)))
        for name in ("vth", "drift"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                value = (value, 0.0, 0.0)
            object.__setattr__(self, name, _floats(value))
        for name in ("x", "v"):
            value = getattr(self, name)
            if value is not None:
                assert np.shape(value) == (self.n, 3), f"{name} must have shape (n, 3)"

    @staticmethod
    def electrons(n, density, vth=(0.0, 0.0, 0.0), name="electrons", **kwargs):
        return Species(name, n, -1.0, mass_electron, density, vth, **kwargs)

    @staticmethod
    def ions(n, density, mass_ratio=1.0, charge=1.0, name="ions", vth=None,
             temperature_ratio=1.0, electrons=None, **kwargs):
        """Ions of mass ``mass_ratio`` proton masses. If ``vth`` is not given
        it is derived from the electron species as
        :math:`v_{th,i} = v_{th,e}\\sqrt{(T_i/T_e)(m_e/m_i)}`."""
        mass = mass_ratio * mass_proton
        if vth is None:
            assert electrons is not None, "give vth or the electron species"
            factor = np.sqrt(temperature_ratio * mass_electron / mass)
            vth = tuple(factor * np.asarray(electrons.vth))
        return Species(name, n, charge, mass, density, vth, **kwargs)

    @property
    def charge_si(self):
        return self.charge * elementary_charge


@pytree_dataclass(static=("algorithm", "field_solver", "relativistic", "filter_passes",
                          "filter_strides", "picard_iterations", "substeps"))
class Solver:
    """Numerical choices.

    Args:
        algorithm: ``"explicit"`` (leapfrog with the Boris pusher) or
            ``"implicit"`` (Crank-Nicolson with Picard iteration).
        field_solver: ``"ampere"`` advances :math:`E_x` with Ampere's law and the
            charge-conserving current; ``"gauss"`` recomputes it from the charge
            density every step.
        relativistic: Relativistic Boris pusher.
        filter_passes: Binomial smoothing passes on the sources (0 disables).
            Each pass is followed by one compensation pass.
        filter_alpha: Centre weight of the three-point filter.
        filter_strides: Cell offsets of the filter stencil.
        picard_iterations: Fixed-point iterations of the implicit scheme.
        substeps: Particle sub-steps per field step in the implicit scheme.
    """
    algorithm: str = "explicit"
    field_solver: str = "ampere"
    relativistic: bool = False
    filter_passes: int = 0
    filter_alpha: float = 0.5
    filter_strides: tuple = (1,)
    picard_iterations: int = 8
    substeps: int = 2

    def __post_init__(self):
        assert self.algorithm in ("explicit", "implicit")
        assert self.field_solver in ("ampere", "gauss")
        assert self.filter_passes >= 0 and self.picard_iterations >= 1 and self.substeps >= 1
        object.__setattr__(self, "filter_alpha", _float(self.filter_alpha))
        object.__setattr__(self, "filter_strides", tuple(int(s) for s in self.filter_strides))


@pytree_dataclass(static=("pairs",))
class Collisions:
    """Binary Coulomb collisions (Takizuka and Abe, 1977).

    Args:
        pairs: Tuple of ``(name_a, name_b)`` species pairs to collide, including
            ``(name, name)`` for self-collisions. ``None`` collides every pair.
        coulomb_log: Coulomb logarithm; ``None`` uses the NRL formulary value
            from the electron density and temperature at the start of the run.
    """
    pairs: object = None
    coulomb_log: object = None

    def __post_init__(self):
        if self.pairs is not None:
            object.__setattr__(self, "pairs", tuple(tuple(p) for p in self.pairs))
        object.__setattr__(self, "coulomb_log", _float(self.coulomb_log))
