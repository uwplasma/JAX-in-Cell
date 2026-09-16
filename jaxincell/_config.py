"""Configuration objects: the domain, the particle species, the solver and the
collision model.

Each is a frozen dataclass registered as a JAX pytree. Physical quantities
(lengths, temperatures, drifts, densities, the filter weight, the coefficients of
restitution and reflection) are pytree leaves and are therefore traced: they can
be changed without recompiling and differentiated with respect to. Structural
settings (particle counts, cell counts, boundary types, algorithm switches) and
functions, such as a velocity-dependent reflection law, are static and become
part of the compiled program.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields

import jax
import numpy as np

# physical constants in SI units (CODATA 2018)
epsilon_0 = 8.8541878128e-12      # vacuum permittivity, F/m
mu_0 = 1.25663706212e-6           # vacuum permeability, H/m
speed_of_light = 299792458.0      # m/s
elementary_charge = 1.602176634e-19   # C
mass_electron = 9.1093837015e-31      # kg
mass_proton = 1.67262192369e-27       # kg
boltzmann_constant = 1.380649e-23     # J/K

__all__ = ["Domain", "Species", "Solver", "Source", "Collisions", "BOUNDARIES"]

BOUNDARIES = {"periodic": 0, "reflective": 1, "absorbing": 2, "thermal": 3, "open": 4}


def pytree_dataclass(static=()):
    """Turn a class into a frozen dataclass that JAX treats as a pytree, with
    the named fields kept static (they must be hashable). A function is not an
    array, so a field holding one is static too, and so is each function inside a
    tuple, while the numbers beside it stay leaves.

    JAX rebuilds an object from its leaves without calling ``__init__``, so what a
    tree operation puts there -- stacked arrays, tracers, ``None`` -- is stored as
    it is. ``__post_init__`` normalises and checks what is constructed or passed to
    ``replace``."""
    static = tuple(static)

    def wrap(cls):
        cls = dataclass(frozen=True)(cls)
        cls._static_fields = static
        leaves = tuple(f.name for f in fields(cls) if f.name not in static)

        def flatten(obj):
            parts = [_split(getattr(obj, n)) for n in leaves]
            return [p[0] for p in parts], (tuple(p[1] for p in parts), tuple(getattr(obj, n) for n in static))

        def unflatten(aux, values):
            functions, fixed = aux
            obj = object.__new__(cls)
            for name, value, function in zip(leaves, values, functions):
                object.__setattr__(obj, name, _join(value, function))
            for name, value in zip(static, fixed):
                object.__setattr__(obj, name, value)
            return obj

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
        cls.replace = dataclasses.replace
        return cls

    return wrap


def _split(value):
    """A field value as its traced part and its static part: a function is
    static, and so is each function in a tuple, with ``None`` in its place."""
    if callable(value):
        return None, value
    if isinstance(value, tuple) and any(callable(v) for v in value):
        return tuple(None if callable(v) else v for v in value), tuple(v if callable(v) else None for v in value)
    return value, None


def _join(value, static):
    """Inverse of :func:`_split`."""
    if static is None:
        return value
    if callable(static):
        return static
    return tuple(v if s is None else s for v, s in zip(value, static))


def _template(obj):
    """Whether ``obj`` is a template made by a tree operation, such as the
    ``in_axes`` of a ``vmap`` built as ``jax.tree.map(lambda _: None, obj).replace(density=0)``:
    ``None`` where a configuration always holds a number (a leaf field whose
    default is not ``None``). A template is stored as given, so that its integer
    axes do not become floats and its placeholders are not checked."""
    values = [getattr(obj, f.name) for f in fields(obj) if f.name not in obj._static_fields and f.default is not None]
    return any(v is None for v in jax.tree_util.tree_leaves(values, is_leaf=lambda v: v is None))


def _require(condition, message):
    """Validation that ``python -O`` keeps."""
    if not condition:
        raise ValueError(message)


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


def _plain(value):
    """Whether a value is a Python number, so that a check may look at it. A tracer or
    an array is not: its value is not known until the program runs."""
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))


def _walls(value, name):
    """One value for both walls, or a ``(left, right)`` pair, as a pair of
    floats (functions and arrays pass through); plain numbers must lie in [0, 1]."""
    pair = _floats(value if isinstance(value, (tuple, list)) else (value, value))
    _require(len(pair) == 2 and not any(isinstance(v, float) and not 0 <= v <= 1 for v in pair),
             f"{name} takes a number in [0, 1] or a (left, right) pair of them, not {value!r}")
    return pair


def _components(value, name):
    """A velocity as three components. A bare number is the x component, the
    direction the grid resolves, and the other two are zero; the last axis of an
    array holds the components."""
    if not isinstance(value, (tuple, list)):
        value = (value, 0.0, 0.0) if np.ndim(value) == 0 else tuple(value[..., i] for i in range(np.shape(value)[-1]))
    _require(len(value) == 3, f"{name} takes a number (the x component) or three components, not {len(value)}")
    return _floats(value)


def _boundary_codes(value, name):
    """Wall names as a ``(left, right)`` pair of codes. Codes pass through,
    because ``replace`` converts the stored value again."""
    pair = (value, value) if isinstance(value, str) else tuple(value)
    codes = tuple(BOUNDARIES.get(v) if isinstance(v, str) else v for v in pair)
    _require(len(codes) == 2 and all(c in BOUNDARIES.values() for c in codes),
             f"{name} takes one of {', '.join(map(repr, BOUNDARIES))} or a (left, right) pair of them, not {value!r}")
    return tuple(int(c) for c in codes)


@pytree_dataclass(static=("side", "emit", "beam"))
class Source:
    """A maintained inflow of one species through one wall: a reservoir of plasma
    behind the plane that supplies a prescribed flux, independently of what leaves.

    The distribution behind the plane is a Maxwellian at rest or a cold beam; the
    flux that crosses is the velocity density weighted by the normal speed
    (:mod:`~jaxincell._sources`). A fixed number of particles is emitted every step
    with a continuous weight :math:`\\Gamma\\Delta t/N_{\\rm emit}`, so the emitted
    weight is exactly the prescribed flux and is differentiable in ``density`` and
    ``vth``.

    Args:
        density: Reservoir number density :math:`n_{\\rm in}`, :math:`\\mathrm{m^{-3}}`.
        vth: Thermal speed per component of the reservoir, :math:`\\sqrt{2k_BT/m}`, m/s,
            given as :class:`Species` takes it. Zero makes a cold beam.
        drift: Drift velocity of the reservoir, m/s. Its normal component must
            vanish unless ``vth`` does: the drifting crossing distribution needs a
            sampler that is not implemented, and a Rayleigh sample plus a drift is
            not it.
        side: ``"left"`` or ``"right"``, the wall the plasma enters through.
        beam: Whether the reservoir is cold, so that the sampler draws a beam rather
            than a Maxwellian. It is worked out from ``vth`` when the object is built
            and is then static, because it selects a branch: read live from ``vth`` it
            would be a traced value, and inside ``jit`` every source would look cold.
        emit: Particles emitted per step. They occupy the dead slots of the
            species, so the species needs enough of them: ``n`` must exceed
            ``emit`` times the longest residence time in steps.
        min_weight: Fraction of the emitted weight below which a wall collects
            what is left of a particle instead of reflecting it again, freeing its
            slot. A wall that returns the fraction :math:`R` of each impact would
            otherwise hold a particle for ever at a weight falling as :math:`R^k`.
            The remainder is given to the wall, so the charge and energy ledgers
            stay exact; what changes is where the last :math:`10^{-3}` of a
            particle lands.
    """
    density: float = 0.0
    vth: tuple = (0.0, 0.0, 0.0)
    drift: tuple = (0.0, 0.0, 0.0)
    side: str = "left"
    emit: int = 0
    beam: object = None
    min_weight: float = 1e-3

    def __post_init__(self):
        if _template(self):
            return
        object.__setattr__(self, "density", _float(self.density))
        object.__setattr__(self, "min_weight", _float(self.min_weight))
        for name in ("vth", "drift"):
            object.__setattr__(self, name, _components(getattr(self, name), name))
        _require(self.side in ("left", "right"), f"side is 'left' or 'right', not {self.side!r}")
        _require(self.emit >= 1, "a Source emits at least one particle per step")
        warm = any(_plain(u) and u != 0 for u in self.vth)
        if self.beam is None:
            _require(warm or all(_plain(u) for u in self.vth),
                     "a Source built from a traced vth must say whether it is a beam: pass beam=True or False")
            object.__setattr__(self, "beam", not warm)
        if warm and _plain(self.drift[0]) and self.drift[0] != 0:
            raise ValueError("a Source is a Maxwellian at rest or a cold beam: the crossing distribution of a "
                             "drifting Maxwellian, proportional to v exp[-(v-u)^2/2 sigma^2] on v > 0, has no "
                             "sampler here. Give vth=0 for a beam, or drift=(0, u_y, u_z).")

    @property
    def sigma(self):
        """Component spread :math:`\\sigma = v_{th}/\\sqrt2` of the reservoir."""
        return jax.numpy.asarray(self.vth[0]) / jax.numpy.sqrt(2.0)


@pytree_dataclass(static=("cells", "particle_bc", "field_bc"))
class Domain:
    """The simulation box.

    Args:
        length: Box length :math:`L` in metres; the box spans :math:`[-L/2, L/2]`.
        cells: Number of cells :math:`N_x`.
        dt_over_dx_c: Time step as :math:`c\\,\\Delta t/\\Delta x`.
        particle_bc: ``"periodic"``, ``"reflective"``, ``"absorbing"`` or
            ``"thermal"``, or a ``(left, right)`` pair. A thermal wall re-emits
            every particle that reaches it from the half-Maxwellian flux of its
            species, at the species' thermal speed.
        field_bc: The electrical condition at each wall: ``"periodic"``;
            ``"reflective"``, a symmetry plane where :math:`E_x` vanishes;
            ``"absorbing"``, a conductor that holds the charge it collects, two of
            them being short-circuited to each other; or ``"open"``, the plane a
            :class:`Source` supplies through, which imposes nothing and leaves the
            collector opposite to close the problem. A particle boundary and an
            electrical one are separate choices: a wall that absorbs particles is
            not necessarily a conductor, and a symmetry plane for the field does not
            reflect particles.
        restitution: Coefficient of restitution of the walls, or a
            ``(left, right)`` pair: whatever a wall sends back has its normal
            velocity multiplied by ``-restitution``. It applies to everything a
            reflective wall returns and to the part an absorbing wall reflects
            (see ``Species.reflection``).
        length_y, length_z: Periods of the ignorable coordinates.
    """
    length: float = 1e-2
    cells: int = 64
    dt_over_dx_c: float = 1.0
    particle_bc: object = "periodic"
    field_bc: object = "periodic"
    restitution: object = 1.0
    length_y: float = 1e-2
    length_z: float = 1e-2

    def __post_init__(self):
        if _template(self):
            return
        for name in ("length", "dt_over_dx_c", "length_y", "length_z"):
            object.__setattr__(self, name, _float(getattr(self, name)))
        object.__setattr__(self, "restitution", _walls(self.restitution, "restitution"))
        object.__setattr__(self, "particle_bc", _boundary_codes(self.particle_bc, "particle_bc"))
        object.__setattr__(self, "field_bc", _boundary_codes(self.field_bc, "field_bc"))
        _require(self.cells >= 4, "need at least four cells")
        for bc in (self.particle_bc, self.field_bc):
            _require((0 in bc) == (bc == (0, 0)), "a periodic wall needs a periodic partner")
        _require(3 not in self.field_bc, "a thermal wall re-emits particles; give the fields a reflective one")
        _require(4 not in self.field_bc or self.field_bc == (4, 2),
                 "field_bc='open' is the source plane of a box closed by a collector opposite: "
                 "use field_bc=('open', 'absorbing')")

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


@pytree_dataclass(static=("name", "n", "active", "plus_minus", "quiet", "random_positions"))
class Species:
    """One population of pseudo-particles.

    Velocities are Maxwellian with thermal speed ``vth`` per component, defined
    by :math:`f(v) \\propto \\exp(-v^2/v_{th}^2)`, that is
    :math:`v_{th} = \\sqrt{2 k_B T/m}`. The pseudo-particle weight is
    ``density * length / n``.

    Args:
        name: Label used in the output.
        n: Number of pseudo-particle slots. Without a source every slot holds a
            particle and ``n`` is the population; with one it is a capacity, and the
            slots a wall empties are the ones the source refills.
        active: Slots filled at :math:`t = 0`, ``n`` by default. They spread over the
            whole box and carry the weight ``density * length / active``; the rest
            start dead, parked beyond a wall with no weight, which is the headroom a
            source needs. Use it to start from a plasma rather than an empty box
            without giving the source a full pool.
        charge: Charge in units of the elementary charge.
        mass: Mass in kilograms.
        density: Number density in :math:`\\mathrm{m^{-3}}`.
        vth: Thermal speed per component, m/s: three numbers, an array whose last
            axis holds the three, or a bare number, which is the x component
            alone (the other two are then zero).
        drift: Drift velocity per component, m/s, given the same way.
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
        source: A :class:`Source` that maintains this species through one wall,
            or ``None``. The particles it emits occupy the species' own dead slots,
            so ``n`` is a capacity rather than a fixed population.
        reflection: Fraction of each particle that an absorbing wall sends back
            instead of collecting: a number, a function of the normal impact
            speed :math:`|v_x|` in m/s that returns the fraction, or a
            ``(left, right)`` pair of either. The wall keeps the rest of the
            particle's weight. A number is traced like any physical parameter; a
            function is compiled into the program.
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
    active: object = None
    plus_minus: bool = False
    quiet: bool = False
    random_positions: bool = False
    x: object = None
    v: object = None
    source: object = None
    reflection: object = 0.0

    def __post_init__(self):
        if _template(self):
            return
        _require(self.n > 0, "a species needs at least one particle")
        object.__setattr__(self, "active", self.n if self.active is None else int(self.active))
        _require(0 <= self.active <= self.n, f"active must be between 0 and n = {self.n}, not {self.active}")
        for name in ("charge", "mass", "density", "perturbation_amplitude", "perturbation_mode"):
            object.__setattr__(self, name, _float(getattr(self, name)))
        object.__setattr__(self, "reflection", _walls(self.reflection, "reflection"))
        for name in ("vth", "drift"):
            object.__setattr__(self, name, _components(getattr(self, name), name))
        for name in ("x", "v"):      # a leading axis is an ensemble, stacked by a tree operation
            value = getattr(self, name)
            _require(value is None or np.shape(value)[-2:] == (self.n, 3), f"{name} must have shape (n, 3)")

    @staticmethod
    def electrons(n, density, vth=(0.0, 0.0, 0.0), name="electrons", **kwargs):
        return Species(name, n, -1.0, mass_electron, density, vth, **kwargs)

    @staticmethod
    def ions(n, density, mass_ratio=1.0, charge=1.0, name="ions", vth=None,
             temperature_ratio=1.0, electrons=None, **kwargs):
        """Ions of mass ``mass_ratio`` proton masses. If ``vth`` is not given
        it is derived from the electron species as
        :math:`v_{th,i} = v_{th,e}\\sqrt{(T_i/T_e)(m_e/m_i)}`, which stays
        differentiable with respect to the electron thermal speed."""
        mass = mass_ratio * mass_proton
        if vth is None:
            if electrons is None:
                raise TypeError("Species.ions needs vth or the electron species")
            factor = (temperature_ratio * mass_electron / mass) ** 0.5
            vth = tuple(factor * u for u in electrons.vth)
        return Species(name, n, charge, mass, density, vth, **kwargs)

    @property
    def charge_si(self):
        return self.charge * elementary_charge


@pytree_dataclass(static=("algorithm", "model", "field_solver", "relativistic", "filter_passes",
                          "filter_strides", "picard_iterations", "substeps"))
class Solver:
    """Numerical choices.

    Args:
        algorithm: ``"explicit"`` (leapfrog with the Boris pusher) or
            ``"implicit"`` (Crank-Nicolson with Picard iteration).
        model: ``"electromagnetic"`` solves Maxwell's equations for all six field
            components; ``"electrostatic"`` solves :math:`\\partial_x E_x = \\rho/\\epsilon_0`
            alone. An electrostatic run keeps all three velocity components and any
            external field, but the plasma's own transverse fields are not evolved and
            the transverse currents are not deposited, which is cheaper and removes the
            light-wave time-step limit. Choose it whenever the physics is
            :math:`\\mathbf E = -\\nabla\\phi`: waves along the grid, sheaths, beam
            instabilities.
        field_solver: how an electromagnetic run advances :math:`E_x`: ``"ampere"``
            with Ampere's law and the charge-conserving current, or ``"gauss"`` by
            recomputing it from the charge density every step. An electrostatic run
            always solves the Gauss law.
        relativistic: Relativistic Boris pusher.
        filter_passes: Binomial smoothing passes on the sources (0 disables).
            Each pass is followed by one compensation pass.
        filter_alpha: Centre weight of the three-point filter.
        filter_strides: Cell offsets of the filter stencil.
        picard_iterations: Fixed-point iterations of the implicit scheme.
        substeps: Particle sub-steps per field step in the implicit scheme.
    """
    algorithm: str = "explicit"
    model: str = "electromagnetic"
    field_solver: str = "ampere"
    relativistic: bool = False
    filter_passes: int = 0
    filter_alpha: float = 0.5
    filter_strides: tuple = (1,)
    picard_iterations: int = 8
    substeps: int = 2

    def __post_init__(self):
        _require(self.algorithm in ("explicit", "implicit"),
                 f"algorithm is 'explicit' or 'implicit', not {self.algorithm!r}")
        _require(self.model in ("electromagnetic", "electrostatic"),
                 f"model is 'electromagnetic' or 'electrostatic', not {self.model!r}")
        _require(self.field_solver in ("ampere", "gauss"),
                 f"field_solver is 'ampere' or 'gauss', not {self.field_solver!r}")
        _require(self.filter_passes >= 0 and self.picard_iterations >= 1 and self.substeps >= 1,
                 "filter_passes cannot be negative, and picard_iterations and substeps must be at least one")
        object.__setattr__(self, "filter_alpha", _float(self.filter_alpha))
        object.__setattr__(self, "filter_strides", tuple(int(s) for s in self.filter_strides))

    @property
    def electrostatic(self):
        """Whether the plasma's own transverse fields are left out of the run."""
        return self.model == "electrostatic"


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
