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

__all__ = ["Domain", "Species", "Solver", "Source", "Collisions", "Impacts", "BOUNDARIES"]

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


@pytree_dataclass(static=("side", "emit", "model", "every"))
class Source:
    """A maintained inflow of one species through one wall: a reservoir of plasma
    behind the plane that supplies a prescribed flux, independently of what leaves.

    The distribution behind the plane is a Maxwellian at rest or a cold beam; the
    flux that crosses is the velocity density weighted by the normal speed
    (:mod:`~jaxincell._sources`). A fixed number of particles is emitted once every
    ``every`` steps with a continuous weight :math:`\\Gamma k\\Delta t/N_{\\rm emit}`,
    :math:`k` being ``every``, so the emitted weight is exactly the prescribed flux and is
    differentiable in ``density`` and ``vth``.

    Args:
        density: Reservoir number density :math:`n_{\\rm in}`, :math:`\\mathrm{m^{-3}}`.
        vth: Thermal speed per component of the reservoir, :math:`\\sqrt{2k_BT/m}`, m/s,
            given as :class:`Species` takes it. All three components are used: the normal
            one sets the crossing distribution and the two tangential ones are drawn from
            their own Maxwellians. Zero makes a cold beam.
        drift: Drift velocity of the reservoir, m/s. The tangential components ride along
            unchanged; the normal one is the drift of the reservoir towards or away from
            the plane, and its **sign** is physical -- a reservoir drifting away still sends
            some flux across, and a cold beam pointing away from the plane sends none and is
            refused rather than reflected.
        side: ``"left"`` or ``"right"``, the wall the plasma enters through.
        model: Which crossing distribution the sampler draws from: ``"beam"`` for a cold
            reservoir, ``"maxwellian"`` for a Maxwellian with no normal drift,
            ``"drifting"`` for one with a normal drift, or ``"sampled"`` for a reservoir
            given as ``samples``. It is worked out from ``vth``, ``drift`` and ``samples``
            when the object is built and is then static, because it selects a
            branch: read live from the leaves it would be a traced value, and inside
            ``jit`` every source would take the same branch whatever it holds.
        samples: Velocities of the reservoir itself, ``(k, 3)`` in m/s, for a distribution
            that is none of the three closed forms -- one computed by another code, or
            measured. They are the distribution **behind** the plane, not the flux across
            it: the sampler weights them by their inward normal component, which is what
            makes fast particles cross more often, and ignores those that do not cross.
            ``density`` is still the reservoir's density, and the flux follows from the two.
            ``vth`` and ``drift`` are then unused. It is data rather than a model, so the
            emitted weight is differentiable in ``density`` and not in the samples.
        emit: Particles emitted at each emission. They occupy the dead slots of the
            species, so the species needs enough of them: ``n`` must exceed
            ``emit / every`` times the longest residence time in steps.
        every: Steps between emissions, :math:`k`. One emits on every step. More emits
            ``emit`` particles once every :math:`k` steps, each carrying :math:`k` steps'
            worth of flux and spread over the window they stand for: the one that crossed
            the plane at the fraction :math:`s_j = (j+\\tfrac12)/N_{\\rm emit}` of it is placed
            where its orbit has taken it since. The flux is unchanged and the pool the
            species needs falls by :math:`k`, which is what a species whose residence is
            hundreds of thousands of steps needs. The field at the plane is held over the
            window, so :math:`k` is bounded by the entry's own resolution: the distance an
            entering particle covers in :math:`k\\Delta t` small against a cell, and its
            gyro-angle :math:`|\\Omega|k\\Delta t` small against one. Within
            :math:`v_xk\\Delta t` of the plane the density is short by a ramp, because a
            particle is only in the box from the emission after it crossed. The first
            emission is on step :math:`k-1`, so that after any whole number of windows the
            emitted weight equals ``every = 1``'s. Static, since it sets the schedule.
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
    model: object = None
    min_weight: float = 1e-3
    samples: object = None
    every: int = 1

    def __post_init__(self):
        if _template(self):
            return
        object.__setattr__(self, "density", _float(self.density))
        object.__setattr__(self, "min_weight", _float(self.min_weight))
        for name in ("vth", "drift"):
            object.__setattr__(self, name, _components(getattr(self, name), name))
        _require(self.side in ("left", "right"), f"side is 'left' or 'right', not {self.side!r}")
        _require(self.emit >= 1, "a Source emits at least one particle per emission")
        _require(isinstance(self.every, (int, np.integer)) and not isinstance(self.every, bool)
                 and self.every >= 1,
                 f"every is a whole number of steps between emissions, at least one, not {self.every!r}")
        object.__setattr__(self, "every", int(self.every))
        _require(not _plain(self.density) or self.density >= 0,
                 f"a Source density cannot be negative, not {self.density!r}")
        _require(not _plain(self.min_weight) or 0 <= self.min_weight <= 1,
                 f"min_weight is a fraction of the emitted weight, in [0, 1], not {self.min_weight!r}")
        if self.samples is not None:
            object.__setattr__(self, "samples", jax.numpy.asarray(self.samples))
            _require(jax.numpy.ndim(self.samples) == 2 and jax.numpy.shape(self.samples)[-1] == 3
                     and jax.numpy.shape(self.samples)[0] >= 1,
                     "Source samples are the reservoir's velocities, of shape (k, 3), not "
                     f"{jax.numpy.shape(self.samples)}")
        if self.model is None:
            object.__setattr__(self, "model", "sampled" if self.samples is not None
                               else self._model_from_leaves())
        _require(self.model in ("beam", "maxwellian", "drifting", "sampled"),
                 f"model is 'beam', 'maxwellian', 'drifting' or 'sampled', not {self.model!r}")
        _require((self.model == "sampled") == (self.samples is not None),
                 "model='sampled' is the one that draws from samples, and the one that needs them: "
                 f"model is {self.model!r} and samples are {'given' if self.samples is not None else 'not'}")
        inward = 1.0 if self.side == "left" else -1.0
        if self.model == "beam" and _plain(self.drift[0]):
            _require(inward * self.drift[0] > 0,
                     f"a cold beam entering through the {self.side} wall needs a normal drift towards the box, "
                     f"which is {'positive' if inward > 0 else 'negative'} here, not {self.drift[0]!r}. A beam "
                     "pointing away from the plane sends no flux across it.")

    def _model_from_leaves(self):
        """Which crossing distribution the leaves describe, decided once, at construction.

        Both branches read leaves, so both have to be settled here: inside ``jit`` a
        comparison against a leaf is a traced array and every source would take one branch
        whatever it holds. A traced leaf whose branch matters therefore has to be named."""
        if not all(_plain(u) for u in self.vth):
            raise ValueError("a Source built from a traced vth must say which crossing distribution it is: "
                             "pass model='beam', 'maxwellian' or 'drifting'")
        if not any(u != 0 for u in self.vth):
            return "beam"
        if not _plain(self.drift[0]):
            raise ValueError("a Source built from a traced normal drift must say which crossing distribution it "
                             "is: pass model='maxwellian' for a reservoir at rest or model='drifting' for one "
                             "that drifts towards or away from the plane")
        return "drifting" if self.drift[0] != 0 else "maxwellian"

    @property
    def beam(self):
        """Whether the reservoir is cold."""
        return self.model == "beam"

    @property
    def sigma(self):
        """The three component spreads :math:`\\sigma = v_{th}/\\sqrt2` of the reservoir."""
        return jax.numpy.asarray(self.vth) / jax.numpy.sqrt(2.0)


@pytree_dataclass(static=("energy_bins", "angle_bins"))
class Impacts:
    """Fixed bins for the energy and incidence of what reaches each wall.

    A snapshot of the particles near a wall is not a spectrum of impacts: it repeats a
    particle across frames, counts outgoing ones, and weights by how many happen to be
    there rather than by how many crossed. This accumulates one entry per crossing, at the
    moment of the crossing and at the velocity that carried the particle there, so summing
    it back gives exactly the fluence ``Wall.arrived``.

    Args:
        energy_max: Top of the last resolved energy bin, in joules per particle. Everything
            above it lands in one overflow bin, which is kept rather than clipped into the
            last resolved bin, so a spectrum says when its range was too small.
        energy_bins: Number of equal bins over :math:`[0, E_{\\max}]`. The accumulator has
            one more, the overflow.
        angle_bins: Number of equal bins of the incidence angle
            :math:`\\theta = \\arctan(|v_t|/v_n)` over :math:`[0, \\pi/2]`, zero being normal
            incidence. The angle is bounded, so it needs no overflow bin.
    """
    energy_max: float
    energy_bins: int = 32
    angle_bins: int = 18

    def __post_init__(self):
        if _template(self):
            return
        object.__setattr__(self, "energy_max", _float(self.energy_max))
        _require(not _plain(self.energy_max) or self.energy_max > 0,
                 f"energy_max is the top of the last resolved bin and must be positive, not {self.energy_max!r}")
        _require(self.energy_bins >= 1 and self.angle_bins >= 1, "a spectrum needs at least one bin of each")

    @property
    def energy_edges(self):
        """The ``energy_bins + 1`` edges of the resolved bins; the overflow bin is above the last."""
        return jax.numpy.arange(self.energy_bins + 1) * (self.energy_max / self.energy_bins)

    @property
    def angle_edges(self):
        """The ``angle_bins + 1`` edges of the incidence bins, radians."""
        return jax.numpy.arange(self.angle_bins + 1) * (jax.numpy.pi / 2 / self.angle_bins)


@pytree_dataclass(static=("cells", "particle_bc", "field_bc"))
class Domain:
    """The simulation box.

    Args:
        length: Box length :math:`L` in metres; the box spans :math:`[-L/2, L/2]`.
        cells: Number of cells :math:`N_x`.
        time_step: Time step in seconds. Give this or ``dt_over_dx_c``, not both;
            ``Domain.dt`` is the step in seconds either way.
        dt_over_dx_c: Time step as :math:`c\\,\\Delta t/\\Delta x`, the Courant number of a
            light wave, and the natural input for an electromagnetic run, whose stability
            limit is one. An electrostatic run has no light wave and is set by the plasma
            frequency instead, where ``time_step`` says directly what it means.
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
    dt_over_dx_c: object = None
    time_step: object = None
    particle_bc: object = "periodic"
    field_bc: object = "periodic"
    restitution: object = 1.0
    length_y: float = 1e-2
    length_z: float = 1e-2

    def __post_init__(self):
        if _template(self):
            return
        _require(self.time_step is None or self.dt_over_dx_c is None,
                 "give Domain either time_step, in seconds, or dt_over_dx_c, the light-wave Courant "
                 "number, not both")
        if self.time_step is None and self.dt_over_dx_c is None:
            object.__setattr__(self, "dt_over_dx_c", 1.0)
        for name in ("length", "dt_over_dx_c", "time_step", "length_y", "length_z"):
            object.__setattr__(self, name, _float(getattr(self, name)))
        object.__setattr__(self, "restitution", _walls(self.restitution, "restitution"))
        object.__setattr__(self, "particle_bc", _boundary_codes(self.particle_bc, "particle_bc"))
        object.__setattr__(self, "field_bc", _boundary_codes(self.field_bc, "field_bc"))
        _require(self.cells >= 4, "need at least four cells")
        for name in ("length", "length_y", "length_z"):
            value = getattr(self, name)
            _require(not _plain(value) or value > 0, f"{name} must be positive, not {value!r}")
        step = self.time_step if self.dt_over_dx_c is None else self.dt_over_dx_c
        _require(not _plain(step) or step > 0,
                 f"the time step must be positive, not {step!r}; a run goes forwards")
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
        """The time step in seconds, however it was given."""
        return self.time_step if self.dt_over_dx_c is None else self.dt_over_dx_c * self.dx / speed_of_light

    @property
    def courant(self):
        """:math:`c\\,\\Delta t/\\Delta x`, however the step was given: the Courant number of a
        light wave, which the explicit electromagnetic field update needs below one. It is
        returned as given rather than recomputed, so that a run asking for exactly one gets
        exactly one and not the round-off above it."""
        return self.time_step * speed_of_light / self.dx if self.dt_over_dx_c is None else self.dt_over_dx_c

    @property
    def grid(self):
        """Cell centres :math:`x_i = -L/2 + (i + 1/2)\\Delta x`, where the densities and the
        deposited moments live."""
        import jax.numpy as jnp
        return -self.length / 2 + (jnp.arange(self.cells) + 0.5) * self.dx

    @property
    def faces(self):
        """The stored cell faces :math:`x_{i+1/2} = -L/2 + (i + 1)\\Delta x`, the right face of
        each cell, where :math:`E_x` and the potential live. The left wall face is not stored."""
        return self.grid + self.dx / 2


@pytree_dataclass(static=("name", "n", "active", "plus_minus", "sampling"))
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
        sampling: How the initial phase space is drawn. Three states, and they were two
            booleans, ``quiet`` and ``random_positions``, of which one combination -- both
            true -- silently meant the first:

            * ``"quiet"``, the quiet start: equally spaced positions and velocities at the
              quantiles of the Maxwellian, ordered by a bit-reversed sequence, which is what
              makes the discrete-particle noise low enough to follow a Landau decay over
              three e-foldings;
            * ``"lattice"``, the default: equally spaced positions and random velocities;
            * ``"random"``: uniformly random positions and random velocities.
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
    sampling: str = "lattice"
    x: object = None
    v: object = None
    source: object = None
    reflection: object = 0.0

    def __post_init__(self):
        if _template(self):
            return
        _require(self.n > 0, "a species needs at least one particle")
        _require(self.sampling in ("quiet", "lattice", "random"),
                 f"sampling is 'quiet', 'lattice' or 'random', not {self.sampling!r}")
        object.__setattr__(self, "active", self.n if self.active is None else int(self.active))
        _require(0 <= self.active <= self.n, f"active must be between 0 and n = {self.n}, not {self.active}")
        for name in ("charge", "mass", "density", "perturbation_amplitude", "perturbation_mode"):
            object.__setattr__(self, name, _float(getattr(self, name)))
        _require(not _plain(self.mass) or self.mass > 0, f"mass must be positive, not {self.mass!r}")
        _require(not _plain(self.density) or self.density >= 0,
                 f"density cannot be negative, not {self.density!r}")
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
            the transverse currents are not deposited, which removes the light-wave
            time-step limit and is 1.19 times faster, measured at 200000 particles on 256
            cells on a CPU. Choose it whenever the physics is
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
