"""The simulation: initial state, the time loop, and the output."""
from __future__ import annotations

import os
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.scipy.special import erfinv

from ._collisions import collide, coulomb_logarithm
from ._config import Collisions, Domain, Solver, Species, pytree_dataclass
from ._config import elementary_charge, epsilon_0, mass_electron, mass_proton, speed_of_light as c
from ._core import (PARITY, PARK, E_x_from_rho, apply_particle_bc, boris, boris_relativistic, s2_weights,
                    current_from_continuity, curl_B, curl_E, deposit, gather, half_step_fields, smooth,
                    to_centres, to_faces, wall_faces_E, with_ghosts, wrap_positions)
from ._sources import check_sources, crossing_flux, inject


def _enable_double_precision(environ):
    """Double precision unless the environment asks otherwise through JAX's own switch,
    ``JAX_ENABLE_X64=0``."""
    if "JAX_ENABLE_X64" not in environ:
        jax.config.update("jax_enable_x64", True)


_enable_double_precision(os.environ)

_BETA2_MAX = 1 - 1e-5     # largest v^2/c^2 of a velocity entering a relativistic run; see Simulation._momentum

__all__ = ["Simulation", "Output", "State", "Wall", "load_toml", "quiet_start"]


@pytree_dataclass(static=())
class Wall:
    """What the two walls have exchanged with each species since the run began.

    Every field has shape ``(species, side)``, side 0 the left wall and side 1 the
    right, in the units of a planar run: a weight is physical particles per unit area,
    :math:`\\mathrm{m^{-2}}`, and an energy is :math:`\\mathrm{J/m^2}`. Multiply a weight
    by the species' charge for a collected charge, or divide by the elapsed time for a flux.

    Every exchange is recorded **after** the wall's law has acted, so a thermal wall's
    ``energy_out`` is the energy of the particle it re-emitted and not of the bounce that
    preceded the redraw. Kinetic energy is :math:`m|\\mathbf u|^2/(\\gamma+1)`, which is
    :math:`m v^2/2` in a Newtonian run and the relativistic energy from the carried momentum
    otherwise, so the ledger does not silently change meaning with ``Solver(relativistic=)``.

    Attributes:
        arrived: Weight that reached the wall, counting every impact of a particle that
            bounces more than once.
        collected: The part of it the wall kept. ``arrived - collected`` went back.
        injected: Weight a :class:`~jaxincell.Source` emitted through the wall.
        energy_in: Kinetic energy carried to the wall, at the velocity of the numerical
            drift segment on which the particle crossed.
        energy_out: Kinetic energy carried back out by what the wall returned. The
            difference is what the wall absorbed, including the loss to a coefficient of
            restitution below one and the heat a thermal wall gives or takes.
        energy_injected: Kinetic energy a :class:`~jaxincell.Source` carried in.
        truncated: Weight a wall kept only because ``Source.min_weight`` stopped the orbit,
            which its reflection law would otherwise have sent back. It is the cost of the
            cutoff, in the units the rest of the ledger is in, so a run can say what it was
            rather than assume it was nothing.
        momentum: Momentum delivered to the wall, ``(species, side, 3)``, in
            :math:`\\mathrm{kg\\,m^{-1}s^{-1}}`: what arrived less what went back out. Its
            sign is the direction the wall is pushed, so the left wall's x component is
            negative for a plasma pressing outwards on both sides.
        momentum_injected: Momentum a :class:`~jaxincell.Source` carried in, the same shape.
        spectrum: Weight that arrived in each energy and incidence bin,
            ``(species, side, energy_bins + 1, angle_bins)``, when
            :class:`~jaxincell.Simulation` was given :class:`~jaxincell.Impacts`, and
            ``None`` otherwise. The last energy bin is the overflow. Summing over the two
            bin axes gives ``arrived`` exactly: every crossing is entered once, at the
            velocity that carried it there.
        overflow: Largest live weight a source has overwritten, zero while the pool of
            dead slots holds. A positive value means the capacity ``Species.n`` is too
            small and particles were destroyed to make room.
    """
    arrived: object
    collected: object
    injected: object
    energy_in: object
    energy_out: object
    energy_injected: object
    momentum: object
    momentum_injected: object
    truncated: object
    spectrum: object
    overflow: object

    def charge(self, charge_per_particle):
        """Charge on each wall, C/m^2, from ``charge_per_particle`` per species."""
        return jnp.sum(jnp.asarray(charge_per_particle)[:, None] * self.collected, axis=0)


@pytree_dataclass(static=())
class State:
    """Everything the time loop carries from one step to the next, and all that a
    restart needs. ``run(..., state=out.state)`` continues from it: the absolute time
    and step count, the fields, the particles with their weights, the charge density the
    step begins with, the random key and the wall ledger all go on unbroken.

    ``time`` and ``steps`` are both absolute and both counted from the start of the first
    run, not of this one. A cumulative diagnostic divides by a difference of them, never by
    the length of an array.

    ``x`` is the half-step position of the leapfrog and ``u`` the momentum per unit
    mass of a relativistic run, or the velocity otherwise; the implicit scheme carries
    integer-time positions instead. ``sigma`` is the charge on the collector at the time
    ``rho`` is for, collected and overlapping together, which is what makes the boundary
    current a difference rather than a guess. ``moments`` is the running sum of
    :meth:`Simulation.moments`, or ``None`` when ``run(moments=False)``.
    """
    E: object
    B: object
    x: object
    u: object
    w: object
    qm: object
    rho: object
    sigma: object
    key: object
    time: object
    steps: object
    wall: object
    moments: object


@pytree_dataclass(static=("names", "counts", "relativistic", "field_bc"))
class Output:
    """Result of :meth:`Simulation.run`. Histories have the stored step as their
    first axis; ``t`` is the time of each stored state. ``charge`` and ``mass``
    are those of one physical particle, and ``weight`` is the history of the
    pseudo-particle weights, which fall as absorbing walls collect the particles.
    ``wall`` is the history of the :class:`Wall` ledger, running totals of what the two
    walls and any sources have exchanged. ``state`` is the final :class:`State` and can be
    passed back to :meth:`Simulation.run` to continue; in a relativistic run it carries the
    momentum per unit mass :math:`\\gamma\\mathbf v` where ``v`` has the velocity.

    ``t`` and ``steps`` are absolute: a continued run goes on from the time and the step
    count its state had reached, so the histories of a run split into chunks join without a
    shift, and the window between two stored states is a difference of them rather than a
    count of array entries. ``moments`` is the history of the running sums of
    :meth:`Simulation.moments` when ``run(moments=True)`` asked for them, and ``None``
    otherwise.

    ``sigma`` is the charge on each wall, ``(stored, 2)``: what it has collected plus the part
    of the live clouds that reaches past it. With the charge on the grid it is everything the box
    holds, which is what :func:`~jaxincell.charge_balance` checks against what went in and out.

    ``grid``, ``faces`` and ``walls`` are the three coordinate arrays of the staggered grid.
    Densities and deposited moments are on ``grid``; :math:`E_x` and the potential are on
    ``faces``."""
    t: object
    steps: object
    sigma: object
    x: object
    v: object
    E: object
    B: object
    J: object
    rho: object
    grid: object
    dx: object
    dt: object
    length: object
    charge: object
    mass: object
    weight: object
    wall: object
    moments: object
    species: object
    state: object
    names: tuple
    counts: tuple
    relativistic: bool
    field_bc: tuple

    @property
    def overflow(self):
        """Largest live weight a source has had to overwrite, at each stored step: zero while
        the pool of dead slots holds, and a running maximum once it has not, so it never falls
        back to zero and a run cannot look valid because it recovered later."""
        return self.wall.overflow

    @property
    def problems(self):
        """What makes this run unusable, as a tuple of sentences, empty when nothing does.

        It reads values, so it belongs on the host: inside ``jit`` or ``grad`` there is nothing
        to read. A differentiated objective takes :attr:`overflow` out with its result instead
        and rejects the trial itself; :meth:`validate` is the host-side shortcut."""
        spilt = float(jnp.max(jnp.asarray(self.overflow)))
        if spilt <= 0:
            return ()
        return (f"a source overwrote live particles: the largest weight destroyed was {spilt:.3g}, "
                f"against {float(jnp.sum(jnp.asarray(self.weight)[-1])) if self.weight is not None else 0:.3g} "
                "left alive. Species.n is a capacity and has to hold every particle alive at once, so it "
                "grows with the emission rate and with how long a particle lives -- a smaller time step and "
                "a longer box both lengthen that. Nothing this run reports is trustworthy.",)

    def validate(self):
        """Raise :class:`RuntimeError` if :attr:`problems` is not empty, and return the output
        otherwise, so that a script can write ``out = simulation.run(...).validate()``."""
        if self.problems:
            raise RuntimeError(" ".join(self.problems))
        return self

    @property
    def faces(self):
        """The cell faces the staggered grid stores, :math:`x_{i+1/2} = -L/2 + (i+1)\\Delta x`:
        the right face of each cell. The left wall face :math:`-L/2` is not among them, which is
        why :func:`~jaxincell.potential` and the field solver take it separately."""
        return self.grid + self.dx / 2

    @property
    def walls(self):
        """The two wall faces, :math:`-L/2` and :math:`+L/2`."""
        return jnp.stack([-self.length / 2, self.length / 2])

    def particles(self, name):
        """Positions and velocities ``(S, n, 3)`` of the species called ``name``."""
        start = sum(self.counts[: self.names.index(name)])
        stop = start + self.counts[self.names.index(name)]
        return self.x[:, start:stop], self.v[:, start:stop]


def _van_der_corput(n, base):
    q, denominator, i = np.zeros(n), 1.0, np.arange(1, n + 1)
    while i.any():
        denominator *= base
        q += (i % base) / denominator
        i //= base
    return q


def quiet_start(n, length, vth=(0.0, 0.0, 0.0), drift=(0.0, 0.0, 0.0)):
    """Positions and velocities of a quiet start, as plain arrays.

    Equally spaced positions and velocities at the quantiles of a bit-reversed
    (van der Corput) sequence, which is what ``Species(quiet=True)`` uses. It is
    exposed because custom initial conditions are often a quiet start plus a
    coherent seed -- a transverse current for the Weibel instability, say -- and
    building that by hand otherwise means reproducing the sampling.

    The sequence itself is built on the host, since it is a fixed set of quadrature
    nodes and nothing differentiates with respect to it; the length, the thermal speed
    and the drift that scale and shift it are ordinary JAX values, so a gradient with
    respect to any of them passes through the advertised way of building an initial
    state, and the whole function works inside ``jit``. That makes the arrays JAX arrays,
    which are immutable, so a coherent seed is added with ``.at[]`` rather than in place::

        x, v = quiet_start(n, length, vth=vth)
        v = v.at[:, 2].add(0.01 * vth[2] * jnp.sin(2 * jnp.pi * x[:, 0] / length))
        electrons = Species.electrons(n=n, density=n_e, vth=vth).replace(x=x, v=v)

    Args:
        n: Number of pseudo-particles.
        length: Box length; positions fill ``[-L/2, L/2]``.
        vth: Thermal speed per component, ``sqrt(2 k_B T / m)``.
        drift: Drift velocity per component.

    Returns:
        tuple: ``x`` and ``v``, both ``(n, 3)``, ready for
        ``species.replace(x=x, v=v)``.
    """
    lattice = jnp.asarray((np.arange(n) + 0.5) / n - 0.5)                 # host: the fixed nodes
    nodes = jnp.asarray(np.stack([_van_der_corput(n, base) for base in (2, 3, 5)], axis=1))
    x = jnp.zeros((n, 3)).at[:, 0].set(length * lattice)
    return x, erfinv(2 * nodes - 1) * jnp.asarray(vth) + jnp.asarray(drift)


@pytree_dataclass(static=())
class Simulation:
    """A one-dimensional, three-velocity particle-in-cell simulation.

    Args:
        domain: The box, grid, time step and boundaries.
        species: The particle populations.
        solver: Integrator, field solver and filter.
        collisions: Binary-collision model, or ``None``.
        external_E, external_B: Static external fields as arrays of shape
            ``(cells, 3)`` on the faces (E) and centres (B), or ``None``.
        impacts: :class:`~jaxincell.Impacts` bins for the energy and incidence of what
            reaches each wall, or ``None`` for no spectrum. It costs one scatter per
            species per wall per step.

    The object is a JAX pytree: every physical parameter is a leaf, so
    ``jax.grad`` and ``jax.vmap`` apply to functions of it directly, and changing
    a physical parameter never recompiles the program.
    """
    domain: Domain
    species: tuple
    solver: Solver = Solver()
    collisions: object = None
    external_E: object = None
    external_B: object = None
    impacts: object = None

    def __post_init__(self):
        object.__setattr__(self, "species", tuple(self.species))
        if not self.species:
            raise ValueError("a Simulation needs at least one species")
        names = [s.name for s in self.species]
        if len(set(names)) != len(names):
            raise ValueError(f"species names must be distinct, got {names}")
        self._check_implicit()
        self._check_collisions()
        self._check_courant()
        check_sources(self.species, self.solver, self.domain)

    def _check_implicit(self):
        """Refuse the solver switches the Crank-Nicolson scheme would otherwise ignore.

        A filter keeps its energy conservation only if the same filter acts on the current
        and on the field gathered at the particles, as a transpose pair that respects the
        parity of each component at the walls; that pair is not implemented. The Gauss
        solve, and the electrostatic model that always uses it, would overwrite E_x after
        the update that conserves energy, which is the property the scheme is there for.
        Silently skipping a switch, as the scheme once did, makes a run look filtered or
        electrostatic when it is neither."""
        s = self.solver
        if s.algorithm != "implicit":
            return
        if s.filter_passes:
            raise ValueError("the implicit scheme has no filter: conserving energy needs the same filter on the "
                             "current and on the gathered field, which is not implemented. Use filter_passes=0, "
                             "or algorithm='explicit'.")
        if self.domain.field_bc == (4, 2):
            raise ValueError("field_bc=('open', 'absorbing') is the source plane of a box closed by a "
                             "collector, and the implicit scheme carries no surface charge to close it "
                             "with: its continuity current would be anchored at nothing. A Source needs "
                             "algorithm='explicit' in any case.")
        for setting in ("field_solver='gauss'" if s.field_solver == "gauss" else None,
                        "model='electrostatic'" if s.electrostatic else None):
            if setting is not None:
                raise ValueError(f"{setting} would take E_x from the charge density and so replace the update "
                                 "that makes the implicit scheme conserve energy, which is the property it is "
                                 "there for; it is available with algorithm='explicit' only.")

    def _check_collisions(self):
        """The default Coulomb logarithm is taken from the lightest negatively charged species,
        so collisions without a given ``coulomb_log`` need one. Traced charges are skipped, as
        in :meth:`_check_courant`, since their sign is not known until the program runs.

        A relativistic pusher is refused outright: Takizuka and Abe pair particles by their
        lab-frame relative velocity and rotate it through an angle whose variance is the
        nonrelativistic Coulomb one, so the operator is not the relativistic binary collision
        and combining the two would report a rate that belongs to neither."""
        if self.collisions is None:
            return
        if self.solver.relativistic:
            raise ValueError("Collisions() is the nonrelativistic Takizuka-Abe operator and "
                             "Solver(relativistic=True) is the relativistic pusher; the combination is not "
                             "implemented. Use one or the other.")
        if self.collisions.coulomb_log is not None:
            return
        charges = [s.charge for s in self.species]
        if any(isinstance(q, jax.core.Tracer) for q in charges):
            return
        if not any(q < 0 for q in charges):
            raise ValueError("Collisions() takes its default Coulomb logarithm from the negatively charged species, "
                             "and there is none: give Collisions(coulomb_log=...).")

    def _check_courant(self):
        """The explicit field update is unstable for ``c dt > dx``. The electrostatic
        model has no light wave to be unstable, and an electromagnetic run whose
        particles carry no transverse velocity never seeds one, so warn only when a
        run could. Traced values are skipped, so the check happens when the object is
        first built and not on every rebuild."""
        def plain(v):
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        courant = self.domain.courant
        if (self.solver.algorithm != "explicit" or self.solver.electrostatic
                or not plain(courant) or courant <= 1):
            return

        def transverse(s):
            if s.v is not None:                      # given as an array: look at it
                v = np.asarray(s.v) if not isinstance(s.v, jax.core.Tracer) else None
                return v is None or bool(np.any(v[:, 1:]))
            return any(plain(u) and u != 0 for u in tuple(s.vth[1:]) + tuple(s.drift[1:]))

        if any(transverse(s) for s in self.species):
            warnings.warn(f"c dt / dx = {courant:g} exceeds one while the particles carry transverse "
                          "velocity: the explicit field solver is unstable for electromagnetic waves. "
                          "Use dt_over_dx_c <= 1, or algorithm='implicit'.", stacklevel=3)

    # -- derived quantities -------------------------------------------------------
    @property
    def blocks(self):
        starts = np.cumsum([0] + [s.n for s in self.species])
        return tuple((int(a), int(s.n)) for a, s in zip(starts, self.species))

    def plasma_frequency(self, name=None):
        """Plasma frequency of a species (the first by default), rad/s."""
        s = self.species[0] if name is None else self.species[[t.name for t in self.species].index(name)]
        return jnp.sqrt(s.density * s.charge_si / epsilon_0 * (s.charge_si / s.mass))   # no product below 1e-38

    def debye_length(self, name=None):
        s = self.species[0] if name is None else self.species[[t.name for t in self.species].index(name)]
        return jnp.max(jnp.asarray(s.vth)) / (jnp.sqrt(2.0) * self.plasma_frequency(name))

    def _reflection(self, v):
        """Fraction of each particle's weight that the left and the right wall send
        back, from the law of its species at its normal speed."""
        speed = jnp.abs(v[:, 0])

        def law(r, start, n):
            return jnp.broadcast_to(r(speed[start:start + n]) if callable(r) else r, (n,))

        return tuple(jnp.clip(jnp.concatenate([law(s.reflection[side], *block)
                                               for s, block in zip(self.species, self.blocks)]), 0.0, 1.0)
                     for side in (0, 1))

    def _thermalise(self, key, x, u):
        """Redraw the velocity of every particle that crossed a thermal wall from the
        half-Maxwellian flux of its species: the normal speed from the Rayleigh
        distribution :math:`\\sigma\\sqrt{-2\\ln U}`, pointing into the box, and the
        tangential components from the Maxwellian, with :math:`\\sigma = v_{th}/\\sqrt2`.
        Takes and returns the carried ``u`` (:meth:`_momentum`)."""
        d = self.domain
        if 3 not in d.particle_bc:
            return u
        sigma = jnp.concatenate([jnp.broadcast_to(jnp.asarray(s.vth) / jnp.sqrt(2.0), (s.n, 3))
                                 for s in self.species])
        k_normal, k_tangential = random.split(key)
        uniform = random.uniform(k_normal, (u.shape[0],), minval=jnp.finfo(u.dtype).tiny)
        left = x[:, 0] < -d.length / 2
        new = (sigma * random.normal(k_tangential, u.shape)).at[:, 0].set(
            jnp.where(left, 1.0, -1.0) * sigma[:, 0] * jnp.sqrt(-2 * jnp.log(uniform)))
        hit = (left & (d.particle_bc[0] == 3)) | ((x[:, 0] > d.length / 2) & (d.particle_bc[1] == 3))
        return jnp.where(hit[:, None], self._momentum(new), u)

    @staticmethod
    def _split_step_key(key):
        """The keys of one step: the key carried to the next step, the collisions', the
        thermal wall's and the sources'. Every key is split once and then either split
        again or drawn from, never both, and there is no ``fold_in``: in JAX's threefry keys
        ``fold_in(k, 1)`` is ``split(k)[1]``, and ``split(k, 2)[i]`` is ``split(k, 5)[i]``,
        so keys derived from one parent in two ways coincide and two consumers draw
        the same numbers. That same identity makes the first three keys here independent
        of whether the fourth is asked for, so adding sources left every other stream
        where it was."""
        return random.split(key, 4)

    def _gamma(self, u):
        """The Lorentz factor of the carried ``u``, ``(N, 1)``, and one in a Newtonian run."""
        if not self.solver.relativistic:
            return 1.0
        return jnp.sqrt(1 + jnp.sum(u * u, axis=1, keepdims=True) / c ** 2)

    def _velocity(self, u):
        """The velocity of the carried ``u``, which is :math:`\\gamma\\mathbf v` in a
        relativistic run and the velocity itself otherwise."""
        return u / self._gamma(u)

    def _mean_velocity(self, u, u_new):
        """The velocity that carries a particle through a push from ``u`` to ``u_new``,
        :math:`(\\mathbf u + \\mathbf u')/(\\gamma + \\gamma')`, along which the Boris step changes the
        kinetic energy by exactly the work of E, Newtonian (the mean velocity) or relativistic."""
        return (u + u_new) / (self._gamma(u) + self._gamma(u_new))

    def _momentum(self, v):
        """The carried ``u`` of a velocity: :math:`\\gamma\\mathbf v` in a relativistic run,
        the velocity itself otherwise.

        A drawn velocity can reach or pass :math:`c` -- the tail of a Maxwellian sampled as
        if Newtonian, or a drift given too close to it -- and is then not a velocity. In a
        relativistic run a speed with :math:`v^2/c^2 > 1 - 10^{-5}` is brought back to that
        speed along its own direction, which caps :math:`\\gamma` at 316. The margin
        :math:`10^{-5}` is about a hundred times the resolution of single precision, where
        :math:`\\gamma` is then still known to 1 %. A Newtonian run has no speed limit and
        leaves velocities, and their derivatives, alone."""
        if not self.solver.relativistic:
            return v
        beta2 = jnp.sum(v * v, axis=1, keepdims=True) / c ** 2
        v = v * jnp.sqrt(_BETA2_MAX / jnp.maximum(beta2, _BETA2_MAX))
        return v / jnp.sqrt(1 - jnp.sum(v * v, axis=1, keepdims=True) / c ** 2)

    def _collide_momenta(self, key, x, u, w, qm, m, dt):
        """The collision operator acts on velocities; a relativistic run converts to it and back."""
        if self.collisions is None:
            return u
        return self._momentum(self._collide(key, x, self._velocity(u), w, qm, m, dt))

    # -- sources, walls and the field model ---------------------------------------------

    @property
    def sources(self):
        """The species that a :class:`~jaxincell.Source` maintains, with their blocks."""
        return tuple((sp, block) for sp, block in zip(self.species, self.blocks) if sp.source is not None)

    def _weight_floor(self):
        """Per particle, the weight at or below which a wall keeps the remainder instead of
        reflecting it again (:func:`~jaxincell._core.apply_particle_bc`). It is a fraction of
        what the species' source emits, and zero without one, which leaves every run that has
        no source exactly as it was."""
        if not self.sources:
            return 0.0
        floors = [jnp.broadcast_to(sp.source.min_weight * crossing_flux(sp.source) * self.domain.dt / sp.source.emit
                                   if sp.source is not None else 0.0, (sp.n,)) for sp in self.species]
        return jnp.concatenate(floors)

    def _empty_wall(self):
        zeros = jnp.zeros((len(self.species), 2))
        vectors = jnp.zeros((len(self.species), 2, 3))
        bins = self.impacts
        spectrum = None if bins is None else jnp.zeros((len(self.species), 2,
                                                        bins.energy_bins + 1, bins.angle_bins))
        return Wall(zeros, zeros, zeros, zeros, zeros, zeros, vectors, vectors, zeros, spectrum,
                    jnp.zeros(()))

    def _spectrum(self, wall, arrived, m, u_in):
        """Add this step's crossings to the energy-incidence accumulator.

        The energy is the kinetic energy of the segment on which the particle crossed and the
        incidence is :math:`\\theta = \\arctan(|v_t|/v_n)` from the normal, both taken from the
        velocity **before** the wall acted. Energies above ``Impacts.energy_max`` go to the
        overflow bin rather than into the last resolved one, so a range that was too small
        shows up instead of piling on the end."""
        bins = self.impacts
        if bins is None:
            return wall.spectrum
        energy = self._kinetic(m, u_in)
        width = bins.energy_max / bins.energy_bins
        level = jnp.clip(jnp.floor(energy / width).astype(jnp.int32), 0, bins.energy_bins)
        rows = []
        for i, ((start, n), sp) in enumerate(zip(self.blocks, self.species)):
            block, block_level = slice(start, start + n), level[start:start + n]
            sides = []
            for side, inward in ((0, 1.0), (1, -1.0)):
                normal = inward * -u_in[block, 0]          # towards that wall, positive on a crossing
                tangential = jnp.sqrt(jnp.sum(u_in[block, 1:] ** 2, axis=1))
                theta = jnp.arctan2(tangential, jnp.maximum(normal, 0.0))
                index = jnp.clip((theta / (jnp.pi / 2 / bins.angle_bins)).astype(jnp.int32),
                                 0, bins.angle_bins - 1)
                sides.append(jnp.zeros((bins.energy_bins + 1, bins.angle_bins))
                             .at[block_level, index].add(arrived[side, block]))
            rows.append(jnp.stack(sides))
        return wall.spectrum + jnp.stack(rows)

    def _kinetic(self, m, u):
        """Kinetic energy per unit weight of the carried ``u``, :math:`m|u|^2/(\\gamma+1)`.

        In a Newtonian run :math:`\\gamma` is one and this is :math:`mv^2/2`. In a relativistic
        one it is :math:`(\\gamma-1)mc^2` written so that it does not subtract two large numbers,
        which at :math:`v \\ll c` would leave nothing but round-off."""
        gamma = self._gamma(u)                 # 1.0, a scalar, in a Newtonian run
        return m * jnp.sum(u * u, axis=1) / (jnp.reshape(gamma, (-1,)) + 1 if jnp.ndim(gamma) else gamma + 1)

    def _at_impact(self, u, hits, fields, qm, dt):
        """The carried momentum of each particle at the moment it met a wall.

        A particle is pushed once over the whole step and then drifts, so one that meets a wall
        part-way through the drift is recorded, without this, in the state it reached by the end
        of it: it has taken the whole step's push where only part of it belongs before the
        impact. The value is wrong by :math:`O(\\Delta t)`, which is why it looks harmless, but
        the **derivative** is wrong by a fixed fraction that does not fall with the time step,
        because it is taken at a fixed step index instead of at the crossing -- the
        :math:`d\\tau/d\\theta` term of an event observable, missing. Running the same pusher
        backwards over the part of the step that follows the impact puts it back, and the
        control in ``test_gradients`` then converges: the error falls as the time step, from
        1.1e-3 to 1.3e-4 over a factor of eight, where before it sat at 6.2e-2 whatever the step.
        """
        fraction = hits[3]
        # a particle meets at most one wall in a step; elsewhere a half means no correction
        interval = (jnp.where(fraction[0] != 0.5, fraction[0], fraction[1]) - 0.5) * dt
        return self._accelerate(u, fields, qm, interval[:, None])

    def _record(self, wall, hits, m, u_in, u_out):
        """Add one step's impacts to the ledger. ``hits`` is what
        :func:`~jaxincell._core.apply_particle_bc` returned, and ``u_in`` and ``u_out`` are the
        carried momenta of the particles before the wall acted and after it has finished acting
        -- after a thermal wall's redraw, not before it -- so that the energy and momentum each
        wall received and returned follow without a second pass over the walls."""
        def per_species(per_particle):
            return jnp.stack([jnp.sum(per_particle[:, a:a + n], axis=1) for a, n in self.blocks])

        arrived, kept, truncated, _ = hits
        returned = arrived - kept
        return wall.replace(spectrum=self._spectrum(wall, arrived, m, u_in),
                            arrived=wall.arrived + per_species(arrived),
                            collected=wall.collected + per_species(kept),
                            truncated=wall.truncated + per_species(truncated),
                            energy_in=wall.energy_in + per_species(arrived * self._kinetic(m, u_in)),
                            energy_out=wall.energy_out + per_species(returned * self._kinetic(m, u_out)),
                            momentum=wall.momentum + jnp.stack(
                                [per_species(arrived * (m * u_in[:, k]) - returned * (m * u_out[:, k]))
                                 for k in range(3)], axis=-1))

    def _inject(self, key, x, u, w, qm, wall, E, B, rho):
        """Emit one step's worth of every source into the dead slots of its species.

        The particles enter at a quiet quadrature of times across the interval that ends at
        the position the leapfrog carries, and each is given the partial trajectory of
        :func:`~jaxincell._sources.inject` in the field at its entry plane, so that the
        deposit at the end of this step already sees it and the one full-step push the loop
        applies afterwards is the push it should have had."""
        if not self.sources:
            return x, u, w, qm, wall
        d = self.domain

        def push(velocity, fields, charge_over_mass, interval):
            """A partial Boris step on a velocity, whichever pusher the run uses."""
            return self._velocity(self._accelerate(self._momentum(velocity), fields,
                                                   charge_over_mass, interval))
        injected, energy, momentum = wall.injected, wall.energy_injected, wall.momentum_injected
        overflow = wall.overflow
        for i, (sp, block) in enumerate(zip(self.species, self.blocks)):
            if sp.source is None:
                continue
            key, k = random.split(key)
            plane = jnp.full((1, 3), (-1.0 if sp.source.side == "left" else 1.0) * d.length / 2)
            x, v, w, qm, weight, entering, spill = inject(k, sp.source, block, x, self._velocity(u), w, qm,
                                                          sp.charge_si / sp.mass, d.dt, d.length,
                                                          self._fields_at(plane, E, B, rho)[0], push)
            u = self._momentum(v)
            side = 0 if sp.source.side == "left" else 1
            carried = self._momentum(entering)      # the velocities as the pusher will carry them
            injected = injected.at[i, side].add(weight * sp.source.emit)
            energy = energy.at[i, side].add(weight * jnp.sum(self._kinetic(sp.mass, carried)))
            momentum = momentum.at[i, side].add(weight * sp.mass * jnp.sum(carried, axis=0))
            overflow = jnp.maximum(overflow, spill)
        return x, u, w, qm, wall.replace(injected=injected, energy_injected=energy,
                                         momentum_injected=momentum, overflow=overflow)

    def moments(self, x, v, w):
        """Density, particle flux and kinetic energy density of each species on the grid,
        ``(species, 3, cells)``, in :math:`\\mathrm{m^{-3}}`, :math:`\\mathrm{m^{-2}s^{-1}}`
        and :math:`\\mathrm{J/m^3}`.

        They use the deposit's own shape function, so a profile lines up with the charge
        density the field solver saw. ``run(moments=True)`` sums them over every step and
        stores the running sum, which is how a mean over a long window is had without a
        particle history: divide the difference of two stored sums by the number of steps
        between them. It costs three passes over the particles per species per step.
        """
        d = self.domain
        rows = []
        for (start, n), sp in zip(self.blocks, self.species):
            xs, vs, ws = x[start:start + n, 0], v[start:start + n], w[start:start + n]

            def density_of(amount, xs=xs):
                return deposit(xs, amount, d.grid[0], d.dx, d.cells, d.particle_bc)

            rows.append(jnp.stack([density_of(ws), density_of(ws * vs[:, 0]),
                                   density_of(0.5 * sp.mass * ws * jnp.sum(vs ** 2, axis=1))]))
        return jnp.stack(rows)

    def _accumulate(self, totals, x, v, w):
        return None if totals is None else totals + self.moments(x, v, w)

    def _current_closure(self, current_per_particle):
        """The constant of the continuity current for a closure that does not take a
        boundary value: the mean current the particles carry, which a periodic box has no
        wall to replace (:func:`~jaxincell._core.current_from_continuity`)."""
        return jnp.sum(current_per_particle) / self.domain.length

    def _collector_current(self, sigma_old, sigma_new, interval):
        """Conduction current at the collector face, :math:`\\mathrm{A/m^2}`, which closes the
        continuity current of a box whose other wall is an open plane.

        Ampere's law makes the total current uniform across a one-dimensional box, so
        :math:`J_{\\rm cond} + \\epsilon_0\\partial_t E_x` is the current in the external
        circuit. A floating collector is connected to nothing, so that total is zero and the
        conduction current at its face is :math:`-\\epsilon_0\\partial_t E_x = \\dot\\sigma_w`,
        the rate at which the electrode's charge changes -- collected and overlapping alike,
        since both are on the surface. Taken as a difference over the same interval the density
        change spans, that is exact for the discrete continuity relation rather than an
        approximation to it. Anchoring it at zero instead, as this did, leaves an internal
        transport measured from the source plane and not an absolute current.
        """
        return (sigma_new - sigma_old) / interval

    def _surface_charge(self, wall, x, w):
        """Charge on each wall, ``(left, right)``: what it has taken plus the clouds reaching
        past it. The two together with the charge on the grid are everything the box holds, which
        is what :func:`~jaxincell.charge_balance` checks against what went in and out."""
        return wall.charge([sp.charge_si for sp in self.species]) + self._overlap_charge(x, w)

    def _overlap_charge(self, x, w):
        """Charge of the parts of the live particle clouds that reach past each wall,
        ``(left, right)`` in :math:`\\mathrm{C/m^2}`.

        A cloud is one and a half cells wide, so it crosses the wall before its centre does,
        and :func:`~jaxincell._core.deposit` drops the part outside the grid. That part is not
        gone: it is charge the wall already sees, and it is reversible, because the particle
        may still turn round. Left out of both the volume and the surface, it makes the total
        charge of a particle crossing the wall swing between a half and one and a half of
        itself, and the field inside jump by half a particle at the crossing. It is taken from
        the same shape function and the same positions the deposit used, so the two partition
        each particle exactly.

        Only a wall the deposit actually drops charge at has any: a periodic wall wraps it to
        the far end and a reflective one clamps it into the boundary cell, and in both it is
        already on the grid. Counting it twice there would make the charge balance drift by a
        part in ten thousand with nothing wrong.
        """
        d = self.domain
        index, weights = s2_weights(x[:, 0], d.grid[0], d.dx)
        charges = jnp.concatenate([jnp.full((sp.n,), sp.charge_si) for sp in self.species])
        outside = (index < 0, index >= d.cells)
        return jnp.stack([jnp.sum(charges * w * jnp.sum(jnp.where(side, weights, 0.0), axis=1))
                          if code in (2, 4) else jnp.zeros(())
                          for side, code in zip(outside, d.particle_bc)])

    def _electrode_field(self, wall, overlap=0.0):
        """:math:`E_x` at the collector face, :math:`-\\sigma_w/\\epsilon_0`, from the charge it
        has collected and the part of the live clouds that reaches past it. It closes the Gauss
        solve of a box whose other wall is an open source plane; with a symmetry plane opposite
        and nothing crossing it, the same number comes out of global charge conservation instead.

        Only this closure takes the overlap. The other absorbing closures fix their constant from
        a potential difference rather than from a surface charge, and still lose the truncated
        part; in a ten-Debye-length sheath that is 0.4 % of the electron charge and 0.6 % of the
        ion charge, and 0.29 % of what the collector holds.
        """
        if self.domain.field_bc != (4, 2):
            return 0.0
        return -(wall.charge([sp.charge_si for sp in self.species])[1] + overlap) / epsilon_0

    def _electrode_overlap(self, sigma, wall):
        """The collector's share of ``sigma`` that is not collected charge."""
        return sigma[1] - wall.charge([sp.charge_si for sp in self.species])[1]

    def _advance_fields(self, E, B, J, dt_half, rho, wall, electric_first, overlap=0.0):
        """Half a step of the field equations.

        An electrostatic run solves :math:`\\partial_x E_x = \\rho/\\epsilon_0` for the field at
        the time of the density it is given, and leaves the transverse components and the
        plasma's own magnetic field alone: with :math:`\\mathbf E = -\\nabla\\phi` in one
        dimension there is nothing else to solve, and the light-wave time-step limit goes with
        it. An electromagnetic run takes the symmetric half step of Maxwell's equations."""
        d, bc = self.domain, self.domain.field_bc
        if self.solver.electrostatic:
            return E.at[:, 0].set(E_x_from_rho(rho, d.dx, bc, self._electrode_field(wall, overlap))), B
        E, B = half_step_fields(E, B, J, dt_half, d.dx, bc, electric_first)
        if self.solver.field_solver == "gauss" and not electric_first:
            E = E.at[:, 0].set(E_x_from_rho(rho, d.dx, bc))
        return E, B

    # -- initial state ------------------------------------------------------------------
    def initial_state(self, key):
        d = self.domain
        L, dx, dt = d.length, d.dx, d.dt
        xs, vs, qs, ms, ws = [], [], [], [], []
        for s in self.species:
            key, k_x, k_y, k_v = random.split(key, 4)
            if s.x is not None:
                x = jnp.asarray(s.x)
            else:
                if s.random_positions and not s.quiet:
                    x1 = random.uniform(k_x, (s.n,), minval=-L / 2, maxval=L / 2)
                else:
                    # `active` is static, so `spread` is a Python int: at active = 0 every
                    # slot is dead and its position is overwritten by the parking below, but
                    # the expression is still traced and a division by zero would put an
                    # infinity in the untaken branch of the weight's `where`, whose cotangent
                    # is NaN. An empty start is a source-driven run's natural beginning.
                    spread = max(s.active, 1)
                    x1 = -L / 2 + (jnp.arange(s.n) % spread + 0.5) * (L / spread)
                k = 2 * jnp.pi * s.perturbation_mode / L
                x1 = x1 + s.perturbation_amplitude * jnp.sin(k * x1)
                yz = (jnp.zeros((s.n, 2)) if s.quiet else
                      random.uniform(k_y, (s.n, 2), minval=-0.5, maxval=0.5) * jnp.array([d.length_y, d.length_z]))
                x = jnp.concatenate([x1[:, None], yz], axis=1)
            if s.v is not None:
                v = jnp.asarray(s.v)
            else:
                if s.quiet:
                    # With plus_minus the two beams are alternate particles, and the base-2
                    # van der Corput value is below one half exactly when the index is even,
                    # which would hand each beam one half of the Maxwellian. Drawing n/2
                    # quantiles and giving each to both beams makes them mirror images, each
                    # sampling the whole distribution.
                    m = (s.n + 1) // 2 if s.plus_minus else s.n
                    u = jnp.stack([jnp.asarray(_van_der_corput(m, b)) for b in (2, 3, 5)], axis=1)
                    u = jnp.repeat(u, 2, axis=0)[: s.n] if s.plus_minus else u
                    v = jnp.asarray(s.vth) * erfinv(2 * u - 1)
                else:
                    v = jnp.asarray(s.vth) / jnp.sqrt(2.0) * random.normal(k_v, (s.n, 3))
                v = v + jnp.asarray(s.drift)
                if s.plus_minus:
                    v = v.at[:, 0].multiply(jnp.where(jnp.arange(s.n) % 2 == 0, 1.0, -1.0))
            xs.append(x)
            vs.append(v)
            ws.append(jnp.where(jnp.arange(s.n) < s.active, s.density * L / max(s.active, 1), 0.0))
            qs.append(jnp.full((s.n,), s.charge_si))
            ms.append(jnp.full((s.n,), s.mass))
        x, v = jnp.concatenate(xs), jnp.concatenate(vs)
        w, q, m = jnp.concatenate(ws), jnp.concatenate(qs), jnp.concatenate(ms)
        u = self._momentum(v)
        v = self._velocity(u)
        qm = q / m
        box = (L, d.length_y, d.length_z)
        if self.sources or any(sp.active < sp.n for sp in self.species):
            # A species a source maintains gives `n` as a capacity, and a slot of no weight
            # is a dead one: park it beyond the wall with no charge-to-mass ratio, exactly
            # where a wall leaves a particle it has collected, so that the source finds it
            # free. `density=0` is then a physically empty start that fills from the source.
            dead = w <= 0
            x = x.at[:, 0].set(jnp.where(dead, -L / 2 - PARK * dx, x[:, 0]))
            qm = jnp.where(dead, 0.0, qm)
        wall = self._empty_wall()
        if self.solver.algorithm == "explicit":
            # The leapfrog carries the half-step position and reconstructs the
            # integer-time one as wrap(x - dt v / 2). A particle that meets a wall in
            # that first half step has to meet it the way every later step would, and
            # the initial field has to be built from the density the first step will
            # actually see; otherwise the discrete Gauss law starts out violated and
            # stays that way for the whole run.
            x_free = x + 0.5 * dt * v
            x, u_out, w, qm, hits = apply_particle_bc(
                x_free, u, w, qm, box, d.particle_bc, d.restitution, self._reflection(v), dx,
                self._weight_floor())
            if 3 in d.particle_bc:                # the same wall law as every later step, key and all
                key, k_wall = random.split(key)
                u_out = self._thermalise(k_wall, x_free, u_out)
            wall = self._record(wall, hits, m, u, u_out)
            u = u_out
            x_integer = wrap_positions(x - 0.5 * dt * self._velocity(u), w, box, d.particle_bc, dx)
        else:
            x_integer = x
        rho = self._smooth(deposit(x_integer[:, 0], q * w, d.grid[0], dx, d.cells, d.particle_bc))
        E = jnp.zeros((d.cells, 3)).at[:, 0].set(
            E_x_from_rho(rho, dx, d.field_bc, self._electrode_field(wall, self._overlap_charge(x_integer, w)[1])))
        B = jnp.zeros((d.cells, 3))
        return (State(E, B, x, u, w, qm, rho, self._surface_charge(wall, x_integer, w), key, jnp.zeros(()),
                      jnp.zeros((), jnp.int32), wall, None), (m, q))

    def _smooth(self, f):
        s = self.solver
        return smooth(f, s.filter_passes, s.filter_alpha, s.filter_strides, self.domain.field_bc)

    # -- one step ------------------------------------------------------------------------------
    def _sources(self, x, v, q, dt_half, wall_current, rho_old):
        """Charge density and current over a half step from the motion into positions ``x``
        with velocities ``v``. An electrostatic run has no use for the transverse currents
        and does not deposit them, which is two passes over the particles saved per half step;
        the longitudinal current is kept, since it is a diagnostic in its own right."""
        d = self.domain
        rho_new = self._smooth(deposit(x[:, 0], q, d.grid[0], d.dx, d.cells, d.particle_bc))
        J_x = current_from_continuity(rho_old, rho_new, dt_half, d.dx, wall_current, d.field_bc)
        if self.solver.electrostatic:
            return rho_new, jnp.stack([J_x, jnp.zeros_like(J_x), jnp.zeros_like(J_x)], axis=1)
        J_y = to_faces(self._smooth(deposit(x[:, 0], q * v[:, 1], d.grid[0], d.dx, d.cells, d.particle_bc)), d.field_bc)
        J_z = to_faces(self._smooth(deposit(x[:, 0], q * v[:, 2], d.grid[0], d.dx, d.cells, d.particle_bc)), d.field_bc)
        return rho_new, jnp.stack([J_x, J_y, J_z], axis=1)

    def _fields_at(self, x, E, B, rho):
        """E and B at the particles, as ``(N, 6)``.

        E is averaged from the faces to the centres, where B and the charge live, and both
        are gathered from there with the deposit's own spline, which makes the gather the
        transpose of the deposit: a particle exerts no force on itself, two exert equal and
        opposite forces on each other, and near a wall a particle feels the image the wall
        implies (:func:`~jaxincell._core.with_ghosts`). Gathering E straight from the faces
        does neither; in a periodic box a lone particle pushed itself with up to 8 % of its
        own field. ``rho`` gives the field at an absorbing left wall. External fields are
        added as given, continued unchanged beyond a wall."""
        d, bc = self.domain, self.domain.field_bc
        F = with_ghosts(jnp.concatenate([to_centres(E, *wall_faces_E(E, B, rho, d.dx, bc)), B], axis=1), bc,
                        jnp.asarray(PARITY))
        if self.external_E is not None or self.external_B is not None:
            E_ext = jnp.zeros_like(E) if self.external_E is None else jnp.asarray(self.external_E)
            B_ext = jnp.zeros_like(B) if self.external_B is None else jnp.asarray(self.external_B)
            F = F + with_ghosts(jnp.concatenate([to_centres(E_ext, E_ext[0], E_ext[-1]), B_ext], axis=1), bc)
        return gather(F, x[:, 0], d.grid[0], d.dx)

    def _accelerate(self, v, fields, qm, dt):
        """The Boris step in the fields ``(N, 6)`` gathered at the particles."""
        push = boris_relativistic if self.solver.relativistic else boris
        return push(v, fields[:, :3], fields[:, 3:], qm[:, None], dt)

    def _collide(self, key, x, v, w, qm, m, dt):
        d = self.domain
        names = [s.name for s in self.species]
        pairs = (self.collisions.pairs if self.collisions.pairs is not None
                 else tuple((a, b) for a in names for b in names if names.index(a) <= names.index(b)))
        pairs = tuple((names.index(a), names.index(b)) for a, b in pairs)
        ln_lambda = self.collisions.coulomb_log
        if ln_lambda is None:
            # The NRL logarithm is the electrons': the lightest negatively charged species, at its
            # density and at the temperature m v_th^2 / 2 of its largest thermal-speed component.
            # construction has checked that a negatively charged species exists; inside the run the
            # charges are traced, so the choice is made with array operations
            charges = [s.charge for s in self.species]
            charge = jnp.stack([jnp.asarray(q, float) for q in charges])
            mass = jnp.stack([jnp.asarray(s.mass, float) for s in self.species])
            e = jnp.argmin(jnp.where(charge < 0, mass, jnp.inf))
            density = jnp.stack([jnp.asarray(s.density, float) for s in self.species])[e]
            vth = jnp.stack([jnp.max(jnp.asarray(s.vth, float)) for s in self.species])[e]
            kT_ev = mass[e] * vth ** 2 / 2 / elementary_charge
            ln_lambda = jnp.where(jnp.any(charge < 0), coulomb_logarithm(density, kT_ev), jnp.nan)
        return collide(key, x, v, w, m, qm * m, self.blocks, pairs, ln_lambda, dt, d.dx, d.length, d.cells)

    def _explicit_step(self, st, extra):
        d, dt, dx, L = self.domain, self.domain.dt, self.domain.dx, self.domain.length
        box = (L, d.length_y, d.length_z)
        m, q = extra
        key, k_collide, k_wall, k_source = self._split_step_key(st.key)
        # What the sources supplied over the interval ending at the position the leapfrog
        # carries. They enter first, so the deposit below already counts them and no charge
        # appears between the two halves of the step.
        x_half, u, w, qm, wall = self._inject(k_source, st.x, st.u, st.w, st.qm, st.wall, st.E, st.B, st.rho)
        v = self._velocity(u)
        # First half step: sources from the motion x^n -> x^{n+1/2}. The density at x^n
        # is the one the previous step ended on (or the initial one), carried in the
        # state rather than deposited again from wrap(x^{n+1/2} - dt v/2), which is
        # the same positions, velocities and weights and so the same density.
        sigma_half = self._surface_charge(wall, x_half, w)
        closure = (self._collector_current(st.sigma[1], sigma_half[1], dt / 2) if d.field_bc == (4, 2)
                   else self._current_closure(q * w * v[:, 0]))
        rho_half, J1 = self._sources(x_half, v, q * w, dt / 2, closure, st.rho)
        E, B = self._advance_fields(st.E, st.B, J1, dt / 2, rho_half, wall, electric_first=True,
                                    overlap=self._electrode_overlap(sigma_half, wall))
        # push with the fields at t^{n+1/2}
        fields = self._fields_at(x_half, E, B, rho_half)
        u = self._accelerate(u, fields, qm, dt)
        u = self._collide_momenta(k_collide, x_half, u, w, qm, m, dt)
        v = self._velocity(u)
        x_free = x_half + dt * v
        incident = qm                     # the wall zeroes it for what it collects; the impact had it
        x_next_half, u_out, w, qm, hits = apply_particle_bc(x_free, u, w, qm, box, d.particle_bc, d.restitution,
                                                            self._reflection(v), dx, self._weight_floor(),
                                                            displacement=dt * v[:, 0])
        u_out = self._thermalise(k_wall, x_free, u_out)
        # The ledger records the state the particle arrived in, which is its state at the
        # crossing and not at the end of the step it overshot to. The two differ by the part
        # of the push that belongs after the impact, and the difference does not go away with
        # the time step: the derivative of the recorded energy is off by a fixed fraction,
        # 6 % in the control of test_gradients, because it is taken at a fixed step index
        # rather than at the wall. Undoing that part is the dtau/dtheta term of an event.
        wall = self._record(wall, hits, m, self._at_impact(u, hits, fields, incident, dt), u_out)
        u = u_out
        v = self._velocity(u)
        x_next = wrap_positions(x_next_half - 0.5 * dt * v, w, box, d.particle_bc, dx)
        # Second half step, x^{n+1/2} -> x^{n+1}, starting from the charge density the
        # first half already ended on. Depositing it again here would use the weights
        # that apply_particle_bc has just reduced, so the density at x^{n+1/2} would jump
        # by the charge collected at the wall with no current to account for it, and the
        # discrete Gauss law would drift by that much every step.
        sigma_next = self._surface_charge(wall, x_next, w)
        closure = (self._collector_current(sigma_half[1], sigma_next[1], dt / 2) if d.field_bc == (4, 2)
                   else self._current_closure(q * w * v[:, 0]))
        rho_next, J2 = self._sources(x_next, v, q * w, dt / 2, closure, rho_half)
        E, B = self._advance_fields(E, B, J2, dt / 2, rho_next, wall, electric_first=False,
                                    overlap=self._electrode_overlap(sigma_next, wall))
        totals = self._accumulate(st.moments, x_next, v, w)
        state = State(E, B, x_next_half, u, w, qm, rho_next, sigma_next, key, st.time + dt, st.steps + 1,
                      wall, totals)
        return state, (x_next, v, w, E, B, 0.5 * (J1 + J2), rho_next)

    def _implicit_step(self, st, extra):
        """Crank-Nicolson step solved by a fixed number of Picard iterations (docs/numerics/implicit.md).

        Each sub-step moves a particle on a straight line at :meth:`_mean_velocity`. Its current is the
        continuity current of the deposits at the two ends, which keeps the discrete Gauss law, and E_x at
        the particle is the discrete gradient of the potential the transposes of that current and of the
        deposit make of E_x, so the work equals the energy the current takes from the field (Kormann and
        Sonnendruecker 2021). E_y, E_z and B are gathered at the mid-point, with the transpose of that
        gather as their current (Chen, Chacon and Barnes 2011). The end and velocity of each sub-step are
        carried from one Picard iteration to the next; the step returns the last iteration's state."""
        d, dt, dx, L = self.domain, self.domain.dt, self.domain.dx, self.domain.length
        box, bc = (L, d.length_y, d.length_z), d.field_bc
        m, q = extra
        E, B, x, u, w, qm, rho, key = st.E, st.B, st.x, st.u, st.w, st.qm, st.rho, st.key
        n_sub = self.solver.substeps
        dtau = dt / n_sub
        # below this shift E_x is the potential's slope at the mid-point, off the quotient by (shift/dx)^2 = eps
        tiny = jnp.sqrt(jnp.finfo(x.dtype).eps) * dx

        # one thermal-wall key per sub-step, the same in every Picard iteration, so that
        # the wall re-emits a particle identically each time the orbit is recomputed
        key, k_collide, k_wall, _ = self._split_step_key(key)
        keys = random.split(k_wall, n_sub)

        def deposit_x(positions, amounts):
            return deposit(positions, amounts, d.grid[0], dx, d.cells, d.particle_bc)

        def substeps(E_half, B_half, orbits):
            # E_x as a potential at the particles: the transpose of the continuity current, then of the deposit
            to_current = partial(current_from_continuity, jnp.zeros_like(rho), dt=1.0, dx=dx, wall_current=0.0, bc=bc)
            phi = jax.linear_transpose(to_current, rho)(E_half[:, 0])[0]
            E_mean = jnp.mean(E_half[:, 0]) if bc[0] == 0 else 0.0     # the work of the periodic mean current

            def potential(positions):
                return dx * jax.linear_transpose(partial(deposit_x, positions), w)(phi)[0]

            def one(state, inputs):
                (x_end, v_bar), k_sub = inputs
                xs, us, ws, qms, rho_s, wall, x_start, phi_start, J_acc = state
                shift = dtau * v_bar[:, 0]
                x_mid = wrap_positions(x_start + 0.5 * dtau * v_bar, ws, box, d.particle_bc, dx)
                # the gather without the self-consistent E_x, which the discrete gradient below replaces
                gather = partial(self._fields_at, x_mid, B=B_half, rho=jnp.zeros_like(rho))
                fields, transpose = jax.vjp(gather, E_half.at[:, 0].set(0.0))
                phi_end = potential(x_end[:, 0])
                slope = jax.jvp(potential, (x_mid[:, 0],), (jnp.ones_like(shift),))[1]
                small = jnp.abs(shift) < tiny
                E_x = jnp.where(small, slope, (phi_end - phi_start) / jnp.where(small, tiny, shift)) + E_mean
                u_new = self._accelerate(us, fields.at[:, 0].add(E_x), qms, dtau)
                v_new = self._mean_velocity(us, u_new)
                x_free = xs + dtau * v_new
                x_new, u_bounced, w_new, qms, hits = apply_particle_bc(x_free, u_new, ws, qms, box, d.particle_bc,
                                                                       d.restitution, self._reflection(v_new), dx,
                                                                       self._weight_floor())
                u_bounced = self._thermalise(k_sub, x_free, u_bounced)
                wall = self._record(wall, hits, m, u_new, u_bounced)
                u_new = u_bounced
                rho_new = deposit_x(x_new[:, 0], q * w_new)
                J = transpose(jnp.concatenate([(q * ws)[:, None] * v_new, jnp.zeros_like(v_new)], axis=1))[0] / dx
                mean_current = jnp.sum(q * ws * v_new[:, 0]) / L
                J = J.at[:, 0].set(current_from_continuity(rho_s, rho_new, dtau, dx, mean_current, bc))
                return (x_new, u_new, w_new, qms, rho_new, wall, x_end, phi_end, J_acc + J / n_sub), (x_new, v_new)

            # the ledger starts from the state's, so that only the Picard iteration the step
            # accepts, the last one, adds its impacts to it
            init = (x, u, w, qm, rho, st.wall, x, potential(x[:, 0]), jnp.zeros((d.cells, 3)))
            state, orbits = lax.scan(one, init, (orbits, keys))
            return state[:6], state[-1], orbits

        def picard(state, _):
            E_new, orbits, _ = state
            E_half = 0.5 * (E + E_new)
            B_half = B - 0.5 * dt * curl_E(E_half, B, dx, bc)
            particles, J, orbits = substeps(E_half, B_half, orbits)
            return (E + dt * (c ** 2 * curl_B(B_half, E_half, dx, bc) - J / epsilon_0), orbits, (particles, J)), None

        v = self._velocity(u)      # the first guess: every particle streams freely at its present velocity
        free = jax.vmap(lambda s: wrap_positions(x + s * dtau * v, w, box, d.particle_bc, dx))
        orbits = (free(jnp.arange(1.0, n_sub + 1)), jnp.broadcast_to(v, (n_sub,) + v.shape))
        state, _ = lax.scan(picard, (E, orbits, ((x, u, w, qm, rho, st.wall), jnp.zeros_like(E))), None,
                            length=self.solver.picard_iterations)
        E_new, _, ((x, u, w, qm, rho_next, wall), J) = state
        B_new = B - dt * curl_E(0.5 * (E + E_new), B, dx, bc)
        u = self._collide_momenta(k_collide, x, u, w, qm, m, dt)
        v = self._velocity(u)
        return (State(E_new, B_new, x, u, w, qm, rho_next, st.sigma, key, st.time + dt, st.steps + 1, wall,
                      self._accumulate(st.moments, x, v, w)),
                (x, v, w, E_new, B_new, J, rho_next))

    # -- the run ---------------------------------------------------------------------------------
    def run(self, steps, seed=0, store_every=1, store_particles=True, moments=False, state=None):
        """Advance ``steps`` time steps and return an :class:`Output`.

        Args:
            steps: Number of time steps; a multiple of ``store_every``.
            seed: Integer seed of the random numbers (traced, so ``jax.vmap``
                over seeds gives an ensemble with one compilation).
            store_every: Keep every ``store_every``-th state in the output.
            store_particles: Keep the particle histories (the bulk of the memory).
            moments: Sum :meth:`moments` over every step and store the running sums, so
                that a mean profile over a long window needs no particle history. It
                costs three passes over the particles per species per step.
            state: A previous :class:`State`, normally ``Output.state``, to continue
                from. It carries the absolute time, the particles, the fields, the
                random key, the source remainders and the wall ledger, so a run split
                into chunks is the run taken whole. The simulation it is passed to must
                have the same grid, species layout and integrator; physical parameters
                may differ, which is how an experiment changes a control part-way through.
        """
        if store_every < 1 or steps % store_every:
            raise ValueError(f"steps ({steps}) must be a multiple of store_every ({store_every}), "
                             "which must be at least one")
        return _run(self, steps, seed, store_every, store_particles, moments, state)


@partial(jax.jit, static_argnames=("steps", "store_every", "store_particles", "moments"))
def _run(sim, steps, seed, store_every, store_particles, moments, state):
    key = random.PRNGKey(seed)
    carry0, extra = sim.initial_state(key)
    if state is not None:
        carry0 = state
    if moments:
        carry0 = carry0.replace(moments=jnp.zeros((len(sim.species), 3, sim.domain.cells))
                                if carry0.moments is None else carry0.moments)
    step = sim._explicit_step if sim.solver.algorithm == "explicit" else sim._implicit_step
    step = partial(step, extra=extra)
    # an output, overwritten before it is read
    placeholder = (carry0.x, carry0.u, carry0.w, carry0.E, carry0.B, jnp.zeros_like(carry0.E), carry0.rho)

    def advance(pair, _):
        return step(pair[0]), None

    def chunk(carry, _):
        # The output of the last step rides along with the state, so that the step is
        # traced once, not once for the first store_every - 1 steps and again for the last.
        (carry, (x, v, w, E, B, J, rho)), _ = lax.scan(advance, (carry, placeholder), None, length=store_every)
        if not store_particles:
            x = v = w = None
        return carry, (x, v, w, E, B, J, rho, carry.wall, carry.time, carry.steps, carry.sigma,
                       carry.moments)

    chunks = steps // store_every
    carry, (x, v, w, E, B, J, rho, wall, t, n, sigma, totals) = lax.scan(chunk, carry0, None, length=chunks)
    d = sim.domain
    m, q = extra
    return Output(t=t, steps=n, sigma=sigma, x=x, v=v, E=E, B=B, J=J, rho=rho, grid=d.grid, dx=d.dx, dt=d.dt,
                  length=d.length, charge=q, mass=m, weight=w, wall=wall, moments=totals,
                  species=jnp.concatenate([jnp.full((s.n,), i) for i, s in enumerate(sim.species)]),
                  state=carry, names=tuple(s.name for s in sim.species), counts=tuple(s.n for s in sim.species),
                  relativistic=sim.solver.relativistic, field_bc=d.field_bc)


def load_toml(path):
    """Build a :class:`Simulation` and the run settings from a TOML file.

    The file has ``[domain]``, ``[solver]`` and ``[[species]]`` tables whose keys
    are the constructor arguments, an optional ``[collisions]`` table, and a
    ``[run]`` table with ``steps``, ``seed`` and ``store_every``. Every species
    gives ``mass``, as a number in kilograms or as ``"electron"`` or ``"proton"``,
    optionally multiplied by ``mass_ratio``, and ``charge`` in units of e.

    Raises:
        ValueError: If a species has no ``mass`` or names an unknown one.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    species, named = [], {"electron": mass_electron, "proton": mass_proton}
    for s in raw.get("species", []):
        s = dict(s)
        mass = s.pop("mass", None)
        if mass is None or (isinstance(mass, str) and mass not in named):
            raise ValueError(f"species {s.get('name')!r} needs a mass: a number in kilograms, "
                             f"or \"electron\" or \"proton\", not {mass!r}")
        species.append(Species(mass=named.get(mass, mass) * s.pop("mass_ratio", 1.0), **s))
    collisions = Collisions(**raw["collisions"]) if "collisions" in raw else None
    sim = Simulation(Domain(**raw.get("domain", {})), species, Solver(**raw.get("solver", {})), collisions)
    return sim, raw.get("run", {})
