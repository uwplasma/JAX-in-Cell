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
from ._constants import elementary_charge, epsilon_0, mass_electron, mass_proton, speed_of_light as c
from ._core import (PARITY, E_x_from_rho, apply_particle_bc, boris, boris_relativistic, current_from_continuity,
                    curl_B, curl_E, deposit, gather, half_step_fields, smooth, to_centres, to_faces, wall_faces_E,
                    with_ghosts, wrap_positions)


def _enable_double_precision(environ):
    """Double precision unless the environment asks otherwise through JAX's own switch,
    ``JAX_ENABLE_X64=0``."""
    if "JAX_ENABLE_X64" not in environ:
        jax.config.update("jax_enable_x64", True)


_enable_double_precision(os.environ)

__all__ = ["Simulation", "Output", "load_toml", "quiet_start"]


@pytree_dataclass(static=("names", "counts", "relativistic", "field_bc"))
class Output:
    """Result of :meth:`Simulation.run`. Histories have the stored step as their
    first axis; ``t`` is the time of each stored state. ``charge`` and ``mass``
    are those of one physical particle, and ``weight`` is the history of the
    pseudo-particle weights, which fall as absorbing walls collect the particles.
    ``state`` is the final loop state and can be passed back to
    :meth:`Simulation.run` to continue."""
    t: object
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
    species: object
    state: object
    names: tuple
    counts: tuple
    relativistic: bool
    field_bc: tuple

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

    Args:
        n: Number of pseudo-particles.
        length: Box length; positions fill ``[-L/2, L/2]``.
        vth: Thermal speed per component, ``sqrt(2 k_B T / m)``.
        drift: Drift velocity per component.

    Returns:
        tuple: ``x`` and ``v``, both ``(n, 3)``, ready for
        ``species.replace(x=x, v=v)``.
    """
    x = np.zeros((n, 3))
    x[:, 0] = -length / 2 + (np.arange(n) + 0.5) * (length / n)
    u = np.stack([_van_der_corput(n, base) for base in (2, 3, 5)], axis=1)
    return x, np.asarray(erfinv(2 * u - 1)) * np.asarray(vth) + np.asarray(drift)


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

    def _check_implicit(self):
        """Refuse the two solver switches the Crank-Nicolson scheme would otherwise ignore.

        A filter keeps its energy conservation only if the same filter acts on the current
        and on the field gathered at the particles, as a transpose pair that respects the
        parity of each component at the walls; that pair is not implemented. The Gauss
        solve would overwrite E_x after the update that conserves energy, which is the
        property the scheme is there for. Silently skipping either switch, as the scheme
        once did, makes a run look filtered or electrostatic when it is neither."""
        s = self.solver
        if s.algorithm != "implicit":
            return
        if s.filter_passes:
            raise ValueError("the implicit scheme has no filter: conserving energy needs the same filter on the "
                             "current and on the gathered field, which is not implemented. Use filter_passes=0, "
                             "or algorithm='explicit'.")
        if s.field_solver == "gauss":
            raise ValueError("field_solver='gauss' would replace E_x after the energy-conserving update of the "
                             "implicit scheme; it is available with algorithm='explicit' only.")

    def _check_collisions(self):
        """The default Coulomb logarithm is taken from the lightest negatively charged species,
        so collisions without a given ``coulomb_log`` need one. Traced charges are skipped, as
        in :meth:`_check_courant`, since their sign is not known until the program runs."""
        if self.collisions is None or self.collisions.coulomb_log is not None:
            return
        charges = [s.charge for s in self.species]
        if any(isinstance(q, jax.core.Tracer) for q in charges):
            return
        if not any(q < 0 for q in charges):
            raise ValueError("Collisions() takes its default Coulomb logarithm from the negatively charged species, "
                             "and there is none: give Collisions(coulomb_log=...).")

    def _check_courant(self):
        """The explicit field update is unstable for ``c dt > dx``. Electrostatic
        runs never excite the transverse fields and are often stepped above that
        limit on purpose, so warn only when the particles carry the transverse
        velocity that would seed a light wave. Traced values are skipped, so the
        check happens when the object is first built and not on every rebuild."""
        def plain(v):
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        courant = self.domain.dt_over_dx_c
        if self.solver.algorithm != "explicit" or not plain(courant) or courant <= 1:
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

    def _thermalise(self, key, x, v):
        """Redraw the velocity of every particle that crossed a thermal wall from the
        half-Maxwellian flux of its species: the normal speed from the Rayleigh
        distribution :math:`\\sigma\\sqrt{-2\\ln U}`, pointing into the box, and the
        tangential components from the Maxwellian, with :math:`\\sigma = v_{th}/\\sqrt2`."""
        d = self.domain
        if 3 not in d.particle_bc:
            return v
        sigma = jnp.concatenate([jnp.broadcast_to(jnp.asarray(s.vth) / jnp.sqrt(2.0), (s.n, 3))
                                 for s in self.species])
        k_normal, k_tangential = random.split(key)
        u = random.uniform(k_normal, (v.shape[0],), minval=jnp.finfo(v.dtype).tiny)
        left = x[:, 0] < -d.length / 2
        new = (sigma * random.normal(k_tangential, v.shape)).at[:, 0].set(
            jnp.where(left, 1.0, -1.0) * sigma[:, 0] * jnp.sqrt(-2 * jnp.log(u)))
        hit = (left & (d.particle_bc[0] == 3)) | ((x[:, 0] > d.length / 2) & (d.particle_bc[1] == 3))
        return jnp.where(hit[:, None], new, v)

    @staticmethod
    def _split_step_key(key):
        """The keys of one step: the key carried to the next step, the collisions', and
        the thermal wall's. Every key is split once and then either split again or drawn
        from, never both, and there is no ``fold_in``: in JAX's threefry keys
        ``fold_in(k, 1)`` is ``split(k)[1]``, and ``split(k, 2)[i]`` is ``split(k, 5)[i]``,
        so keys derived from one parent in two ways coincide and two consumers draw
        the same numbers."""
        return random.split(key, 3)

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
                    x1 = -L / 2 + (jnp.arange(s.n) + 0.5) * (L / s.n)
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
            ws.append(jnp.full((s.n,), s.density * L / s.n))
            qs.append(jnp.full((s.n,), s.charge_si))
            ms.append(jnp.full((s.n,), s.mass))
        x, v = jnp.concatenate(xs), jnp.concatenate(vs)
        w, q, m = jnp.concatenate(ws), jnp.concatenate(qs), jnp.concatenate(ms)
        limit = 0.99 * c
        v = jnp.clip(v, -limit, limit)
        qm = q / m
        box = (L, d.length_y, d.length_z)
        if self.solver.algorithm == "explicit":
            # The leapfrog carries the half-step position and reconstructs the
            # integer-time one as wrap(x - dt v / 2). A particle that meets a wall in
            # that first half step has to meet it the way every later step would, and
            # the initial field has to be built from the density the first step will
            # actually see; otherwise the discrete Gauss law starts out violated and
            # stays that way for the whole run.
            x, v, w, qm = apply_particle_bc(x + 0.5 * dt * v, v, w, qm, box, d.particle_bc, d.restitution,
                                            self._reflection(v), dx)
            x_integer = wrap_positions(x - 0.5 * dt * v, w, box, d.particle_bc, dx)
        else:
            x_integer = x
        rho = self._smooth(deposit(x_integer[:, 0], q * w, d.grid[0], dx, d.cells, d.particle_bc))
        E = jnp.zeros((d.cells, 3)).at[:, 0].set(E_x_from_rho(rho, dx, d.field_bc))
        B = jnp.zeros((d.cells, 3))
        return (E, B, x, v, w, qm, rho, key), (m, q)

    def _smooth(self, f):
        s = self.solver
        return smooth(f, s.filter_passes, s.filter_alpha, s.filter_strides, self.domain.field_bc)

    # -- one step ------------------------------------------------------------------------------
    def _sources(self, x, v, q, dt_half, mean_current, rho_old):
        """Current over a half step from the motion into positions ``x`` with velocities ``v``."""
        d = self.domain
        rho_new = self._smooth(deposit(x[:, 0], q, d.grid[0], d.dx, d.cells, d.particle_bc))
        J_x = current_from_continuity(rho_old, rho_new, dt_half, d.dx, mean_current, d.field_bc)
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
        if self.collisions is None:
            return v
        d = self.domain
        names = [s.name for s in self.species]
        pairs = (self.collisions.pairs if self.collisions.pairs is not None
                 else tuple((a, b) for a in names for b in names if names.index(a) <= names.index(b)))
        pairs = tuple((names.index(a), names.index(b)) for a, b in pairs)
        ln_lambda = self.collisions.coulomb_log
        if ln_lambda is None:
            # The NRL logarithm is the electrons': the lightest negatively charged species, at its
            # density and at the temperature m v_th^2 / 2 of its largest thermal-speed component.
            charges = [s.charge for s in self.species]
            if not any(isinstance(q, jax.core.Tracer) for q in charges) and not any(q < 0 for q in charges):
                raise ValueError("Collisions(coulomb_log=None) takes the Coulomb logarithm from the electrons, "
                                 "but no species has a negative charge: give coulomb_log explicitly.")
            # traced inside the run, where the choice has to be made with array operations
            charge = jnp.stack([jnp.asarray(q, float) for q in charges])
            mass = jnp.stack([jnp.asarray(s.mass, float) for s in self.species])
            e = jnp.argmin(jnp.where(charge < 0, mass, jnp.inf))
            density = jnp.stack([jnp.asarray(s.density, float) for s in self.species])[e]
            vth = jnp.stack([jnp.max(jnp.asarray(s.vth, float)) for s in self.species])[e]
            kT_ev = mass[e] * vth ** 2 / 2 / elementary_charge
            ln_lambda = jnp.where(jnp.any(charge < 0), coulomb_logarithm(density, kT_ev), jnp.nan)
        return collide(key, x, v, w, m, qm * m, self.blocks, pairs, ln_lambda, dt, d.dx, d.length, d.cells)

    def _explicit_step(self, carry, extra):
        d, dt, dx, L = self.domain, self.domain.dt, self.domain.dx, self.domain.length
        box = (L, d.length_y, d.length_z)
        m, q = extra
        E, B, x_half, v, w, qm, rho_n, key = carry
        # First half step: sources from the motion x^n -> x^{n+1/2}. The density at x^n
        # is the one the previous step ended on (or the initial one), carried in the
        # state rather than deposited again from wrap(x^{n+1/2} - dt v/2), which is
        # the same positions, velocities and weights and so the same density.
        rho_half, J1 = self._sources(x_half, v, q * w, dt / 2, jnp.sum(q * w * v[:, 0]) / L, rho_n)
        E, B = half_step_fields(E, B, J1, dt / 2, dx, d.field_bc, electric_first=True)
        # push with the fields at t^{n+1/2}
        v = self._accelerate(v, self._fields_at(x_half, E, B, rho_half), qm, dt)
        key, k_collide, k_wall = self._split_step_key(key)
        v = self._collide(k_collide, x_half, v, w, qm, m, dt)
        x_free = x_half + dt * v
        x_next_half, v, w, qm = apply_particle_bc(x_free, v, w, qm, box, d.particle_bc, d.restitution,
                                                  self._reflection(v), dx)
        v = self._thermalise(k_wall, x_free, v)
        x_next = wrap_positions(x_next_half - 0.5 * dt * v, w, box, d.particle_bc, dx)
        # Second half step, x^{n+1/2} -> x^{n+1}, starting from the charge density the
        # first half already ended on. Depositing it again here would use the weights
        # that apply_particle_bc has just reduced, so the density at x^{n+1/2} would jump
        # by the charge collected at the wall with no current to account for it, and the
        # discrete Gauss law would drift by that much every step.
        rho_next, J2 = self._sources(x_next, v, q * w, dt / 2, jnp.sum(q * w * v[:, 0]) / L, rho_half)
        E, B = half_step_fields(E, B, J2, dt / 2, dx, d.field_bc, electric_first=False)
        if self.solver.field_solver == "gauss":
            E = E.at[:, 0].set(E_x_from_rho(rho_next, dx, d.field_bc))
        return (E, B, x_next_half, v, w, qm, rho_next, key), (x_next, v, w, E, B, 0.5 * (J1 + J2), rho_next)

    def _implicit_step(self, carry, extra):
        """Crank-Nicolson step solved by a fixed number of Picard iterations.

        Fields are advanced with their time-centred averages and the particles
        are sub-stepped in those averaged fields. The midpoint positions of every
        sub-step, at which the fields are gathered and the current is deposited,
        are carried across the iterations, so that at convergence gather and
        deposit use the same time-centred orbit (Chen, Chacon and Barnes 2011)."""
        d, dt, dx, L = self.domain, self.domain.dt, self.domain.dx, self.domain.length
        box = (L, d.length_y, d.length_z)
        m, q = extra
        E, B, x, v, w, qm, rho, key = carry
        n_sub = self.solver.substeps
        dtau = dt / n_sub

        # one thermal-wall key per sub-step, the same in every Picard iteration, so that
        # the wall re-emits a particle identically each time the orbit is recomputed
        key, k_collide, k_wall = self._split_step_key(key)
        keys = random.split(k_wall, n_sub)

        def substeps(E_half, B_half, x_mid_all):
            def one(state, inputs):
                x_mid, k_sub = inputs
                xs, vs, ws, qms, J_acc = state
                # the current is the transpose of the gather of E, the condition for energy conservation
                fields, transpose = jax.vjp(lambda E: self._fields_at(x_mid, E, B_half, rho), E_half)
                v_new = self._accelerate(vs, fields, qms, dtau)
                v_mid = 0.5 * (vs + v_new)
                x_free = xs + dtau * v_mid
                x_new, v_new, ws, qms = apply_particle_bc(x_free, v_new, ws, qms, box, d.particle_bc,
                                                          d.restitution, self._reflection(v_mid), dx)
                v_new = self._thermalise(k_sub, x_free, v_new)
                new_mid = wrap_positions(x_new - 0.5 * dtau * v_mid, ws, box, d.particle_bc, dx)
                carried = jnp.concatenate([(q * ws)[:, None] * v_mid, jnp.zeros_like(v_mid)], axis=1)
                J = transpose(carried)[0] / dx
                return (x_new, v_new, ws, qms, J_acc + J / n_sub), new_mid
            init = (x, v, w, qm, jnp.zeros((d.cells, 3)))
            (xs, vs, ws, qms, J_avg), new_mids = lax.scan(one, init, (x_mid_all, keys))
            return xs, vs, ws, qms, J_avg, new_mids

        def picard(state, _):
            E_new, x_mid_all = state
            E_half = 0.5 * (E + E_new)
            B_new = B - dt * curl_E(E_half, B, dx, d.field_bc)
            B_half = 0.5 * (B + B_new)
            _, _, _, _, J, x_mid_all = substeps(E_half, B_half, x_mid_all)
            E_next = E + dt * (c ** 2 * curl_B(B_half, E_half, dx, d.field_bc) - J / epsilon_0)
            return (E_next, x_mid_all), None

        x_mid0 = jnp.broadcast_to(wrap_positions(x + 0.5 * dtau * v, w, box, d.particle_bc, dx), (n_sub,) + x.shape)
        (E_new, x_mid_all), _ = lax.scan(picard, (E, x_mid0), None, length=self.solver.picard_iterations)
        E_half = 0.5 * (E + E_new)
        B_new = B - dt * curl_E(E_half, B, dx, d.field_bc)
        x, v, w, qm, J, _ = substeps(E_half, 0.5 * (B + B_new), x_mid_all)
        v = self._collide(k_collide, x, v, w, qm, m, dt)
        rho_next = deposit(x[:, 0], q * w, d.grid[0], dx, d.cells, d.particle_bc)
        return (E_new, B_new, x, v, w, qm, rho_next, key), (x, v, w, E_new, B_new, J, rho_next)

    # -- the run ---------------------------------------------------------------------------------
    def run(self, steps, seed=0, store_every=1, store_particles=True, state=None):
        """Advance ``steps`` time steps and return an :class:`Output`.

        Args:
            steps: Number of time steps; a multiple of ``store_every``.
            seed: Integer seed of the random numbers (traced, so ``jax.vmap``
                over seeds gives an ensemble with one compilation).
            store_every: Keep every ``store_every``-th state in the output.
            store_particles: Keep the particle histories (the bulk of the memory).
            state: A previous ``Output.state`` to continue from.
        """
        if store_every < 1 or steps % store_every:
            raise ValueError(f"steps ({steps}) must be a multiple of store_every ({store_every}), "
                             "which must be at least one")
        return _run(self, steps, seed, store_every, store_particles, state)


@partial(jax.jit, static_argnames=("steps", "store_every", "store_particles"))
def _run(sim, steps, seed, store_every, store_particles, state):
    key = random.PRNGKey(seed)
    carry0, extra = sim.initial_state(key)
    if state is not None:
        carry0 = state
    step = sim._explicit_step if sim.solver.algorithm == "explicit" else sim._implicit_step
    step = partial(step, extra=extra)
    E, B, x, v, w, _, rho, _ = carry0
    placeholder = (x, v, w, E, B, jnp.zeros_like(E), rho)     # an output, overwritten before it is read

    def advance(pair, _):
        return step(pair[0]), None

    def chunk(carry, _):
        # The output of the last step rides along with the state, so that the step is
        # traced once, not once for the first store_every - 1 steps and again for the last.
        (carry, (x, v, w, E, B, J, rho)), _ = lax.scan(advance, (carry, placeholder), None, length=store_every)
        if not store_particles:
            x = v = w = None
        return carry, (x, v, w, E, B, J, rho)

    carry, (x, v, w, E, B, J, rho) = lax.scan(chunk, carry0, None, length=steps // store_every)
    d = sim.domain
    m, q = extra
    kept = (jnp.arange(steps // store_every) + 1) * store_every
    return Output(t=kept * d.dt, x=x, v=v, E=E, B=B, J=J, rho=rho, grid=d.grid, dx=d.dx, dt=d.dt,
                  length=d.length, charge=q, mass=m, weight=w,
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
