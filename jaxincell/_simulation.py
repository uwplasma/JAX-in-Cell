"""The simulation: initial state, the time loop, and the output."""
from __future__ import annotations

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
from ._core import (E_x_from_rho, apply_particle_bc, boris, boris_relativistic, current_from_continuity,
                    curl_B, curl_E, deposit, gather, half_step_fields, smooth, to_faces, wrap_positions)

jax.config.update("jax_enable_x64", True)

__all__ = ["Simulation", "Output", "load_toml", "quiet_start"]


@pytree_dataclass(static=("names", "counts", "relativistic"))
class Output:
    """Result of :meth:`Simulation.run`. Histories have the stored step as their
    first axis; ``t`` is the time of each stored state. ``state`` is the final
    loop state and can be passed back to :meth:`Simulation.run` to continue."""
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
        assert len(self.species) > 0, "at least one species is needed"
        names = [s.name for s in self.species]
        assert len(set(names)) == len(names), "species names must be distinct"
        self._check_courant()

    def _check_courant(self):
        """The explicit field update is unstable for ``c dt > dx``. Electrostatic
        runs never excite the transverse fields and are often stepped above that
        limit on purpose, so warn only when the particles carry the transverse
        velocity that would seed a light wave. Traced values are skipped, so the
        check happens when the object is first built and not on every rebuild."""
        plain = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        courant = self.domain.dt_over_dx_c
        if self.solver.algorithm != "explicit" or not plain(courant) or courant <= 1:
            return
        if any(s.v is not None or any(plain(u) and u != 0 for u in tuple(s.vth[1:]) + tuple(s.drift[1:]))
               for s in self.species):
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
        return jnp.sqrt(s.density * s.charge_si ** 2 / (epsilon_0 * s.mass))

    def debye_length(self, name=None):
        s = self.species[0] if name is None else self.species[[t.name for t in self.species].index(name)]
        return jnp.max(jnp.asarray(s.vth)) / (jnp.sqrt(2.0) * self.plasma_frequency(name))

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
            w = s.density * L / s.n
            xs.append(x); vs.append(v); ws.append(jnp.full((s.n,), w))
            qs.append(jnp.full((s.n,), s.charge_si)); ms.append(jnp.full((s.n,), s.mass))
        x, v = jnp.concatenate(xs), jnp.concatenate(vs)
        w, q, m = jnp.concatenate(ws), jnp.concatenate(qs), jnp.concatenate(ms)
        limit = 0.99 * c
        v = jnp.clip(v, -limit, limit)
        q_pseudo, qm = q * w, q / m
        rho = self._smooth(deposit(x[:, 0], q_pseudo, d.grid[0], dx, d.cells, d.particle_bc))
        E = jnp.zeros((d.cells, 3)).at[:, 0].set(E_x_from_rho(rho, dx, d.field_bc))
        B = jnp.zeros((d.cells, 3))
        if self.solver.algorithm == "explicit":
            x = wrap_positions(x + 0.5 * dt * v, (L, d.length_y, d.length_z), d.particle_bc, dx)
        return (E, B, x, v, q_pseudo, qm, key), (w, m, q)

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

    def _push(self, x, v, q, qm, m_pseudo, E, B, dt):
        d = self.domain
        E_p = gather(E if self.external_E is None else E + self.external_E, x[:, 0], d.grid[0] + d.dx / 2, d.dx, d.field_bc)
        B_p = gather(B if self.external_B is None else B + self.external_B, x[:, 0], d.grid[0], d.dx, d.field_bc)
        if self.solver.relativistic:
            return boris_relativistic(v, E_p, B_p, q[:, None], m_pseudo[:, None], dt)
        return boris(v, E_p, B_p, qm[:, None], dt)

    def _collide(self, key, x, v, q, w, m, dt):
        if self.collisions is None:
            return v
        d = self.domain
        names = [s.name for s in self.species]
        pairs = (self.collisions.pairs if self.collisions.pairs is not None
                 else tuple((a, b) for a in names for b in names if names.index(a) <= names.index(b)))
        pairs = tuple((names.index(a), names.index(b)) for a, b in pairs)
        if self.collisions.coulomb_log is None:
            e = self.species[0]
            kT_ev = e.mass * jnp.max(jnp.asarray(e.vth)) ** 2 / 2 / elementary_charge
            ln_lambda = coulomb_logarithm(e.density, kT_ev)
        else:
            ln_lambda = self.collisions.coulomb_log
        return collide(key, x, v, w, m, q / w, self.blocks, pairs, ln_lambda, dt, d.dx, d.length, d.cells)

    def _explicit_step(self, carry, extra):
        d, dt, dx, L = self.domain, self.domain.dt, self.domain.dx, self.domain.length
        box = (L, d.length_y, d.length_z)
        w, m, _ = extra
        E, B, x_half, v, q, qm, key = carry
        # first half step: sources from the motion x^n -> x^{n+1/2}
        x_n = wrap_positions(x_half - 0.5 * dt * v, box, d.particle_bc, dx)
        rho_n = self._smooth(deposit(x_n[:, 0], q, d.grid[0], dx, d.cells, d.particle_bc))
        _, J1 = self._sources(x_half, v, q, dt / 2, jnp.sum(q * v[:, 0]) / L, rho_n)
        E, B = half_step_fields(E, B, J1, dt / 2, dx, d.field_bc, electric_first=True)
        # push with the fields at t^{n+1/2}
        v = self._push(x_half, v, q, qm, m * w, E, B, dt)
        key, k_c = random.split(key)
        v = self._collide(k_c, x_half, v, q, w, m, dt)
        x_next_half = x_half + dt * v
        x_next_half, v, q, qm = apply_particle_bc(x_next_half, v, q, qm, box, d.particle_bc, d.restitution, dx)
        x_next = wrap_positions(x_next_half - 0.5 * dt * v, box, d.particle_bc, dx)
        # second half step: sources from x^{n+1/2} -> x^{n+1}
        rho_half = self._smooth(deposit(x_half[:, 0], q, d.grid[0], dx, d.cells, d.particle_bc))
        rho_next, J2 = self._sources(x_next, v, q, dt / 2, jnp.sum(q * v[:, 0]) / L, rho_half)
        E, B = half_step_fields(E, B, J2, dt / 2, dx, d.field_bc, electric_first=False)
        if self.solver.field_solver == "gauss":
            E = E.at[:, 0].set(E_x_from_rho(rho_next, dx, d.field_bc))
        return (E, B, x_next_half, v, q, qm, key), (x_next, v, E, B, 0.5 * (J1 + J2), rho_next)

    def _implicit_step(self, carry, extra):
        """Crank-Nicolson step solved by a fixed number of Picard iterations.

        Fields are advanced with their time-centred averages and the particles
        are sub-stepped in those averaged fields. The midpoint positions of every
        sub-step, at which the fields are gathered and the current is deposited,
        are carried across the iterations, so that at convergence gather and
        deposit use the same time-centred orbit (Chen, Chacon and Barnes 2011)."""
        d, dt, dx, L = self.domain, self.domain.dt, self.domain.dx, self.domain.length
        box = (L, d.length_y, d.length_z)
        w, m, _ = extra
        E, B, x, v, q, qm, key = carry
        n_sub = self.solver.substeps
        dtau = dt / n_sub

        def substeps(E_half, B_half, x_mid_all):
            def one(state, x_mid):
                xs, vs, qs, qms, J_acc, mids = state
                v_new = self._push(x_mid, vs, qs, qms, m * w, E_half, B_half, dtau)
                v_mid = 0.5 * (vs + v_new)
                x_new, v_new, qs, qms = apply_particle_bc(xs + dtau * v_mid, v_new, qs, qms, box,
                                                          d.particle_bc, d.restitution, dx)
                new_mid = wrap_positions(x_new - 0.5 * dtau * v_mid, box, d.particle_bc, dx)
                J = jnp.stack([deposit(x_mid[:, 0], qs * v_mid[:, i], d.grid[0] + dx / 2, dx, d.cells, d.particle_bc)
                               for i in range(3)], axis=1)
                return (x_new, v_new, qs, qms, J_acc + J / n_sub, mids), new_mid
            init = (x, v, q, qm, jnp.zeros((d.cells, 3)), None)
            (xs, vs, qs, qms, J_avg, _), new_mids = lax.scan(one, init, x_mid_all)
            return xs, vs, qs, qms, J_avg, new_mids

        def picard(state, _):
            E_new, x_mid_all = state
            E_half = 0.5 * (E + E_new)
            B_new = B - dt * curl_E(E_half, B, dx, d.field_bc)
            B_half = 0.5 * (B + B_new)
            _, _, _, _, J, x_mid_all = substeps(E_half, B_half, x_mid_all)
            E_next = E + dt * (c ** 2 * curl_B(B_half, E_half, dx, d.field_bc) - J / epsilon_0)
            return (E_next, x_mid_all), None

        x_mid0 = jnp.broadcast_to(wrap_positions(x + 0.5 * dtau * v, box, d.particle_bc, dx), (n_sub,) + x.shape)
        (E_new, x_mid_all), _ = lax.scan(picard, (E, x_mid0), None, length=self.solver.picard_iterations)
        E_half = 0.5 * (E + E_new)
        B_new = B - dt * curl_E(E_half, B, dx, d.field_bc)
        x, v, q, qm, J, _ = substeps(E_half, 0.5 * (B + B_new), x_mid_all)
        key, k_c = random.split(key)
        v = self._collide(k_c, x, v, q, w, m, dt)
        rho = deposit(x[:, 0], q, d.grid[0], dx, d.cells, d.particle_bc)
        return (E_new, B_new, x, v, q, qm, key), (x, v, E_new, B_new, J, rho)

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
        assert steps % store_every == 0, "steps must be a multiple of store_every"
        return _run(self, steps, seed, store_every, store_particles, state)


@partial(jax.jit, static_argnames=("steps", "store_every", "store_particles"))
def _run(sim, steps, seed, store_every, store_particles, state):
    key = random.PRNGKey(seed)
    carry0, extra = sim.initial_state(key)
    if state is not None:
        carry0 = state
    step = sim._explicit_step if sim.solver.algorithm == "explicit" else sim._implicit_step
    step = partial(step, extra=extra)

    def chunk(carry, start):
        carry, _ = lax.scan(lambda c, i: (step(c)[0], None), carry, None, length=store_every - 1)
        carry, out = step(carry)
        x, v, E, B, J, rho = out
        if not store_particles:
            x = v = None
        return carry, (x, v, E, B, J, rho)

    carry, (x, v, E, B, J, rho) = lax.scan(chunk, carry0, jnp.arange(steps // store_every))
    d = sim.domain
    w, m, q = extra
    kept = (jnp.arange(steps // store_every) + 1) * store_every
    return Output(t=kept * d.dt, x=x, v=v, E=E, B=B, J=J, rho=rho, grid=d.grid, dx=d.dx, dt=d.dt,
                  length=d.length, charge=carry[4], mass=m * w, weight=w,
                  species=jnp.concatenate([jnp.full((s.n,), i) for i, s in enumerate(sim.species)]),
                  state=carry, names=tuple(s.name for s in sim.species), counts=tuple(s.n for s in sim.species),
                  relativistic=sim.solver.relativistic)


def load_toml(path):
    """Build a :class:`Simulation` and the run settings from a TOML file.

    The file has ``[domain]``, ``[solver]`` and ``[[species]]`` tables whose keys
    are the constructor arguments, an optional ``[collisions]`` table, and a
    ``[run]`` table with ``steps``, ``seed`` and ``store_every``. A species may
    give ``mass`` as a number in kilograms or as ``"electron"``/``"proton"``,
    optionally multiplied by ``mass_ratio``, and ``charge`` in units of e.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    species = []
    for s in raw.get("species", []):
        s = dict(s)
        mass = s.pop("mass", "proton")
        mass = {"electron": mass_electron, "proton": mass_proton}.get(mass, mass) * s.pop("mass_ratio", 1.0)
        species.append(Species(mass=mass, **s))
    collisions = Collisions(**raw["collisions"]) if "collisions" in raw else None
    sim = Simulation(Domain(**raw.get("domain", {})), species, Solver(**raw.get("solver", {})), collisions)
    return sim, raw.get("run", {})
