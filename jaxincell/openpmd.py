"""openPMD export of a :class:`jaxincell.Output` (optional dependency ``openpmd-api``).

One group-based series, one iteration per exported stored step ``s`` (``time = out.t[s]``,
``dt = out.dt``, ``timeUnitSI = 1``); float64 SI data with ``unitSI = 1`` and ``unitDimension``.

Layout:
    Meshes ``E``, ``B``, ``J`` (components ``x``/``y``/``z``) and the scalar ``rho`` on the 1D
    Cartesian grid: ``axis_labels=["x"]``, ``grid_spacing=[dx]``, ``grid_global_offset=[-length/2]``;
    ``E`` and ``J`` sit on cell faces (component ``position=[0.5]``), ``B`` and ``rho`` on centres
    (``position=[0.0]``). Particles: one species per ``out.names`` with vector records
    ``position``, ``positionOffset`` (zero) and ``momentum`` (``m * gamma * v``, gamma from ``v``)
    plus scalar ``weighting`` (pseudo-particle weight), ``charge`` and ``mass`` (physical, i.e.
    ``out.charge / out.weight`` and ``out.mass / out.weight``); skipped when ``out.x is None``.

Example:
    >>> write_openpmd(sim.run(steps=1000, store_every=10), "run.json", every=5)  # or .h5 / .bp
"""
import os

import numpy as np

from ._constants import speed_of_light

try:
    from .version import __version__
except ImportError:  # source tree without the generated version file
    __version__ = "unknown"

__all__ = ["write_openpmd"]
_DIMS = {"E": dict(L=1, M=1, T=-3, I=-1), "B": dict(M=1, T=-2, I=-1), "J": dict(L=-2, I=1),
         "rho": dict(L=-3, T=1, I=1), "position": dict(L=1), "positionOffset": dict(L=1),
         "momentum": dict(L=1, M=1, T=-1), "weighting": {}, "charge": dict(T=1, I=1), "mass": dict(M=1)}
_FACES = ("E", "J")  # records on cell faces (in-cell position 0.5); B and rho are on centres (0.0)


def _describe(io, record, name, particle=False):
    record.unit_dimension = {getattr(io.Unit_Dimension, k): float(v) for k, v in _DIMS[name].items()}
    if particle:
        record.set_attribute("macroWeighted", np.uint32(name == "weighting"))
        record.set_attribute("weightingPower", 0.0 if name.startswith("position") else 1.0)


def _store(io, component, data, keep):
    # np.array copies, which matters twice: a JAX array converts to a read-only
    # buffer that store_chunk refuses, and the buffer has to stay alive and
    # writeable until the series is flushed, which is what ``keep`` is for.
    data = np.array(data, dtype=np.float64, order="C")
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)
    component.unit_SI = 1.0
    keep.append(data)  # the buffer must outlive store_chunk until the series is flushed


def _write_meshes(io, it, out, s, keep):
    for name in ("E", "B", "J", "rho"):
        data, mesh = np.asarray(getattr(out, name)[s], dtype=np.float64), it.meshes[name]
        mesh.geometry, mesh.axis_labels, mesh.grid_unit_SI = io.Geometry.cartesian, ["x"], 1.0
        mesh.grid_spacing, mesh.grid_global_offset = [float(out.dx)], [-0.5 * float(out.length)]
        _describe(io, mesh, name)
        for label, column in zip("xyz", data.T) if data.ndim == 2 else [(io.Record_Component.SCALAR, data)]:
            mesh[label].position = [0.5 if name in _FACES else 0.0]
            _store(io, mesh[label], column, keep)


def _write_particles(io, it, out, s, keep):
    x, v = (np.asarray(a[s], dtype=np.float64) for a in (out.x, out.v))
    w, species = np.asarray(out.weight, dtype=np.float64), np.asarray(out.species)
    q, m = (np.divide(np.asarray(a, dtype=np.float64), w, out=np.zeros_like(w), where=w != 0)
            for a in (out.charge, out.mass))
    gamma = 1.0 / np.sqrt(np.clip(1.0 - np.sum(v ** 2, axis=1) / speed_of_light ** 2, 1e-15, None))
    p, zero = (m * gamma)[:, None] * v, np.zeros_like(x)
    for i, name in enumerate(out.names):
        sel, sp = species == i, it.particles[name]
        for rec, data in (("position", x[sel]), ("positionOffset", zero[sel]), ("momentum", p[sel])):
            _describe(io, sp[rec], rec, particle=True)
            for label, column in zip("xyz", data.T):
                _store(io, sp[rec][label], column, keep)
        for rec, data in (("weighting", w[sel]), ("charge", q[sel]), ("mass", m[sel])):
            _describe(io, sp[rec], rec, particle=True)
            _store(io, sp[rec][io.Record_Component.SCALAR], data, keep)


def write_openpmd(out, path, every=1, meshes=True, particles=True):
    """Write ``out`` to the openPMD series ``path`` and return the path.

    Args:
        out: :class:`jaxincell.Output` (or any object with the same attributes).
        path: Output file; ``.json`` (default, appended if no extension), ``.h5`` or ``.bp`` pick the backend.
        every: Export every ``every``-th stored step (the iteration index is the stored-step index).
        meshes: Write the ``E``, ``B``, ``J`` and ``rho`` meshes.
        particles: Write the per-species particle records (skipped when ``out.x is None``).

    Raises:
        ImportError: If the optional dependency ``openpmd-api`` is missing.
    """
    try:
        import openpmd_api as io
    except ImportError as exc:
        raise ImportError("write_openpmd needs the optional openpmd-api: pip install openpmd-api") from exc
    root, ext = os.path.splitext(os.fspath(path))
    path = root + (ext or ".json")
    series = io.Series(path, io.Access.create)
    series.set_software("JAX-in-Cell", __version__)
    for s in range(0, len(out.t), every):
        it, keep = series.iterations[s], []
        it.time, it.dt, it.time_unit_SI = float(out.t[s]), float(out.dt), 1.0
        if meshes:
            _write_meshes(io, it, out, s, keep)
        if particles and out.x is not None:
            _write_particles(io, it, out, s, keep)
        series.flush()
    series.close()
    return path
