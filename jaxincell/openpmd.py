"""openPMD export of a :class:`jaxincell.Output` (optional dependency ``openpmd-api``).

One group-based series, one iteration per exported stored step ``s`` (``time = out.t[s]``,
``dt = out.dt``, ``timeUnitSI = 1``); float64 SI data with ``unitSI = 1`` and ``unitDimension``.

Layout:
    Meshes ``E``, ``B``, ``J`` (components ``x``/``y``/``z``) and the scalar ``rho`` on the 1D
    Cartesian grid: ``axis_labels=["x"]``, ``grid_spacing=[dx]``, ``grid_unit_SI=1``.

    The standard places a component at
    :math:`x_i = (\\mathrm{gridGlobalOffset} + (i + \\mathrm{position})\\,\\mathrm{gridSpacing})
    \\times\\mathrm{gridUnitSI}`, with ``position`` in :math:`[0,1)` measured from the **lower
    corner** of the cell. So a cell centre is ``position=0.5``, not ``0.0``; ``0.0`` is the left
    face. ``B`` and ``rho`` live on the centres and are written with ``grid_global_offset =
    [-length/2]`` and ``position=[0.5]``. ``E`` and ``J`` live on the faces this code stores,
    which are the **right** face of each cell, :math:`-L/2 + (i+1)\\Delta x`; that is
    ``position=1.0``, which the standard does not allow, so their offset is shifted a cell
    instead: ``grid_global_offset = [-length/2 + dx]`` with ``position=[0.0]``. Both records then
    read back at the coordinates the code puts them at, by the formula above and nothing else.

    (This was the wrong way round: centres at ``0.0`` and faces at ``0.5``, which put every
    quantity half a cell from where it belongs and the two of them a whole cell apart. Note that
    WarpX's default openPMD output writes ``0.5`` on every component because it cell-centres
    before writing, so it is not a reference for staggering; PIConGPU is.)

    Particles: one species per ``out.names`` with the vector records
    ``position`` and ``momentum`` and the scalar ``weighting`` per particle, and ``positionOffset``
    (zero), ``charge`` and ``mass`` (of one physical particle) as constant records; skipped when
    ``out.x is None``. The momentum is the one the pusher advances: ``m * gamma * v`` for a
    relativistic run and ``m * v`` otherwise. ``weighting`` is ``out.weight * area``, with ``area``
    recorded as the attribute ``transverseArea`` (m^2), since a 1D weight counts particles per unit
    area of the y-z plane.
"""
import os

import numpy as np

from . import __version__
from ._config import speed_of_light

__all__ = ["write_openpmd"]
_DIMS = {"E": dict(L=1, M=1, T=-3, I=-1), "B": dict(M=1, T=-2, I=-1), "J": dict(L=-2, I=1),
         "rho": dict(L=-3, T=1, I=1), "position": dict(L=1), "positionOffset": dict(L=1),
         "momentum": dict(L=1, M=1, T=-1), "weighting": {}, "charge": dict(T=1, I=1), "mass": dict(M=1)}
_FACES = ("E", "J")  # on the right face of each cell; B and rho are on the centres


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
    keep.append(data)


def _constant(io, component, value, count):
    """A record component with the same value for all ``count`` particles."""
    component.reset_dataset(io.Dataset(np.dtype(np.float64), [count]))
    component.make_constant(float(value))
    component.unit_SI = 1.0


def _write_meshes(io, it, out, s, keep):
    for name in ("E", "B", "J", "rho"):
        data, mesh = np.asarray(getattr(out, name)[s], dtype=np.float64), it.meshes[name]
        mesh.geometry, mesh.axis_labels, mesh.grid_unit_SI = io.Geometry.cartesian, ["x"], 1.0
        # x_i = (offset + (i + position) * spacing); position is in [0, 1) from the lower corner,
        # so a centre is 0.5 and the right face of cell i is the lower corner of cell i + 1
        faces = name in _FACES
        mesh.grid_spacing = [float(out.dx)]
        mesh.grid_global_offset = [-0.5 * float(out.length) + (float(out.dx) if faces else 0.0)]
        _describe(io, mesh, name)
        for label, column in zip("xyz", data.T) if data.ndim == 2 else [(io.Record_Component.SCALAR, data)]:
            mesh[label].position = [0.0 if faces else 0.5]
            _store(io, mesh[label], column, keep)


def _momentum(out, s):
    """Momentum of one physical particle, as the pusher defines it."""
    v, m = np.asarray(out.v[s], dtype=np.float64), np.asarray(out.mass, dtype=np.float64)
    gamma = 1.0 / np.sqrt(1.0 - np.sum(v ** 2, axis=1) / speed_of_light ** 2) if out.relativistic else 1.0
    return (m * gamma)[:, None] * v


def _write_particles(io, it, out, s, area, keep):
    x, p = np.asarray(out.x[s], dtype=np.float64), _momentum(out, s)
    w = np.asarray(out.weight[s], dtype=np.float64) * area
    start = 0
    for name, count in zip(out.names, out.counts):
        block, first, start = slice(start, start + count), start, start + count
        sp = it.particles[name]
        for rec, data in (("position", x[block]), ("momentum", p[block])):
            _describe(io, sp[rec], rec, particle=True)
            for label, column in zip("xyz", data.T):
                _store(io, sp[rec][label], column, keep)
        _describe(io, sp["positionOffset"], "positionOffset", particle=True)
        for label in "xyz":
            _constant(io, sp["positionOffset"][label], 0.0, count)
        _describe(io, sp["weighting"], "weighting", particle=True)
        sp["weighting"].set_attribute("transverseArea", float(area))
        _store(io, sp["weighting"][io.Record_Component.SCALAR], w[block], keep)
        for rec, values in (("charge", out.charge), ("mass", out.mass)):
            _describe(io, sp[rec], rec, particle=True)
            _constant(io, sp[rec][io.Record_Component.SCALAR], values[first], count)


def write_openpmd(out, path, every=1, meshes=True, particles=True, area=1.0):
    """Write ``out`` to the openPMD series ``path`` and return the path.

    Args:
        out: :class:`jaxincell.Output` (or any object with the same attributes).
        path: Output file; ``.json`` (default, appended if no extension), ``.h5`` or ``.bp`` pick the backend.
        every: Export every ``every``-th stored step (the iteration index is the stored-step index).
        meshes: Write the ``E``, ``B``, ``J`` and ``rho`` meshes.
        particles: Write the per-species particle records (skipped when ``out.x is None``).
        area: Transverse area in m^2 that the one-dimensional run stands for; ``weighting`` is
            ``out.weight * area``, a number of physical particles (see the module docstring).

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
            _write_particles(io, it, out, s, area, keep)
        series.flush()
    series.close()
    return path
