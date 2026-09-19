"""Writing a run's state to a file and reading it back, so that a restart survives the process.

``Output.state`` already continues a run exactly -- that is what
``run(..., state=out.state)`` does, and what makes a run split into chunks the run taken
whole. It lives in memory, so it does not survive the process that made it, and a long
campaign is a sequence of processes.

This writes it as a **named, versioned** archive: one ``.npz`` with one array per field of
:class:`~jaxincell._simulation.State` and of the :class:`~jaxincell._simulation.Wall` ledger
inside it, under the names they have in the code, plus the shape of the run that made it. It
is not a pickle: nothing here executes what it reads, and a field added or removed later is a
name that is present or absent rather than a class that no longer unpickles.

It is also not openPMD. :func:`~jaxincell.write_openpmd` writes what an analysis tool reads --
the fields and the particles at the steps that were stored -- and that is not enough to carry
on from: it has no random key, no wall ledger, no source bookkeeping, no charge density at the
step the loop is about to begin, and its positions are at integer times while the explicit
loop carries half-step ones. The two files answer different questions and both are worth
having.
"""
import dataclasses

import jax.numpy as jnp
import numpy as np

__all__ = ["save_state", "load_state"]

FORMAT = 1      #: Version of the archive layout, written into every file and checked on reading.


def _fields(obj):
    return tuple(f.name for f in dataclasses.fields(obj))


def save_state(path, state, simulation=None):
    """Write ``state`` to ``path`` (``.npz`` appended if it has no suffix) and return the path.

    ``simulation`` is the run the state came from; what it is for is the reading, where it says
    whether the archive and the simulation are the same shape. It is stored as the counts and
    names that have to match, not as the object.

    Arrays that a run did not keep -- the moment sums when ``moments=False``, the impact spectrum
    when no :class:`~jaxincell.Impacts` was given -- are absent rather than zero, and come back
    as ``None``.
    """
    path = str(path)
    path = path if path.endswith(".npz") else path + ".npz"
    arrays = {"format": np.asarray(FORMAT)}
    for name in _fields(state):
        value = getattr(state, name)
        if name == "wall":
            for inner in _fields(value):
                if getattr(value, inner) is not None:
                    arrays[f"wall.{inner}"] = np.asarray(getattr(value, inner))
        elif value is not None:
            arrays[name] = np.asarray(value)
    if simulation is not None:
        arrays["names"] = np.asarray([s.name for s in simulation.species])
        arrays["counts"] = np.asarray([s.n for s in simulation.species])
        arrays["cells"] = np.asarray(simulation.domain.cells)
        arrays["algorithm"] = np.asarray(simulation.solver.algorithm)
    np.savez(path, **arrays)
    return path


def load_state(path, simulation=None):
    """Read an archive written by :func:`save_state` and return the
    :class:`~jaxincell._simulation.State`.

    With ``simulation``, the archive is checked against it first -- the species names and counts,
    the cell count and the integrator -- because a state restored into a differently shaped run
    fails somewhere later and less clearly. Without it, the state is returned as it stands.

    Raises:
        ValueError: If the archive is of a format this version does not read, if a field the
            state needs is missing, or if it does not match ``simulation``.
    """
    from ._simulation import State, Wall

    with np.load(str(path), allow_pickle=False) as data:
        stored = {key: data[key] for key in data.files}
    version = int(stored.pop("format", -1))
    if version != FORMAT:
        raise ValueError(f"{path} is a format {version} archive and this is jaxincell's format "
                         f"{FORMAT}; it was written by a different version and is not read here")
    if simulation is not None:
        wanted = {"names": np.asarray([s.name for s in simulation.species]),
                  "counts": np.asarray([s.n for s in simulation.species]),
                  "cells": np.asarray(simulation.domain.cells),
                  "algorithm": np.asarray(simulation.solver.algorithm)}
        for key, want in wanted.items():
            if key in stored and not np.array_equal(stored[key], want):
                raise ValueError(f"{path} was written by a run whose {key} is {stored[key].tolist()!r}, "
                                 f"and this simulation's is {want.tolist()!r}: a state restored into a "
                                 "differently shaped run is not the run it came from")
    wall = Wall(*[jnp.asarray(stored[f"wall.{name}"]) if f"wall.{name}" in stored else None
                  for name in _fields(Wall)])
    values = []
    for name in _fields(State):
        if name == "wall":
            values.append(wall)
        elif name in stored:
            values.append(jnp.asarray(stored[name]))
        elif name == "moments":
            values.append(None)
        else:
            raise ValueError(f"{path} has no {name!r}, which a state needs; it is not a state archive")
    return State(*values)
