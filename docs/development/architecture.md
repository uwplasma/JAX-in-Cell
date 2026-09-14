# Architecture

The package is eight modules and about 1600 lines. Each one has a single job, and the
dependency graph is a straight line with no cycles.

```
_constants.py    CODATA values
_config.py       Domain, Species, Solver, Collisions -- frozen pytree dataclasses
_core.py         the numerical kernels: shape function, deposit, gather, curls,
                 Maxwell update, Gauss solve, Boris pushers, boundaries, filter
_collisions.py   Takizuka-Abe binary collisions
_simulation.py   Simulation, Output, the time loop, TOML input, quiet_start
_diagnostics.py  energies, momentum, Gauss residual, temperatures, frequency
_plot.py         the animated overview figure and the movie writer
openpmd.py       optional openPMD export
```

`__init__.py` re-exports the public names and holds `main`, the `jaxincell` command.
`plot` is imported on first use, since matplotlib would otherwise take a third of the
import time, and `__version__` is `"unknown"` in a source tree that was never
installed, since setuptools_scm writes `jaxincell/version.py` at build time.

## Everything is a pytree

`pytree_dataclass(static=(...))` in `_config.py` is twenty lines and does the work: it
makes a frozen dataclass, registers it with `jax.tree_util`, and splits the fields into
leaves and static metadata. Leaves are traced, so they can change without
recompilation and be differentiated with respect to; static fields become part of the
treedef and therefore of the cache key. A field that holds a function, such as a
velocity-dependent reflection law, is moved into the static metadata whatever its
declaration, since a function is not an array; in a tuple such as `(law, 0.3)` only the
function moves, and the number stays a leaf.

The split is the main design decision in the package. A parameter is static if the
*shape* of the computation depends on it — particle counts, cell counts, boundary
types, the algorithm name, the number of Picard iterations — and a leaf otherwise.
Getting it wrong shows up immediately: a static physical parameter recompiles on every
change, and a leaf that controls a shape fails to trace.

JAX rebuilds an object from its leaves without calling `__init__`, so `__post_init__`
normalises and validates only what is constructed or passed to `replace`, and tree
operations can put anything in the leaves: stacked ensembles, tracers, `None`. Since
`replace` converts the stored values again, the conversions must be idempotent (the
boundary-name conversion accepts codes as well as names) and must not force a traced
value (the Courant check skips values that are not plain Python numbers). An object with
`None` where it always holds a number is a template, such as a `vmap` `in_axes`, and is
stored as given.

## The time loop

`_run` is jitted with `steps`, `store_every` and `store_particles` as static
arguments. Inside, `lax.scan` runs the chunks and an inner `lax.scan` runs the
`store_every - 1` steps that are not kept, so thinning the history costs nothing and
the whole loop is a single XLA program with no Python in it.

The step function itself is a method on `Simulation`, chosen once from
`solver.algorithm`. Because `Simulation` is a pytree and `self` is traced, the method
closes over the traced parameters without capturing them as constants.

## Adding something

**A diagnostic**: a function of `Output` in `_diagnostics.py`, added to the dictionary
that `diagnostics` returns. Nothing else has to change; it can be computed on a stored
run.

**A boundary condition**: a code in `BOUNDARIES`, a branch in `map_indices`,
`apply_particle_bc`, `_left_ghost_E`, `_right_ghost_B` and `_shift`. The branches are
resolved at trace time because the codes are static, so they cost nothing at run time.
A wall that needs random numbers, as the thermal wall does to redraw velocities, is a
position map in `apply_particle_bc` and a method on `Simulation`, which holds the key.

**A field solver or an integrator**: a branch in `Solver` and a method on
`Simulation` with the same signature as `_explicit_step`. Keep any iteration a
`lax.scan` of fixed length rather than a `lax.while_loop`, or reverse-mode
differentiation stops working.

**A species initialisation**: `Species.replace(x=..., v=...)` covers most of it from
outside the package; {func}`~jaxincell.quiet_start` exists so that a custom condition
can start from the quiet sampling.

## What the design gives up

Shapes are static, so particles cannot be created or destroyed. Absorption zeroes a
particle's weight and parks it outside the grid rather than removing it, which keeps
the arrays rectangular at the cost of memory for dead particles, and partial reflection
lowers the weight instead of splitting the particle in two. Ionisation and
injection would need the same treatment, with a pool of inactive particles.

The geometry is one-dimensional. Two and three dimensions would change `_core.py`
thoroughly, the rest much less: the configuration objects, the loop, the diagnostics
and the differentiability are not specific to one dimension.
