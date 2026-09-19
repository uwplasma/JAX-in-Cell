# Running a simulation

```python
output = simulation.run(steps, seed=0, store_every=1, store_particles=True, state=None)
```

| argument | meaning |
|---|---|
| `steps` | number of time steps; must be a multiple of `store_every` |
| `seed` | integer seed of the random numbers; a traced value, so `jax.vmap` over it gives an ensemble from one compilation |
| `store_every` | keep every n-th state |
| `store_particles` | keep the particle histories, which are the bulk of the memory |
| `state` | a previous `Output.state` to continue from |
| `verbose` | report progress while the run goes on: `True`, or anything to call with `(done, total)` |

The whole loop — initialisation, deposition, field solve, push, boundaries,
diagnostics — is one `jax.jit`-compiled program built around `lax.scan`. The first
call compiles, which takes a second or two; subsequent calls with the same
`steps`, `store_every` and `store_particles` reuse it, even when the physical
parameters change.

## Saying how far it has got

```python
output = simulation.run(200000, verbose=True)
```

writes a line to stderr that is rewritten as the run goes — steps done, per cent, rate,
elapsed and an estimate of what is left — and a fresh line each time when the stream is a
log rather than a terminal.

The meter is on the **host**, outside anything traced. The run is split into about twenty
groups, each the same compiled program, and the state carries everything between them, so a
verbose run is a silent run bit for bit; that is the same guarantee a restart rests on. It
costs about 1.3 %, measured on a 2000-step run of 200000 particles. How often it reports is
its own choice and not `store_every`'s.

Putting a bar inside the loop with `jax.debug.callback` instead is the obvious thing and it
does not work: `run` is `jit`-ed, so its body runs once per *compilation*, and a bar built
there is trace-time state — a second identical call reuses the compiled program and the bar
closed at the end of the first, and reports nothing. Beyond that a debug callback fires on
the forward pass only under `grad`, is unrolled across the mapped axis under `vmap`, raises
on more than one device when ordered, and dispatches asynchronously.

`verbose` takes anything callable, which is how a `tqdm` bar goes in without `tqdm` becoming
a dependency of this package:

```python
bar = tqdm.tqdm(total=steps)
out = simulation.run(steps, verbose=lambda done, total: bar.update(done - bar.n))
```

Under `jax.jit`, `jax.grad` or `jax.vmap` the meter turns itself off without being asked:
there is nothing to report from inside a trace, and a side effect has no business in the
differentiated path.

## What is static and what is not

`steps`, `store_every` and `store_particles` are static arguments, and so are the
particle counts, the cell count, the boundary types and every switch in
{class}`~jaxincell.Solver`. Change one and the program is rebuilt. Everything
physical — lengths, densities, drifts, thermal speeds, the filter weight, the
restitution, the external fields — is a pytree leaf, so it can be changed freely:

```python
hotter = simulation.replace(species=(electrons.replace(vth=(2e6, 0, 0)), ions))
output = hotter.run(1000, seed=0)     # no recompilation
```

## Where a quantity lives

The grid is staggered, and the output publishes both coordinate arrays rather than
leaving each call site to rebuild one:

| array | length | holds |
|---|---|---|
| `output.grid` | `cells` | cell centres $x_i = -L/2 + (i+\tfrac12)\Delta x$: `rho`, the deposited moments, $B$ |
| `output.faces` | `cells` | the stored faces $x_{i+1/2} = -L/2 + (i+1)\Delta x$, the right face of each cell: $E$, the potential |
| `output.walls` | 2 | the two wall faces, $\pm L/2$ |

The left wall face $-L/2$ is not among `faces`, which is why
{func}`~jaxincell.potential` and the field solver take its value separately.
{class}`~jaxincell.Domain` carries `grid` and `faces` too, so a script can lay out its
axes before it runs anything.

`output.t` and `output.steps` are both absolute: they count from the beginning of the
first run, not of this one, so a restart's histories join on without a shift and a mean
over a window is a difference divided by a difference of `steps`, never by the length of
an array.

## Memory

The particle history is almost always the binding constraint, not speed. Its size is

```{math}
\text{bytes} = \frac{\text{steps}}{\text{store\_every}} \times N \times 8 \times (3 + 3 + 1)
```

for the positions, the velocities and the weights: six thousand steps of sixty thousand
particles is 20 GB, while the same run's fields on a 128-cell grid are under a hundred
megabytes. Two ways out:

```python
output = simulation.run(20000, store_every=20)        # 1/20 of the samples
output = simulation.run(20000, store_particles=False) # fields only
```

`store_particles=False` keeps the fields and the charge density, which is all the
field diagnostics need, and is the right choice for a growth-rate measurement. Note
that `kinetic`, `total`, `momentum` and `temperatures` are then absent from
{func}`~jaxincell.diagnostics`, because they cannot be computed without the particles.

## Restarts

`Output.state` is the full loop state. Feed it back to continue exactly where the
previous call stopped:

```python
first  = simulation.run(1000, seed=2)
second = simulation.run(1000, seed=2, state=first.state)
```

The result is bit-identical to a single 2000-step run. This is how to keep the memory
bounded on a long run: process or write each chunk, then discard it.

A state lives in memory, and a long campaign is a sequence of processes, so it can be
written out and read back:

```python
save_state("checkpoint.npz", first.state, simulation)
second = simulation.run(1000, seed=2, state=load_state("checkpoint.npz", simulation))
```

and the result is still bit-identical. The archive is named and versioned — one array per
field of `State` and of the `Wall` ledger inside it, under the names they have in the code
— and nothing in it is executed on reading. Passing the simulation checks that the archive
and the run are the same shape, which a restart into a different one is not; a state that
does not match is refused rather than left to fail somewhere later.

That is not what {func}`~jaxincell.openpmd.write_openpmd` is for. openPMD carries what an
analysis tool reads, the fields and the particles at the steps that were stored, and it is
not enough to carry on from: no random key, no wall ledger, no source bookkeeping, no charge
density at the step the loop is about to begin, and positions at integer times where the
explicit loop carries half-step ones. The two files answer different questions.

## Ensembles

`seed` is traced, so `jax.vmap` over it produces an ensemble from a single compiled
program:

```python
import jax, jax.numpy as jnp

fields = jax.vmap(lambda s: simulation.run(500, seed=s).E[-1, :, 0])(jnp.arange(16))
print(fields.shape)      # (16, cells)
```

Scanning a physical parameter works the same way, and needs no recompilation because
the parameter is a leaf.

## Input files

A run can be written as TOML and started from the command line:

```bash
jaxincell examples/input.toml
```

The tables map onto the constructors: `[domain]` to {class}`~jaxincell.Domain`,
`[solver]` to {class}`~jaxincell.Solver`, each `[[species]]` to a
{class}`~jaxincell.Species`, `[collisions]` to {class}`~jaxincell.Collisions`, and
`[run]` carries `steps`, `seed`, `store_every` and `plot`. Every species states its
`mass`, by name or in kilograms, optionally times `mass_ratio`; a species without one,
or with a name other than `"electron"` or `"proton"`, is a `ValueError` rather than a
silent proton.

```toml
[domain]
length = 0.01
cells = 64
dt_over_dx_c = 4.5

[solver]
algorithm = "explicit"
filter_passes = 2

[[species]]
name = "electrons"
n = 20000
charge = -1
mass = "electron"          # required: "electron", "proton", or a number in kilograms
density = 4.37e17
vth = [1.5e7, 0.0, 0.0]
drift = [6e7, 0.0, 0.0]
plus_minus = true
perturbation_amplitude = 5e-7
perturbation_mode = 1

[run]
steps = 1200
seed = 0
```

{func}`~jaxincell.load_toml` returns the simulation and the `[run]` table, so a file
can also be the starting point of a script:

```python
from jaxincell import load_toml

simulation, run = load_toml("input.toml")
output = simulation.run(run["steps"], seed=run.get("seed", 0))
```

## Reproducibility

Two runs with the same seed and the same parameters give bit-identical results on the
same machine and JAX version. `seed` changes the random velocities of a non-quiet
start and the collision partners; a fully quiet, collisionless run is deterministic
regardless of it.
