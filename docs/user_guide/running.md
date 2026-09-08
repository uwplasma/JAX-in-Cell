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

The whole loop — initialisation, deposition, field solve, push, boundaries,
diagnostics — is one `jax.jit`-compiled program built around `lax.scan`. The first
call compiles, which takes a second or two; subsequent calls with the same
`steps`, `store_every` and `store_particles` reuse it, even when the physical
parameters change.

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

## Memory

The particle history dominates. Its size is

```{math}
\text{bytes} = \frac{\text{steps}}{\text{store\_every}} \times N \times 3 \times 8 \times 2
```

for the positions and the velocities together. Ten thousand steps of a hundred
thousand particles is 48 GB, which will not fit anywhere. Two ways out:

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
`[run]` carries `steps`, `seed`, `store_every` and `plot`.

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
mass = "electron"          # or "proton", or a number in kilograms
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
