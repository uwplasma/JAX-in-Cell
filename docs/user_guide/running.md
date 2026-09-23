# Running a simulation

The whole loop — initialisation, deposition, field solve, push, boundaries, diagnostics —
is one `jax.jit` program built around `lax.scan`. The first call compiles in a second or
two; later calls with the same `steps`, `store_every` and `store_particles` reuse it, even
when the physical parameters change.

```python
output = simulation.run(steps, seed=0, store_every=1, store_particles=True,
                        moments=False, state=None, verbose=False)
```

## Arguments

| argument | meaning | default |
|---|---|---|
| `steps` | time steps; must be a multiple of `store_every` | — |
| `seed` | seed of the random numbers; traced, so `jax.vmap` over it gives an ensemble from one compilation | `0` |
| `store_every` | keep every n-th state | `1` |
| `store_particles` | keep the particle histories, the bulk of the memory | `True` |
| `moments` | running velocity-moment sums ({doc}`sources`) | `False` |
| `state` | a previous `Output.state` to continue from | `None` |
| `verbose` | report progress: `True`, or anything to call with `(done, total)` | `False` |

## Static and traced

| | |
|---|---|
| static — changing one rebuilds the program | `steps`, `store_every`, `store_particles`, the particle counts, the cell count, the boundary types, every switch in {class}`~jaxincell.Solver` |
| pytree leaves — free to change | lengths, densities, drifts, thermal speeds, the filter weight, the restitution, the external fields |

```python
hotter = simulation.replace(species=(electrons.replace(vth=(2e6, 0, 0)), ions))
output = hotter.run(1000, seed=0)     # no recompilation
```

## Progress

```python
output = simulation.run(200000, verbose=True)
```

* Writes a line to stderr, rewritten as the run goes: steps done, per cent, rate, elapsed,
  and an estimate of what is left. A log rather than a terminal gets a fresh line each time.
* The meter is on the **host**, outside anything traced. The run is split into about twenty
  groups of the same compiled program, with the state carried between them, so a verbose
  run is a silent run bit for bit — the guarantee a restart rests on too.
* Costs about 1.3 %, measured on a 2000-step run of 200000 particles.
* How often it reports is its own choice, not `store_every`'s.
* Turns itself off under `jax.jit`, `jax.grad` and `jax.vmap`.

`verbose` takes anything callable, so a `tqdm` bar goes in without `tqdm` becoming a
dependency of this package:

```python
bar = tqdm.tqdm(total=steps)
out = simulation.run(steps, verbose=lambda done, total: bar.update(done - bar.n))
```

A bar inside the loop with `jax.debug.callback` does not work. `run` is `jit`-ed, so its
body runs once per *compilation*, and a bar built there is trace-time state: a second
identical call reuses the compiled program and reports nothing, into a bar closed at the
end of the first. A debug callback also fires on the forward pass only under `grad`, is
unrolled across the mapped axis under `vmap`, raises on more than one device when ordered,
and dispatches asynchronously.

## Where a quantity lives

The grid is staggered, and the output publishes both coordinate arrays.

| array | length | holds |
|---|---|---|
| `output.grid` | `cells` | cell centres $x_i = -L/2 + (i+\tfrac12)\Delta x$: `rho`, the deposited moments, $B$ |
| `output.faces` | `cells` | the stored faces $x_{i+1/2} = -L/2 + (i+1)\Delta x$, the right face of each cell: $E$, the potential |
| `output.walls` | 2 | the two wall faces, $\pm L/2$ |

* The left wall face $-L/2$ is not among `faces`, so {func}`~jaxincell.potential` and the
  field solver take its value separately.
* {class}`~jaxincell.Domain` carries `grid` and `faces` too, so a script can lay out its
  axes before it runs anything.
* `output.t` and `output.steps` are absolute: they count from the first run, not this one.
  A restart's histories join on without a shift, and a mean over a window is a difference
  divided by a difference of `steps`, never by the length of an array.

## Memory

The particle history is almost always the binding constraint, not speed:

```{math}
\text{bytes} = \frac{\text{steps}}{\text{store\_every}} \times N \times 8 \times (3 + 3 + 1)
```

for the positions, the velocities and the weights. Six thousand steps of sixty thousand
particles is 20 GB; the same run's fields on a 128-cell grid are under a hundred megabytes.
Two ways out:

```python
output = simulation.run(20000, store_every=20)        # 1/20 of the samples
output = simulation.run(20000, store_particles=False) # fields only
```

`store_particles=False` keeps the fields and the charge density, which is all the field
diagnostics need, and is the right choice for a growth-rate measurement. `kinetic`,
`total`, `momentum` and `temperatures` are then absent from
{func}`~jaxincell.diagnostics`: they cannot be computed without the particles.

## Restarts

`Output.state` is the full loop state. Fed back, it continues exactly where the previous
call stopped, bit-identically to a single 2000-step run — which is how to keep the memory
bounded on a long run: process or write each chunk, then discard it.

```python
first  = simulation.run(1000, seed=2)
second = simulation.run(1000, seed=2, state=first.state)
```

A state can also be written out and read back, still bit-identical:

```python
save_state("checkpoint.npz", first.state, simulation)
second = simulation.run(1000, seed=2, state=load_state("checkpoint.npz", simulation))
```

* The archive is named and versioned — one array per field of `State` and of the `Wall`
  ledger inside it, under the names they have in the code — and nothing in it is executed
  on reading.
* Passing the simulation checks that the archive and the run are the same shape; a state
  that does not match is refused rather than left to fail somewhere later.
* {func}`~jaxincell.openpmd.write_openpmd` is not for this. openPMD carries what an
  analysis tool reads — the fields and the particles at the stored steps — and is not
  enough to carry on from: no random key, no wall ledger, no source bookkeeping, no charge
  density at the step the loop is about to begin, and positions at integer times where the
  explicit loop carries half-step ones.

## Ensembles

`seed` is traced, so `jax.vmap` over it produces an ensemble from a single compiled
program. Scanning a physical parameter works the same way and needs no recompilation,
because the parameter is a leaf.

```python
import jax, jax.numpy as jnp

fields = jax.vmap(lambda s: simulation.run(500, seed=s).E[-1, :, 0])(jnp.arange(16))
print(fields.shape)      # (16, cells)
```

## Input files

A run can be written as TOML and started from the command line:

```bash
jaxincell examples/input.toml
```

Every table is a constructor.

| table | builds |
|---|---|
| `[domain]` | {class}`~jaxincell.Domain` |
| `[solver]` | {class}`~jaxincell.Solver` |
| `[[species]]` | {class}`~jaxincell.Species` |
| `[species.source]` | {class}`~jaxincell.Source`, optional |
| `[collisions]` | {class}`~jaxincell.Collisions` |
| `[impacts]` | {class}`~jaxincell.Impacts` |
| `[external]` | uniform external fields, `E` and `B` as three components broadcast over the grid |
| `[run]` | the arguments of `run` — `steps`, `seed`, `store_every`, `store_particles`, `moments`, `verbose` — and `plot`, which the command line reads |

* Every species states its `mass`, by name or in kilograms, optionally times `mass_ratio`.
  A species without one, or with a name other than `"electron"` or `"proton"`, is a
  `ValueError` rather than a silent proton.
* **Nothing is ignored.** A table or a key that nothing reads is an error, checked before
  anything is built, so a misspelling is reported as a misspelling and not as whatever the
  half-built object goes on to complain about: a misspelled `vth` is a different plasma.
* Scans, optimisation and movie scripting are deliberately absent: they are programs, not
  configuration. Load the `Simulation` from a file and write the loop around it in Python.

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

{func}`~jaxincell.load_toml` returns the simulation and the `[run]` table, so a file can
also start a script:

```python
from jaxincell import load_toml

simulation, run = load_toml("input.toml")
output = simulation.run(run["steps"], seed=run.get("seed", 0))
```

## Reproducibility

Two runs with the same seed and the same parameters give bit-identical results on the same
machine and JAX version. `seed` changes the random velocities of a non-quiet start and the
collision partners; a fully quiet, collisionless run is deterministic regardless of it.
