# Input files

Runs you can start without writing any Python:

```bash
jaxincell inputs/two_stream.toml                        # run and animate
jaxincell inputs/landau_damping.toml --no-plot          # headless
jaxincell inputs/weibel.toml --save weibel --movie weibel.mp4
jaxincell inputs/sheath_unmagnetized.toml --steps 960 --seed 1 --no-plot
```

| file | what it runs | the script that measures it |
|---|---|---|
| `two_stream.toml` | two counter-streaming beams go unstable | `examples/1_basic/two_stream.py` |
| `landau_damping.toml` | a wave damped by resonant electrons | `examples/1_basic/landau_damping.py` |
| `langmuir_wave.toml` | one Langmuir wave at $k\lambda_D = 0.3$ | `examples/1_basic/langmuir_wave.py` |
| `bump_on_tail.toml` | a beam on the tail drives a wave to saturation | `examples/2_intermediate/bump_on_tail.py` |
| `weibel.toml` | a temperature anisotropy grows a magnetic field | `examples/2_intermediate/weibel.py` |
| `collisions.toml` | Coulomb collisions inside a full run | `examples/2_intermediate/collisions.py` |
| `conservation_implicit.toml` | the energy-conserving implicit scheme | `examples/3_advanced/conservation.py` |
| `sheath_unmagnetized.toml` | a maintained source-to-collector sheath | `examples/1_basic/sheath_unmagnetized.py` |
| `sheath_magnetized.toml` | the same sheath in an oblique magnetic field | `examples/2_intermediate/sheath_magnetized.py` |

Every table in a file is a constructor — `[domain]`, `[solver]`, each `[[species]]` and its
`[species.source]`, `[collisions]`, `[impacts]`, `[external]` — and `[run]` holds the arguments
of `Simulation.run`. **Nothing is ignored**: a key nothing reads is an error, so a misspelling
is caught in the file and not in the answer.

## Flags

| flag | what it does |
|---|---|
| `--steps N`, `--seed S` | override `[run]`; `N` must be a multiple of `store_every` |
| `--save DIR` | write `fields.npz`, `run.json` and a copy of the input file |
| `--movie FILE` | write the animation as a video |
| `--plot`, `--no-plot` | override `[run] plot` |

`run.json` carries the settings, the results and the versions, precision, device and commit
that produced them, so a folder says what was run as well as what came out.

## What a file cannot do

Scans, optimisation and movie scripting are not here on purpose. Those are programs — a loop,
an objective, a schedule — and a configuration file that grows a control flow is a worse
programming language than the one it is written in. Build the `Simulation` from a file with
`load_toml` and write the loop around it in Python; the examples do exactly that.
