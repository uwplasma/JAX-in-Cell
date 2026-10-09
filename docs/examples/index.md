# Examples

The `examples/` directory contains runnable scripts and input files; the collision
example lives in the repository root. The pages below explain their physics and
how to run them, with reference figures produced under `docs/scripts/` where available.

```{toctree}
:maxdepth: 1

two_stream
relativistic_two_stream
landau_damping
langmuir_wave
bump_on_tail
weibel
energy_conservation
autodiff
optimisation
inference
scaling
collisions
```

| script | what it shows | run time on a laptop CPU |
|---|---|---|
| `two-stream_instability.py`, `input.toml` | two counter-streaming beams, the default configuration | seconds |
| `relativistic_two_stream.py` | relativistic beam growth against cold theory and relativistic energy | seconds |
| `Landau_damping.py` | damping of a Langmuir wave in a warm plasma | seconds |
| `Langmuir_wave.py` | plasma oscillations at the plasma frequency | seconds |
| `bump-on-tail.py`, `bump-on-tail.toml` | four populations, a weak beam on a Maxwellian, explicit or implicit | tens of seconds |
| `Weibel_instability.py` | magnetic field generation from a temperature anisotropy | tens of seconds |
| `3d_field_runs.py` | prescribed x, x-y, x-z and x-y-z fields checked against gyro-orbits; see {doc}`../user_guide/external_fields` | seconds |
| `auto-differentiability.py` | gradient of a diagnostic with respect to the drift speed, against finite differences | a minute |
| `optimize_two_stream_saturation.py` | minimise the saturated field energy over the ion temperature | minutes |
| `inference_two_stream.py` | recover the drift speed from the growth rate with forward-mode derivatives | minutes |
| `openpmd_export.py` | optional openPMD export of particles and staggered fields | seconds |
| `scaling_energy_time.py` | run time and energy error against resolution | minutes |
| `source_particles.py` | matched particle injection, fractional collection and reservoir budgets; see {doc}`../user_guide/external_fields` | seconds |
| `mixed_bc.py`, `bc_parameter_comparison.py` | collection, fractional marker return and normal restitution; see {doc}`../user_guide/boundaries` | seconds |
| `twospecies_tempdiff.py` (repository root) | Coulomb relaxation and a collision-only energy control | depends on steps and particles |

Run any of them from the repository root, for example

```bash
python examples/Landau_damping.py
```

The scripts open matplotlib windows; set the environment variable `MPLBACKEND=Agg`
to run them without a display.
