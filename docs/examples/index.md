# Examples

The `examples/` directory of the repository contains runnable scripts and input files.
Each page below shows the script, explains the physics it sets up and shows the result
as produced by the documentation build scripts under `docs/scripts/`.

```{toctree}
:maxdepth: 1

two_stream
landau_damping
langmuir_wave
bump_on_tail
weibel
energy_conservation
autodiff
optimisation
inference
scaling
```

| script | what it shows | run time on a laptop CPU |
|---|---|---|
| `two-stream_instability.py`, `input.toml` | two counter-streaming beams, the default configuration | seconds |
| `Landau_damping.py` | damping of a Langmuir wave in a warm plasma | seconds |
| `Langmuir_wave.py` | plasma oscillations at the plasma frequency | seconds |
| `bump-on-tail.py`, `bump-on-tail.toml` | four populations, a weak beam on a Maxwellian, explicit or implicit | tens of seconds |
| `Weibel_instability.py` | magnetic field generation from a temperature anisotropy | tens of seconds |
| `auto-differentiability.py` | gradient of a diagnostic with respect to the drift speed, against finite differences | a minute |
| `optimize_two_stream_saturation.py` | minimise the saturated field energy over the ion temperature | minutes |
| `inference_two_stream.py` | recover the drift speed from the growth rate with forward-mode derivatives | minutes |
| `scaling_energy_time.py` | run time and energy error against resolution | minutes |

Run any of them from the repository root, for example

```bash
python examples/Landau_damping.py
```

The scripts open matplotlib windows; set the environment variable `MPLBACKEND=Agg`
to run them without a display.
