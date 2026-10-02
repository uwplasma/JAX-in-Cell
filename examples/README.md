# Examples

Each script stands on its own: editable parameters at the top, the physical setup
printed, a number compared with something the code does not itself compute, and a
figure. Run one from anywhere:

```bash
python examples/1_basic/two_stream.py
```

They are ordered by how much of the code they use, not by how interesting they are.
`JAX_ENABLE_X64=0` in front of any of them switches the whole run to single precision.

## 1_basic

| script | what it shows | compared with | runs in |
|---|---|---|---|
| `two_stream.py` | two counter-streaming beams go unstable | the cold-beam growth rate, Buneman (1959) | ~1 min |
| `langmuir_wave.py` | the frequency of an electron plasma wave against `k` | the kinetic dispersion relation | ~30 s |
| `parameters_and_sampling.py` | physical and numerical inputs, derived scales, and three loadings | the Poisson spread of random loading | ~30 s |
| `landau_damping.py` | a wave damped by resonant electrons, with no collisions | the least-damped kinetic root | ~30 s |
| `sheath_unmagnetized.py` | a maintained source-to-collector sheath | the kinetic floating potential, in closed form | ~2 min |

## 2_intermediate

| script | what it shows | compared with | runs in |
|---|---|---|---|
| `bump_on_tail.py` | a beam on the tail of a Maxwellian drives a wave | the kinetic growth rate | ~1 min |
| `weibel.py` | a temperature anisotropy grows a magnetic field: the cutoff, then every mode of a twelve-wavelength box | the transverse kinetic root, mode by mode | a few min on a GPU; `--quick` ~2 min |
| `output_and_restart.py` | save, restart and openPMD round trip | the uninterrupted run, bit for bit | ~20 s |
| `compare_models.py` | one two-stream problem with seven solver settings | the kinetic growth rate; energy conservation | ~2 min on a GPU; `--quick` ~2 min |
| `collisions.py` | Coulomb slowing-down and perpendicular diffusion | the NRL formulary rates | ~20 s |
| `relativistic_two_stream.py` | relativistic counter-streaming beams | the cold and warm dispersion relations | default control |
| `wall_reflection.py` | a velocity-dependent wall, through its flux average | `u^2/(u^2+sigma^2)`, in closed form | ~20 s |
| `sheath_magnetized.py` | the sheath in a field oblique to the wall | its own limits, and the impact distributions | ~1 h |
| `sheath_reflection.py` | a wall that returns part of the electron flux | Hobbs and Wesson (1967) | ~4 min |
| `external_fields_3d.py` | an external field on an (x, y, z) grid: grad-B drift and a mirror bounce | guiding-centre theory | ~40 s; `--quick` ~20 s |

## 3_advanced

| script | what it shows | compared with | runs in |
|---|---|---|---|
| `electron_field.py` | driven electron waves with uniform, frozen and mobile backgrounds | the accelerated-frame control and kinetic dielectric | `--quick` smoke preset |
| `invariants.py` | periodic quintic shapes and orbit integration | conservation and refinement controls | `--quick` smoke preset |
| `conservation.py` | the implicit scheme conserves energy and charge at once | round-off | ~1 min |
| `optimize_two_stream.py` | gradient ascent on a growth rate, through the whole run | `sqrt(3/8)`, the cold-beam optimum | ~2 min |
| `sheath_optimization.py` | a wall's reflectivity recovered from the sheath it holds | the value the target was made at | ~11 min |

`sheath_optimization.py` also takes `--oblique`, which runs the same experiment in a
magnetic field 30 degrees to the wall, and writes its own folder of results.

Scripts offering `--quick` provide a smoke run with fewer particles or a shorter time:
`sheath_unmagnetized.py` in 10 seconds, `sheath_magnetized.py` in two minutes and
`sheath_optimization.py` in forty seconds, which is what continuous integration runs. It
checks that they execute and reproduces the structure, with more noise, and each of them
says so when it starts rather than letting a smoke run be quoted as a measurement.

`input.toml` is for the command line, `jaxincell examples/input.toml`.
