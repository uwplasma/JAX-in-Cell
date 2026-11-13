# Examples

Every script is a plain top-to-bottom program with no arguments. Settings sit at the top;
the defaults finish in about a minute and show the structure with more noise. Where a longer
run is more informative, the setting carries a `research` comment with the value the
documentation's numbers were made with. Each script prints a number compared with something the
code does not compute and shows a figure. Run one from anywhere:

```bash
python examples/1_basic/two_stream.py
```

The directories are ordered by how much of the code they use. Scripts accept `JAX_ENABLE_X64=0`
to run in single precision; `beam_laser.py` and `rf_source_control.py` require float64. The
runtimes are those of the continuous-integration runner with four scripts running at once.


## 1_basic

| script | what it shows | compared with | runs in |
|---|---|---|---|
| `two_stream.py` | two counter-streaming beams go unstable | the cold-beam growth rate (Buneman 1959) | 11 s |
| `langmuir_wave.py` | the frequency of an electron plasma wave against `k` | the kinetic dispersion relation | 38 s |
| `parameters_and_sampling.py` | physical and numerical inputs, derived scales, three particle loadings | the Poisson spread of random loading | 28 s |
| `landau_damping.py` | a wave damped by resonant electrons, no collisions | the least-damped kinetic root (Canosa 1972) | 35 s |
| `sheath_unmagnetized.py` | a maintained source-to-collector sheath | the kinetic floating potential in closed form | 33 s |

## 2_intermediate

| script | what it shows | compared with | runs in |
|---|---|---|---|
| `bump_on_tail.py` | a beam on the tail of a Maxwellian drives a wave | the kinetic growth rate | 22 s |
| `weibel.py` | temperature anisotropy grows a magnetic field: the cutoff, then mode by mode | the transverse kinetic root | 33 s |
| `output_and_restart.py` | save, restart and openPMD round trip | the uninterrupted run, bit for bit | 7 s |
| `compare_models.py` | one two-stream problem with seven solver settings | the kinetic growth rate; energy conservation | 32 s |
| `collisions.py` | Takizuka-Abe slowing-down and perpendicular diffusion | the NRL formulary rates | 5 s |
| `relativistic_two_stream.py` | relativistic counter-streaming beams | the cold and warm dispersion relations | 9 s |
| `wall_reflection.py` | a velocity-dependent wall, through its flux average | `u^2/(u^2+sigma^2)` in closed form | 28 s |
| `sheath_magnetized.py` | the sheath in a field oblique to the wall | its own limits and the impact distributions | 33 s |
| `sheath_reflection.py` | a wall that returns part of the electron flux | Hobbs and Wesson (1967) | 60 s |
| `external_fields_3d.py` | an external field on an (x, y, z) grid: grad-B drift and a mirror bounce | guiding-centre theory | 8 s |
| `electron_cooling.py` | classical and quantum-corrected radiation reaction, beam cooling, ring formation | exact uniform-B solution, Niel et al. Gaunt factor, Bilbao and Silva (2023) | 25 s |

## 3_advanced

| script | what it shows | compared with | runs in |
|---|---|---|---|
| `electron_field.py` | driven electron waves on uniform, frozen and mobile backgrounds | the accelerated-frame control and kinetic dielectric | 43 s |
| `invariants.py` | quintic shapes and orbit integration | conservation and refinement controls | 16 s |
| `conservation.py` | explicit and implicit schemes against energy, momentum and charge | round-off | 17 s |
| `optimize_two_stream.py` | gradient ascent on a growth rate through the whole run | `sqrt(3/8)`, the cold-beam optimum | 25 s |
| `sheath_optimization.py` | a wall reflectivity recovered from the sheath it holds | the value the target was made at | 52 s |
| `grazing_sheath.py` | a sheath at grazing magnetic incidence | GYRAZE (Geraldini et al., arXiv:2508.09067), when `reference` is set | 67 s |
| `runaway_rf.py` | a prescribed RF envelope optimized through the time loop | directional derivatives and refined trajectories | 10 s |
| `runaway_baseline.py` | a no-RF runaway baseline with moment-restored collisions | the reduced test-particle ODE | 9 s |
| `runaway_rf_minimum.py` | the minimum RF energy to hold a runaway tail below 10 MeV | resolved wave windows and frozen-control checks | 13 s |
| `beam_laser.py` | finite-spread beam cooling in a laser pulse, reduced and full LL | exact LL endpoint, phase ODE and analytic AD | 18 s |
| `rf_source_control.py` | supplied RF energy for a cold-plasma deposition target | the Maxwell source and cross-energy budget | 37 s |

## What each part of the code is shown by

| feature | scripts |
|---|---|
| explicit and implicit solvers, shape orders, conservation | `compare_models`, `conservation`, `invariants` |
| relativistic pushers and Maxwell solver | `relativistic_two_stream`, `weibel` |
| Coulomb collisions | `collisions`, `runaway_baseline` |
| radiation reaction: `ll_reduced`, `ll_full`, `ll_quantum` | `electron_cooling`, `beam_laser` |
| RF control, prescribed fields and gradients | `runaway_rf`, `rf_source_control`, `runaway_rf_minimum`, `optimize_two_stream` |
| walls, reflection, sources and sheaths | `wall_reflection`, `sheath_reflection`, `sheath_unmagnetized`, `sheath_magnetized`, `sheath_optimization`, `grazing_sheath` |
| gridded external fields | `external_fields_3d` |
| loading, output, restart and openPMD | `parameters_and_sampling`, `output_and_restart` |

`input.toml` is for the command line, `jaxincell examples/input.toml`.
