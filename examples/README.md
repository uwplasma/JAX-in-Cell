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
| `landau_damping.py` | a wave damped by resonant electrons, with no collisions | the least-damped kinetic root | ~30 s |
| `sheath_unmagnetized.py` | a maintained source-to-collector sheath | the kinetic floating potential, in closed form | ~4 min |

## 2_intermediate

| script | what it shows | compared with | runs in |
|---|---|---|---|
| `bump_on_tail.py` | a beam on the tail of a Maxwellian drives a wave | the kinetic growth rate | ~1 min |
| `weibel.py` | a temperature anisotropy grows a magnetic field | the growth rates of six modes | ~1 min |
| `collisions.py` | Coulomb slowing-down and perpendicular diffusion | the NRL formulary rates | ~20 s |
| `wall_reflection.py` | a velocity-dependent wall, through its flux average | `u^2/(u^2+sigma^2)`, in closed form | ~20 s |
| `sheath_magnetized.py` | the sheath in a field oblique to the wall | its own limits, and the impact distributions | ~6 min |
| `sheath_reflection.py` | a wall that returns part of the electron flux | Hobbs and Wesson (1967) | ~3 min |

## 3_advanced

| script | what it shows | compared with | runs in |
|---|---|---|---|
| `conservation.py` | the implicit scheme conserves energy and charge at once | round-off | ~1 min |
| `optimize_two_stream.py` | gradient ascent on a growth rate, through the whole run | `sqrt(3/8)`, the cold-beam optimum | ~2 min |
| `sheath_optimization.py` | a wall's reflectivity recovered from the sheath it holds | the value the target was made at | ~15 min |

`sheath_optimization.py` also takes `--oblique`, which runs the same experiment in a
magnetic field 30 degrees to the wall.

The three sheath scripts take `--quick`, which runs a smaller version with the same
structure and more noise; that is what continuous integration runs.

`input.toml` is for the command line, `jaxincell examples/input.toml`.
