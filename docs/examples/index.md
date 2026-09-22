# Examples

Every script in `examples/` runs on its own and reproduces a result the code does not
itself compute. They are grouped below by velocity dimensionality; the directory a script
sits in (`1_basic`, `2_intermediate`, `3_advanced`) says how much of the code it uses.

```bash
git clone https://github.com/uwplasma/JAX-in-Cell
cd JAX-in-Cell
python examples/1_basic/two_stream.py
```

Where an input file is listed the run needs no Python: `jaxincell inputs/<file>.toml`.
Four scripts — `sheath_unmagnetized.py`, `sheath_magnetized.py`, `sheath_optimization.py`
and `grazing_sheath.py` — take `--quick`, a smoke run of ten seconds to two minutes with
far fewer particles. It reproduces the structure with more noise, and each says so when it
starts, so that a smoke run is not quoted as a measurement.

## 1D1V: electrostatic, one velocity component

| case | what it shows | script | input file |
|---|---|---|---|
| {doc}`landau_damping` | a wave damped without collisions, at $k\lambda_D = 0.5$ | `1_basic/landau_damping.py` | `landau_damping.toml` |
| {doc}`langmuir_wave` | the kinetic dispersion relation scanned in $k$ | `1_basic/langmuir_wave.py` | `langmuir_wave.toml` |
| {doc}`two_stream` | growth, saturation and the phase-space vortex | `1_basic/two_stream.py` | `two_stream.toml` |
| {doc}`bump_on_tail` | a beam-driven wave and the quasilinear plateau | `2_intermediate/bump_on_tail.py` | `bump_on_tail.toml` |
| {doc}`wall_reflection` | a wall returns the flux average of its reflection law | `2_intermediate/wall_reflection.py` | — |
| {doc}`conservation` | energy, momentum and charge in both schemes | `3_advanced/conservation.py` | `conservation_implicit.toml` |
| {doc}`optimize_two_stream` | gradient ascent through the solver finds the fastest beam | `3_advanced/optimize_two_stream.py` | — |

## 1D2V: two velocity components, the magnetic field grown by the plasma

| case | what it shows | script | input file |
|---|---|---|---|
| {doc}`weibel` | a temperature anisotropy driving transverse magnetic modes | `2_intermediate/weibel.py` | `weibel.toml` |

## 1D3V: three velocity components — sheaths, oblique fields, collisions

| case | what it shows | script | input file |
|---|---|---|---|
| {doc}`sheath_unmagnetized` | a maintained sheath against the kinetic floating potential | `1_basic/sheath_unmagnetized.py` | `sheath_unmagnetized.toml` |
| {doc}`sheath_magnetized` | the magnetic presheath, and what the wall is struck by | `2_intermediate/sheath_magnetized.py` | `sheath_magnetized.toml` |
| {doc}`sheath_reflection` | the Bohm criterion and the Hobbs-Wesson sheath drop | `2_intermediate/sheath_reflection.py` | — |
| {doc}`grazing_sheath` | the two-layer transition, set up against the gyrokinetic code GYRAZE | `3_advanced/grazing_sheath.py` | — |
| {doc}`collisions` | the Takizuka-Abe operator against the Fokker-Planck rates | `2_intermediate/collisions.py` | `collisions.toml` |
| {doc}`sheath_optimization` | a wall's reflectivity recovered from the sheath it holds | `3_advanced/sheath_optimization.py` | — |

## Where the numbers come from

The scripts that draw the figures in this documentation live in `docs/scripts/` and record
the numbers the pages quote in `measurements.json`; see {doc}`../numerics/verification`.
Each example page names the script its figure and numbers come from. Most of those scripts
run exactly the example's setup, some adding a scan around it. Two do not: the two-stream
figure uses a quiet start at another drift, so that its growth rate can be fitted, and the
sheath figure uses more particles.

There is also `examples/input.toml`, which runs the two-stream case from the command line:

```bash
jaxincell examples/input.toml
```

```{toctree}
:hidden:

landau_damping
langmuir_wave
two_stream
bump_on_tail
wall_reflection
conservation
optimize_two_stream
weibel
sheath_unmagnetized
sheath_magnetized
sheath_reflection
grazing_sheath
collisions
sheath_optimization
```
