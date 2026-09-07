# API reference

Everything importable from `jaxincell` is listed here, grouped by what it does. The
top-level names are the public interface; the modules they come from start with an
underscore and are not part of the documented API.

```{toctree}
:maxdepth: 1

simulation
diagnostics
kernels
constants
```

## Overview

| name | purpose |
|---|---|
| {class}`jaxincell.Simulation` | build, configure and run a simulation |
| {func}`jaxincell.load_parameters` | read a TOML parameter file |
| {func}`jaxincell.diagnostics` | energies, species split, dominant frequency |
| {func}`jaxincell.plot` | animated overview figure, optional MP4 |
| {func}`jaxincell._algorithms.Boris_step`, {func}`jaxincell._algorithms.CN_step` | one time step of the explicit and implicit schemes |
| {func}`jaxincell.boris_step`, {func}`jaxincell.boris_step_relativistic` | the particle pushers |
| {func}`jaxincell.fields_to_particles_grid` | field interpolation with the quadratic spline |
| {func}`jaxincell.calculate_charge_density`, {func}`jaxincell.current_density` | source deposition |
| {func}`jaxincell.curlE`, {func}`jaxincell.curlB`, {func}`jaxincell.field_update1`, {func}`jaxincell.field_update2` | finite-difference field update |
| {func}`jaxincell.E_from_Gauss_1D_FFT`, {func}`jaxincell.E_from_Poisson_1D_FFT`, {func}`jaxincell.E_from_Gauss_1D_Cartesian` | electrostatic solvers |
| {func}`jaxincell.set_BC_particles`, {func}`jaxincell.set_BC_positions` | particle boundary conditions |
| {func}`jaxincell.filter_scalar_field`, {func}`jaxincell.filter_vector_field` | digital filter |
| `jaxincell.epsilon_0`, `jaxincell.speed_of_light`, ... | physical constants |
