# Your first simulation

This page runs the two-stream instability from `examples/input.toml` and explains each
choice along the way. The same file is used on the {doc}`quickstart` page; here the
focus is on why the numbers are what they are and what to look at in the output.

## The physical setup

Two electron beams stream through each other at $\pm v_d$ with $v_d = 0.2c$
(`drift_speed_x = 6e7` m/s) and thermal speed $v_{th} = 0.05c$ (`vth_over_c_x = 0.05`).
Protons with the same temperature as the electrons provide a neutralising background.
The box is periodic, and a small sinusoidal displacement of the electrons
(`perturbation_amplitude_x = 5e-7` m, `perturbation_wavenumber_x = 1`) seeds the longest
wavelength that fits in the box. Cold-beam theory predicts an instability when
$k v_d < \omega_{pe}$; the {doc}`../numerics/verification` page compares the measured growth
rate with the kinetic dispersion relation.

The relevant parameters are collected in three sections:

```toml
[domain_parameters]
length = 0.01                          # box length in metres
number_grid_points = 70
timestep_over_spatialstep_times_c = 4.5   # c dt / dx
total_steps = 1100

[species_parameters.electrons.electrons0]
number_pseudoparticles = 3500
grid_points_per_Debye_length = 0.50265482457   # sets the density through lambda_D
vth_over_c_x = 0.05
drift_speed_x = 6e7
velocity_plus_minus_x = true           # half the particles get -drift_speed_x
perturbation_amplitude_x = 0.0000005
perturbation_wavenumber_x = 1

[species_parameters.ions.ions0]
number_pseudoparticles = 3500
grid_points_per_Debye_length = 0.50265482457
mass_over_proton_mass = 1
vth_over_c_x = "_electrons0"           # thermal speed from the electron temperature
ion_temperature_over_electron_temperature_x = 1
```

Three conventions are worth knowing from the start:

* **Density is set through the Debye length.** There is no density parameter. The
  pseudo-particle weight of each population is chosen so that the electron Debye
  length equals `1 / grid_points_per_Debye_length` cells; see
  {doc}`../user_guide/species` for the formula. With $\lambda_D \approx 2\,\Delta x$
  the box holds $L/\lambda_D \approx 35$ Debye lengths.
* **Thermal speed** $v_{th}$ is defined by $f(v) \propto \exp(-v^2/v_{th}^2)$, that is
  $v_{th} = \sqrt{2 k_B T / m}$. Velocities are sampled with standard deviation
  $v_{th}/\sqrt{2}$ per component.
* **A string value refers to another species.** `"_electrons0"` for an ion thermal speed
  means "the value that gives the temperature ratio
  `ion_temperature_over_electron_temperature_x` relative to the population labelled
  `electrons0`", with the mass ratio taken into account. The leading underscore is part
  of the canonical label that the code assigns to each population.

## Running

```python
from jaxincell import Simulation, load_parameters, diagnostics, plot
from jax import block_until_ready

parameters = load_parameters("examples/input.toml")
sim = Simulation(parameters)
output = block_until_ready(sim.run())
```

The first call to `run` traces and compiles the whole time loop, which takes a few
seconds; later calls on the same `Simulation` object reuse the compiled program. With
`print_info = true` the run starts by printing the derived quantities:

```text
Length of the simulation box: 35.18583771989999 Debye lengths or 1.244007222673535 Skin Depths
Density of electrons: 4.370228556770184e+17 m^-3
Electron temperature: 638.7486896859264 eV
Ion temperature / Electron temperature: 1.0
Debye length: 0.0002842052555237108 m
Skin depth: 0.008038538537186855 m
Wavenumber * Debye length: 0.0002842052555237108
Pseudoparticles per cell: 50.0
Pseudoparticle weight: 1248636730505.7668
Steps at each plasma frequency: 12.50439328006844
Total time: 87.96908217477141 / plasma frequency
Number of particles on a Debye cube: 10032298.935126843
Relativistic gamma factor: Maximum 1.0540053444235398, Average 1.0106887144507677
Charge x External electric field x Debye Length / Temperature: 0.0
```

`Steps at each plasma frequency` is $1/(\omega_{pe}\Delta t)$: about twelve steps per
inverse plasma frequency, so one plasma period is resolved by roughly eighty steps.
`Total time` is the simulated duration in units of $\omega_{pe}^{-1}$. The line
`Wavenumber * Debye length` multiplies the mode number by the Debye length in metres
rather than by $2\pi/L$; the dimensionless $k\lambda_D$ of this run is
$2\pi \times 1 \times 2.84\times10^{-4}/0.01 = 0.179$.

```{note}
`timestep_over_spatialstep_times_c = 4.5` exceeds the light-wave Courant limit
$c\,\Delta t/\Delta x \le 1$ of the explicit field solver. The run is stable only because
no transverse field is ever excited: all velocities are along $x$, so $J_y = J_z = 0$ and
the transverse Maxwell equations stay identically zero. If you give the particles a
$y$ or $z$ thermal spread, or an external magnetic field, reduce this parameter to one
or below. The {doc}`../numerics/stability` page lists all the constraints.
```

## Reading the diagnostics

```python
diagnostics(output)
print(output["plasma_frequency"])        # rad/s
print(output["electric_field_energy"])   # (steps,) J/m^2, epsilon_0/2 * integral of E^2 dx
print(output["total_energy"][-1] / output["total_energy"][0] - 1)
```

`diagnostics` computes the field and kinetic energies at every step, splits the
particles into electrons and ions (and into a per-population list under
`output["species"]`), and finds the dominant frequency of $E_x$ at the box centre.
It modifies the dictionary in place. The relative change of the total energy is a quick
check of the run: for this configuration it stays below $4\times 10^{-3}$ with the
explicit scheme and at round-off with the implicit one.

To see the instability grow, plot the electrostatic energy on a logarithmic scale:

```python
import matplotlib.pyplot as plt
t = output["time_array"] * output["plasma_frequency"]
plt.semilogy(t, output["electric_field_energy"])
plt.xlabel(r"$t\,\omega_{pe}$"); plt.ylabel(r"$\epsilon_0/2 \int E_x^2\,dx$")
```

The energy first sits at the noise level set by the finite number of pseudo-particles,
then grows exponentially, then saturates when the beams trap in the wave and form the
phase-space vortex shown on the landing page.

## Plotting and animating

```python
plot(output, animation_interval=5)
plot(output, save_mp4="two_stream.mp4", fps=50, dpi=150, save_stride=5, show=False)
```

`plot` opens a figure with heat maps of the non-zero field components, the velocity
distributions of electrons and ions, and their phase space, animated over time.
`save_mp4` writes the animation with `ffmpeg`; `save_stride` keeps every n-th frame to
reduce the file size. The options are described in {doc}`../user_guide/plotting`.

## Saving the output

The output is a dictionary of arrays and can be stored with NumPy:

```python
import numpy as np
np.savez("two_stream.npz", **output)
data = dict(np.load("two_stream.npz", allow_pickle=True))
```

Nested dictionaries (the parameter sections) are stored as object arrays, hence
`allow_pickle=True` when loading.

## Changing the physics

Everything about the run is controlled by the parameter tree. To see Landau damping
instead, remove the drift, lower the perturbation wavenumber to `1.02`, raise the
electron thermal speed and make the ions heavy: that is `examples/Landau_damping.py`.
To add a beam, add a second electron population with its own label: that is
`examples/bump-on-tail.toml`. The {doc}`../examples/index` describe each case and
the {doc}`../user_guide/species` page explains multiple populations.
