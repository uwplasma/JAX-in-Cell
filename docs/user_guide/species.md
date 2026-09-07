# Species

The `species_parameters` section holds one entry per population. There are two types,
`electrons` and `ions`; each type can contain any number of labelled populations. The
type fixes the mass model (electron mass, or a multiple of the proton mass) and the
default sign of the charge. Everything else is per population.

```toml
[species_parameters.electrons.bulk]
number_pseudoparticles = 12000
vth_over_c_x = 0.0707

[species_parameters.electrons.beam]
number_pseudoparticles = 12000
grid_points_per_Debye_length = 0.444    # 3 % of the bulk density, see below
drift_speed_x = 7.5e7

[species_parameters.ions.protons]
number_pseudoparticles = 12000
vth_over_c_x = "_electrons0"
```

## Parameters common to all populations

| parameter | default | differentiable | meaning |
|---|---|---|---|
| `number_pseudoparticles` | `500` | no | Number of pseudo-particles $N_s$. |
| `grid_points_per_Debye_length` | `2` | yes | $\Delta x/\lambda_{D}$ evaluated with this population's density and the reference electron temperature. Sets the density, see below. |
| `weight` | `0` | yes | Number of physical particles per pseudo-particle, $w_s$. `0` means "compute from `grid_points_per_Debye_length`". |
| `charge_over_elementary_charge` | `-1` (electrons), `1` (ions) | yes | Charge $q_s/e$. |
| `vth_over_c_x`, `vth_over_c_y`, `vth_over_c_z` | `0` | yes | Thermal speed per component, $v_{th}/c$, with $f \propto \exp(-v^2/v_{th}^2)$. A string value refers to another population, see below. |
| `drift_speed_x`, `drift_speed_y`, `drift_speed_z` | `0` | yes | Drift velocity per component in m/s. |
| `velocity_plus_minus_x`, `_y`, `_z` | `false` | no | If true, every second particle has its velocity component negated, which turns one drifting population into two counter-streaming beams of half density each. |
| `perturbation_amplitude_x`, `_y`, `_z` | `0.0` | yes | Amplitude $a$ (metres) of a sinusoidal displacement $x \to x + a\sin(k x)$. |
| `perturbation_wavenumber_x`, `_y`, `_z` | `0` | yes | Mode number $m$ of the displacement, $k = 2\pi m/L$. |
| `random_positions_x` | `false` | no | Uniform random positions instead of equally spaced ones. |
| `random_positions_y`, `random_positions_z` | `true` | no | Same for $y$ and $z$. |
| `seed_position_override`, `seed_position` | `false`, `None` | no | Use `seed_position` as the position seed of this population instead of the derived one. |
| `initial_positions`, `initial_velocities` | `None` | yes | Arrays of shape `(number_pseudoparticles, 3)` that replace the generated phase space entirely. |

Ions have four more:

| parameter | default | differentiable | meaning |
|---|---|---|---|
| `mass_over_proton_mass` | `1` | yes | $m_s/m_p$. Electrons always have mass $m_e$. |
| `ion_temperature_over_electron_temperature_x`, `_y`, `_z` | `1` | yes | $T_i/T_e$ per component; used only when `vth_over_c_*` is a reference to an electron population. |

### Defaults of the first populations

The first electron population and the first ion population (`_electrons0`, `_ions0`)
start from a different set of defaults, chosen so that `Simulation()` with no arguments
runs a two-stream instability:

| parameter | first electron population | first ion population |
|---|---|---|
| `perturbation_amplitude_x` | `1e-7` | `1e-7` |
| `perturbation_wavenumber_x` | `8` | `0` |
| `vth_over_c_x` | `0.05` | `"_electrons0"` |
| `vth_over_c_y`, `vth_over_c_z` | `0` | `"_electrons0"` |
| `drift_speed_x` | `1e8` | `0` |
| `velocity_plus_minus_x` | `true` | `false` |

Any further population uses the table above (cold, at rest, unperturbed).

## Density and pseudo-particle weight

The code has no density parameter. Instead, the electron Debye length is prescribed in
units of the cell size, and the weight follows. Let $v_{th,e}$ be the largest of the
three thermal speeds of the first electron population and $q_e$ its charge. For a
population $s$ with $N_s$ pseudo-particles and $g_s$ = `grid_points_per_Debye_length`,

```{math}
w_s = \frac{\epsilon_0\, m_e c^2}{q_e^2}\,
      \frac{N_x^2\, g_s^2}{2\, L\, N_s}\left(\frac{v_{th,e}}{c}\right)^2 ,
\qquad
n_s = \frac{N_s w_s}{L} = \frac{\epsilon_0 m_e v_{th,e}^2}{2 q_e^2 \lambda_{D,s}^2},
\quad \lambda_{D,s} = \frac{\Delta x}{g_s}.
```

In words: $g_s$ is the number of grid points per Debye length that a plasma of density
$n_s$ and temperature $k_B T_e = m_e v_{th,e}^2/2$ would have. For the first electron
population this is exactly the Debye length of the run. For any other population it is
a convenient way to set a density ratio: because $n_s \propto g_s^2$, a beam with 3 % of
the bulk density uses $g_{beam} = \sqrt{0.03}\, g_{bulk}$, which is what
`examples/bump-on-tail.toml` does. Setting `weight` to a positive number bypasses the
formula.

Charge neutrality is not enforced. With equal $N$ and equal $g$ for electrons and
singly charged ions, the densities match; otherwise check that
$\sum_s q_s n_s = 0$ yourself, or expect a uniform background field to build up.

The quantities printed at the start of a run with `print_info = true` (density,
temperature, Debye length, plasma frequency, particles per cell) all refer to the first
electron population.

## Thermal speeds and temperature ratios

Each velocity component is drawn from a normal distribution with standard deviation
$v_{th}/\sqrt{2}$, then the drift is added, then the sign is flipped for every second
particle if `velocity_plus_minus` is set. With $v_{th} = \sqrt{2 k_B T/m}$ this gives a
Maxwellian of temperature $T$ in that component. Different values per component produce
a bi-Maxwellian, which is how the Weibel example sets up its anisotropy.

A string value for `vth_over_c_*` names another population, using its canonical label.
For an ion population referring to electrons the thermal speed becomes

```{math}
v_{th,i} = v_{th,e}\sqrt{\frac{T_i}{T_e}}\sqrt{\frac{m_e}{m_i}},
```

with the temperature ratio taken from `ion_temperature_over_electron_temperature_*`
of the ion population. An electron population may refer to an ion population in the
same way (the inverse formula is used). A referenced value must itself be a number;
chains of references are rejected. Because `vth_over_c_x` of the first electron
population defaults to `0.05`, the common pattern `"vth_over_c_x": "_electrons0"` for
ions works without further input.

After initialisation every velocity component is clipped to $\pm 0.99c$.

## Positions and perturbations

Positions are equally spaced over $[-L/2, L/2]$ unless `random_positions_x` is true, in
which case they are uniform random numbers. The displacement
$x \to x + a\sin(2\pi m x/L)$ then imposes a density perturbation
$\delta n/n = -a k\cos(kx)$ to first order in $ak$. For a linear-theory test keep
$ak \ll 1$; for a strong perturbation note that the sampling noise of $N$ particles per
mode is of order $1/\sqrt{N}$. The same displacement is applied in $y$ and $z$ with
their own amplitude and mode number, but nothing depends on $y$ or $z$.

## Random seeds

All randomness comes from `jax.random` keys derived from `solver_parameters.seed`.
The first electron population uses `seed` for positions and `seed + 3` for velocities;
the first ion population `seed` and `seed + 6`; every additional population gets a seed
of its own derived from its position in the input. Two runs with the same parameters
are bit-for-bit reproducible on the same hardware. `seed_position_override` lets two
populations share the same random positions, which `examples/bump-on-tail.toml` uses so
that the beam electrons and their neutralising ions start at the same places.

## Supplying the phase space directly

`initial_positions` and `initial_velocities` accept arrays of shape `(N_s, 3)` in SI
units. They replace the generated phase space after the weight has been computed, so
`grid_points_per_Debye_length` and `vth_over_c_*` still set the density and are still
printed; make sure the supplied velocities are consistent with them if you rely on the
derived quantities. Both arrays are differentiable inputs, which allows gradients with
respect to the full initial condition. A quiet start built this way is used in
{doc}`../numerics/verification` to measure Landau damping with a low noise floor.

## Accessing per-species output

After {func}`jaxincell.diagnostics`, `output["species"]` is a list with one entry per
distinct (charge, mass) pair, each holding `positions` and `velocities` of shape
`(steps, N, 3)`; two populations with the same charge and mass (such as a bulk and a
beam of electrons) are merged into one entry. To separate them use
`output["species_integer_index"]`, an integer per pseudo-particle in input order
(`_electrons0`, `_electrons1`, ..., `_ions0`, ...), together with `output["weights"]`,
`output["charge_integer_lookup"]` and `output["mass_integer_lookup"]`.
