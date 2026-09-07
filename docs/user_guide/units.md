# Units and normalisations

JAX-in-Cell works in SI units throughout. Lengths are in metres, times in seconds,
velocities in metres per second, charge in coulombs, mass in kilograms, electric field
in volts per metre, magnetic field in tesla, charge density in coulombs per cubic
metre and current density in amperes per square metre. Energies are per unit area
(J/m²) because the box is one-dimensional in space.

The physical constants are available as module attributes:

| name | value |
|---|---|
| `jaxincell.epsilon_0` | $8.854\,187\,82\times10^{-12}$ F/m |
| `jaxincell.mu_0` | $1.256\,637\,06\times10^{-6}$ H/m |
| `jaxincell.speed_of_light` | $2.997\,924\,58\times10^{8}$ m/s |
| `jaxincell.elementary_charge` | $1.602\,176\,63\times10^{-19}$ C |
| `jaxincell.mass_electron` | $9.109\,383\,71\times10^{-31}$ kg |
| `jaxincell.mass_proton` | $1.672\,621\,93\times10^{-27}$ kg |
| `jaxincell.boltzmann_constant` | $1.380\,649\times10^{-23}$ J/K |

## Inputs that are dimensionless

Several inputs are ratios so that a configuration can be scaled without recomputing
densities and fields:

| input | definition |
|---|---|
| `vth_over_c_*` | $v_{th}/c$ with $v_{th} = \sqrt{2 k_B T/m}$ |
| `grid_points_per_Debye_length` | $\Delta x/\lambda_D$ |
| `timestep_over_spatialstep_times_c` | $c\,\Delta t/\Delta x$ |
| `perturbation_wavenumber_*` | mode number $m$, $k = 2\pi m/L$ |
| `charge_over_elementary_charge`, `mass_over_proton_mass` | $q/e$, $m/m_p$ |
| `ion_temperature_over_electron_temperature_*` | $T_i/T_e$ |

The density follows from the Debye length, see {doc}`species`. The perturbation
amplitude and the drift speeds are dimensional (metres, metres per second).

## Derived quantities

For the first electron population, with $n_e$ its density, $T_e$ its temperature from
the largest thermal speed, and $\lambda_D = \Delta x/g$:

```{math}
\omega_{pe} = \sqrt{\frac{n_e e^2}{\epsilon_0 m_e}}, \qquad
\lambda_D = \frac{v_{th,e}}{\sqrt 2\,\omega_{pe}} = \sqrt{\frac{\epsilon_0 k_B T_e}{n_e e^2}}, \qquad
d_e = \frac{c}{\omega_{pe}}, \qquad
k_B T_e = \frac{m_e v_{th,e}^2}{2}.
```

The summary printed at the start of a run with `print_info = true` lists $L/\lambda_D$,
$L/d_e$, $n_e$, $k_B T_e$ in eV, $T_i/T_e$, $\lambda_D$, $d_e$, the number of
pseudo-particles per cell, the weight, $1/(\omega_{pe}\Delta t)$, the total simulated
time in units of $\omega_{pe}^{-1}$, $n_e\lambda_D^3$, the maximum and mean Lorentz
factor of the initial velocities and the normalised external electric field
$-q_e E_0\lambda_D/k_B T_e$. The line labelled `Wavenumber * Debye length` prints the
mode number times $\lambda_D$ in metres; multiply by $2\pi/L$ to obtain $k\lambda_D$.

## Converting a physical problem

To simulate a plasma with density $n$ and temperature $T$:

1. compute $v_{th} = \sqrt{2 k_B T/m_e}$ and set `vth_over_c_x = v_th / c`;
2. compute $\lambda_D$ and choose the cell size, for example $\Delta x = \lambda_D/2$,
   which gives `grid_points_per_Debye_length = 2`;
3. choose the box in Debye lengths and set `length = L` in metres and
   `number_grid_points = L / dx`;
4. choose the time step from $\omega_{pe}\Delta t$ and set
   `timestep_over_spatialstep_times_c = c dt / dx`.

The density that results is the one you started from, because the weight formula
inverts the Debye-length relation. Check the printed summary against your numbers.

## Time axis of the output

`time_array` is in seconds. Multiply by `plasma_frequency` for times in units of
$\omega_{pe}^{-1}$, which is what all figures in this documentation use.
