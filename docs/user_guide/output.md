# Output and diagnostics

`Simulation.run` returns a plain dictionary. It contains the time histories, the
particle bookkeeping, the derived quantities and a copy of every parameter section.
The tables below use `S` for `total_steps`, `N` for the total number of pseudo-particles
of all populations and `G` for `number_grid_points`.

## Time histories

| key | shape | unit | location |
|---|---|---|---|
| `positions` | `(S, N, 3)` | m | particle positions at integer times $t^n$ |
| `velocities` | `(S, N, 3)` | m/s | particle velocities at $t^n$ |
| `electric_field` | `(S, G, 3)` | V/m | cell faces $x_{i+1/2}$, all three components |
| `magnetic_field` | `(S, G, 3)` | T | cell centres $x_i$ |
| `current_density` | `(S, G, 3)` | A/m² | $J_x$ at cell faces, $J_y$, $J_z$ at cell centres |
| `charge_density` | `(S, G)` | C/m³ | cell centres |
| `time_array` | `(S,)` | s | `(arange(S) + 1) * dt` |

Entry `n` of each history is the state after step `n + 1`; the initial state is
available as `initial_positions`, `initial_velocities` and `fields`
(a tuple `(E, B)` of the initial fields). The first stored time is `dt`.
The velocities of the explicit scheme are
defined at integer times and the stored positions are the integer-time positions
reconstructed from the half-step ones, so the two are synchronous. The particle axis is
ordered by population in input order; `species_integer_index` tells which population
each particle belongs to.

## Particle bookkeeping

| key | shape | meaning |
|---|---|---|
| `charges` | `(N, 1)` | nominal marker charge, $q_s w_s$, including reserved source slots |
| `masses` | `(N, 1)` | nominal marker mass, $m_s w_s$ |
| `charge_to_mass_ratios` | `(N, 1)` | nominal $q_s/m_s$ |
| `weights` | `(N, 1)` | $w_s$ |
| `species_integer_index` | `(N,)` | population index in input order |
| `charge_integer_lookup`, `mass_integer_lookup`, `charge_mass_integer_lookup` | `(P,)` | per-population $q_s$, $m_s$, $q_s/m_s$ for the `P` populations |
| `number_pseudoelectrons` | int | pseudo-particles in the first electron population |

Active sources additionally return `masses_over_time`, `charges_over_time` and
`charge_to_mass_ratios_over_time`, each `(S, N, 1)`, plus `weights_over_time` and
`alive_particles`, each `(S, N)`. These contain the actual live state: unborn and
absorbed markers have zero mass, charge, weight and charge/mass ratio. A partial
wall return reduces the live weight. `source_birth_steps` is `(N,)`, with `-1` for
initial particles and the scheduled birth step for each reserved marker.

`injected_weight`, `injected_charge`, `injected_energy`, `lost_weight`, `lost_charge`,
`lost_energy`, `wall_energy_transfer` and `source_field_work` are cumulative `(S,)`
histories; `injected_momentum`, `lost_momentum` and `wall_momentum_transfer` are
`(S, 3)`. Injection records the prescribed birth state. Loss records the removed
weight's incoming kinetic energy and momentum at each physical half drift.
Wall transfer records the full before/after kinetic energy and momentum change,
including restitution and the impulse of reflected markers. `source_field_work`
records the self-field energy change from the Gauss projection at birth; it does
not include imposed-field work, outgoing electromagnetic energy, or time-integration
error. Particle weights are numbers per unit area, charge is C/m², energy J/m²
and momentum kg/(m s). These ledgers describe an open system; they do not by
themselves certify a closed total-energy budget.

`diagnostics` and `plot` use the live masses and weights when these histories are
present, so dormant and absorbed slots do not appear as extra cold particles.

## Grid, time step and derived quantities

| key | meaning |
|---|---|
| `grid` | cell centres, shape `(G,)` |
| `dx`, `dt`, `length`, `box_size` | cell size, time step, $L$, $(L, L_y, L_z)$ |
| `plasma_frequency` | $\omega_{pe} = \sqrt{n_e e^2/(\epsilon_0 m_e)}$ of the first electron population, rad/s |
| `max_initial_vth_electrons`, `vth_electrons_over_c` | largest thermal speed of the first electron population, in m/s and in units of $c$ |
| `charge_electrons` | charge of one physical electron of the first population, C |
| `external_electric_field`, `external_magnetic_field` | the arrays that were added to the fields, `(G, 3)` |
| `number_grid_points`, `total_steps` | copies of the inputs |

Every key of every parameter section is also copied to the top level (for example
`output["filter_passes"]`), and the sections themselves are available under
`domain_parameters`, `species_parameters`, `solver_parameters`,
`external_field_parameters`, `source_parameters` and `parameter_sections`.

## What `diagnostics` adds

{func}`jaxincell.diagnostics` post-processes the dictionary in place. It is not part of
the compiled run, so it can use NumPy and Python control flow.

Species split
: `position_electrons`, `velocity_electrons`, `mass_electrons`, `charge_electrons`
  (all particles with negative charge) and the same four keys with `_ions` (non-negative
  charge, which includes absorbed particles whose charge was set to zero). `species` is
  a list of dictionaries, one per distinct (charge, mass) pair, with `name`, `charge`,
  `mass`, `positions` and `velocities`. The combined `positions`, `velocities`, `masses`
  and `charges` arrays are deleted afterwards; if you need them, copy them first or do
  not call `diagnostics`.

Energies, all as functions of time with shape `(S,)`
: `electric_field_energy` $= \tfrac{\epsilon_0}{2}\sum_i |\mathbf E_i|^2 \Delta x$,
  `magnetic_field_energy` $= \tfrac{1}{2\mu_0}\sum_i |\mathbf B_i|^2 \Delta x$,
  the corresponding `external_*_energy` for the external arrays,
  `kinetic_energy_electrons`, `kinetic_energy_ions` and their sum `kinetic_energy`,
  computed as $\sum_p \tfrac12 m_p |\mathbf v_p|^2$ (non-relativistic), and
  `total_energy`, the sum of all of the above. The energy densities
  `electric_field_energy_density` and `magnetic_field_energy_density` have shape
  `(S, G)`. All energies are per unit area (J/m²) because the box is one-dimensional.

Dominant frequency
: `dominant_frequency` is the angular frequency of the largest peak in the power
  spectrum of $E_x$ at the centre cell, computed from the whole time series. Its
  resolution is $2\pi/(S\,\Delta t)$, which is coarse for short runs; for accurate
  frequencies fit the signal directly, as in the {doc}`../numerics/verification` page.

Charge and momentum conservation
: `gauss_error_Linf` is $\max_i|(E_{x,i} - E_{x,i-1})/\Delta x - \rho_i/\epsilon_0|$ at every
  step, and `gauss_error_Linf_rel` the same divided by $\max_i|\rho_i/\epsilon_0|$ (added
  when the output holds `charge_density`). `total_momentum`, shape `(S, 3)`, is
  $\sum_p m_p\mathbf v_p$ and `momentum_error_rel` is $|\mathbf P(t) - \mathbf P(0)|$ divided
  by $\sum_p m_p|\mathbf v_p(0)|$. {func}`jaxincell.plot` draws both relative errors on
  the energy panel, next to the relative energy error.

The relative energy error $|\mathcal E(t) - \mathcal E(0)|/\mathcal E(0)$ built from
`total_energy` is the standard check of a run. The explicit scheme is expected to
drift by $10^{-3}$ to $10^{-2}$ over hundreds of plasma periods, the implicit scheme
to stay at round-off. For the charge error it is the other way round: the explicit
scheme keeps Gauss's law at round-off (with periodic boundaries), the implicit one
does not, see {doc}`../numerics/implicit`.

## Memory

Every step is stored. The phase-space histories take $2 \times 3 \times 8$ bytes per
particle per step in double precision, so $10^5$ particles over $10^4$ steps need
48 GB. Reduce `total_steps`, the particle count, or run in chunks by building the
initial phase space of the next run from the last stored state through
`initial_positions` and `initial_velocities`.

## Saving

```python
import numpy as np
np.savez("run.npz", **output)
loaded = dict(np.load("run.npz", allow_pickle=True))
```

The nested parameter dictionaries become zero-dimensional object arrays; access them
with `loaded["domain_parameters"].item()`.
