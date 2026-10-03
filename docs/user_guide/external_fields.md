# External fields and sources

## External fields

A static external electric or magnetic field can be added to the self-consistent
fields felt by the particles. It is supplied as an array with one row per cell in the
`external_field_parameters` section:

```python
import numpy as np

G = 70
B_external = np.zeros((G, 3))
B_external[:, 0] = 0.1          # 0.1 T along x, uniform

parameters["external_field_parameters"] = {
    "external_magnetic_field": {"B": B_external},
    # "external_electric_field": {"E": E_external},   # same shape, V/m
}
```

The arrays have shape `(number_grid_points, 3)` and are interpreted on the same
staggered locations as the self-consistent fields (electric field at cell faces,
magnetic field at cell centres). They are constant in time, are added to $\mathbf E$
and $\mathbf B$ before the fields are interpolated to the particles, and do not enter
Maxwell's equations. The external field energies are reported separately by
{func}`jaxincell.diagnostics`. The arrays are stored in single precision.

```{warning}
The scalar parameters `external_electric_field_amplitude`,
`external_electric_field_wavenumber`, `external_magnetic_field_amplitude`,
`external_magnetic_field_wavenumber`, `external_electric_field_function` and
`external_magnetic_field_function` are accepted and validated, but on the `main`
branch they do not create a field. The electric-field amplitude only appears in the
`print_info` summary as the normalised field strength
$-q_e E_0 \lambda_D / k_B T_e$. Use the array form above to apply an external field.
```

| parameter | default | effect on `main` |
|---|---|---|
| `external_electric_field` | absent | `{"E": array}` adds the array to $\mathbf E$ at every step. |
| `external_magnetic_field` | absent | `{"B": array}` adds the array to $\mathbf B$ at every step. |
| `external_electric_field_amplitude` | `0.0` | Printed only. |
| `external_electric_field_wavenumber` | `0.0` | None. |
| `external_magnetic_field_amplitude` | `0.0` | None. |
| `external_magnetic_field_wavenumber` | `0.0` | None. |
| `external_electric_field_function` | `None` | None. |
| `external_magnetic_field_function` | `None` | None. |

None of these are differentiable inputs.

A uniform magnetic field along $x$ is the simplest way to study magnetised plasma
waves: particles gyrate in the $y$-$z$ plane while the fields remain functions of $x$
only. Keep $c\,\Delta t/\Delta x \le 1$ in that case, because the transverse currents
excite electromagnetic waves, and resolve the gyration with
$\Omega_c \Delta t \ll 1$, where $\Omega_c = |q| B / m$.

## Sources

The `source_parameters` section describes particle injection: which populations are
sourced, how often, at what rate, where in the box and with what velocity.

| parameter | default |
|---|---|
| `source_term_active` | `0` |
| `source_species` | `1` |
| `how_often_source_should_produce_quasiparticles` | `20` |
| `source_particles_per_second` | `1e16` |
| `location_of_source` | `0` (`0` centre, `1` left, `2` right, `3` whole box) |
| `width_of_source` | `1` |
| `injection_speed_x`, `injection_speed_y`, `injection_speed_z` | `1e7`, `0`, `0` |

Set `source_term_active = 1` to inject markers with the nonrelativistic Boris
integrator and `field_solver = 2` (Cartesian Gauss). Other integrators and field
solvers reject active sources: their charge-creation current is not implemented.
The electromagnetic transverse update still requires the explicit light-wave
Courant limit. Source parameters are static configuration values.

`source_species` indexes the same ordered populations as `species_integer_index`:
all named electron populations followed by all named ion populations. Each source
parameter accepts a scalar or a tuple matching `source_species`; separate sources
may target the same population. Injection velocities specify all three components,
with norm below $c$.

A batch is born at the beginning of steps `0, cadence, 2*cadence, ...`, including
the last batch before `total_steps`. All slots are reserved before compilation;
unborn slots have zero live charge, mass and velocity. Birth positions are grid
centres with $y=z=0$. Left/right sources occupy `width_of_source` centres, whole-box
sources occupy every centre. A centred source with mismatched grid/width parity
uses one extra centre and half-weight endcaps to retain symmetry.
`source_particles_per_second` is the rate **per grid site per unit area**. Each
marker carries `rate * cadence * dt`, multiplied by its endcap factor. Thus the
effective total rate is `width * rate` (or `G * rate` for the whole box).
Batches retain their full cadence weight, including the last partial run interval;
approximating a continuous source requires cadence and timestep refinement.

Gauss's law is recomputed immediately after birth and at the end of every step.
Periodic fields use a uniform compensating background for net charge; use matched
electron/ion sources when that background is unwanted. Wall fields retain the
Cartesian solver's zero left-face field convention; this is not a collector/sheath
boundary model. Births exchange energy and momentum with an external reservoir,
so total simulation energy is not a closed-system invariant. See the separate
birth, wall-loss and field-projection histories in {doc}`output`.

For example, add matched sources to an existing two-population parameter tree:

```python
parameters["solver_parameters"].update(field_solver=2, relativistic=False,
                                       time_evolution_algorithm=0)
parameters["source_parameters"] = dict(
    source_term_active=1, source_species=(0, 1),
    how_often_source_should_produce_quasiparticles=5,
    source_particles_per_second=1e12, location_of_source=0, width_of_source=1,
    injection_speed_x=1e7, injection_speed_y=0., injection_speed_z=0.,
)
output = Simulation(parameters).run()
```

A standalone small control is available from the repository root:

```bash
MPLBACKEND=Agg python examples/source_particles.py
```

It injects co-moving electron/ion batches into the centre of the box, returns half
of each marker at the right wall with normal restitution `0.8`, and writes
`source_particles.png`. The matched positions, rates and velocities keep the
charge density and self-fields zero. The script checks live plus collected weight
against injected weight, and live kinetic energy/momentum plus wall transfer against
the injected totals. The plotted wall energy includes both collection and restitution
loss; `lost_energy` alone excludes dissipation of the returned fraction. This is a
near-ballistic reservoir control rather than a plasma sheath model. Increasing the
cadence changes batch sizes as well as injection times; reduce cadence and time step
together when approximating a continuous source.
