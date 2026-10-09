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
branch they do not create a field. Setting a non-zero amplitude or a field function
raises a `UserWarning` at construction saying so, because the run would otherwise
proceed silently with no external field. The electric-field amplitude only appears in
the `print_info` summary as the normalised field strength
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

The section is validated (lengths of the per-source tuples must match
`source_species`) and copied into the output, but no code path on `main` creates
particles from it. Setting `source_term_active = 1` raises a `UserWarning` saying that
no particles will be injected. The implementation lives on the `ds/source_particles`
branch of the repository. Leave the section out, or keep `source_term_active = 0`.
