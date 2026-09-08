# The solver

{class}`~jaxincell.Solver` collects every numerical choice. All of its fields except
`filter_alpha` are static, so changing one recompiles the program.

```python
from jaxincell import Solver

solver = Solver(algorithm="explicit", field_solver="ampere", filter_passes=2)
```

| argument | meaning | default |
|---|---|---|
| `algorithm` | `"explicit"` (Boris leapfrog) or `"implicit"` (Crank-Nicolson) | `"explicit"` |
| `field_solver` | `"ampere"` advances $E_x$ from the current; `"gauss"` recomputes it from $\rho$ | `"ampere"` |
| `relativistic` | relativistic Boris pusher | `False` |
| `filter_passes` | binomial smoothing passes on the sources; `0` disables | `0` |
| `filter_alpha` | centre weight of the three-point filter (a pytree leaf) | `0.5` |
| `filter_strides` | cell offsets of the filter stencil | `(1,)` |
| `picard_iterations` | fixed-point iterations of the implicit scheme | `8` |
| `substeps` | particle sub-steps per field step, implicit only | `2` |

## Which integrator

`"explicit"` is a second-order leapfrog with the Boris rotation. It is fast, its energy
error is bounded rather than growing, and it is stable while
$\omega_p\Delta t \lesssim 2$, $c\Delta t \le \Delta x$ and
$\Delta x \lesssim \lambda_D$.

`"implicit"` is the energy-conserving Crank-Nicolson scheme of Chen, Chacón and
Barnes. It is unconditionally stable and drives the energy error to round-off, at
about {{ scaling_implicit_over_explicit }} times the cost per step. Use it when the
energy budget matters, when the step you want breaks an explicit limit, or when the
Debye length cannot be resolved.

```python
Solver(algorithm="implicit", picard_iterations=8, substeps=2)
```

The energy error falls geometrically with `picard_iterations` —
{{ energy_error_max_implicit_1 }} at one, {{ energy_error_max_implicit_4 }} at four,
{{ energy_error_max_implicit_8 }} at eight — so raising it is the first thing to try
if the budget is not tight enough. `substeps` resolves orbits that turn inside one
field step without refining the field grid. Both schemes are differentiable;
{doc}`../numerics/implicit` explains why the iteration count is fixed rather than
adaptive.

## Which field solver

`"ampere"` advances $E_x$ in time with the charge-conserving current, which keeps the
discrete Gauss law satisfied to round-off with no elliptic solve. That is the default
and almost always the right choice.

`"gauss"` recomputes $E_x$ from the charge density every step. It is useful as a
cross-check and for problems posed as a charge distribution. See
{doc}`../numerics/field_solvers`.

## Filtering

`filter_passes` smoothing passes, each followed by an automatic compensation pass that
restores the long wavelengths. It suppresses the grid-scale noise of a finite particle
count and pushes back the finite-grid instability, at the price of damping genuinely
short-wavelength physics.

```python
Solver(filter_passes=2, filter_strides=(1, 2, 4))   # cuts everything below ~20 cells
```

:::{note}
`filter_passes=1` at the default `filter_alpha=0.5` is very nearly the identity: one
pass and its own compensation. Use `0` to disable, `2` or more for an effect.
:::

The verification runs use no filtering at all, because the modes they measure are long
compared with the cell and the quiet start already keeps the noise low. Turn it on for
production runs with modest particle counts. {doc}`../numerics/filtering` has the
transfer function.

## Relativity

`relativistic=True` switches the pusher to act on $\mathbf p = \gamma m\mathbf v$ and
makes the kinetic-energy diagnostic use $(\gamma-1)mc^2$ automatically. Initial
velocities are clipped to $0.99c$ so that $\gamma$ is finite. Everything else — the
deposit, the field solve, the boundaries — is unchanged, since they act on positions
and velocities.
