# The solver

{class}`~jaxincell.Solver` collects every numerical choice. All of its fields except
`filter_alpha` are static, so changing one recompiles the program.

```python
from jaxincell import Solver

solver = Solver(algorithm="explicit", model="electromagnetic", field_solver="ampere",
                filter_passes=2)
```

## Arguments

| argument | meaning | default |
|---|---|---|
| `algorithm` | `"explicit"` (Boris leapfrog) or `"implicit"` (energy- and charge-conserving Crank-Nicolson) | `"explicit"` |
| `model` | `"electromagnetic"` solves all six field components; `"electrostatic"` solves $\partial_x E_x = \rho/\epsilon_0$ alone | `"electromagnetic"` |
| `field_solver` | `"ampere"` advances $E_x$ from the current; `"gauss"` recomputes it from $\rho$ | `"ampere"` |
| `relativistic` | relativistic Boris pusher | `False` |
| `filter_passes` | binomial smoothing passes on the sources; `0` disables | `0` |
| `filter_alpha` | centre weight of the three-point filter (a pytree leaf) | `0.5` |
| `filter_strides` | cell offsets of the filter stencil | `(1,)` |
| `picard_iterations` | fixed-point iterations of the implicit scheme | `8` |
| `substeps` | particle sub-steps per field step, implicit only | `2` |

## Which field model

`"electrostatic"` solves $\partial_x E_x = \rho/\epsilon_0$ and nothing else.

* All three velocity components remain, and an external magnetic field acts as before.
  What goes is the plasma's own transverse field, not evolved, and the transverse
  currents, not deposited.
* Choose it whenever the physics is $\mathbf E = -\nabla\phi$: waves along the grid, beam
  instabilities, sheaths.
* It removes the light-wave time-step limit and is 1.19 times faster, measured at 200000
  particles on 256 cells on a CPU.
* The two models differ in one place: a periodic box has no wall to fix the constant of
  integration, so Ampere's law carries the mean field the net particle current drives,
  while the Gauss solve sets the mean to zero. Between walls they agree to round-off.
* A {class}`~jaxincell.Source` needs the electrostatic model, for the reason in
  {doc}`sources`.

## Which integrator

| `algorithm` | scheme | conserves | stability |
|---|---|---|---|
| `"explicit"` | second-order leapfrog with the Boris rotation; fast | energy error bounded rather than growing | $\omega_p\Delta t \lesssim 2$, $c\Delta t \le \Delta x$, $\Delta x \lesssim \lambda_D$ |
| `"implicit"` | Crank-Nicolson, relativistic or not; about {{ scaling_implicit_over_explicit }} times an explicit step | energy and the discrete Gauss law to round-off; gives up the exact momentum | unconditional |

Use the implicit scheme when the energy budget matters, when the step you want breaks an
explicit limit, or when the Debye length cannot be resolved.

```python
Solver(algorithm="implicit", picard_iterations=8, substeps=2)
```

* The Gauss law holds whatever the iteration count.
* The energy error falls geometrically with `picard_iterations` —
  {{ energy_error_max_implicit_1 }} at one, {{ energy_error_max_implicit_4 }} at four,
  {{ energy_error_max_implicit_8 }} at eight — so raise it first if the budget is not
  tight enough.
* `substeps` resolves orbits that turn inside one field step without refining the field
  grid.
* Both schemes are differentiable; {doc}`../numerics/implicit` explains why the iteration
  count is fixed rather than adaptive.

## Which field solver

| `field_solver` | what it does | when |
|---|---|---|
| `"ampere"` | advances $E_x$ in time with the charge-conserving current, keeping the discrete Gauss law satisfied to round-off with no elliptic solve | the default, and almost always the right choice |
| `"gauss"` | recomputes $E_x$ from the charge density every step | as a cross-check, and for problems posed as a charge distribution |

See {doc}`../numerics/field_solvers`.

## Filtering

`filter_passes` smoothing passes, each followed by an automatic compensation pass that
restores the long wavelengths.

```python
Solver(filter_passes=2, filter_strides=(1, 2, 4))   # cuts everything below ~20 cells
```

* It suppresses the grid-scale noise of a finite particle count and pushes back the
  finite-grid instability, at the price of damping genuinely short-wavelength physics.
* The verification runs use none: their modes are long compared with the cell and the
  quiet start already keeps the noise low. Turn it on for production runs with modest
  particle counts.
* {doc}`../numerics/filtering` has the transfer function.

:::{note}
`filter_passes=1` at the default `filter_alpha=0.5` is very nearly the identity: one
pass and its own compensation. Use `0` to disable, `2` or more for an effect.
:::

## Relativity

`relativistic=True` switches the pusher to act on $\mathbf p = \gamma m\mathbf v$ and
makes the kinetic-energy diagnostic use $(\gamma-1)mc^2$ automatically.

* The loop carries the momentum per unit mass $\mathbf u = \gamma\mathbf v$, so
  $\gamma = \sqrt{1 + u^2/c^2}$ never has to be recovered from $1 - v^2/c^2$, which loses
  a fraction $\gamma^2\epsilon$ of it at every conversion: converting each step, a
  single-precision particle at $\gamma = 1000$ drifted to 1423 in a thousand field-free
  steps.
* `Output.v` still holds velocities; `Output.state` holds $\mathbf u$, and passing it
  back to a relativistic run continues it.
* Velocities enter as velocities, from `drift`, `vth` or `Species.v`, and are converted
  once.
* A drawn velocity can reach or pass $c$ — a Maxwellian tail sampled as if Newtonian, or
  a drift too close to light. A speed with $v^2/c^2 > 1 - 10^{-5}$ is brought back to
  that speed along its own direction, capping the initial $\gamma$ at 316: about a
  hundred times the resolution of single precision, where $\gamma$ is still known to
  1 %. Keep the thermal spread well below $c$ for the Maxwellian sampling to mean
  anything.
* A Newtonian run has no speed limit and leaves velocities alone.
* At a wall, restitution scales the normal component of $\mathbf u$. The deposit, the
  field solve and the boundaries are unchanged, since they act on positions and
  velocities.
