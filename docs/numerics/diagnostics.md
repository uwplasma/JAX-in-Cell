# Diagnostics

{func}`~jaxincell.diagnostics` computes everything below from a stored
{class}`~jaxincell.Output`; the individual functions are also exported so that only
what is needed has to be computed. They are plain functions of the stored arrays, so
they can be recomputed at will and applied to a reloaded run.

```python
from jaxincell import diagnostics

d = diagnostics(output)
print(float(d["energy_error"].max()), float(d["momentum_error"].max()), float(d["gauss_residual"].max()))
```

The three errors of the conservation laws are relative, each to a scale that does not vanish
for the plasmas a particle-in-cell code is used on, and all three are measured from the
first stored step; {doc}`../examples/conservation` plots them for both schemes.

## Energies

Per unit area, in J/m², at every stored step:

```{math}
W_E = \frac{\epsilon_0}{2}\sum_i |\mathbf E_i|^2\Delta x, \qquad
W_B = \frac{1}{2\mu_0}\sum_i |\mathbf B_i|^2\Delta x, \qquad
W_P = \sum_p \frac{m_p |\mathbf v_p|^2}{2},
```

as `electric`, `magnetic` and `kinetic`, with `kinetic_<name>` per species and `total`
their sum. With `Solver(relativistic=True)` the kinetic energy becomes
$\sum_p (\gamma_p - 1)m_pc^2$, which is the quantity the relativistic pusher conserves;
the diagnostic follows the solver automatically.

`energy_error` is $|W(t) - W(0)|/W(0)$ for $W$ = `total`, the single most informative
number about a run. A bounded oscillation of a few parts in $10^{4}$ is what the explicit
scheme does and round-off what the implicit one does; a steady rise means a resolution
problem ({doc}`stability`). Walls that collect, re-emit or slow particles change the energy
physically.

## Momentum

`momentum` is $\mathbf P = \sum_p \gamma_p m_p\mathbf v_p$, shape `(steps, 3)`, the momentum
the pusher conserves, and `momentum_error` is

```{math}
|\mathbf P(t) - \mathbf P(0)| \Big/ \sum_p \gamma_p m_p|\mathbf v_p| \text{ at } t = 0 ,
```

relative to the sum of the magnitudes of the particle momenta rather than to the total,
which vanishes for two counter-streaming beams. In a periodic electrostatic run it should
stay small: over the two-stream run it reaches {{ momentum_error_relative }} with the
explicit scheme, whose gather makes the forces between particles antisymmetric
({doc}`deposition`), and {{ momentum_error_implicit }} with the implicit one, which gives
that up for the energy ({doc}`implicit`).

## Gauss-law residual

`gauss_residual` is the error in the conservation of charge, the violation of
{eq}`discrete-gauss` at every step relative to the density of one sign of charge,

```{math}
\max_i\left|\frac{E_{x,i+1/2}-E_{x,i-1/2}}{\Delta x} - \frac{\rho_i}{\epsilon_0}\right|
\Big/ \frac{e n}{\epsilon_0}, \qquad
e n = \frac1L\max\Big(\sum_{q_p>0} q_p w_p,\ \sum_{q_p<0} |q_p| w_p\Big),
```

from the weights of that step, or of the final state when the particles were not stored.
The net density is no scale: in a neutral plasma it is the particle noise, which a quiet
start makes as small as it likes, so a residual relative to $\max_i|\rho_i|$ measures the
start as much as the solver. $E_{-1/2}$ is taken as the solver takes it: the far end of the
box for a periodic wall, zero otherwise. Measuring it as periodic regardless reports a
violation in the first cell that the solver never committed, which at an absorbing wall —
where the field at the far end is the sheath field and nowhere near zero — can be larger
than the density itself.

It should sit at round-off for the whole run, at every wall type and in both schemes:
{{ gauss_residual_max_explicit }} explicit and {{ gauss_residual_max_implicit }} implicit
over the two-stream run. It will not if the current deposit is bypassed, which makes it a
good regression check.

## Potential

{func}`~jaxincell.potential` integrates the longitudinal field from the left wall with
the trapezoidal rule over the faces,
$\phi_{i+1/2} = -\Delta x\sum_{j\le i} (E_{x,j-1/2} + E_{x,j+1/2})/2$, so entry $i$ is
the potential relative to that wall and the last entry is the potential of the right
wall. The field at the left wall face is not stored: it is the field at the far end of a
periodic box, zero at a reflective wall, and at an absorbing wall the field of the
collected charge, $E_{x,1/2} - \Delta x\,\rho_0/\epsilon_0$ from the Gauss law of the
first cell. The field solver closes two absorbing walls with the same rule
({doc}`field_solvers`), so between those short-circuited conductors the last entry stays
at zero to round-off and the bulk floats above both — see {doc}`../examples/sheath_reflection`. A
periodic box has no wall, so the mean is set to zero instead.

## Temperatures

{func}`~jaxincell.temperatures` returns, per species, the temperature per component in
electronvolts from the velocity variance about the mean,
$k_BT = m\,\mathrm{var}(v)$. Note the convention of {doc}`initialization`:
$v_{th} = \sqrt{2k_BT/m}$, so the variance is $v_{th}^2/2$ and a species initialised
with `vth` reports $T = m v_{th}^2/2k_B$.

## Dominant frequency

{func}`~jaxincell.dominant_frequency` takes the FFT of $E_x$ at the box centre and
returns the angular frequency of the strongest peak other than the mean, in rad/s, and
NaN for a run that stored fewer than two steps, which have no frequency. It
is a quick check that a Langmuir wave landed where it should; for a careful
measurement use the mode amplitude directly, as the verification scripts do:

```python
import numpy as np
amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, mode])
```

## What is not stored

Only what {meth}`~jaxincell.Simulation.run` was asked to keep. With
`store_particles=False` the particle history is dropped, so `kinetic`, `total`,
`momentum`, their errors and `temperatures` are absent from the dictionary and only the field
diagnostics are available — which is usually the right trade when the run is long,
since the particle history dominates the memory
({doc}`../user_guide/performance`).
