# Diagnostics

{func}`~jaxincell.diagnostics` computes everything below from a stored
{class}`~jaxincell.Output`; the individual functions are also exported so that only
what is needed has to be computed. They are plain functions of the stored arrays, so
they can be recomputed at will and applied to a reloaded run.

```python
from jaxincell import diagnostics

d = diagnostics(output)
print(float(abs(d["total"][-1] / d["total"][0] - 1)))
```

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

The relative change of `total` is the single most informative number about a run. A
bounded oscillation of a few parts in $10^{4}$ is what the explicit scheme does; a
steady rise means a resolution problem ({doc}`stability`).

## Momentum

`momentum` is $\sum_p \gamma_p m_p\mathbf v_p$, shape `(steps, 3)`. In a periodic box
its $x$ component should stay put: over the two-stream run it drifts by
{{ momentum_error_relative }} of $\sum_p m_p|v_{x,p}|$ ({doc}`deposition`).

## Gauss-law residual

`gauss_residual` is the relative violation of {eq}`discrete-gauss` at every step,

```{math}
\max_i\left|\frac{E_{x,i+1/2}-E_{x,i-1/2}}{\Delta x} - \frac{\rho_i}{\epsilon_0}\right|
\Big/ \max_i\left|\frac{\rho_i}{\epsilon_0}\right| .
```

It should sit at round-off, {{ gauss_residual_max_explicit }}, for the whole run. It
will not if the current deposit is bypassed, which makes it a good regression check.

## Temperatures

{func}`~jaxincell.temperatures` returns, per species, the temperature per component in
electronvolts from the velocity variance about the mean,
$k_BT = m\,\mathrm{var}(v)$. Note the convention of {doc}`initialization`:
$v_{th} = \sqrt{2k_BT/m}$, so the variance is $v_{th}^2/2$ and a species initialised
with `vth` reports $T = m v_{th}^2/2k_B$.

## Dominant frequency

{func}`~jaxincell.dominant_frequency` takes the FFT of $E_x$ at the box centre and
returns the angular frequency of the strongest peak other than the mean, in rad/s. It
is a quick check that a Langmuir wave landed where it should; for a careful
measurement use the mode amplitude directly, as the verification scripts do:

```python
import numpy as np
amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, mode])
```

## What is not stored

Only what {meth}`~jaxincell.Simulation.run` was asked to keep. With
`store_particles=False` the particle history is dropped, so `kinetic`, `total`,
`momentum` and `temperatures` are absent from the dictionary and only the field
diagnostics are available — which is usually the right trade when the run is long,
since the particle history dominates the memory
({doc}`../user_guide/performance`).
