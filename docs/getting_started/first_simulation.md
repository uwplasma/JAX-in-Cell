# Your first simulation, in detail

This page takes the two-stream run of `examples/1_basic/two_stream.py` apart and explains
every number in it. Two electron beams stream through one another at $\pm v_0$ on a
background of protons; each beam Landau-resonates with the space-charge wave carried by
the other, and the pair is unstable until the beams trap each other and the phase space
folds into a vortex. It is the standard first test of a kinetic code, because the growth
rate is known in closed form {cite}`buneman1959`.

```{figure} ../_static/figures/two_stream.png
:width: 100%
:alt: Growth of the seeded two-stream mode and the electron phase space after saturation

What this run produces. (a) $|E_{k=1}(t)|$ growing exponentially, with the fitted rate
(dashed) and the kinetic rate (dotted). (b) The electron phase space after saturation.
{doc}`../examples/two_stream` measures it.
```

## Choosing the parameters

```python
import numpy as np
from jaxincell import (Domain, Simulation, Solver, Species, diagnostics, plot,
                       epsilon_0, mass_electron, elementary_charge as e, speed_of_light as c)

length, cells = 0.01, 64
density, drift, vth = 4.37e17, 6e7, 0.05 * c
```

Everything else follows from those five numbers.

| choice | what it fixes |
|---|---|
| **density** | $\omega_{pe} = \sqrt{n e^2/\epsilon_0 m_e}$, the clock of the problem. At $4.37\times10^{17}\,\mathrm{m^{-3}}$ it is $3.7\times10^{10}$ rad/s, so a plasma period is 0.17 ns and the interesting physics takes a few tens of them. |
| **thermal speed** | $\lambda_D = v_{th}/(\sqrt2\,\omega_{pe}) = 2.8\times10^{-4}$ m. The cell must resolve it, or the explicit scheme heats itself through the finite-grid instability. Here $\Delta x/\lambda_D = {{ two_stream_dx_over_debye }}$ — comfortably below one. |
| **box length** | which mode is seeded. Mode 1 has $k = 2\pi/L$, and the beams are most unstable near $kv_0/\omega_{pe} = \sqrt{3/8}$. Solving for $v_0$ at fixed $L$ and $n$ is how the drift was picked; at $v_0 = 6\times10^7$ m/s the mode sits at $kv_0/\omega_{pe} = 1.01$, just past the peak — a deliberately unexciting choice for a first run. |
| **time step** | `dt_over_dx_c=4.5` gives $\omega_{pe}\Delta t = {{ two_stream_omega_pe_dt }}$, well inside the accuracy limit of 0.2. |

`dt_over_dx_c=4.5` is above the light-wave Courant limit of one, which is allowed here
only because a purely electrostatic run never excites the transverse fields — see
{doc}`../numerics/stability`. Adding a magnetic field, an isotropic temperature or
collisions would make this choice diverge.

## Building it

```python
electrons = Species.electrons(n=8000, density=density, vth=(vth, 0, 0),
                              drift=(drift, 0, 0), plus_minus=True,
                              perturbation_amplitude=5e-7, perturbation_mode=1)
ions = Species.ions(n=8000, density=density, electrons=electrons)

simulation = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=4.5),
                        [electrons, ions], Solver(filter_passes=2))
```

* `plus_minus=True` negates $v_x$ on every second particle, so one population becomes two
  counter-streaming beams of density $n/2$ each.
* `Species.ions` derives the proton thermal speed from the electrons, so the two start in
  thermal equilibrium.
* `perturbation_amplitude` is a **displacement in metres**, not a density:
  $5\times10^{-7}$ m over a 0.01 m box with $k = 628\ \mathrm{m^{-1}}$ makes
  $ak = 3\times10^{-4}$, a small enough seed for the linear phase to be several
  e-foldings long.
* Eight thousand pseudo-particles per species is 125 per cell — enough to see the
  instability, not enough to measure its rate precisely. The verification runs use five
  times more and a quiet start ({doc}`../numerics/verification`).
* `filter_passes=2` smooths the deposited sources, suppressing the grid-scale noise of a
  modest particle count. It also damps genuinely short-wavelength physics, so it is off in
  the runs that measure rates.

## Checking before running

```python
print(f"omega_pe dt   = {float(simulation.plasma_frequency() * simulation.domain.dt):.3f}")
print(f"dx / lambda_D = {float(simulation.domain.dx / simulation.debye_length()):.2f}")
```

Two numbers, both of which should be below one. Printing them costs nothing and catches
most bad runs before they start.

## Running and reading the result

```python
output = simulation.run(1200, seed=0, store_every=2)

energy = diagnostics(output)
print(f"energy drift {abs(float(energy['total'][-1] / energy['total'][0]) - 1):.2e}")
print(f"electric energy grew by {float(energy['electric'].max() / energy['electric'][0]):.3g}")
plot(output, omega=float(simulation.plasma_frequency()))
```

* The first call compiles for a second or two, then runs.
* `store_every=2` halves the stored history. 1200 steps of 16 000 particles at every step
  would be 460 MB, which is fine; the same run at 100 000 particles would not be. The
  memory formula is in {doc}`../user_guide/running`.
* The **energy drift** is the health check: a few parts in $10^5$ means the resolution is
  fine, and a steady rise would mean it is not.
* The **electric energy growing by four orders of magnitude** is the instability.
* The animation shows the space-time map of $E_x$ and $\rho$, the velocity distribution,
  and the electron phase space rolling into its vortex.

## Measuring the growth rate

The rate is the slope of $\ln|E_k|$ during the linear phase:

```python
t = np.asarray(output.t) * float(simulation.plasma_frequency())
amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1])
peak = int(np.argmax(amplitude))
window = ((amplitude > 10 * amplitude[0]) & (amplitude < 0.1 * amplitude[peak])
          & (np.arange(t.size) < peak))
print(np.polyfit(t[window], np.log(amplitude[window]), 1)[0])
```

Setting the window by amplitude rather than by time is what makes this robust: it follows
the same part of the growth whatever the rate turns out to be. With this particle count
expect a few tens of per cent scatter. {doc}`../numerics/verification` shows what a quiet
start and more particles buy — {{ two_stream_scan_mean_deviation_percent }} per cent
agreement with kinetic theory across the unstable range.

## What to change next

| change | what happens |
|---|---|
| `drift` | moves along the growth-rate curve |
| `n` | the noise floor falls as $1/\sqrt N$ |
| `sampling="quiet"` | the noise floor falls much faster |
| `Solver(algorithm="implicit")` | the energy error drops to round-off |
| `cells` | the finite-grid instability appears when $\Delta x$ passes $\lambda_D$ |
