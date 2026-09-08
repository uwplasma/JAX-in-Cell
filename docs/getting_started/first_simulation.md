# Your first simulation, in detail

This page takes the two-stream run apart and explains every number in it. The script
is `examples/two_stream.py`.

## The physics

Two electron beams stream through one another at $\pm v_0$ on a background of
protons. Each beam Landau-resonates with the space-charge wave carried by the other,
and the pair is unstable: a small density perturbation grows exponentially until the
beams trap each other and the phase space folds into a vortex. It is the standard
first test of a kinetic code, because the growth rate is known in closed form
{cite}`buneman1959`.

## Choosing the parameters

```python
import numpy as np
from jaxincell import (Domain, Simulation, Solver, Species, diagnostics, plot,
                       epsilon_0, mass_electron, elementary_charge as e, speed_of_light as c)

length, cells = 0.01, 64
density, drift, vth = 4.37e17, 6e7, 0.05 * c
```

Everything else follows from these five numbers.

**Density → plasma frequency.** $\omega_{pe} = \sqrt{n e^2/\epsilon_0 m_e}$ is the
clock of the problem. At $n = 4.37\times10^{17}\,\mathrm{m^{-3}}$ it is
$3.7\times10^{10}$ rad/s, so a plasma period is 0.17 ns and the interesting physics
takes a few tens of them.

**Thermal speed → Debye length.** $\lambda_D = v_{th}/(\sqrt2\,\omega_{pe})$ is
$2.8\times10^{-4}$ m here. The cell must resolve it, or the explicit scheme heats
itself through the finite-grid instability. With `cells=64` and `length=0.01`,
$\Delta x/\lambda_D = ${{ two_stream_dx_over_debye }} — comfortably below one.

**Box length → the mode that is seeded.** Mode 1 has $k = 2\pi/L$, and the beams are
most unstable near $kv_0/\omega_{pe} = \sqrt{3/8}$. Solving for $v_0$ at fixed $L$ and
$n$ is how the drift was picked; at $v_0 = 6\times10^7$ m/s the mode sits at
$kv_0/\omega_{pe} = 1.01$, just past the peak, which is a deliberately unexciting
choice for a first run.

**Time step.** `dt_over_dx_c=4.5` gives $\omega_{pe}\Delta t = $
{{ two_stream_omega_pe_dt }}, well inside the accuracy limit of 0.2. It is above the
light-wave Courant limit of one, which is allowed here because a purely electrostatic
run never excites the transverse fields — see {doc}`../numerics/stability`, and note
that adding a magnetic field, an isotropic temperature or collisions would make this
choice diverge.

## Building it

```python
electrons = Species.electrons(n=8000, density=density, vth=(vth, 0, 0),
                              drift=(drift, 0, 0), plus_minus=True,
                              perturbation_amplitude=5e-7, perturbation_mode=1)
ions = Species.ions(n=8000, density=density, electrons=electrons)
```

`plus_minus=True` negates $v_x$ on every second particle, so one population becomes
two counter-streaming beams of density $n/2$ each. `Species.ions` derives the proton
thermal speed from the electrons, so the two start in thermal equilibrium.

`perturbation_amplitude` is a **displacement** in metres, not a density: $5\times10^{-7}$
m over a 0.01 m box with $k = 628\ \mathrm{m^{-1}}$ makes $ak = 3\times10^{-4}$, a
small enough seed for the linear phase to be several e-foldings long.

Eight thousand pseudo-particles per species is 125 per cell. That is enough to see the
instability but not enough to measure its rate precisely; the verification runs use
five times more and a quiet start ({doc}`../numerics/verification`).

```python
simulation = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=4.5),
                        [electrons, ions], Solver(filter_passes=2))
```

`filter_passes=2` smooths the deposited sources, which suppresses the grid-scale noise
of a modest particle count. It also damps genuinely short-wavelength physics, so it is
off in the runs that measure rates.

## Checking before running

```python
print(f"omega_pe dt   = {float(simulation.plasma_frequency() * simulation.domain.dt):.3f}")
print(f"dx / lambda_D = {float(simulation.domain.dx / simulation.debye_length()):.2f}")
```

Two numbers, both of which should be below one. Getting into the habit of printing
them costs nothing and catches most bad runs before they start.

## Running

```python
output = simulation.run(1200, seed=0, store_every=2)
```

The first call compiles for a second or two, then runs. `store_every=2` halves the
stored history: 1200 steps of 16 000 particles at every step would be 460 MB, which is
fine, but the same run at 100 000 particles would not be. The memory formula is in
{doc}`../user_guide/running`.

## Reading the result

```python
energy = diagnostics(output)
print(f"energy drift {abs(float(energy['total'][-1] / energy['total'][0]) - 1):.2e}")
print(f"electric energy grew by {float(energy['electric'].max() / energy['electric'][0]):.3g}")
```

The energy drift is the health check: a few parts in $10^5$ means the resolution is
fine, and a steady rise would mean it is not. The electric energy growing by four
orders of magnitude is the instability.

```python
plot(output, omega=float(simulation.plasma_frequency()))
```

The animation shows the space-time map of $E_x$ and $\rho$, the velocity distribution,
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

Setting the window by amplitude rather than by time is what makes this robust: it
follows the same part of the growth whatever the rate turns out to be. With this
particle count expect a few tens of per cent scatter; {doc}`../numerics/verification`
shows what a quiet start and more particles buy — {{ two_stream_scan_mean_deviation_percent }}
per cent agreement with kinetic theory across the unstable range.

## What to change next

* `drift`, to move along the growth-rate curve.
* `n`, to watch the noise floor fall as $1/\sqrt N$.
* `quiet=True`, to watch it fall much faster.
* `Solver(algorithm="implicit")`, to see the energy error drop to round-off.
* `cells`, to see the finite-grid instability appear when $\Delta x$ passes
  $\lambda_D$.
