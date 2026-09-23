# Collisions

The Takizuka-Abe collision operator checked against the Fokker-Planck relaxation rates, in
the limit where they are closed-form and contain no adjustable constant. A beam
{{ collisions_beam_over_background }} times faster than the background it scatters off
slows down and spreads in angle at rates that follow from theory alone.

```{figure} ../_static/figures/collisions.png
:width: 100%
:alt: Beam slowing down and perpendicular diffusion against theory

(a) Slowing down: $\langle v_x\rangle/v_{\rm beam}$ against $t\nu_0$, for equal masses and
for a background 100 times heavier, with $e^{-\nu_s t}$ drawn through each. (b)
Perpendicular diffusion: $\langle v_\perp^2\rangle/v_{\rm beam}^2$ against $\nu_\perp t$.
Lines are theory, points the operator.
```

## What is measured against what

Ratios of the measured rate to the Fokker-Planck rate, so 1 is exact agreement:

| rate | equal masses | background 100× heavier | reference |
|---|---|---|---|
| slowing down $\nu_s$ | {{ collisions_nu_slow_ratio_equal_mass }} | {{ collisions_nu_slow_ratio_heavy }} | $(1 + m_a/m_b)\,\nu_0$ |
| perpendicular $\nu_\perp$ | {{ collisions_nu_perp_ratio_equal_mass }} | {{ collisions_nu_perp_ratio_heavy }} | $2\nu_0$ |

The worst of the four is {{ collisions_max_deviation_percent }} per cent away, with

```{math}
\nu_0 = \frac{q_a^2q_b^2 n_b\ln\Lambda}{4\pi\epsilon_0^2 m_a^2 v^3}
```

{cite}`trubnikov1965,nrl2019`. Matching $\nu_\perp$ is what fixes the variance of the
scattering angle in the first place ({doc}`../numerics/collisions`), so $\nu_s$ and the
mass dependence are a real check of the implementation rather than a tautology.

## The setup

| | |
|---|---|
| particles | {{ collisions_particles }} |
| $\ln\Lambda$ | {{ collisions_coulomb_log }} |
| beam speed over background | {{ collisions_beam_over_background }} |

The example calls the operator directly rather than through a simulation, because
collisional and plasma timescales are five orders of magnitude apart in a weakly coupled
plasma and a run that resolved both would be enormous. That separation is physics, not a
limitation of the code.

## Using collisions in a simulation

```python
from jaxincell import Collisions
simulation = Simulation(domain, [electrons, ions], solver,
                        collisions=Collisions(pairs=(("electrons", "ions"),)))
```

Keep `dt_over_dx_c <= 1` when collisions are on: they scatter velocity into the transverse
directions, which excites the light-wave branch that an electrostatic run was safely
ignoring. See {doc}`../user_guide/collisions`.

## How to run

```bash
python examples/2_intermediate/collisions.py
jaxincell inputs/collisions.toml
```

The figure and the numbers come from `docs/scripts/fig_collisions.py`, which runs this
setup and repeats it against a background a hundred times heavier.
