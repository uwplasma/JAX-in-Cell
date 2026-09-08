# Plasma sheath

`examples/sheath.py`

Put a plasma between two absorbing walls and it does not stay neutral for long.
Electrons are $\sqrt{m_i/m_e}$ times faster than ions, so they reach the walls first
and charge them negative. The plasma floats positive until the field it sets up holds
back enough electrons for the two fluxes to match. What is left is the structure every
bounded plasma has: a quasi-neutral bulk, a pre-sheath that accelerates the ions, and a
thin positively charged sheath at each wall.

```{figure} ../_static/figures/sheath.png
:width: 100%
:alt: Potential and charge density profile, absorbed particle excess, and ion flow speed

(a) The potential, averaged over the second half of the run, with the charge density on
the right axis: flat and neutral in the middle, a spike of net positive charge a few
Debye lengths thick at each wall. (b) The excess of electrons over ions absorbed, which
falls as the sheath grows and throttles the electron flux. (c) The ion flow speed where
the sheath begins, approaching the Bohm speed.
```

## What is being tested

Two closed-form results, neither of which has a free parameter.

**The floating potential.** Equating the ion flux entering the sheath at the Bohm speed
with the electron flux that gets over the barrier gives {cite}`lieberman2005`

```{math}
\phi_{\rm plasma} - \phi_{\rm wall} = \frac{T_e}{2e}\ln\!\frac{m_i}{2\pi m_e},
```

which for the mass ratio used here, $m_i/m_e = $ {{ sheath_mass_ratio }}, is
{{ sheath_drop_theory }} $T_e/e$. The run gives {{ sheath_drop_measured }} $\pm$
{{ sheath_drop_spread }}.

**The Bohm criterion.** Ions must enter the sheath at no less than
$c_s = \sqrt{T_e/m_i}$, which the pre-sheath field accelerates them to
{cite}`bohm1949sheath,riemann1991`. After one ion transit the measured flow at the sheath
edge is {{ sheath_bohm_ratio }} $c_s$.

And one exact check: the two walls are short-circuited conductors, so the far wall stays
at the near wall's potential to {{ sheath_wall_potential }} of $T_e/e$
({doc}`../numerics/boundaries`).

## Why the potential comes out low

{{ sheath_drop_deviation_percent }} per cent below the formula, and the reason is worth
knowing before trusting a bounded-plasma run.

Nothing sustains this plasma. The walls take the fast electrons preferentially, so the
bulk cools — from 1 eV to {{ sheath_temperature_final }} eV over the run, which is why
the comparison uses the temperature measured as it goes rather than the one it started
with — and, more importantly, the distribution loses the tail the flux balance is
derived from. By the end there is nothing left beyond about two standard deviations,
where a Maxwellian would still have 4.5 per cent of its electrons. A truncated tail
carries less flux, so a smaller barrier suffices.

Closing that gap means sustaining the plasma, which is what the bounded-plasma
literature does: an ionisation source that replaces the ions lost to the walls
{cite}`reboul2022`, or collisions frequent enough to refill the tail from the bulk.
Either turns this into a study of a discharge; neither is needed to show where a sheath
comes from.

## Running it

```bash
python examples/sheath.py
```

About thirty seconds: {{ sheath_particles }} particles on {{ sheath_cells }} cells over
{{ sheath_box_debye }} Debye lengths, run for {{ sheath_steps }} steps, roughly one ion
transit. The mass ratio is reduced to {{ sheath_mass_ratio }} for exactly that reason —
the ion transit is what sets the cost, and it scales as $\sqrt{m_i/m_e}$.

The run is electrostatic, so the time step follows $\omega_{pe}\Delta t = 0.2$ rather
than the light-wave limit, which is a factor of nearly three hundred in step size here.
That is safe only while nothing excites the transverse fields; see
{doc}`../numerics/stability`.

## Things to try

* Change the mass ratio and check that the drop follows $\ln(m_i/m_e)$.
* Make the box longer in Debye lengths: the sheath stays a few $\lambda_D$ wide while
  the quasi-neutral bulk grows, which is the separation of scales the theory assumes.
* Set `particle_bc=("reflective", "absorbing")`: one symmetry plane and one electrode,
  a half-domain problem, and the sheath forms only at the absorbing end.
