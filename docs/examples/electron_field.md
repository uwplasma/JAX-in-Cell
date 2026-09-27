# Electron-field instability

A periodic plasma in a static, uniform electric field: do electron plasma waves grow? Beving,
Hopkins and Baalrud {cite}`beving2023` report that they do, at a rate proportional to the
field, and compare it with the dielectric function of electrons in a field of Fried et al.
{cite}`fried1960` ({func}`jaxincell.theory.field_epsilon`). The example runs their case with
the ions taken three ways, each with the field on and off, because one of them decides what
the growth is.

```{figure} ../_static/figures/electron_field.png
:width: 100%
:alt: Fluctuation energy for five runs, the change-of-frame check, the helium spectrum and the fitted rates

(a) The fluctuation energy $\langle\mathcal E\rangle$ (the paper's eq. 2 averaged over the box,
and here over one plasma period), ensemble mean of {{ efi_realisations }} realisations, for
the exactly uniform background, frozen ion macro-particles and mobile helium; dotted, the
same without the field. (b) The driven uniform run moved back by $at^2/2$ against the
undriven one with the same particles. (c) The spectrum of the helium run, with the paper's
eq. 12 for the fastest-growing wavenumber. (d) The slope of $\ln\langle\mathcal E\rangle$ in the
paper's window, against twice the growth rate of eq. 10 and against the paper's own fit.
```

## What is measured against what

| quantity | measured | reference |
|---|---|---|
| $d\ln\langle\mathcal E\rangle/dt$, helium, $t\omega_{pe}$ = {{ efi_fit }} | {{ efi_helium_rate }} $\omega_{pe}$ (gain {{ efi_helium_gain }}) | $2\gamma$ of eq. 10 at the mean drift, {{ efi_theory }}; the paper's fit, {{ efi_paper }} |
| the same, frozen ion macro-particles | {{ efi_frozen_rate }} (gain {{ efi_frozen_gain }}) | as above |
| the same, exactly uniform background | {{ efi_uniform_rate }} (gain {{ efi_uniform_gain }}) | 0: the change of frame removes the field |
| undriven, uniform and frozen | {{ efi_uniform_still_rate }} and {{ efi_frozen_still_rate }} | 0 |
| driven uniform run moved by $at^2/2$, against the undriven one, $k\lambda_{De}\le0.3$ | {{ efi_frame_long }} (all wavelengths: {{ efi_frame }}) | 0 |
| peak of the spectrum, $k\lambda_{De}\,\kappa_e\lambda_{De}\,t\omega_{pe}$, helium and frozen | {{ efi_helium_peak }} and {{ efi_frozen_peak }} | 1 (eq. 12) |
| work of $E_0$ against the kinetic and field energy gained, uniform | {{ efi_ledger }} relative | 0 |

## What the controls say

For collisionless Vlasov-Poisson on an exactly uniform, immobile background the change of
variables $x' = x - at^2/2$, $v' = v - at$, with $a = -eE_0/m_e$, removes the field from the
equations. The driven and undriven uniform runs start from the same particles, and after the
shift they agree to {{ efi_frame_long }} on the wavelengths the instability is about; the
remainder, and the larger difference at grid scale, is the grid, which stays in the
laboratory frame while the electrons move 40 $v_{Te}$ across it. Neither grows. Nor does the
frozen-ion run without a field.

Growth appears only when the field and discrete ions are both present, and the frozen
macro-particles already give three quarters of the helium rate. Its wavenumber follows
eq. 12, $k^*\lambda_{De} = 1/(\kappa_e\lambda_{De}\,t\,\omega_{pe})$, which is
$k^* = \omega_{pe}/(at)$: the electron plasma wave whose phase velocity in the electron frame
equals the drift of the ions through it, that is, a wave standing still in the ion frame. That is the
Cherenkov wake of the ions' charge noise, swept through resonance as the electrons
accelerate, and it depends on the number of ion macro-particles, not on the field alone.
Here, then, the growth is not the continuum instability of the Fried dielectric: in this code
the continuum limit of the same setup, the uniform background, shows none. Its rate,
{{ efi_helium_rate }}, is of the size the paper reports for the same fit ({{ efi_paper }})
and below $2\gamma$ of eq. 10 ({{ efi_theory }}).

What is not settled by this page: whether the growth in the helium run scales with the
number of ion macro-particles per cell, as the wake picture implies; and whether a physical, non-discrete ion noise level, far
below 400 per cell, would still drive it. Collisions and walls, which also break the change
of frame, are not included.

## The setup

| | |
|---|---|
| plasma | $n = 3\times10^{14}$ m$^{-3}$, $T_e$ = 3 eV, $T_i$ = 0.026 eV, He$^+$ |
| field | $E_0 = -800$ V/m, $\kappa_e\lambda_{De} = eE_0\lambda_{De}/T_e$ = {{ efi_kappa }} |
| box | {{ efi_debye_lengths }} $\lambda_{De}$, five cells per $\lambda_{De}$, periodic, electrostatic |
| particles | {{ efi_per_cell }} per cell per species, positions at random |
| step | $\omega_{pe}\Delta t$ = {{ efi_dt }}, {{ efi_steps }} steps to $t\omega_{pe}$ = {{ efi_t_end }} |
| realisations | {{ efi_realisations }} (the paper used sixteen) |

The field is `Simulation(external_E=...)`, uniform on the faces. The paper's thermal
speed is $v_{Te} = \sqrt{T_e/m_e}$, and {func}`jaxincell.theory.field_epsilon` keeps it;
`Species` takes $\sqrt{2T/m}$. The frozen ions are ion macro-particles of $10^9$ proton
masses; the uniform background is no ion species at all, which the periodic field solve
neutralises exactly.

`python examples/3_advanced/electron_field.py` runs the full preset (about three and a half
hours on one A4000); `--quick` runs a 240 $\lambda_{De}$ box with 40 particles per cell in
about a minute, which shows the same separation and is not a measurement. The figure and
numbers here come from `docs/scripts/fig_electron_field.py`, which runs the example.
