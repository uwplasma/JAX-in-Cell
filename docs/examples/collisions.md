# Coulomb relaxation

`twospecies_tempdiff.py` and `twospecies_temp.toml` in the repository root are the
author's hot-electron/cold-positron example. The densities and masses are equal;
all three velocity components are thermal. Enable the optional collision operator
with `solver_parameters.collisions = true`. It supports Newtonian periodic Boris
runs; implicit, relativistic, wall and particle-source coupling are rejected.

Run the collision-only control first, from the repository root:

```bash
MPLBACKEND=Agg python twospecies_tempdiff.py --collision-only --particles 1000 --steps 4000 --store-every 20 --output relaxation.png
```

This keeps the initialized positions fixed and advances only binary scattering.
`--particles` changes each population's marker count while preserving its density.
`--store-every` bounds collision-only history memory and retains the final step,
even when the interval does not divide the step count. Seed and Coulomb logarithm
come from the same configuration as full PIC. Omitting `coulomb_logarithm` uses the
initial electron-ion NRL estimate, floored at 2. The floor prevents numerical NaNs;
weak coupling and sufficiently small scattering steps are still required.

The figure shows temperatures, their difference and relative total-energy change.
Temperatures are physical, marker-weighted velocity variances in kelvin, with each
population's bulk velocity removed. The equilibrium reference comes from the
initial three-dimensional kinetic energy in the common centre-of-mass frame:

```{math}
T_{\rm eq}=\frac{\sum_p m_p w_p|\mathbf v_p-\mathbf V_{\rm COM}|^2}
                  {3 k_B\sum_p w_p}.
```

Equal marker weights conserve energy and momentum per collision, including unequal
cell populations. Unequal weights conserve them statistically; individual runs can
drift and require seed ensembles. Pair-local density normalization follows
[Higginson et al. (2020)](https://doi.org/10.1016/j.jcp.2020.109450).
Temperatures can cross the reference through finite-sample fluctuations; inspect
energy and momentum before diagnosing heating. A one-seed temperature curve or
fitted exponential is insufficient to validate a collision rate. For equal masses
and densities the coupled Maxwellian temperature difference decays at
`gamma = 4*nu_inter/3`. Refine the step, increase particles and compare multiple
seeds; `tests/test_collisions.py` contains independent conservation and rate controls.
Trajectory derivatives are tested, while weight-dependent discrete acceptance does
not give an unbiased derivative of ensemble expectations.

For the coupled particle-field control, omit `--collision-only`:

```bash
MPLBACKEND=Agg python twospecies_tempdiff.py --particles 1000 --steps 200 --output relaxation-pic.png
```

The original dense benchmark under-resolves the Debye length. This run therefore
tests functionality, and its kinetic plus field-energy change must be checked before
interpreting late temperatures as collisional equilibration. Collision-only energy
checks isolate the operator from particle-field heating. Both plots include the
initial state; the PIC energy reference includes the initialized fields.
