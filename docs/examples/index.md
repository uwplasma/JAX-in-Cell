# Examples

Every script in `examples/` runs on its own on a laptop in seconds to about a minute,
and each one reproduces a result from the literature rather than making a picture for
its own sake.

```bash
git clone https://github.com/uwplasma/JAX-in-Cell
cd JAX-in-Cell/examples
python two_stream.py
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Two-stream instability
:link: two_stream
:link-type: doc
`two_stream.py` — growth, saturation and the phase-space vortex. Buneman 1959.
:::

:::{grid-item-card} Landau damping
:link: landau_damping
:link-type: doc
`landau_damping.py` — the damping rate and frequency at $k\lambda_D=0.5$. Landau 1946.
:::

:::{grid-item-card} Langmuir waves
:link: langmuir_wave
:link-type: doc
`langmuir_wave.py` — the Bohm-Gross dispersion relation, scanned in $k$.
:::

:::{grid-item-card} Bump-on-tail
:link: bump_on_tail
:link-type: doc
`bump_on_tail.py` — a beam-driven instability and the quasilinear plateau.
:::

:::{grid-item-card} Weibel instability
:link: weibel
:link-type: doc
`weibel.py` — a temperature anisotropy driving magnetic modes. Weibel 1959.
:::

:::{grid-item-card} Energy conservation
:link: energy_conservation
:link-type: doc
`energy_conservation.py` — explicit against implicit as the Picard iteration converges.
:::

:::{grid-item-card} Collisions
:link: collisions
:link-type: doc
`collisions.py` — the Takizuka-Abe operator against the Fokker-Planck rates.
:::

:::{grid-item-card} Wall reflection
:link: wall_reflection
:link-type: doc
`wall_reflection.py` — a wall returns the flux average of its reflection law.
:::

:::{grid-item-card} Plasma sheath
:link: sheath
:link-type: doc
`sheath.py` — the Bohm criterion and the sheath drop of Hobbs and Wesson, with and without reflection.
:::

:::{grid-item-card} Optimisation
:link: optimisation
:link-type: doc
`optimisation.py` — gradient ascent through the whole solver finds the fastest beam.
:::

::::

```{toctree}
:hidden:

two_stream
landau_damping
langmuir_wave
bump_on_tail
weibel
energy_conservation
collisions
wall_reflection
sheath
optimisation
```

There is also `input.toml`, which runs the two-stream case from the command line:

```bash
jaxincell examples/input.toml
```

The scripts that produce the figures in this documentation live in `docs/scripts/` and
record the numbers the pages quote in `measurements.json`; see
{doc}`../numerics/verification`. Each example page names the script its figure and
numbers come from. Most of those scripts run exactly the example's setup, some adding a
scan around it. Two do not: the two-stream figure uses a quiet start at another drift, so
that its growth rate can be fitted, and the sheath figure uses more particles.
