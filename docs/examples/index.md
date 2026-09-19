# Examples

Every script in `examples/` runs on its own and reproduces a result the code does not
itself compute, rather than making a picture for its own sake. They are in three
directories by how much of the code they use, not by how interesting they are, and
`examples/README.md` lists what each teaches and how long it takes.

```bash
git clone https://github.com/uwplasma/JAX-in-Cell
cd JAX-in-Cell
python examples/1_basic/two_stream.py
```

Three of them take `--quick`, a smoke run of between ten seconds and two minutes with far
fewer particles: it checks that they execute and reproduces the structure, with more
noise, and each says so when it starts so that a smoke run is not quoted as a
measurement.

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

:::{grid-item-card} A maintained sheath
:link: sheath_unmagnetized
:link-type: doc
`sheath_unmagnetized.py` — a source-to-collector sheath against the kinetic floating
potential, in closed form.
:::

:::{grid-item-card} An oblique magnetic field
:link: sheath_magnetized
:link-type: doc
`sheath_magnetized.py` — the magnetic presheath, and what the wall is struck by.
:::

:::{grid-item-card} Recovering a wall's reflectivity
:link: sheath_optimization
:link-type: doc
`sheath_optimization.py` — an inverse problem solved with the gradient of the whole
calculation, and the horizon over which that gradient is useful.
:::

:::{grid-item-card} A sheath at grazing incidence
:link: grazing_sheath
:link-type: doc
`grazing_sheath.py` — the magnetic presheath and the Debye sheath, set up to be compared
with the gyrokinetic code GYRAZE.
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

:::{grid-item-card} Conservation laws
:link: conservation
:link-type: doc
`conservation.py` — energy, momentum and charge in both schemes, periodic and between absorbing walls.
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

:::{grid-item-card} A wall that reflects electrons
:link: sheath_reflection
:link-type: doc
`sheath_reflection.py` — the Bohm criterion and the sheath drop of Hobbs and Wesson, with and without reflection.
:::

:::{grid-item-card} Optimisation
:link: optimize_two_stream
:link-type: doc
`optimize_two_stream.py` — gradient ascent through the whole solver finds the fastest beam.
:::

::::

```{toctree}
:hidden:

two_stream
landau_damping
langmuir_wave
sheath_unmagnetized
bump_on_tail
weibel
collisions
wall_reflection
sheath_magnetized
sheath_reflection
conservation
optimize_two_stream
sheath_optimization
grazing_sheath
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
