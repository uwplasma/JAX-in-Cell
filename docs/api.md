# API reference

The top-level names of `jaxincell` are the public interface. The modules they come from
start with an underscore; the numerical kernels are documented too, because the
{doc}`numerics/index` pages refer to them by name, but they may change.

## Configuration

```{eval-rst}
.. autoclass:: jaxincell.Domain
   :members:

.. autoclass:: jaxincell.Species
   :members:

.. autoclass:: jaxincell.Solver
   :members:

.. autoclass:: jaxincell.Collisions
   :members:
```

## Simulation and output

```{eval-rst}
.. autoclass:: jaxincell.Simulation
   :members:

.. autoclass:: jaxincell.Output
   :members:

.. autofunction:: jaxincell.quiet_start

.. autofunction:: jaxincell.load_toml

.. autofunction:: jaxincell.plot

.. autofunction:: jaxincell.save_state

.. autofunction:: jaxincell.load_state

.. autofunction:: jaxincell.openpmd.write_openpmd
```

## Diagnostics

```{eval-rst}
.. automodule:: jaxincell._diagnostics
   :members:
   :member-order: bysource
```

## Numerical kernels

```{eval-rst}
.. automodule:: jaxincell._core
   :members:
   :member-order: bysource

.. automodule:: jaxincell._collisions
   :members:
   :member-order: bysource
```

## Physical constants

`epsilon_0`, `mu_0`, `speed_of_light`, `elementary_charge`, `mass_electron`,
`mass_proton` and `boltzmann_constant` are plain floats importable from `jaxincell`:
the CODATA 2018 values in SI units. Species charges and masses are given as multiples
of `elementary_charge`, `mass_electron` and `mass_proton`, see {doc}`user_guide/species`.
