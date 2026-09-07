# Numerical kernels

The functions on this page are the building blocks of a time step. They are exported
so that they can be tested, reused in a custom loop, or differentiated on their own.
All of them are compiled with `jax.jit` and operate on JAX arrays. The conventions for
array shapes are: `N` particles, `G` grid points, positions and velocities of shape
`(N, 3)`, fields of shape `(G, 3)`.

## Time steps

The two step functions are not re-exported at the package level; import them from
`jaxincell._algorithms`.

```{eval-rst}
.. autofunction:: jaxincell._algorithms.Boris_step

.. autofunction:: jaxincell._algorithms.CN_step
```

## Particle push and interpolation

```{eval-rst}
.. autofunction:: jaxincell.boris_step

.. autofunction:: jaxincell.boris_step_relativistic

.. autofunction:: jaxincell.rotation

.. autofunction:: jaxincell.fields_to_particles_grid

.. autofunction:: jaxincell.fields_to_particles_periodic_CN
```

## Source deposition

```{eval-rst}
.. autofunction:: jaxincell.calculate_charge_density

.. autofunction:: jaxincell.single_particle_charge_density

.. autofunction:: jaxincell.charge_density_BCs

.. autofunction:: jaxincell.current_density

.. autofunction:: jaxincell.current_density_periodic_CN

.. autofunction:: jaxincell.get_S2_weights_and_indices_periodic_CN
```

## Field update and electrostatic solvers

```{eval-rst}
.. autofunction:: jaxincell.curlE

.. autofunction:: jaxincell.curlB

.. autofunction:: jaxincell.field_update

.. autofunction:: jaxincell.field_update1

.. autofunction:: jaxincell.field_update2

.. autofunction:: jaxincell.E_from_Gauss_1D_FFT

.. autofunction:: jaxincell.E_from_Poisson_1D_FFT

.. autofunction:: jaxincell.E_from_Gauss_1D_Cartesian
```

## Boundary conditions

```{eval-rst}
.. autofunction:: jaxincell.set_BC_particles

.. autofunction:: jaxincell.set_BC_single_particle

.. autofunction:: jaxincell.set_BC_positions

.. autofunction:: jaxincell.set_BC_single_particle_positions
```

## Digital filter

```{eval-rst}
.. autofunction:: jaxincell.filter_scalar_field

.. autofunction:: jaxincell.filter_vector_field

.. autofunction:: jaxincell.binomial_filter_3point
```
