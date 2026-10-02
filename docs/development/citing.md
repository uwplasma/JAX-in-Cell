# Citing JAX-in-Cell

If you use JAX-in-Cell in a publication, please cite the software. The repository
contains a `CITATION.cff` file that GitHub renders as a citation, and the entry below
can be pasted into a BibTeX database:

```bibtex
@software{jaxincell,
  author  = {Ma, Longyu and Jorge, Rogerio and Lu, Hongke and Tran, Aaron and Woolford, Christopher},
  title   = {{JAX-in-Cell}: a differentiable particle-in-cell code for plasma physics},
  year    = {2025},
  url     = {https://github.com/uwplasma/JAX-in-Cell},
  note    = {Version 0.1}
}
```

Please also cite JAX {cite}`jax2018`, and the original references of the numerical
methods you rely on: the Boris pusher {cite}`boris1970`, the charge-conserving
current deposit {cite}`villasenor1992,esirkepov2001`, and the energy-conserving
implicit scheme {cite}`chen2011,chen2014` when using the implicit integrator.

JAX-in-Cell was inspired by an earlier particle-in-cell code in JAX by Sean Lim
{cite}`lim2023`. Development is supported by the National Science Foundation under
grant PHY-2409066.
