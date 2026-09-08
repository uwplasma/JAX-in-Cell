# Numerical methods

```{toctree}
:maxdepth: 1

equations
discretization
deposition
explicit
implicit
field_solvers
boundaries
filtering
collisions
initialization
stability
diagnostics
verification
references
```

These pages describe what the code computes, in the order in which a time step
executes: the {doc}`equations`, their {doc}`discretization` on a staggered grid, the
{doc}`deposition` of the sources, the {doc}`explicit` and {doc}`implicit` time
integrators, the {doc}`field_solvers`, the {doc}`boundaries`, the {doc}`filtering` of
the sources and the {doc}`collisions` operator, the {doc}`initialization` of the phase
space, the {doc}`stability` constraints on the resolution, the {doc}`diagnostics` and
the {doc}`verification` against linear kinetic theory. Symbols follow Birdsall and
Langdon {cite}`birdsall1991`; the {doc}`references` page lists the sources.

Every statement about accuracy on these pages is a number produced by a script in
`docs/scripts/`, recorded in `measurements.json` and substituted into the text, so the
prose and the figures always come from the same run.
