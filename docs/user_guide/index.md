# User guide

```{toctree}
:maxdepth: 1

parameters
domain
species
solver
boundaries
external_fields
running
output
plotting
differentiation
performance
units
```

The user guide is organised by what you do with the code: {doc}`parameters` explains
how the input tree is structured and validated, and the following four pages list every
parameter of each section with its default and meaning. {doc}`running` covers the
`Simulation` object, re-running with different inputs, and recompilation.
{doc}`output` documents every key of the result dictionary. {doc}`differentiation` shows
how to take gradients through a run, and {doc}`performance` what to expect from
compilation and scaling.
