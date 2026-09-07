# Quickstart

There are three equivalent ways to run a simulation. All of them build a
{class}`jaxincell.Simulation` object from a parameter tree, call its `run` method and
return a dictionary of arrays.

## From the command line

```bash
jaxincell                      # built-in defaults
jaxincell examples/input.toml  # parameters from a TOML file
```

The command runs the simulation, computes the diagnostics and opens the animated plot.
It is the equivalent of the Python snippet below.

## From a TOML file

`examples/input.toml` in the repository sets up a two-stream instability:

```{literalinclude} ../../examples/input.toml
:language: toml
```

Load it and run:

```python
from jaxincell import Simulation, load_parameters, diagnostics, plot

parameters = load_parameters("examples/input.toml")
sim = Simulation(parameters)
output = sim.run()
diagnostics(output)
plot(output)
```

`load_parameters` is a thin wrapper around `tomllib.load`; the result is an ordinary
nested dictionary that you can edit before constructing the simulation.

## From a Python dictionary

The same parameters can be written directly in Python. Anything not specified takes
the default listed in the {doc}`../user_guide/index`.

```python
from jaxincell import Simulation, diagnostics, plot

parameters = {
    "domain_parameters": {
        "length": 0.01,
        "number_grid_points": 70,
        "total_steps": 1100,
        "timestep_over_spatialstep_times_c": 1.0,
    },
    "species_parameters": {
        "electrons": {
            "electrons0": {
                "number_pseudoparticles": 3500,
                "grid_points_per_Debye_length": 0.5,
                "vth_over_c_x": 0.05,
                "drift_speed_x": 6e7,
                "velocity_plus_minus_x": True,
                "perturbation_amplitude_x": 5e-7,
                "perturbation_wavenumber_x": 1,
            },
        },
        "ions": {
            "ions0": {
                "number_pseudoparticles": 3500,
                "grid_points_per_Debye_length": 0.5,
                "vth_over_c_x": "_electrons0",
            },
        },
    },
    "solver_parameters": {"field_solver": 0, "time_evolution_algorithm": 0},
}

output = Simulation(parameters).run()
diagnostics(output)
plot(output)
```

## What you get back

`run` returns a dictionary. The time histories are arrays whose first axis is the time
step:

| key | shape | meaning |
|---|---|---|
| `positions`, `velocities` | `(steps, N, 3)` | phase space of all pseudo-particles |
| `electric_field`, `magnetic_field` | `(steps, G, 3)` | fields on the grid |
| `current_density`, `charge_density` | `(steps, G, 3)`, `(steps, G)` | deposited sources |
| `time_array`, `grid` | `(steps,)`, `(G,)` | time and cell-centre coordinates |
| `plasma_frequency`, `dt`, `dx` | scalars | derived quantities |

{func}`jaxincell.diagnostics` adds the field and kinetic energies, the dominant
frequency of the electric field and per-species views, and removes the combined
`positions` and `velocities` arrays to save memory. The full list is in
{doc}`../user_guide/output`.

## Where to go next

* {doc}`first_simulation` explains the choices behind the parameters above.
* {doc}`../user_guide/parameters` lists every parameter with its default.
* {doc}`../numerics/index` describes what happens in each time step.
