# Output and restart

A run stopped halfway, written to disk with {func}`~jaxincell.save_state`, read back with
{func}`~jaxincell.load_state` and continued, against the same run done in one go; and the
field written as an openPMD series and read back with `openpmd-api`.

```{figure} ../_static/figures/output_and_restart.png
:width: 100%
:alt: A restarted two-stream run on top of the uninterrupted one, and an openPMD field read back

(a) The seeded two-stream mode of one 400-step run (grey), of its first 200 steps (dashed)
and of the run restarted from the file (dotted). (b) The last $E_x$ in memory and read back
from the openPMD series at the coordinates its attributes give.
```

## What is measured against what

| quantity | measured | reference |
|---|---|---|
| restarted run against the uninterrupted one (`t`, `E`, `B`, `x`, `v`) | bit-identical: {{ restart_bit_identical }} | bit-identical |
| openPMD field read back | largest difference {{ restart_openpmd_field }} V/m | 0 |
| openPMD coordinates, $x_i = $ offset $+ (i + $ position$)\,\Delta x$ | largest difference {{ restart_openpmd_coordinates }} m | the faces the code stores |

`Output.state` is everything a run needs to continue: positions, velocities, weights, the
fields, the random key, the step count and the wall ledger. `load_state` checks the species
names and counts, the cell count and the integrator of the simulation it is restored into,
because a state in a differently shaped run fails later and less clearly.

## How to run

```bash
python examples/2_intermediate/output_and_restart.py
pip install openpmd-api      # optional: the openPMD part is skipped without it
```

It writes `output_and_restart/run.json` and its figure;
`docs/scripts/fig_output_and_restart.py` runs it for this page.
