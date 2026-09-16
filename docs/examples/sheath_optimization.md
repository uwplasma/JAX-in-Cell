# Recovering a wall's reflectivity

`examples/3_advanced/sheath_optimization.py`

A collector returns the fraction $r$ of every electron that reaches it and keeps the
rest. Reflected electrons are not lost, so the wall needs less voltage to hold the rest
back and the sheath in front of it is shallower. Measure the potential in the plasma,
and $r$ follows.

This example generates a measurement at a known reflectivity and asks a bounded
optimiser to find it back from a different start, using the gradient of the whole
particle-in-cell calculation with respect to $r$. Nothing is finite-differenced:
`jax.grad` runs back through the field solve, the deposit, the gather, the Boris push,
the source and the wall for every step of the experiment. The finite differences are
there to check that gradient, not to compute it.

## A short experiment, on purpose

A plasma is prepared once at a fixed reflectivity, **outside** the differentiated
calculation and therefore independent of $r$. The trial value is then applied and the
response is watched for twenty-five steps, about four electron plasma periods: the time
in which the electrons rearrange and the wall's charge follows. That is the window the
measurement lives in, and it is also as long as the gradient can usefully be taken over.

The script measures why. At $r = 0.25$, from a state prepared over 1200 steps:

| horizon | reverse mode | forward mode | best central difference | at $h$ |
|---|---|---|---|---|
| 5 steps | $-1.6457810097\times10^{-1}$ | identical to $8\times10^{-15}$ | $1.5\times10^{-9}$ | $10^{-3}$ |
| 25 steps | $-8.3315086793\times10^{-1}$ | identical to $7\times10^{-16}$ | $1.1\times10^{-9}$ | $10^{-5}$ |
| 100 steps | $+1.7036661618$ | identical to $3\times10^{-14}$ | $1.9\times10^{-7}$ | $10^{-7}$ |

The implementation is exact at every horizon: the two modes agree to round-off and both
agree with a central difference of the same realisation to nine digits. What changes is
the step size at which that agreement holds — $10^{-1}$ over five steps, $10^{-4}$ over
twenty-five, $10^{-7}$ over a hundred — because every absorption at the wall is a branch
of the program, and over a long window a change in $r$ of one part in a million already
flips some.

Past that the derivative of one realisation stops tracking the response of the average.
Over 100 to 800 steps the gradient grows to between five and a hundred times the
measured mean slope and changes sign, while the mean slope stays between $-0.2$ and
$-0.7$. That is the distinction between the derivative of a fixed discretisation and a
fixed realisation and the derivative of a finite-time expectation, and the reason
{cite}`chung2020` build particle sensitivities that do not follow the plasma particles.
Nothing here is broken; the horizon is a modelling choice and the script shows how to
choose it.

## The inverse problem

Two fixed Gaussian sensors of fixed physical width read the potential, one in the plasma
and one in the sheath; neither moves if the grid is refined. The measurement is averaged
over four fixed realisations first and the loss is of that mean, in units of the
scatter between realisations, which is the measurement's own uncertainty and is fixed
before the optimiser starts. Four further realisations are held out and never used to
choose a step.

The script prints an identifiability check before optimising: the response of each
sensor over the admissible interval against the scatter between realisations. The plasma
sensor comes out at a ratio of 0.3 and carries almost no information; the sheath sensor
at 2.6 and carries it all. That is worth seeing rather than hiding.

Starting at $r = 0.08$ with the answer at $r = 0.35$, in the `--quick` preset:

| | $r$ | training loss | held-out loss |
|---|---|---|---|
| start | 0.0800 | 2.146 | 1.425 |
| recovered | 0.3550 | 0.00027 | 0.00223 |
| reference | 0.3500 | 0 | 0 |

The control is recovered to 0.005 absolute in fourteen iterations of projected gradient
descent with backtracking, and the training loss falls by nearly four orders of
magnitude. A coarse scan of the held-out loss, which uses no gradient at all, puts its
minimum at $r = 0.34$; the distance from the reference is the uncertainty of the
recovered control, about 0.01, and not the optimiser's tolerance.

## What this is and is not

The control is an idealised wall-response parameter, represented as a deterministic
fraction of each macro-particle's weight rather than a random hit-or-miss. It is not a
material coating, a sputtering yield or a heat-load optimisation, and this is a
finite-time response calibration rather than the derivative of an asymptotic floating
state.

```bash
python examples/3_advanced/sheath_optimization.py            # about fifteen minutes
python examples/3_advanced/sheath_optimization.py --quick    # about three minutes
```
