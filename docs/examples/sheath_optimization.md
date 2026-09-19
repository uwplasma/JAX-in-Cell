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

```{figure} ../_static/figures/sheath_source.png
:width: 100%
:alt: The sheath potential and densities against kinetic theory, and the gradient against finite differences at three horizons

(c) The reverse-mode gradient against a central difference of the same realisation, over
seven decades of step size, at three lengths of the response window. The floor each
curve reaches is where the two agree; the dotted line is how well forward and reverse
mode agree with each other, which is round-off at every horizon. Panels (a) and (b) are
the forward problem, in {doc}`sheath_unmagnetized`.
```

## A short experiment, on purpose

A plasma is prepared once at a fixed reflectivity, **outside** the differentiated
calculation and therefore independent of $r$. The trial value is then applied and the
response is watched for twenty-five steps at $\omega_{pe}\Delta t = 0.15$, which is
$3.75$ inverse plasma frequencies and so **0.60 of an oscillation**. The two differ by
$2\pi$, and a window quoted in periods when it is inverse frequencies is six times longer
than it sounds; the script prints both. It is the time in which the electrons rearrange
and the wall's charge begins to follow — the beginning of the response, not a settled
one — and it is as long as the gradient can usefully be taken over.

The script measures why. At $r = 0.25$, from a state prepared over 1200 steps:

| horizon | forward against reverse | best central difference | at $h$ |
|---|---|---|---|
| 5 steps | {{ gradient_modes_agree_5 }} | {{ gradient_best_mismatch_5 }} | {{ gradient_best_step_5 }} |
| 25 steps | {{ gradient_modes_agree_25 }} | {{ gradient_best_mismatch_25 }} | {{ gradient_best_step_25 }} |
| 100 steps | {{ gradient_modes_agree_100 }} | {{ gradient_best_mismatch_100 }} | {{ gradient_best_step_100 }} |

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
sensor comes out at a ratio of 0.5 and carries almost no information; the sheath sensor
at 13 and carries it all. That is worth seeing rather than hiding, and it is what put
the sheath sensor where it is — a scan of positions gives 4.7 at 0.8 Debye lengths from
the collector, 3.9 at 1.5, 2.9 at 2.5 and 0.9 at 6, and 1.5 is as close as a Gaussian of
this width can sit without taking part of its reading from the cells the deposit
truncates at the wall.

Starting at $r = 0.08$ with the answer at $r = 0.35$, the script runs the same descent
twice, against two targets that answer two different questions.

The **self-test** fits a target built on the same four realisations and the same
protocol. Its minimum is exactly at the reference by construction, so what it tests is
the differentiated chain end to end — the source, the wall, the electrode closure, the
gradient — and not the ability to infer anything. It recovers $r = 0.3500$ in fourteen
iterations of projected gradient descent with backtracking, with the loss falling from
2.008 by more than six orders of magnitude.

The **inference** fits a target measured on six further realisations the optimiser never
sees. Nothing in it is zero by construction. It stops after ten iterations at
$r = 0.3264$, an error of 0.0236, on a residual loss of 0.0512 — the noise it could not
fit. That number, and the error bar below, are the inference; the four decimal places
above are the self-test.

| | $r$ | training loss | held-out loss |
|---|---|---|---|
| start | 0.0800 | 2.00814 | 2.02135 |
| self-test | 0.3500 | 0.00000 | 0.00000 |
| inference | 0.3264 | 0.01495 | 0.01511 |
| reference | 0.3500 | 0 | 0 |

Both loss columns are measured against the paired target, which is why the inference's
row is not zero there: a hundredth of a scatter unit is what fitting data the optimiser
did not generate costs.

Either descent stops on a criterion it names — here, both times, the step fell below the
$10^{-4}$ the scan can resolve — and evaluates and records the point it ends on, so the
value returned is one it stood on rather than the last one it happened to have measured.

**How well the control is known** is a third question, and three things that were one are
now separate:

* the **scan's spacing**, 0.0200 over 26 points, anchored so that the reference is one of
  them — the earlier grid had no node at $r = 0.35$, so the distance from its nearest node
  was 0.01 whatever the run did, and that was being reported as the uncertainty;
* the **minimum**, refined off the grid by a parabola through the three lowest samples,
  which the spacing does not limit: $r = 0.3443$;
* the **scatter between realisations**, which is the only one of the three that is an
  uncertainty. Each held-out realisation has its own minimum — 0.3156, 0.3046, 0.3818,
  0.3812 — so the control is recovered as $0.3443 \pm 0.0207$, a standard error over four
  realisations, and the reference sits 0.3 of one away.

The honest error bar is therefore twice the number the grid was reporting, and it is an
error bar rather than a resolution. The self-test's value sits 0.0056 from that minimum
and the inferred one 0.0179, so both land inside it, which is the most that four short
held-out realisations support.

`--quick` is a smoke run: a sixth of the particles, a quarter of the preparation and
three realisations instead of four, in about forty seconds. It checks that the script
executes and that the gradient is still the derivative of the calculation. Both descents
run out of their eight iterations rather than converging, the inference lands at
$r = 0.44$, and with three noisy realisations and eight scan points the error bar is
$\pm 0.09$ against the full preset's $\pm 0.02$. That is the check doing its job; the
numbers above are the full preset's, about ten minutes.

## The same experiment in a magnetic field

`--oblique` adds a uniform field 30 degrees to the wall at $\rho_s/\lambda_D = 6$. One
array is passed to the same `Simulation` and nothing else in the script changes; the run
writes its own `sheath_optimization_oblique/` folder, so the two experiments are two
records rather than one overwriting the other.

The gradient stays exact: forward against reverse to $1\times10^{-14}$, and a central
difference of the same realisation agrees to $2\times10^{-10}$ at $h = 10^{-4}$. The
self-test recovers $r = 0.3500$ as before, because its target is still zero at the
reference by construction.

What the field takes away is the measurement. The sheath sensor's response over the
admissible interval falls from 13 times the scatter between realisations to 0.6, and the
plasma sensor's from 0.5 to 0.0 — over the whole interval it does not move at all. The
electrons are magnetised, $\rho_e/\lambda_D = 0.3$ here, so reflecting a fraction of the
electron flux moves the sheath potential far less, while the noise does not fall with it.

The inference then fails, and it is worth seeing what failure looks like. Against the
independent target the optimiser walks to the upper bound and stops: $r = 0.5000$ against
a reference of 0.35, on a residual of 0.33. The held-out scan agrees that there is
nothing there to find — its minimum is $0.3327 \pm 0.1131$, an error bar five times the
field-free one, and the four per-realisation minima are 0.02, 0.34, 0.50 and 0.50, two of
them pinned at a bound. A self-test that still passes beside an inference that does not is
the distinction the two targets exist to draw.

## What this is and is not

The control is an idealised wall-response parameter, represented as a deterministic
fraction of each macro-particle's weight rather than a random hit-or-miss. It is not a
material coating, a sputtering yield or a heat-load optimisation, and this is a
finite-time response calibration rather than the derivative of an asymptotic floating
state.

```bash
python examples/3_advanced/sheath_optimization.py            # about ten minutes
python examples/3_advanced/sheath_optimization.py --quick    # about forty seconds
```
