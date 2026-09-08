# Initialising the phase space

{meth}`~jaxincell.Simulation.initial_state` turns the {class}`~jaxincell.Species`
list into particle arrays and the initial fields. Everything here happens inside the
compiled program, so the sampling is a function of the traced parameters and can be
differentiated through.

## Positions

By default positions are equally spaced,
$x_p = -L/2 + (p + \tfrac12)L/N$, which is the lowest-noise choice for a uniform
plasma: the deposited density is uniform to round-off, whereas random placement leaves
a $1/\sqrt{N}$ density fluctuation in every mode. Set `random_positions=True` for
uniform random placement when the noise itself is the object of study.

The perturbation is a displacement rather than a density change,

```{math}
x_p \to x_p + a\sin\!\left(\frac{2\pi m x_p}{L}\right),
```

with `perturbation_amplitude` $=a$ and `perturbation_mode` $=m$. To first order this
makes a density perturbation $\delta n/n = -a k\cos(kx)$ with $k = 2\pi m/L$, so the
dimensionless seed usually quoted in the literature is $ak$: to seed a one per cent
perturbation of mode $m$, set `perturbation_amplitude = 0.01 * L / (2 * pi * m)`.
Seeding by displacement keeps every pseudo-particle's weight identical, which the
charge-conserving deposit relies on.

## Velocities

Velocities are Maxwellian with the convention

```{math}
f(v) \propto \exp\!\left(-\frac{(v-u)^2}{v_{th}^2}\right), \qquad
v_{th} = \sqrt{\frac{2k_BT}{m}},
```

so `vth` is $\sqrt2$ times the standard deviation, and
$\lambda_D = v_{th}/(\sqrt2\,\omega_p)$. `drift` adds $u$ per component.

**Random start** (default). Each component is drawn from a normal distribution of
standard deviation $v_{th}/\sqrt2$ using the run's PRNG key, so `seed` changes the
realisation.

**Quiet start** (`quiet=True`). The velocity of particle $p$ is placed at a quantile
of the Maxwellian,

```{math}
v_p = v_{th}\,\mathrm{erf}^{-1}(2u_p - 1),
```

where $u_p$ is the $p$-th element of a van der Corput (bit-reversed) sequence, in
bases 2, 3 and 5 for the three components. The sequence fills $[0,1)$ far more evenly
than random numbers, so the sampled distribution matches the Maxwellian to much better
than $1/\sqrt N$ and the noise floor of the run drops by orders of magnitude. That is
what makes the growth rates of {doc}`verification` measurable at all: with a random
start the modes emerge from the noise floor and saturate two e-foldings later, which
is not enough to fit anything.

:::{note}
A quiet start suppresses the noise so thoroughly that unseeded modes have nothing to
grow *out of*. When the point of a run is which modes are unstable rather than how
fast one of them grows, use a random start so that a well-defined noise floor exists;
see the two panels of {doc}`verification`'s Weibel figure.
:::

### Counter-streaming beams

`plus_minus=True` negates $v_x$ on every second particle, turning one drifting
population into two counter-streaming beams of half the density each. Combined with a
quiet start this needs care, and the code handles it explicitly: the base-2 van der
Corput value is below one half exactly when the index is even, which is the same
parity the sign flip uses, so the naive combination would hand one beam the lower half
of the Maxwellian and the other the upper half. Instead $N/2$ quantiles are drawn and
each is given to both beams, making them exact mirror images that each sample the whole
distribution.

### Custom phase space

`Species.replace(x=..., v=...)` substitutes arrays of shape `(n, 3)` for the generated
phase space, which is how anything the generator does not cover is built: a
bi-Maxwellian with a coherent seed, a ring distribution, a slab. The helper
{func}`~jaxincell.quiet_start` returns the equally spaced positions and quantile
velocities as plain arrays so that a custom condition can be built on top of the quiet
start rather than instead of it:

```python
from jaxincell import quiet_start, Species

x, v = quiet_start(n, length, vth=(vx, 0.0, vz))
v[:, 2] += 1e-3 * vz * np.sin(2 * np.pi * x[:, 0] / length)   # seed a Weibel mode
electrons = Species.electrons(n=n, density=n_e, vth=(vx, 0.0, vz)).replace(x=x, v=v)
```

## Fields

$\mathbf B$ starts at zero. $E_x$ is taken from the discrete Gauss law applied to the
charge density deposited from the initial positions ({doc}`field_solvers`), so the
constraint holds from the first step; the transverse components of $\mathbf E$ start
at zero. Static external fields, if given, are added at the gather and never evolve.

Finally, in the explicit scheme, positions are displaced by
$+\tfrac12\Delta t\,\mathbf v$ to set up the leapfrog, and velocities are clipped to
$0.99c$ so that the relativistic $\gamma$ is finite.
