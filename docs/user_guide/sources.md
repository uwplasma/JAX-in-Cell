# Sources and electrodes

A {class}`~jaxincell.Source` is a reservoir of plasma behind one plane of the box, supplying
a prescribed flux independently of what leaves, so a source-to-collector problem reaches a
steady state and can be compared with theory. Without one the box drains: an absorbing wall
empties it, and a thermal wall returns what reaches it but cannot replace what another wall
takes, so the sheath between a thermal wall and a collector is a transient and a run has to
stop before the drain matters.

```{code-block} python
:caption: a maintained sheath, in full

from jaxincell import Domain, Simulation, Solver, Source, Species, mass_electron

domain = Domain(length=length, cells=120,
                particle_bc="absorbing",             # both planes take back what reaches them
                field_bc=("open", "absorbing"))      # left imposes nothing, right is the collector

electrons = Species("electrons", 120000, -1.0, mass_electron, density, (vth_e, 0, 0),
                    active=30000,                    # 30000 of the 120000 slots start filled
                    source=Source(density=density, vth=(vth_e, 0, 0), emit=120))
ions = Species("ions", 120000, 1.0, mass_ion, density, 0.0, (beam, 0, 0), active=30000,
               source=Source(density=density, vth=0.0, drift=(beam, 0, 0), emit=120))

out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(3000)
```

## Arguments

| argument | meaning | default |
|---|---|---|
| `density` | reservoir number density $n_{\rm in}$, m$^{-3}$ | `0.0` |
| `vth` | thermal speed per component of the reservoir; zero makes a cold beam | `(0.0, 0.0, 0.0)` |
| `drift` | drift velocity of the reservoir, m/s | `(0.0, 0.0, 0.0)` |
| `side` | `"left"` or `"right"`, the wall the plasma enters through | `"left"` |
| `emit` | particles emitted per step | `0` |
| `model` | which crossing distribution the sampler draws from; decided when the object is built | `None` |
| `min_weight` | fraction of the emitted weight at or below which a wall keeps the remainder instead of reflecting again | `1e-3` |
| `samples` | velocities of the reservoir itself, `(k, 3)` in m/s | `None` |

## What crosses a plane

The reservoir is specified by its distribution $f_{\rm in}(\mathbf v)$, not by the particles
that appear; what enters is the flux it sends across the plane,

$$\Gamma = \int_{v_n>0} v_n f_{\rm in}(\mathbf v)\,d^3v,
\qquad p_{\rm cross}(\mathbf v) = v_n f_{\rm in}(\mathbf v)/\Gamma,$$

the velocity density weighted by the normal speed: a fast particle crosses more often than
a slow one.

* For a Maxwellian at rest of component spread $\sigma = v_{th}/\sqrt2$ the normal speed is
  Rayleigh, $v_n = \sigma\sqrt{-2\ln U}$, with $\Gamma = n\sigma/\sqrt{2\pi}$, mean normal
  speed $\sigma\sqrt{\pi/2}$ and mean square $2\sigma^2$ — twice the variance of the
  distribution it came from.
* Half a Maxwellian is not the same thing, and a Maxwellian sampled and then cut at $v>0$ is
  a third.
* All three components of `vth` are used: the normal one sets the crossing distribution, the
  two tangential ones are drawn from their own Maxwellians. A tangential drift rides along
  untouched.
* `vth=(v, 0, 0)` is a reservoir with no tangential motion — a different inflow from an
  isotropic one, and in a magnetised sheath a very different one.

## Models

The normal drift $u$ is where the models differ.

| `model` | reservoir | normal speed |
|---|---|---|
| `"beam"` | cold, `vth = 0` | every particle at $u$ |
| `"maxwellian"` | warm, no normal drift | $\sigma\sqrt{-2\ln U}$, Rayleigh |
| `"drifting"` | warm, drifting | the quantile of $p(v)\propto v e^{-(v-u)^2/2\sigma^2}$ |
| `"sampled"` | given as `samples` | weighted by the inward normal component |

* `"drifting"` has no elementary inverse, so it is sampled by inverting its distribution
  function numerically. What is differentiated is the equation $F(v,u)=p$ itself — the
  implicit function theorem, and exact — not the inversion: a comparison has derivative
  zero, and a fixed number of bisections would report no sensitivity to $u$ at all.
* Shifting a Rayleigh sample by $u$ is none of the models; it agrees with `"drifting"` only
  at $u=0$.
* The **sign** of $u$ is physical, not a magnitude. A reservoir drifting away from the plane
  still sends the tail of its distribution across, at the reduced flux
  $\Gamma = n[u\Phi(u/\sigma) + \sigma\varphi(u/\sigma)]$ with $u<0$; a cold beam pointing
  away sends nothing and is refused rather than turned around.
* `model` is static because it selects a branch. Inside `jit` every leaf is a tracer, so
  `vth != 0` is a traced array and not a Python `True`, and a `Source` built from a traced
  `vth` or a traced normal drift has to say which model it is.

## A reservoir that is not a formula

Some inflows are none of the closed forms. The entrance condition of a magnetised presheath
is $F \propto v_\parallel^2\exp(-v_\parallel^2/2 - v_\perp^2/2)$ along a field meeting the
wall at a few degrees {cite}`geraldini2019`: the $v_\parallel^2$ is the kinetic Chodura
condition, which empties the distribution at zero parallel velocity, and the field angle
mixes the parallel and perpendicular directions into every Cartesian component. Neither the
hole nor the correlation survives being written as one normal distribution across the plane
and two along it. `samples` takes the distribution as velocities instead:

```python
Source(density=n, samples=v, emit=120)     # v is (k, 3), the reservoir's own velocities
```

* They are the distribution **behind** the plane, not the flux across it. The sampler
  weights them by their inward normal component — what makes a fast particle cross more
  often — and never draws one going the other way.
* The flux is $\Gamma = n\langle v_n\rangle_+$ over the same samples, so the two cannot
  disagree.
* `vth` and `drift` are then unused, and `model` comes out as `"sampled"`.
* $k$ sets the resolution of the inflow: the flux and the moments it reproduces carry the
  Monte Carlo error of the sample given, a few parts in a thousand at $k = 10^5$.
* It is data rather than a model, so the emitted weight is differentiable in `density` and
  not in the samples: another code's output, a measurement, or a distribution with no
  closed form.

## Weights, not counts

Each step emits a fixed number `emit` of particles carrying the continuous weight

$$w = \Gamma\,\Delta t/N_{\rm emit},$$

so the weight put in over a step is exactly $\Gamma\Delta t$ and is a differentiable
function of the reservoir's density and temperature.

* A source emitting $\lfloor\Gamma\Delta t/w\rfloor$ particles of fixed weight would put in
  the same plasma on average and have no derivative at all: a count is an integer, and the
  derivative of a step function is zero almost everywhere. Continuous weights are what make
  the reservoir an optimisable control.
* Entry times are a quiet quadrature of the step, $s_k = (k+\tfrac12)/N_{\rm emit}$, and each
  particle streams freely for the remaining $(1-s_k)\Delta t$ before its first push.
* A particle's first deposit is therefore about one step's flight inside the plane, biasing
  the density there by of order $\langle v\rangle\Delta t/L$ — a per cent in the examples
  here, and smaller with a smaller time step.

## Capacity, not population

With a source, `Species.n` is a **capacity**: a pool of slots, of which `active` start filled
and the rest start dead.

* A dead slot is parked beyond a wall with no weight and no charge-to-mass ratio, where it
  deposits nothing, feels nothing and collides with nothing; the source refills the emptiest
  slots it can find. Nothing changes shape, so the whole loop stays compiled.
* Give the pool headroom: the steady population is `emit` times the residence time in steps.
* `Output.overflow` is the largest live weight a source has had to overwrite. Zero while the
  pool holds, positive when it does not — a capacity that needs raising rather than a density
  quietly set by an array size. It is a running maximum, so a run cannot look valid because
  it recovered later.
* `Output.problems` turns that into sentences and `Output.validate()` raises on them:
  `out = simulation.run(...).validate()` gives nothing rather than numbers it should not
  use. Both read values, so both belong on the host; a differentiated objective takes
  `Output.overflow` out with its result and rejects the trial itself, or checks the worst
  case of its admissible interval once before it starts.

A wall returning the fraction $R$ of every impact would hold a particle for ever, its weight
falling as $R^k$ and its slot never free. Below `Source(min_weight=...)` the wall keeps the
remainder instead; it goes on the ledger, so the charge and energy balances stay exact, and
what changes is only where the last $10^{-3}$ of a particle lands. `Wall.truncated` is the
weight a wall kept only because the cutoff stopped the orbit, which the reflection law would
otherwise have sent back, so a run says what the cutoff cost. It falls with `min_weight`,
which is how far a result should be refined before it is compared with anything.

## The electrical boundary

A particle boundary and an electrical one are separate choices: `particle_bc` says what
happens to a particle that reaches a wall, `field_bc` what the field does there.

| `field_bc` | the field there |
|---|---|
| `"reflective"` | a symmetry plane, where $E_x$ vanishes |
| `"absorbing"` | a conductor that keeps the charge it collects; two of them are short-circuited to each other, so the potential difference across the box is zero |
| `"open"` | the plane a source supplies through; it imposes nothing at all |

With `"open"` the constant of the Gauss solve comes from the collector opposite, whose field
is the charge it holds,

$$E_x(L^-) = -\sigma_w/\epsilon_0 .$$

The potential is then measured from the source plane, which is the gauge, and the field there
is free to respond to the current the reservoir drives.

* `field_bc=("open", "absorbing")` is the combination a maintained sheath needs. With a
  symmetry plane opposite and nothing crossing it, closing on the collector's charge and
  closing on $E_x=0$ give the same field to the charge the deposit truncates at the walls,
  which falls with the cell size; with a source they do not, and only the electrode closure
  is right.
* A source needs `Solver(model="electrostatic")` and the explicit scheme. Ampere's law
  integrates a current, and a particle that appears inside the box has no trajectory through
  the wall for that current to carry; the electrostatic solve takes $E_x$ from the charge
  density and is exact with a source.
* The implicit scheme would emit a new population inside every Picard iteration, which its
  fixed-point argument does not allow. Both are refused when the `Simulation` is built rather
  than silently half-done.

## What the walls did

`Output.wall` is a {class}`~jaxincell._simulation.Wall`: running totals, per species and per
wall, of the weight that arrived, the part the wall kept, the weight a source emitted, the
kinetic energy carried in, back out and injected, and the momentum delivered and injected.
Differences between two stored steps are what happened in between, so a flux is a difference
divided by a time and a collected charge is a difference times a charge.

* Every exchange is recorded after the wall's law has acted. A thermal wall is a heat bath:
  what it returns is a fresh draw from its own half-Maxwellian flux, so `energy_out` can
  exceed `energy_in`, and the difference is the heat the wall gave the plasma. Recorded
  before the redraw, a thermal wall with restitution one would report the two identical and
  appear to exchange nothing.
* Kinetic energy is $m|\mathbf u|^2/(\gamma+1)$ — $mv^2/2$ in a Newtonian run, the
  relativistic energy from the carried momentum otherwise — so `Solver(relativistic=True)`
  does not quietly change what the ledger means.
* A floating collector draws no net current once it has charged, and how close to zero it
  gets measures how well the run has settled.

```{code-block} python
:caption: the current the collector drew over the second half of a run

collected = out.wall.collected            # (stored, species, side), side 1 the right wall
late = collected[-1, :, 1] - collected[len(out.t) // 2, :, 1]
current = e * (late[1] - late[0]) / (out.t[-1] - out.t[len(out.t) // 2])   # A/m^2
```

## Profiles without a particle history

`run(moments=...)` sums the velocity moments of each species over **every** step and stores
the running sums, so the mean profile of a window is the difference of two of them divided by
the number of steps between. It needs no particle history, which is what makes it affordable
where a run of a hundred thousand particles over three thousand steps cannot store its phase
space.

| `moments=` | rows | gives | cost of a step |
|---|---|---|---|
| `"density"` | 1 | the density | 20 % |
| `"flux"` | 4 | and the mean velocity | 20 % |
| `"full"`, or `True` | 10 | and the pressure and temperature **tensors** | 58 % |

Measured over 200 steps at 120000 slots on 256 cells, each in its own process. What the six
second moments spend is memory traffic rather than deposits — ten deposits are 0.71 ms
against three at 0.42 — because $v_iv_j$ is six more arrays the length of the particle list at
every step. {func}`~jaxincell.moment_profiles` turns a window of the sums into whichever of
the four a run kept.

Take the number of steps from `Output.steps`, not from the shape of the array:

```{code-block} python
:caption: the mean profile over the second half of a run

late = len(out.t) // 2
window = (out.moments[-1] - out.moments[late]) / (out.steps[-1] - out.steps[late])
```

* A sum is stored *after* the chunk it ends, so the sums at stored steps $a$ and $b$ are
  `steps[b] - steps[a]` apart — the half-open window $(t_a, t_b]$ — and not one chunk more.
* Counting the chunks instead is off by one and reads a constant profile back at
  $(b-a-1)/(b-a)$ of itself: 3.3 % low over half of sixty stored steps, in the direction that
  makes a plasma look thinner than it is.
* `steps` is absolute, so the same expression holds across a restart.
* {func}`~jaxincell.bohm_edge` reads a sheath edge off a measured flow profile, interpolating
  the crossing and returning how many crossings there are. None is an answer: a supersonic
  source has no Bohm point, and the function says so rather than returning the first bin and
  a potential drop to go with it.
