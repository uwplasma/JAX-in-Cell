# Sources and electrodes

A wall that absorbs particles empties the box. A thermal wall returns what reaches it,
which keeps a population alive but cannot replace what another wall takes, so a plasma
between a thermal wall and a collector drains: the sheath it holds is a transient, and a
run has to stop before the drain matters.

A {class}`~jaxincell.Source` removes that limit. It is a reservoir of plasma behind one
plane of the box, supplying a prescribed flux that does not depend on what leaves, so a
source-to-collector problem reaches a steady state and can be compared with theory.

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

## What crosses a plane

The reservoir is specified by its distribution $f_{\rm in}(\mathbf v)$, not by the
particles that appear. What enters is the flux that distribution sends across the plane,

$$\Gamma = \int_{v_n>0} v_n f_{\rm in}(\mathbf v)\,d^3v,
\qquad p_{\rm cross}(\mathbf v) = v_n f_{\rm in}(\mathbf v)/\Gamma,$$

the velocity density weighted by the normal speed, because a fast particle crosses a
plane more often than a slow one. For a Maxwellian at rest of component spread
$\sigma = v_{th}/\sqrt2$ that makes the normal speed Rayleigh distributed,
$v_n = \sigma\sqrt{-2\ln U}$, with $\Gamma = n\sigma/\sqrt{2\pi}$, mean normal speed
$\sigma\sqrt{\pi/2}$ and mean square $2\sigma^2$ — twice the variance of the
distribution it came from. Half a Maxwellian is not the same thing, and a Maxwellian
sampled and then cut at $v>0$ is a third.

Two reservoirs are supported: a **Maxwellian at rest**, and a **cold beam**, every
particle at the drift velocity. A drift along the normal together with a finite
temperature is refused when the `Source` is built, because the crossing density of a
drifting Maxwellian is proportional to $v\exp[-(v-u)^2/2\sigma^2]$ on $v>0$ and adding a
drift to a Rayleigh sample is not a sample of it. A drift in the ignorable directions is
free: the plane does not select on it.

## Weights, not counts

Each step emits a fixed number `emit` of particles carrying the continuous weight

$$w = \Gamma\,\Delta t/N_{\rm emit},$$

so the weight put in over a step is exactly $\Gamma\Delta t$ and is a differentiable
function of the reservoir's density and temperature. A source that emitted
$\lfloor\Gamma\Delta t/w\rfloor$ particles of fixed weight would put in the same plasma
on average and have no derivative at all: a particle count is an integer, and the
derivative of a step function is zero almost everywhere. That distinction is the reason
the source is built this way, and it is what makes the reservoir an optimisable control.

Entry times are a quiet quadrature of the step, $s_k = (k+\tfrac12)/N_{\rm emit}$, and
each particle streams freely for the remaining $(1-s_k)\Delta t$ before its first push.
A particle's first deposit is therefore about one step's flight inside the plane, which
biases the density there by of order $\langle v\rangle\Delta t/L$ — a per cent in the
examples here, and smaller with a smaller time step.

## Capacity, not population

With a source, `Species.n` is a **capacity**: a pool of slots, of which `active` start
filled and the rest start dead. A dead slot is parked beyond a wall with no weight and
no charge-to-mass ratio, where it deposits nothing, feels nothing and collides with
nothing, and the source refills the emptiest slots it can find. Nothing ever changes
shape, so the whole loop stays compiled.

Give the pool enough headroom: the steady population is `emit` times the residence time
in steps, and `Output.wall.overflow` is the largest live weight a source has had to
overwrite. It stays at zero while the pool holds and goes positive when it does not,
which is a capacity that needs raising rather than a density quietly set by an array
size.

A wall that returns the fraction $R$ of every impact would hold a particle for ever, its
weight falling as $R^k$ and its slot never free. `Source(min_weight=...)` is the fraction
of the emitted weight at or below which the wall keeps the remainder instead; the
remainder goes on the ledger, so the charge and energy balances stay exact, and what
changes is only where the last $10^{-3}$ of a particle lands.

## The electrical boundary

A particle boundary and an electrical one are separate choices. `particle_bc` says what
happens to a particle that reaches a wall; `field_bc` says what the field does there.

* `"reflective"` is a symmetry plane, where $E_x$ vanishes.
* `"absorbing"` is a conductor that keeps the charge it collects. Two of them are
  short-circuited to each other, so the potential difference across the box is zero.
* `"open"` is the plane a source supplies through. It imposes nothing at all: the
  constant of the Gauss solve comes from the collector opposite instead, whose field is
  the charge it holds,

  $$E_x(L^-) = -\sigma_w/\epsilon_0 .$$

  The potential is then measured from the source plane, which is the gauge, and the
  field there is free to respond to the current the reservoir drives.

`field_bc=("open", "absorbing")` is the combination a maintained sheath needs. With a
symmetry plane opposite and nothing crossing it, closing on the collector's charge and
closing on $E_x=0$ give the same field to the charge the deposit truncates at the walls,
which falls with the cell size; with a source they do not, and only the electrode
closure is right.

A source needs `Solver(model="electrostatic")` and the explicit scheme. Ampere's law
integrates a current, and a particle that appears inside the box has no trajectory
through the wall for that current to carry; the electrostatic solve takes $E_x$ from the
charge density and is exact with a source. The implicit scheme would emit a new
population inside every Picard iteration, which its fixed-point argument does not allow.
Both are refused when the `Simulation` is built rather than silently half-done.

## What the walls did

`Output.wall` is a {class}`~jaxincell._simulation.Wall`: running totals, per species and
per wall, of the weight that arrived, the part the wall kept, the weight a source
emitted, and the kinetic energy carried in and back out. Differences between two stored
steps are what happened in between, so a flux is a difference divided by a time and a
collected charge is a difference times a charge.

```{code-block} python
:caption: the current the collector drew over the second half of a run

collected = out.wall.collected            # (stored, species, side), side 1 the right wall
late = collected[-1, :, 1] - collected[len(out.t) // 2, :, 1]
current = e * (late[1] - late[0]) / (out.t[-1] - out.t[len(out.t) // 2])   # A/m^2
```

A floating collector draws no net current once it has charged, and how close to zero it
gets is a measure of how well the run has settled.

## Profiles without a particle history

`run(moments=True)` sums the density, the particle flux and the kinetic energy density
of each species over **every** step and stores the running sums, so the mean profile of a
long window is the difference of two of them divided by the number of steps between.
That is a mean over every step rather than over the few the output keeps, and it needs
no particle history at all, which is what makes it affordable: a run of a hundred
thousand particles over three thousand steps cannot store its phase space, and does not
have to. It deposits three moments per species per step and costs 16 % of a step measured
at 120000 slots on a CPU, which is less than the passes suggest because a deposit is cheap
beside the push.

{func}`~jaxincell.bohm_edge` reads a sheath edge off a measured flow profile,
interpolating the crossing and returning how many crossings there are. None is an
answer: a supersonic source has no Bohm point, and the function says so rather than
returning the first bin and a potential drop to go with it.
