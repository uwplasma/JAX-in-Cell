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

`vth` has three components and all three are used: the normal one sets the crossing
distribution above, and the two tangential ones are drawn from their own Maxwellians,
since the plane does not select on them. A tangential drift rides along untouched for the
same reason. `vth=(v, 0, 0)` is therefore a reservoir with no tangential motion, which is
a different inflow from an isotropic one — in a magnetised sheath, a very different one.

The normal drift $u$ is where the three models differ, and which one a `Source` is gets
decided once, when it is built, and kept as `model`:

| `model` | reservoir | normal speed |
|---|---|---|
| `"beam"` | cold, `vth = 0` | every particle at $u$ |
| `"maxwellian"` | warm, no normal drift | $\sigma\sqrt{-2\ln U}$, Rayleigh |
| `"drifting"` | warm, drifting | the quantile of $p(v)\propto v e^{-(v-u)^2/2\sigma^2}$ |

The third has no elementary inverse, so it is sampled by inverting its distribution
function numerically. What is differentiated is not the inversion — a comparison has
derivative zero, and a fixed number of bisections would report no sensitivity to $u$ at
all — but the equation $F(v,u)=p$ itself, which is the implicit function theorem and is
exact. Shifting a Rayleigh sample by $u$ is none of the three; it agrees with the drifting
model only at $u=0$.

The **sign** of $u$ is physical, not a magnitude: a reservoir drifting away from the plane
still sends the tail of its distribution across, at the reduced flux
$\Gamma = n[u\Phi(u/\sigma) + \sigma\varphi(u/\sigma)]$ with $u<0$, and a cold beam
pointing away from the plane sends nothing and is refused rather than turned around.

`model` is static because it selects a branch. Inside `jit` every leaf is a tracer, so
`vth != 0` is a traced array and not a Python `True`; a `Source` built from a traced `vth`
or a traced normal drift has to say which model it is.

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
per wall, of the weight that arrived, the part the wall kept, the weight a source emitted,
the kinetic energy carried in, back out and injected, and the momentum delivered and
injected. Differences between two stored steps are what happened in between, so a flux is
a difference divided by a time and a collected charge is a difference times a charge.

Every exchange is recorded after the wall's law has acted. A thermal wall is a heat bath:
what it returns is a fresh draw from its own half-Maxwellian flux, so `energy_out` can
exceed `energy_in` and the difference is the heat the wall gave the plasma. Recorded
before the redraw, a thermal wall with restitution one reports `energy_in` and
`energy_out` identical and appears to exchange nothing at all. Kinetic energy is
$m|\mathbf u|^2/(\gamma+1)$, which is $mv^2/2$ in a Newtonian run and the relativistic
energy from the carried momentum otherwise, so `Solver(relativistic=True)` does not
quietly change what the ledger means.

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
