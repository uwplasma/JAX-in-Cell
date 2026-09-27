# JAX-in-Cell: correctness, teaching interface, and kinetic benchmarks

**Working plan and live checklist.** It supersedes the sheath-only handoff that PR #42 carried
in its description. It specifies work to do; it is not a report that the work is done. Tick a
box only when a commit, a test and a reproducible number exist for it.

## 0. Where the work happens

| | |
|---|---|
| **Development branch** | **`research-release`**, PR #42 -> `main`, head `433d401` at the time of writing. All changes in this plan are made here. |
| Reference only | `rj/additions-to-pr`, PR #43 -> `research-release`, head `ffa5d6f`. Read it for the example changes and the progress-meter attempt, and port what is worth keeping onto `research-release` deliberately. *(Merged into `research-release` on 2026-09-22 at the maintainer's request, keeping this branch's tree: every change it carried was decided below.)* |

Leave both pull requests open. Do not push to `main`, merge, force-push, tag or publish; the
maintainer chooses the integration order. New commits are authored and committed by
`Rogerio Jorge <rogerio.jorge@ist.utl.pt>`, with no AI author or co-author trailers, and no
existing human attribution is rewritten.

What was taken from PR #43, and what was not. Each was measured before deciding, so the
decisions below are results and not preferences.

* **Weibel: not ported; deferred to W9 with a reason.** The enlargement was built and run --
  twelve marginal wavelengths, 24 cells per wavelength, 12000 steps, 120000 particles, 75 s -- and
  it destroys the only quantitative claim the example makes. See E02. What was kept is the
  precision contract: the script said double precision was "what the conservation checks rely on"
  and it has no conservation checks, so the comment now says what is true and what was measured,
  that single precision gives the same gains to five digits and is 7 % faster.
* **Bump-on-tail: ported.** 2000 to 2400 steps. The fit window is anchored on the peak, so the rate
  is unchanged at 0.1370; the extra steps buy the plateau in the right-hand panel.
* **Landau: not ported.** 500 to 800 steps makes the reproduction *worse*, from -0.1523 to -0.1433
  against the kinetic -0.1533, and the reason is a defect in the example rather than in the step
  count. See E01. The example's hard-coded `amplitude[400:]` was replaced by the relative window the
  figure script and the test already use, which is the same index at 500 steps and no longer silently
  tied to it.
* **Magnetised-sheath quick settings: not ported, and W7 settled why.** They do not hold their pool:
  1.6 sound transits is 19200 steps, in which an ion lives 12000, so 30 emitted a step asks for about
  360000 slots and the preset gives 30000 -- an overflow of an order of magnitude, reported as
  numbers. The settings are recorded in the example as history, where someone comparing the two
  branches will find them. Quick is a smoke preset and now says so when it runs; the third angle and
  the longer run are in the full preset, which holds its pool and takes about an hour.
* **The progress meter** is not a port. Section 5 and W4.

### 0.1 What completion means

Correct code and inputs, the requested teaching workflows, independent numerical evidence,
figures regenerated with provenance, every hand-written number in the documentation traced to a
run that produced it, clean package and documentation builds, and a review report. A ticked box, a
green coverage badge, a successful trace, a pretty plot, or two copies of the same formula agreeing
is not evidence.

Where a literature reproduction fails, keep the disagreement and diagnose it. Do not tune a
test until a wrong answer passes, change a physical model quietly, or manufacture an
instability. A documented negative reproduction is a result; a fabricated positive one is not.

### 0.2 Baseline

Everything below was measured on the machine this branch is developed on, before any W1 change,
so that later timings and tolerances have something to be compared with.

| | |
|---|---|
| Host | Apple M4, 24 GiB, macOS 26.6.2 (`arm64`), 1 CPU device, no GPU |
| Python | 3.13.7; JAX and jaxlib 0.11.1; NumPy 2.5.3 |
| Precision | `jax_enable_x64` is off by default; the examples that need it set it themselves |
| Suite | `pytest -q`: **221 passed in 189 s** |
| Library | 2826 lines over 10 modules in `jaxincell/`; 2970 lines of tests |

`ssh office` is available for anything that needs a GPU; nothing in the register does.

The four rows the register carried as *to reproduce* were reproduced here and are now marked
**confirmed** with their numbers. The scripts that produced them are throwaway; what matters is
that each number is reproducible from the row's own description, and the W2/W3/W6 work turns each
into a test that fails on today's code.

## 1. Strategy: extend this small code

These already exist and are to be repaired or extended, not recreated: frozen configuration
pytrees with static structure and traced physics; explicit and implicit integrators; a real
electrostatic model; `Source`, a fixed-capacity pool, a named `State`, a `Wall` ledger;
streaming moments; `load_toml` and the `jaxincell` entry point; `plot` and a blitted ffmpeg
writer; optional openPMD; host-side theory in `jaxincell.sheath` and `docs/scripts`.

The separation is **numerical kernels / run orchestration / input-output and presentation**.
The kernels stay pure and JAX-compatible: no plotting, progress objects, filesystem access or
host fitting inside a differentiated step.

Three targets, distinguished on purpose:

1. **Maintenance** -- one authoritative implementation of each operator, sampler, diagnostic and
   input conversion.
2. **Runtime** -- no redundant deposits, histories, callbacks, sorts or recompiles.
3. **Pedagogy** -- examples keep the line from mathematics to code. Shortening an example by
   hiding its setup behind `run_example(...)` is not an improvement.

A small `jaxincell/_io.py` is justified if it takes I/O out of `_simulation.py`. A shared host
theory module is justified if it removes duplicated theory from docs and examples. Splitting
`_simulation.py` further needs a clearer dependency graph, not a smaller file. No plugin
framework, no task engine, no class per boundary.

Context, checked on 2026-09-18: an arXiv full-text search for "differentiable particle-in-cell"
returns exactly one paper, this one (arXiv:2512.12160, still v1, no journal reference), and a
GitHub topic sweep of particle-in-cell codes returns this repository as the only JAX entry. The
nearest neighbour is `ergodicio/adept`, which has a 1D1V electrostatic `pic-1d` solver and, since
2026-09-15, a WarpX wrapper that shells out to an external binary rather than differentiating it.
That is the position this work has to be worth: the correctness of the claims matters more than
the feature count.

## 2. Defect register

Every row was re-checked against `research-release`. **confirmed** rows carry the check that
found them. Several are in code this project previously reported as validated; they are listed
plainly for that reason.

### 2.1 Sheath physics, sources, boundaries

| ID | Status | Finding | Required correction |
|---|---|---|---|
| S01 | **fixed (W3)** | Cumulative moments are divided by one interval too many: `(stored-late)*steps//stored` spans `stored-1-late` chunks. A constant 1.0000e16 density reads back 9.6667e15, exactly 29/30. The magnetic example gives 19/20. | Every window is `Output.steps[b] - Output.steps[a]`, the absolute count W1 added, in both sheath drivers, the figure script and the test that had the same bug. The window is the half-open `(t_a, t_b]` and the guide says so. The bias was the dominant error in the sheath benchmark's density comparison: the quick preset's worst disagreement with the kinetic relation falls from 0.082 to **0.050** for electrons and from 0.034 to **0.004** for ions. A frozen uncharged population gives every window an exact answer, and the test states the off-by-one as well as the right answer. |
| S02 | **fixed (W2)** | `sheath_magnetized.py` calls `(x > L/2 - 2*dx) & (w > 0)` an impact. That is snapshot occupancy: it repeats particles across frames, includes outgoing ones, and weights by population instead of crossing flux. | `Impacts` bins the energy and incidence of every crossing at the crossing, from the velocity that carried the particle there, into `Wall.spectrum`, with an overflow bin above `energy_max` so a range chosen too small is visible. Summing it gives `Wall.arrived` exactly. Validated against two closed forms in a free-streaming reservoir: a mean impact energy of 2 T_e and the cosine law `p(theta) = sin 2theta`, whose mean incidence is 45 degrees and which it reproduces bin by bin to 0.01 and in the mean to 0.1 degrees. The example builds its spectra from it and no longer stores a particle history at all. The published table is withdrawn -- see S09 for why it was never a measurement. |
| S03 | **fixed (W2)** | `sample_crossing` uses `source.sigma` (from `vth[0]`) for all three components; `Source.vth[1:3]` is ignored. `test_the_sampler_draws_the_flux_and_not_the_velocity_density` asserts the wrong behaviour. | `Source.sigma` is the three spreads and the sampler uses them. Every reservoir in the repository now states all three explicitly, at the isotropic values the bug was giving them, so the change is bit-for-bit invisible in the existing results -- the quick sheath preset returns the same -0.8107 +- 0.0691 -- and the anisotropy is now a choice rather than an accident. The test asserts the components separately, including a degenerate one. |
| S04 | **fixed (W2, W7)** | Magnetic initial particles have transverse spread and a normal ion drift while the source is at rest and isotropic (via S03), and the source does not inherit the species drift. | Every distribution is now explicit: each reservoir states its three spreads, and with S05 a source can be given the drift its species has. The magnetised example's own inconsistency -- initial ions a beam at `c_s` along `x` with no transverse spread, their reservoir an isotropic Maxwellian at rest -- is resolved in W7 by giving both the same thing: Chodura's entrance, `c_s` **along B**, so the normal component is `c_s sin(alpha)` and the reservoir is a drifting Maxwellian, which is what S05's sampler was built for. |
| S05 | **fixed (W2)** | Only an at-rest Maxwellian and a cold beam are supported; the drifting crossing distribution is refused. | `Source.model` is `"beam"`, `"maxwellian"` or `"drifting"`, settled at construction because it selects a branch, and the third inverts `F(v,u)=p` by bisection with a `custom_jvp` that differentiates the equation rather than the comparisons. Checked against quadrature of `p` itself at `u/sigma` of -1, 0.5 and 2, against the Rayleigh closed form at `u=0` to 1e-9, and its derivative against a central difference of the bisection to 1e-6. The sign of `u` is carried: a reservoir drifting away sends the reduced flux, and an outward cold beam is refused. The fast paths are kept -- the drifting branch costs 115 us against 60 us for 120 particles. |
| S06 | **fixed for the electrode closure (W2)** | Cloud charge is truncated outside a wall before the centre crosses, while surface charge is credited at centre crossing. A unit charge walked through an absorbing wall in 0.1 dx steps has volume-plus-surface charge between 0.500 and 1.405: half of it has vanished when the centre is at the wall, and 40 % too much exists one tenth of a cell later. | The part of a live cloud that reaches past the collector is a third category, reversible and on the surface, and `field_bc=('open','absorbing')` closes on the collected charge **plus** it. Deposited plus exterior is then one at every sub-cell offset, and a sheet crossing the wall leaves the field at an interior face unchanged to 1e-12 where it used to jump by 5.20e10 V/m against a half-particle's 5.65e10. In a ten-Debye-length sheath the truncated part was 0.4 % of the electron charge, 0.6 % of the ion charge and 0.29 % of what the collector held; the quick preset's wall potential moves from -0.8107 to -0.8197 against a standard error of 0.059, so the shift is real but not resolved at that size. **The other absorbing closures still truncate**: `(2,2)` and `(2,1)` fix their constant from a potential difference rather than a surface charge and would need the closure rederived. |
| S07 | **fixed (W2)** | Injected particles free-stream for a residual step with no field interaction; source correctness is inferred from the Gauss solve alone. | The staggering is written down in `inject`: a particle enters at `t^n + (s_k - 1/2) dt` and is put in the arrays at the position it reaches by `t^{n+1/2}` and with the velocity it would have had at `t^n`, both from the field at its entry plane -- the position to second order in the flight, the velocity from the same pusher run over the signed interval `(1/2 - s_k) dt`, which is exact for a uniform field and is what keeps a magnetised entry's gyro-phase. Tested against two closed forms rather than against the Gauss solve: a cold beam falling through a prescribed uniform E, where the worst error falls from 1.1e-3 to 2.0e-5, and one turning in a prescribed uniform B at `Omega dt = 0.079`, where the mean falls from 4.0e-2 to 9.4e-4. |
| S08 | **fixed (W2)** | With an open plane the continuity current is anchored at zero, now documented as "the internal transport measured from the source plane". Honest, but not an absolute current. | The closure is the collector's own conduction current, `sigma_w_dot`, taken as a difference over the same interval the density change spans. Ampere's law makes the total current uniform in one dimension, so with a floating electrode `J + eps_0 dE/dt` must vanish **at every face**, and it does, to 9e-15 of the current's own size; anchored at zero the residual was 0.62 of it. The three are now distinguishable in the output: `Output.J` is the conduction current, `eps_0 dE/dt` the displacement current, and their sum the circuit current, which is zero while the collector floats. The implicit scheme carries no surface charge and refuses the open plane rather than anchoring on nothing. |
| S09 | **fixed (W2)** | `wall.overflow` exists and `sheath_unmagnetized.py` warns on it. `sheath_magnetized.py` and `sheath_optimization.py` ignore it, and the library never invalidates a run. | `Output.problems` is the reasons a run is unusable and `Output.validate()` raises on them, so a script writes `run(...).validate()` and gets nothing instead of numbers it should not use. `Output.overflow` is the running maximum, so a run cannot look valid because it recovered. The two sheath examples validate; the optimisation checks the most reflective end of its admissible interval once, on the host, which is the longest electron lifetime any trial can ask for. The per-injection check was already there -- `overflow = max(w[slots])` -- and is what these read. |
| S10 | **fixed (W2)** | The reflection weight cutoff truncates the last part of an orbit and adds a branch. | `Wall.truncated` is the weight a wall kept only because `min_weight` stopped the orbit, which its reflection law would otherwise have returned: the cost of the cutoff, in the ledger's own units. A test holds it to fall by more than ten with the cutoff from 1e-1 to 1e-3, so a result can be refined until the budget is below what it is being compared against. The branch's effect on sensitivities stays with G09 and W3's derivative matrix. |
| S11 | **fixed (W2)** | `_record` ran before `_thermalise`, so `energy_out` was the specular energy of the bounce, not the energy of the redrawn particle. At a thermal wall with restitution 1 the ledger reported `energy_in - energy_out` identically zero, which is not a measurement: a run whose walls returned 1.1159e-08 J/m^2 against 1.0716e-08 received was reported as exchanging nothing. | The record is taken after the wall's law has finished acting, including the redraw, in the explicit step, the implicit sub-step and the initial half step -- which did not thermalise at all, though its own comment said it had to meet a wall the way every later step would. `Wall` carries `energy_injected`, `momentum` and `momentum_injected`. The thermal wall's returned energy is checked against the closed form `m(2 sigma_x^2 + sigma_y^2 + sigma_z^2)/2` per unit weight, and the injected energy and normal momentum against theirs. |
| S12 | **fixed (W1, W2)** | Wall energies use `m v^2/2` on relativistic paths, and nonrelativistic collisions can be combined with a relativistic pusher. | The ledger takes energy from the carried momentum as `K = m|u|^2/(gamma+1)`, which is `m v^2/2` when gamma is one and does not subtract two large numbers when it is not; at 0.9 c it is 3.19 times the Newtonian value, which is what the test asserts. Takizuka-Abe beside the relativistic pusher is refused (W1). |
| S13 | **fixed (W1)** | `active=0` reaches `jnp.arange(n) % s.active` and `L / s.active`. Called directly it is a `ZeroDivisionError`; inside `jit` the infinity lands in the untaken branch of the weight's `where`, the forward run survives, and `jax.grad` returns **NaN** -- on the one configuration a source-driven optimisation would start from. Both divisions now use `max(active, 1)`; `test_an_empty_start_is_safe_under_jit_grad_and_vmap` fails on the old code. `Domain` rejects a non-positive length or step, `Species` a non-positive mass or negative density, `Source` a negative density or a `min_weight` outside [0, 1], and `check_sources` rejects `emit > n`. | Done, except that the capacity check at each injection is S09's and stays with W2. |
| S14 | **fixed (W3)** | `gauss_residual` skips cell 0 because that cell's equation defines the missing boundary field. Documented, but therefore not an independent full check. | `Output.sigma` publishes the charge on each wall -- collected plus the clouds reaching past it -- and `charge_balance` asks whether the deposit and the wall ledger, two different passes over the particles, agree about how much charge the box holds and how much went in. It reads no field at all, covers every cell and both walls, and holds to 1e-16 with periodic, absorbing, reflective and thermal walls alike. Independence is asserted rather than claimed: a ledger perturbed on one step moves `charge_balance` by exactly the amount and leaves `gauss_residual` untouched, and a tilted field does the reverse. It found a defect on its first run -- the cloud overlap was being counted at walls whose deposit wraps or clamps it, where it is already on the grid. |
| S15 | **fixed (W7)** | `sheath_reflection.py` is still an unmaintained thermal-wall run on the default electromagnetic model with four filter passes and a bare `argmax` edge fallback. | The model is named: it had been electromagnetic at a light-wave Courant number of **286**, stable only because nothing there ever seeds a transverse field, and electrostatic gives the same drop to the digit printed. The four filter passes are kept and their measured worth is stated -- 3.05 T_e/e against 3.01 with none, 1.3 %, which is the size of the agreement being claimed. `R_eff` is measured from the ledger, 0.000 and 0.500 against the 0.0 and 0.5 the laws are meant to have, instead of being assumed and handed to the theory. The edge comes from `bohm_edge` with its crossing count. And the run is labelled the transient it is: it **drains 41.5 % of its ions**, against 4.8 % for the same box with a reservoir, which is the comparison the plan asked for and is now a panel. The figure script was repaired the same way and at the same preset, so the documentation and the example are one run. |
| S16 | **fixed (W7)** | "Normal incidence is the field-free case" is printed from the normal-field run itself; no `B=0` control is executed. | A matched `B = 0` run at the same seed **and** a second realisation of the normal-field case, so the comparison is against the scatter between realisations rather than against zero. The first attempt at this asserted round-off, which the quick preset appeared to confirm at 2e-11 and the full preset refuted at 4e-2: an external array takes a different path through the gather than `None`, so `E_x` differs in its last bit, and a plasma with absorbing walls is chaotic. Algebraic identity of `v_x` under a `B` along `x` is not identity of a run. At the full preset the field-free run and the normal-field one differ by at most **0.079 T_e/e** over a drop of 1.87, two seeds of the same physics by **0.137**, so the difference is 0.58 of the realisation scatter and the claim holds -- measured, with the thing it is measured against also measured. |
| S17 | **fixed (W3)** | `floating_potential` says its left side is `1/2` at `phi=0`; it is `1`. The guard therefore rejects `v0` in `[0.3989, 0.7979)`, which have valid roots: `v0=0.5 -> -0.134117`, `v0=0.7 -> -0.012507`. This already forced a benchmark parameter change during the previous round. | The endpoint and the bound are `sqrt(2/pi) = 0.7979`, both roots are pinned in the test, and the docstring separates the algebraic question -- where a root exists -- from admissibility, which is the caller's and which `bohm_edge` is there to answer. |
| S18 | **fixed (W3)**, and the disagreement it hid is now a result | `sheath_unmagnetized.py` hands `np.minimum(phi_profile, 0)` to the reference: in the quick preset **34 of 48** cell centres are positive, up to +0.0505 T_e/e, and every one of them is silently moved to zero. The same call passes the analytic `phi_wall = -0.79926` with a measured profile whose own wall value is -0.73439, and `phi_profile[0]` is a face value used as a centre. | `potential(out, centres=True)` puts the potential where the densities are, including the first cell, whose left face is the zero of the gauge and which was out by a factor of two. `densities` takes the configured amplitude and the measured cutoff separately, because in a measurement they are different numbers, and **raises outside its domain** instead of accepting a clip. Each frame is compared at its own wall potential, so the mean is a mean of sheaths and not the sheath of a mean. With the clip gone the ion agreement improves to 0.003, and the electrons disagree by 0.092 -- both at the quick preset; at the documented one they are 0.000 and 0.021. Where that disagreement sits, and what the relation assumes about the hump, is S20. |

### 2.2 Derivatives, optimisation, tests that cannot fail

| ID | Status | Finding | Required correction |
|---|---|---|---|
| G01 | **fixed (W3)** | `pytest.approx(x, rel=...)` keeps its default `abs=1e-12`, and `np.allclose` its default `atol=1e-8`. Three assertions in `tests/test_gradients.py` compare SI quantities of order `1e-19` to `1e-25` and therefore pass with **zero and with the wrong sign**. | The suite was audited by wrapping both functions and logging every comparison whose expected value the default absolute tolerance swallows: **24 sites, of which 19 were vacuous** -- times of 1e-10 s, masses of 7e-27 kg, a time step of 2e-13 s compared at `rel=1e-15`, wall energies of 1e-19 J. Each now says what absolute tolerance it means, and the wrapper stays in `conftest.py` as a guard, so a silent default on an SI-scale quantity is refused rather than found again later. `test_gradients.discriminating` asserts in so many words that zero and the wrong sign are rejected. |
| G02 | **fixed (W3)**, and it found a defect in the ledger | In `test_the_impact_energy_...` the particle never reaches the wall: `wall.arrived = 0.0`, `energy_in = 0.0`. The run is 0.80 ns against a 2.26 ns transit. It passes only because of G01. The "analytic impact control" reported in PR #42 tests nothing. | The run is long enough (2800 steps against a 2546-step transit), the impact is asserted before any energy is looked at, and every quantity is in the natural units of the problem. With the control actually running it showed that **the ledger recorded a wall impact at the end of the step the particle overshot to**: the value was off by 1e-4, but the derivative by 32 %, and the 32 % did not fall with the time step, because it was taken at a fixed step index rather than at the crossing -- the missing `dtau/dtheta` term. `Simulation._at_impact` runs the same pusher backwards over the part of the step that follows the impact; the value error falls to 5e-9 and the derivative error to 2e-5, both converging with the step. See S19. |
| G03 | **fixed (W3)** | The charge-sheet invariant compares a `1.35e-23` V/m field using `np.allclose` with default `atol=1e-8`. Zero and the wrong sign both pass. Sparse storage can also step over the cloud overlap. | The field is in units of the sheet's own, every step is stored, and the start is swept across a cell in five sub-cell offsets so the crossing happens at every phase. The bound is 1e-11 of the sheet's field, and the test says in an assertion that zero and the wrong sign do not meet it. That the cloud term is needed at all is S06's test, which fails by 5.20e10 V/m without it. |
| G04 | open | Forward/reverse/small-h agreement verifies one realised map, not an ensemble, continuum or stationary response. | Keep the derivative support matrix of section 4.3 and demonstrate each advertised response separately. |
| G05 | **fixed (W7)** | The optimisation builds one target on the training seeds and another on the held-out seeds, so both are exactly zero at the reference by construction. That is a plumbing self-test, not out-of-sample prediction. | Both are run and both are labelled. The paired target is kept and called what it is -- a self-test of the differentiated chain, whose minimum is at the reference by construction -- and a third set of realisations, used for neither training nor validation, supplies an independent target that nothing made zero. On the full preset the self-test recovers the reference exactly, to four decimals, while the inference lands at 0.3264 -- an error of **0.0236** on a residual loss of **0.0512**, which is the noise the model could not fit; the quick preset gives 0.0030 against 0.0879. The difference between those two numbers is the whole point of the row, and `--oblique` makes it plainer still: there the self-test still recovers 0.3500 exactly and the inference **walks to the bound**, 0.5000 against 0.35, with a held-out error bar of +-0.1131 against +-0.0207 field-free and two of four realisations pinned at a bound. A self-test that passes beside an inference that fails is what the two targets exist to show. |
| G06 | **fixed (W3)** | `np.linspace(0.02, 0.50, 25)` has spacing 0.02 and does not contain `r_reference = 0.35`; its nearest point is 0.34. The reported "uncertainty 0.01" is the distance to a grid point. | Three separate things, and they were one: the scan's spacing, which is printed; the minimum, refined off the grid by a parabola through the three lowest samples; and the scatter of that minimum **between realisations**, which is the only one of the three that is an uncertainty and is now a standard error over the held-out seeds. The grid is anchored on the reference so that a scan which does not contain the answer cannot report the distance to its nearest node as an error bar. |
| G07 | **fixed (W3)** | The loop can leave the final accepted point out of `history`, and "the step is below the uncertainty of the control" is a hard-coded threshold reported as convergence. | The point the loop ends on is evaluated and recorded, so the best point returned is one the optimiser stood on; on the quick preset that alone moves the recovered control from 0.0096 to 0.0030 of the reference. Three named tolerances -- projected gradient, step, relative fall -- each with its own message, and the gradient is projected onto the admissible interval so that a slope pushing out of a bound is not read as a direction. A line search out of halvings reports **stalled**, which is not converged. |
| G08 | **fixed (W3)** | The response window is `25 * 0.15 = 3.75` inverse electron plasma frequencies, about 0.60 oscillations. The documentation calls it "about four electron plasma periods". | The script prints both, the docstring and the documentation say both, and both say that 0.60 of an oscillation is the beginning of a response and not a settled one. A window quoted in periods when it is inverse frequencies is `2 pi` longer than it sounds. |
| G09 | open | Source clipping, slot selection, sorting, accept/reject collisions and the weight floor all add derivative discontinuities. | Test and document each. No blanket claim that ensemble-averaged branchwise AD repairs missing event terms. |

### 2.3 Interface, progress, plots, export

| ID | Status | Finding | Required correction |
|---|---|---|---|
| U01 | **fixed (W5)** | `load_toml` passes a few run keys only; sources, external fields, output, movies, restart, scans and optimisation are not reachable from TOML. | Every table is a constructor and every key is checked **before anything is built**, so a misspelling is reported as a misspelling and not as whatever the half-built object complains about next: unknown tables, unknown keys in any of them, and unknown keys in a `[species.source]`. Reachable now: a source, uniform external fields, an impact spectrum, and every argument of `run` -- which the command line passes through, with the meter on by default because that is the one place somebody is watching. Scans, optimisation and movie scripting are **deliberately** not reachable: those are programs, and a configuration file that grows a control flow is a worse programming language than the one it is written in. |
| U02 | **fixed (W4)** | PR #43 builds `tqdm` inside the jitted `_run` and captures it in `jax.debug.callback`. The bar is trace-time state: a cached program reuses a closed bar on the second identical call. | `run(verbose=True)` splits the run into about twenty groups on the host and blocks once per group; the kernels are untouched and callback-free. A verbose run is a silent run **bit for bit**, which the test asserts field by field, and two identical sequential runs both report. Measured cost on the plan's own case, 2000 steps and 200000 particles: 19.234 s silent against 19.489 s, **+1.3 %**. |
| U03 | **fixed (W4)** | `from tqdm import tqdm` is unconditional and `tqdm` is absent from `pyproject.toml`, so a clean install cannot import the package. The bar also prints during library calls inside tests. | No dependency at all: a fifteen-line stderr reporter, and `verbose` takes anything callable, so a `tqdm` bar goes in from the caller's side in three lines. That is a deviation from section 5's "lazy optional tqdm", and deliberate: an optional import is a second code path that CI does not install, so the tested path would not be the one a user with `tqdm` gets. One path, tested. The cadence is its own -- about twenty updates whatever `store_every` is -- and the meter turns itself off when anything is traced, so `jit`, `grad` and `vmap` are silent without being asked. |
| U04 | **fixed (W6)** | `_plot.py` uses `out.weight[-1] > 0` to choose the particles drawn in **every** frame. Particles collected earlier vanish from all frames; refilled slots appear from the start. | A species is every slot that belongs to it, and whether a slot holds a particle at a step is its weight **there**. Every histogram is weighted, so unequal weights count for what they are and a dead slot counts for nothing. The test runs a source into a collector, so slots are emptied and refilled, and asserts that each frame's histogram sums to that frame's live weight. |
| U05 | **fixed (W6)** | `_hist` clips outliers into the edge bins and the phase-space array adds `+1.0` to weighted counts for log display. | An empty bin is masked and drawn as the background rather than given a count of one, which had shifted every other bin by a particle. What falls outside the velocity range is counted and said in the panel's title rather than piled on the end bin, where it reads as a spike. The colour bar says what the numbers are: weight per bin. |
| U06 | **fixed (W1, W6)** | Face quantities (`E_x`, potential) are drawn on `out.grid`, which is cell centres. | `Domain.faces`, `Output.faces` and `Output.walls` are published and documented, and the places that rebuilt `grid + dx/2` by hand ask for them. The plot draws each field on the coordinates it lives on -- `E` and `J` on the faces, `B` and `rho` on the centres -- which is half a cell, and half a cell is where a sheath profile is steepest. |
| U07 | **fixed (W6)** | Energy, momentum, charge and balance histories are not in the general plot. | A `diagnostics` panel, on by default and switchable off, carries the energy and momentum errors, the Gauss residual and the charge balance on one logarithmic axis. Leaving them out left the one thing a glance should catch -- a run whose energy is running away -- to a separate call nobody makes. It costs what `diagnostics(out)` costs, which is a pass over the phase space, and the guide says so. |
| U08 | **fixed (W6)** | `plot` precomputes every frame before drawing any. On a 400-step, 256-cell, 25000-particle run it grew resident memory by **1.1 GB** while the histograms it keeps are 79 MB. The time axis is an `imshow` extent built from `(t[-1] - t[0])/(S - 1)`, so an irregular store schedule is drawn as if it were even, and `save` and `show` are exclusive. | Each frame is built when it is drawn: **21 MB** on that run against 1.1 GB, a factor of 53, and a thousand-frame movie of 25000 particles still takes 7.6 ms a frame with the blitting kept. The particle history is read through NumPy views -- `np.asarray` of a host JAX array shares its memory, while a JAX fancy-index allocated 129 MB for 64 MB of data. Fields are drawn with `pcolormesh` from the stored times themselves, so an irregular schedule is drawn where it happened. Saving and showing are independent. |
| U09 | **fixed (W5, W6)** | `omega` always labels the axis `omega_pe`; `quiet` is ambiguous. | `plot(omega_label=)` says what the frequency is, so an axis given an ion gyro-frequency no longer claims to be in plasma periods. `Species.sampling` is `"quiet"`, `"lattice"` or `"random"`, one name for each of the three things a start can be. It replaces two booleans of which one combination -- both true -- silently meant the first: three states do not fit in two switches without one of them being a lie. Two deviations from this row as written, both deliberate. The name is `"quiet"` and not `"low_noise"`, because a quiet start is the field's own term and `quiet_start` is already a public function here; renaming it would make the package less idiomatic, not more. And there are no migration aliases, because this branch already breaks `Source.beam`, `run(moments=)` and the `Output` fields, and an alias for one of them and not the others is not a migration path, it is an inconsistency. The release notes carry the list. |
| U10 | **fixed (W5)** | The standard is `x_i = (gridGlobalOffset + (i + position)*gridSpacing)*gridUnitSI` with `position` in `[0,1)`, `0.0` at the lower corner of the element; openPMD-viewer, WarpX and PIConGPU all agree. `openpmd.py` sets `grid_global_offset=-L/2` with `position=0.0` for centres and `0.5` for faces -- **exactly backwards**. Centres belong at `0.5`. The stored faces are the *right* faces, `-L/2+(i+1)dx`, which is `position=1.0` and outside the allowed range. | Centres are written at `position=0.5`, and the stored right faces keep `position=0.0` with their own `grid_global_offset` shifted a cell, which is the standard's way of saying the same thing inside `[0,1)`. The test reads the attributes back and applies the published formula to them, so it checks the file and not the writer's intention, and it asserts that every `position` is in range. |
| U11 | **fixed (W5)** | openPMD output is not a restart state and carries no source or wall context or readback example. | `save_state` and `load_state` write the loop state as a named, versioned `.npz` -- one array per field of `State` and of the `Wall` inside it, nothing executed on reading, absent rather than zero for what a run did not keep. A restart through a file is bit-identical to one through memory, which the test asserts field by field. Given the simulation it checks the shape of the run and refuses a state that does not match. The documentation says plainly what openPMD is for and why it cannot do this: no key, no ledger, no source bookkeeping, no charge density at the step the loop is about to begin, and positions at integer times where the explicit loop carries half-step ones. |
| U12 | **fixed for the sheath examples (W7)** | Examples do not systematically save configuration, data, figures and provenance; documentation quotes numbers from different presets. | `jaxincell.provenance()` is in the package -- versions, precision, device, commit, marked dirty -- where it had been a private helper of the figure scripts, which now use it too. All four sheath examples write a folder beside wherever they were run: `run.json` with the settings, the results and the provenance, `profiles.npz` with the arrays behind the figure, and the figure. `fig_sheath.py` was brought onto the example's own preset, so the documentation and the example are one run rather than two that look alike. The remaining examples are W9's. |
| U13 | confirmed by inspection | PR #43's Weibel sets float32 while the comment says float64, stores large histories, and widens the unstable spectrum with no fitted linear benchmark. | Keep the engaging run; fix the precision contract and memory policy; add verified linear-mode measurements in the same script. |
| U14 | open | The requested numerical-comparison and output/restart examples do not exist. | Implement the inventory of section 6 without four copies of the PIC setup. |

### 2.4 Found while the work was under way

| ID | Status | Finding | Required correction |
|---|---|---|---|
| S20 | **settled (W7)**, and the disagreement is not where the row put it | The kinetic relation predicts an electron density rising as `exp(phi)` across the presheath hump, 4.4 % at the quick preset's +0.043 T_e/e, and the measured density is flat there to 0.5 %. The ions agree to 0.003 over the same stretch, so it is not the window, the coordinates or the moments. | Two things were confused, and separating them settles it. **Where**: the example now reports the worst disagreement on the monotonic fall separately from the one over the whole box, and at both presets they are the same number -- 0.092 of 0.092 quick, 0.019 of 0.021 at the documented preset -- so the disagreement is on the stretch the relation describes exactly and not on the hump. **What the relation assumes**: it counts every orbit energy conservation allows, which above the source plane includes orbits bound to the hump that a plasma fed only from the plane would leave empty. Emptying them is a different prediction, 0.77 n_0 against 1.08 at the hump of the test's preset, and a histogram of the electrons finds them about nine tenths full, so the relation's assumption is the right one. The same measurement puts the wall cutoff exactly where the relation puts it: under one per cent of the weight moves back faster than `sqrt(phi - phi_w)`. Both are held by `test_the_electrons_of_a_maintained_sheath_are_the_kinetic_mapping_of_the_source`. **How big**: the residual is 0.021 n_0 at 120 cells over six transits against 0.092 at 48 cells over one, so it falls with resolution and duration rather than standing as a physics disagreement, and the quick preset now says so when it runs. |
| S19 | **fixed (W3)** | A wall recorded what reached it in the state the particle had at the **end of the step it overshot to**, not at the crossing. A particle is pushed once over a whole step and then drifts, so one that meets a wall part-way through the drift has taken the whole step's push where only part of it belongs before the impact. The value is wrong by `O(dt)`, which is why nothing noticed; the **derivative** is wrong by a fixed fraction -- 32 % in the control of `test_gradients`, 6 % in a bare leapfrog with other numbers -- and it does **not** fall with the time step, because it is taken at a fixed step index rather than at the wall. | `apply_particle_bc` returns where on the drift the wall was met, and `Simulation._at_impact` runs the same pusher backwards over the part of the step that follows the impact. The energy error falls from 1e-4 to 5e-9 and the derivative error from 3.2e-1 to 2.2e-5, both now converging with the step, where the 3.2e-1 sat still. It carries `Wall.energy_in`, `Wall.momentum` and the impact spectrum with it. Not yet done: the **outgoing** half of section 3.4 -- the wall law applied at the crossing and the remaining substep advanced -- which matters when the restitution is below one or a thermal wall redraws, and which is where the mirrored overshoot is still used. |

| E01 | **confirmed** | `landau_damping.py` estimates the noise floor from the last fifth of the run, which at 500 steps is still decaying: it returns 92.2 V/m where the floor is nearer 29. The `> 5 x floor` cut then keeps 5 maxima and the fit gives -0.1523 against the kinetic -0.1533, which looks like agreement. At 800 steps the same rule returns 29.3, admits 9 maxima of which four sit on the floor, and the fit degrades to -0.1433. `fig_landau_damping.py` and `test_landau_damping_matches_the_kinetic_root` use the same rule at the same 500 steps, so the three agree only because none of them was ever lengthened. | Measure the floor where the amplitude has stopped falling rather than in a fixed fraction of the run, and drop maxima at or below it, so the step count stops being load-bearing. Re-measure the example, the figure and the test together (W9). |
| E02 | **confirmed** | The wider Weibel box breaks panel (a)'s threshold test, and no run length repairs it. In a box of twelve marginal wavelengths the low-`k` modes saturate the anisotropy before the modes near the cutoff have grown, and the nonlinear stage fills the whole spectrum: over 24 truncations from `t w_pe` = 16 to 321 the smallest gain below the cutoff exceeds the largest gain above it at four, none of them robust. Reading the same `k/k_c` values the present example uses (modes 3, 6, 9 against 12 to 24) it separates from 51 to 131 with a margin of 5.5 against 3.5, and fails again at 107; the four-wavelength box gives 9.11 against 4.09. Peak-over-initial gain does not repair it either: modes at `k/k_c` 1.25 and 1.42 peak at 8.6 and 12.3, above the unstable modes at 0.75 and 0.92, which peak at 7.7 and 8.0. | The threshold demonstration wants a narrow box, and the engaging nonlinear run wants a diagnostic that survives saturation. W9 gets both: keep the four-wavelength threshold test as it stands, and add the wide run with per-mode fitted linear rates against the kinetic root, which needs the dispersion solver that today lives only in `docs/scripts`. A gain ratio is not a growth rate once a mode has saturated. |

## 3. W2 contracts: sources, collectors, events

### 3.1 A source is a boundary distribution

Use the three requested component spreads. At a plane the normal velocity comes from the flux
distribution and the tangential components from their own. For a factorised Maxwellian with
inward normal drift `u` and normal spread `sigma_n`,

```math
\Gamma=n[u\Phi(u/\sigma_n)+\sigma_n\varphi(u/\sigma_n)],\qquad
p(v_n)=\frac{n v_n}{\Gamma\sqrt{2\pi}\sigma_n}e^{-(v_n-u)^2/2\sigma_n^2},\quad v_n>0,
```

with the cumulative numerator from 0 to `v`

```math
u[\Phi((v-u)/\sigma_n)-\Phi(-u/\sigma_n)]+\sigma_n[\varphi(u/\sigma_n)-\varphi((v-u)/\sigma_n)]
```

divided by `Gamma/n`, as an independent CDF and quadrature check. The sign of `u` matters:
`abs(drift[0])` is not a drifting reservoir, and an outward cold beam is rejected rather than
reflected. Keep the zero-drift Rayleigh and cold-beam limits as fast paths. Do not shift a
Rayleigh sample at nonzero drift.

If a numerical inverse is used, validate its **derivative** as well as its value; differentiating
bisection branches gives the wrong sensitivity. A documented implicit-function `custom_jvp` with
a tested transpose is the right mechanism.

Field-aligned non-Maxwellian input needs a compact `(v_par, v_perp, gyrophase)` or invariant
representation with explicit units, measure and normalisation, transformed to Cartesian and
weighted by the **normal crossing flux** `max(v_n,0) f`. Sampling a gyrotropic density and
rejecting `v_n<0` without flux weighting is wrong. Mark tabulated or accept/reject routes
nondifferentiable in the affected parameters until a validated estimator exists.

### 3.2 Two source models

- **Prescribed reservoir**: incoming characteristics set externally, outgoing particles leave;
  neither density nor flux is reset from collector losses.
- **Schwager-Birdsall**, now confirmed from the source and quotable. Schwager and Birdsall,
  *Collector and source sheaths of a finite ion temperature plasma*, Phys. Fluids B **2**(5),
  1057-1068 (1990), doi 10.1063/1.859279; the preprint UCB/ERL **M88/23** is freely readable at
  <https://www2.eecs.berkeley.edu/Pubs/TechRpts/1988/ERL-88-23.pdf> and carries the full text.
  The prescription is:
  1. at `x=0`, inject **equal, steady ion and electron number fluxes**, each a **half-Maxwellian**
     (`v>0` only) at its own source temperature, truncated at `6 v_th`;
  2. an electron returning to `x=0` is **removed and re-injected with a velocity redrawn from that
     half-Maxwellian** -- "refluxing". It is not specularly reflected. **Ions are not refluxed**;
  3. refluxing is what enforces `E(x=0)=0`, by preventing charge accumulating at the source plane;
  4. so the *emitted* electron flux exceeds the *injected* one by `exp(-psi_c)`, and the two must
     be reported separately;
  5. at `x=L` the collector absorbs everything and floats.

  Their own control is worth copying: replacing refluxing by a hard-coded emitted flux ratio
  `exp(psi_c)` reproduced the same potential profile and fluctuation level. Their **simulations**
  use `m_i/m_e = 40 or 100`, not 1836 -- only the theory curves use 1836 -- with `L = 20-50
  lambda_D`, about six grid points per Debye length, at least 400 particle electrons per Debye
  length, and `dt ~ 0.05/omega_p`. The present thermal wall alone is not this model.

A source normalisation derived to match a benchmark is a setup step. It must not become a hidden
function of an optimisation control.

### 3.3 Pool and particle identity

Fixed shapes and continuous emitted weights stay. Exhausted capacity sets a persistent invalid
status and stops safely at a host boundary; jitted and AD paths return a validity result the
optimiser rejects. Record requested and emitted amounts and the overflow count and weight. A
reused slot is not the same particle. Physical observables must be invariant under permutation of
free slots and changes of capacity when nothing overflows.

### 3.4 Injection, reflection, events

Draw the one-step staggering diagram before touching `_explicit_step`. New particles enter at
known substep times, see the appropriate residual force, and contribute consistent trajectories
and boundary current. At a wall, locate the crossing on the numerical trajectory, capture the
incident state, apply the law, and advance the remaining substep; mirroring the overshoot with
the old speed is wrong after thermalisation or restitution. Bounded substepping or a checked step
restriction handles repeated hits.

Record per species and wall: fluence, conduction charge, kinetic energy in and out, momentum
transfer, and optional fixed-bin energy and angle accumulators with overflow bins. Use
`theta_hit = atan2(|v_t|, v_n)` for `v_n > 0`. Each event is weighted once, and histogram
integrals must recover the independently accumulated fluence.

For a crossing `g(x(tau;theta),theta)=0`,

```math
\frac{d\tau}{d\theta}=-\frac{g_x\,\partial_\theta x|_\tau+g_\theta}{g_x\dot x+g_t},
```

and event observables must include that term. Test `K_hit = K_0 + q E (x_w - x_0)` in a uniform
static field, whose derivatives are `q(x_w-x_0)` and `m v_0` -- **after** asserting that the event
happened. A correct value at a fixed impact-step index can still have a wrong derivative.
Event-time treatment does not fix the derivative of a hard count by a fixed time; keep the
ballistic negative control.

### 3.5 Clouds, wall charge, continuity

Distinguish charge on volume cells, reversible shape overlap at a boundary, irreversibly collected
physical charge, and charge exchanged with a reservoir or supply. Deposited interior plus exterior
fractions sum to one per active particle. Do not credit the exterior part to the electrode and then
count it again at centre crossing. For the ideal conductor,

```math
E_x(x_w^-)=-\sigma_w/\epsilon_0,\qquad
\dot\sigma_w=\sum_s q_s(\Gamma_{s,\rm in}-\Gamma_{s,\rm returned})+j_{\rm supply},
```

with `j_supply = 0` meaning floating. A gauge is not an extra boundary condition. The discrete
continuity relation carries the physical boundary current; taking the collector current to zero by
construction is not an absolute-current diagnostic.

## 4. W3 contracts: diagnostics that can fail

### 4.1 Windows and coordinates

A cumulative diagnostic carries its step counter or accumulated duration; averages are differences
divided by the actual difference, never inferred from array length. State whether a window is
`(t_a, t_b]`. Test a constant signal, a linear signal, an irregular schedule, the first and last
interval, restarts, and a final step not divisible by the stride. **Fix S01 in both sheath drivers
and the figure script first**; it is a 3.3 % and 5 % bias, not statistics.

Publish centre, face and boundary coordinate arrays in the output and use them everywhere. `E_x`
and the potential are on faces; densities and deposited moments are on centres.

### 4.2 Moments and conservation

Add streaming density, three first moments and six second moments with consistent weights, so that
pressure and temperature tensors are available without particle histories. Separate the cadences of
compact diagnostics, field snapshots and particle snapshots.

Expose both the raw physical change and the numerical residual. For open runs,

```math
W(t)-W(0)=W_{\rm injected}-W_{\rm escaping}+W_{\rm external}+W_{\rm supply}+W_{\rm other}+R_W,
```

with every sign and domain written down. Normalise by physical scales fixed at initialisation: a
neutral plasma's net charge and a two-stream state's net momentum are not denominators. Separate
`gauss_residual` from global charge conservation and test each with an independent deliberate
perturbation.

### 4.3 Tests that can fail, and the derivative matrix

Audit every `pytest.approx`, `allclose` and fixed tolerance on SI-scale quantities; both carry
absolute defaults that swallow tiny physical values. Each assertion must fail when the observable is
replaced by zero or by the wrong sign. Replace `std/sqrt(frames)` with block means, independent
seeds, or an integrated autocorrelation estimate, and label plain scatter as scatter.

| Observable / path | What may be asserted |
|---|---|
| Fixed-time smooth field sensor, fixed realisation | JVP/VJP and small-step finite-difference agreement. |
| The same sensor averaged over an ensemble | An estimate of the expected response, after ensemble and perturbation-scale checks. |
| One transversely crossing impact | Event-aware derivative, checked against independent trajectory mathematics. |
| Hard count or histogram bin by fixed time | Discontinuous; branchwise AD is not an expected-flux derivative. |
| Long-time stationary average | Needs preparation, duration, sampling and discretisation convergence. |
| Cutoff, creation count, collision acceptance, material branching | Support-changing terms identified; no blanket end-to-end claim. |

## 5. W4: a progress meter that does not live in the kernel

A progress meter is a requirement: a long run must say how far it has got. PR #43 supplies one
and is the right instinct; its mechanism is the thing to replace.

### 5.1 Why the callback route is rejected

PR #43 builds a `tqdm` object inside the jitted `_run` and captures it in
`jax.debug.callback(..., ordered=True)`. Three separate problems:

1. **The bar is trace-time state.** `_run` is `jax.jit`-ed, so its body runs once per
   compilation. The second call with the same static arguments reuses the compiled program and
   the already-closed bar. Two identical sequential runs do not both show a bar.
2. **Callbacks are not guaranteed.** `jax.debug.callback`'s own docstring says the effect "could
   be dropped, duplicated, or potentially reordered in the presence of higher-order primitives
   and transformations". Four consequences that matter here, all documented:
   - under `grad`, a debug callback **fires on the forward pass only**, so a bar inside a scan
     that is later reverse-differentiated silently stops ticking on the backward pass;
   - under `vmap` the callback is unrolled across the mapped axis, so one bar receives `B` times
     its updates and runs to `B` times its total;
   - `ordered=True` registers an effect that is ordered but **not shardable**, so it raises on
     more than one device, where `io_callback(..., ordered=True)` would not;
   - dispatch is asynchronous, so output can appear after the function has returned;
     `jax.effects_barrier()` is needed, and `block_until_ready` is not enough.
3. **It puts a side effect in the differentiated path**, which is exactly what the
   kernel/orchestration separation exists to prevent.

`tqdm` itself is a second, smaller problem: PR #43 imports it unconditionally and it is not in
`pyproject.toml`, so a clean install cannot import the package at all (U03).

The packaged version of this approach, `jax-tqdm`, has the same shape and the same trouble: it
avoids the closed-bar problem only by never capturing a bar, keying them instead in a module
dictionary, and its ordering bug on multiple devices is open and unreleased for over a year. Note
also that `jax.experimental.host_callback`, which older recipes use, was removed in JAX 0.8.0.

### 5.2 The mechanism to use

Own progress on the **host**, per invocation, outside the traced region:

- `verbose=False` (the default whenever the caller is tracing) runs exactly the fused
  `lax.scan` the code runs today. The differentiated path is untouched, byte for byte.
- `verbose=True` splits the same scan into host-driven groups and blocks once per group. The
  state carries everything, so a grouped run is the ungrouped run: this is already guaranteed
  by `test_the_clock_is_absolute_and_a_run_split_in_two_is_the_run_taken_whole`.
- Group count is chosen for a bounded number of updates (order 20-50), **not** tied to the
  snapshot schedule, so progress cadence and `store_every` are independent.
- Auto-disable when any argument is a tracer, so `jax.jit(lambda r: sim.run(...))` and
  `jax.grad` are silent without the user asking.

Measured cost on a 2000-step, 200000-particle electrostatic run (Apple M4, double precision, best
of three): the fused scan takes 13.174 s; host groups cost **+1.2 % at 10 updates, +1.4 % at 20,
+1.5 % at 50**. That is the price of a truthful meter and it is small. The callback variant was
not timed like for like and is rejected on the correctness grounds above, not on speed.

One implementation note: `_run` currently computes `initial_state` and then discards it when a
`state` is supplied. Grouped execution would repeat that per group, which is wasteful at large
particle counts; skip it when continuing from a state.

### 5.3 The reporter itself

No new required dependency. A dependency-free reporter writing `\r`-updated lines to stderr is
about fifteen lines and covers the need: elapsed, fraction, steps per second, estimated
remaining. Use `tqdm` only if it is already importable, behind a lazy optional import, and add
it to an optional extra rather than to the runtime dependencies. A non-interactive stream
(a log, CI) gets periodic plain lines instead of carriage returns.

Progress reports host-side facts only -- steps completed and wall-clock time. It never changes
the simulation, the random numbers or the output, and there is a test that a verbose run and a
silent run produce identical arrays.

**W4 acceptance:** two identical sequential runs both report; `jit`/`grad`/`vmap` of a run are
silent and unchanged; a verbose run equals a silent run exactly; progress cadence is independent
of `store_every`; a clean install without `tqdm` imports and runs; an interrupted run leaves a
usable terminal.

## 6. Examples: repair, then extend

Every example keeps a short physical explanation and reference; imports; editable parameters;
derived scales; construction of public objects; execution with progress; a quantified comparison;
and saved data, figures and an optional movie. Presets are `quick` (smoke, same physics),
`reference` (the documented validation) and optionally `movie`. A quick run is never quoted as a
reference. Output paths name the preset and store the actual configuration.

| File | Required result |
|---|---|
| `1_basic/parameters_and_sampling.py` (new) | Physical versus numerical inputs; random versus low-noise loading; units and derived scales. |
| `1_basic/sheath_unmagnetized.py` | S01, S17, S18 and U06 repairs; boundary and current diagnostics; resolution and duration studies. |
| `2_intermediate/sheath_magnetized.py` | One documented source problem; true impact spectra (S02); matched `B=0` and normal-field controls (S16). Port PR #43's larger quick settings as a reproducible historical input, not a target. |
| `2_intermediate/sheath_reflection.py` | Maintained source, explicit electrostatic model, measured `R_eff`, the validated edge helper. |
| `2_intermediate/weibel.py` | Port PR #43's larger domain, broader spectrum and longer run; fix the precision contract and history size (U13); add mode-by-mode linear growth against `docs/scripts/dispersion.py`. |
| `2_intermediate/compare_models.py` (new) | Explicit/implicit, filtered/unfiltered, collisional/collisionless, relativistic/not -- four short pairs, not a sixteen-case product. |
| `2_intermediate/output_and_restart.py` (new) | Native save, load and restart; optional openPMD write and read back. |
| `3_advanced/sheath_optimization.py` | G05-G08 repairs: independent fixed target, honest uncertainty, fixed optimiser bookkeeping. |
| `3_advanced/grazing_sheath.py` (new) | Matched GYRAZE case, then a controlled finite-ordering study. |
| `3_advanced/electron_field_instability.py` (new) | Published setup plus the accelerating-frame and ion-background controls. |
| Existing `conservation.py`, `optimize_two_stream.py`, `collisions.py`, `landau_damping.py`, `bump_on_tail.py` | Preserved and improved on shared machinery; port PR #43's longer runs where they help. |

Shared algorithms -- moment averaging, impact histograms, root and fit calculations, saving and
loading, field and source profiles -- become public functions. Host theory moves out of
`docs/scripts` into a small reference location rather than being imported by a `sys.path` hack.
SciPy stays an optional example and validation dependency, not a core one.

## 7. W5: one TOML and CLI path

Extend the existing command; do not add a second application. One normalised resolver builds the
same objects the Python API builds, from the same vocabulary, and an unknown or conflicting key is
an error rather than a silent default. It must reach sources, external fields, output selection,
movies, restart and scans, and support `--describe`/`--validate-only` that resolves derived scales
without running. Native versioned archives hold exact restart state; openPMD holds interoperable
analysis data with coordinates and weights that an independent reader reproduces (U10, U11).

Parameter vocabulary is fixed once and used in constructors, TOML, plots, exports and documentation:
`sampling` (`low_noise`/`random`, not `quiet`), `verbose`, `temperature_ev`, `density_m3`,
`particles`, `capacity`, `emit_per_step`, `mass`/`mass_kg`, `charge_number`, `drift_m_s`, exactly one
of `time_step_s`/`dt_omega_pe`/`courant`, exactly one of `length_m`/`length_debye`,
`reference_frequency` for plots, `potential_v` for electrodes. Publish a migration table, accept
legacy names during migration, and reject conflicting pairs. Never warn from inside a trace.
`steps_per_plasma_period` in the current scripts actually means `1/(omega_pe dt)`; rename it.

## 8. Benchmarks

### 8.1 Grazing incidence against GYRAZE (W8)

Reference: Geraldini, Ewart, Brunner and Parra, *Characteristics of monotonic sheaths near a wall
with grazing magnetic incidence*, arXiv:2508.09067v1 (2025-08-12, 82 pp). Checked on 2026-09-18:
still v1, no `journal_ref`, no publisher DOI, no Crossref record -- **treat it as unpublished**.
The code is GYRAZE, <https://github.com/alessandrogeraldini/GYRAZE>, C, commit
`bcc42e1450ca287cbb2d4e77fae8fe80f353e39f` (2025-10-02, HEAD of `main`, not a tag). Active work
is on branch `pk/gkeyll_fixes`.

**GYRAZE has no licence file at all**, so it is legally all-rights-reserved. Run it as an
external reference tool in its own directory and record its outputs; do not vendor, copy or
derive from its source, and do not redistribute its data without the authors' permission.

It is a grazing-angle, separated-scale kinetic model -- magnetic presheath of thickness `rho_S`
and Debye sheath of thickness `lambda_D`, solved as asymptotically separate systems as
`lambda_D/rho_S -> 0` -- with a monotonic-potential assumption. Electrons are gyrokinetic **in
the Debye sheath** with finite `gamma = rho_e/lambda_D` retained; `gamma_ref = 0` reduces it to
adiabatic Boltzmann electrons and the magnetic presheath alone, which is the 2019-paper regime.
It is not a Boltzmann-electron solver missing only kinetic electrons.

Its README states its own limits, and they bound what can be benchmarked: it "transitions to
being very inaccurate at magnetic field angles of **5-8 degrees**", `tau` must stay **above about
0.2**, and it converges only for monotonic potentials, failing below a critical angle that grows
with `gamma`.

First target: figure 6, `M=3600`, `Z=1`, `bar_T_i/T_e=1`, `alpha=2.5 deg`,
`gamma=rho_e/lambda_{D,DS}=0.3`, zero net wall current. The `gamma=0.7` case sits at a critical
boundary and is not the first validation. Any angle scan stays well below 5 degrees.

Three traps found by inspecting the source rather than the documentation:

- **`gammaflag`: the README has it backwards, the source and figure 6 agree.** The README says `1`
  defines `gamma` at the Debye-sheath entrance and `0` at the magnetic-presheath entrance. The C does
  the opposite: flag `0` sets `gamma_DS = gamma_ref`, so `gamma_ref` *is* `rho_e/lambda_D` at the
  Debye-sheath entrance, and flag `1` sets `gamma_DS = gamma_ref sqrt(n_e,DSE)`, the entrance value
  carried in (a flag-1 run given `0.3/sqrt(0.071033)` printed `gamma_DS = 0.299745`). Figure 6 --
  `gamma = rho_e/lambda_{D,DS} = 0.3` -- reproduces with flag `0`. Use `0` (flag `1` re-evaluates
  `gamma_DS` inside the loop and had not converged on the figure-6 case after 410 iterations), record
  in the manifest that `gamma_ref` is the Debye-sheath-entrance value, and give the PIC run the entrance
  value `gamma_DS/sqrt(n_e,DSE)` (W8, "The references, generated", below).
- **The wall-potential sign differs between the printout and the file.** The code prints
  `eφ_W/T_e = -0.5 v_cut^2` but stores `+0.5 v_cut^2` in `misc_output.txt`. The stored number is
  a magnitude.
- **The Python post-processor's column names do not match the C write order.** `misc_output.txt`
  is written as net current, `0.5 v_cut^2`, `Q_e`, `sum Q_i`, `flux_e`, `sum flux_i`; the
  post-processor unpacks the heat fluxes under names suggesting particle fluxes. Trust the C.

Comparable outputs: `phi_n_MP.txt` and `phi_n_DS.txt` (`x, phi, n_i, n_e` on each scale),
`Fi_W.txt` (the ion distribution at the wall, which the shipped post-processor turns into an
energy-angle distribution), and `misc_output.txt`. Generate reference numbers by running the
pinned commit or by asking the authors; do not digitise rendered figures.

A benchmark manifest is mandatory and fails fast when incomplete: code commit and paper version;
the selected case and the provenance and licence of the reference data; both coordinate
orientations and potential references; all normalisations; mass ratio and charge; wall current
**or** prescribed potential, not both; the incoming distributions with normalisation, support,
Jacobian and gyrophase convention; the Debye reference density location; and the asymptotic
assumptions of the reference against the finite parameters of full-orbit PIC.

`gamma` uses the electron density at the **Debye-sheath entrance**:

```math
\gamma_{\rm DS}=\gamma_{\rm upstream}\sqrt{n_{e,\rm DS}/n_{e,\rm upstream}},\qquad
\frac{\rho_S}{\lambda_{D,\rm DS}}=\sqrt{M(1+\bar T_i/T_e)}\,\gamma_{\rm DS}.
```

These parameters are not independent: varying `epsilon = lambda_D/rho_S` while holding `M`, `gamma`
and the temperature ratio fixed is impossible. Any convergence sequence must say which dimensionless
parameters move and regenerate the reference for them.

**Three normalisation traps, each verified against the papers, and each enough on its own to
invalidate a comparison:**

1. **`rho_S` is the ion sound gyroradius, not the Bohm gyroradius.** The paper defines
   `rho_S = sqrt(m_i(Z T_e + T_i))/(Z e B) = c_S/Omega_i`, while `rho_B = v_B/Omega_i` with
   `v_B = sqrt(Z T_e/m_i)`. For `Z=1`, `rho_S/rho_B = sqrt(1+tau)`. **At the benchmark's
   `tau = 1` they differ by `sqrt(2)`.** A PIC run normalising lengths to the cold-electron
   `rho_B` and compared against a GYRAZE profile normalised to `rho_S` has every length wrong by
   that factor. A third scale, the thermal `rho_i = sqrt(m_i T_i)/(ZeB)`, appears in the 2018
   abstract; check which one a given figure axis uses.
2. **The thermal-speed convention carries a factor of two.** The 2019 paper uses
   `v_t,i = sqrt(2 T_i/m_i)`; this code uses `sqrt(T/m)`. Every velocity-space width differs by
   `sqrt(2)` unless converted. The 2025 paper is not even self-consistent with the 2019 one on
   this point, using `v_t,i = sqrt(bar_T_i/m_i)` in section 5.1.
3. **`bar_T_i` is a width parameter, not a temperature.** The paper says so explicitly: the
   ad-hoc distribution is non-Maxwellian and `bar_T_i` "is strictly not the ion temperature as
   conventionally defined from a Maxwellian velocity distribution". So `tau = 1` in the ADHOC
   runs means the width parameter equals `Z T_e`, **not** that the ions are a Maxwellian at
   `T_i = T_e`. Injecting a genuine Maxwellian at `T_i = T_e` is a different problem.
   Additionally the entrance distribution must satisfy the kinetic Chodura condition, which
   forces `F_i(mu, Omega_i mu) = 0` -- no ions with zero parallel velocity at the presheath
   entrance -- and a drifting Maxwellian generally violates it.

Sequence: single-particle orbit and source-flux checks; the matched monotonic absorbing case with
zero net collector current, stationary inventory, no source-position dependence, no overflow and
closed balances; profile, flux, drop and impact-distribution comparison on common coordinates with
predeclared tolerances; then a scan **within** the reference's valid range.

#### What the reference tool actually does (measured, W8)

Cloned at the pinned commit into `~/local/GYRAZE-ref`, outside this repository, and built against
Homebrew's GSL with `make -B PROFILE=release CFLAGS="-Wall -O3 -I/opt/homebrew/include"
LDFLAGS="-O3 -L/opt/homebrew/lib -lgsl -lgslcblas -lm"`. **Figure 6 reproduces**: `ADHOC`,
`alpha=2.5`, `gammaflag=0`, `gamma=0.3`, one species, `ni:ne=1`, `Ti:Te=1`, `mi:me=3600`,
`set_current=1`, `jwall=0` converges in 43 iterations and 11 s to

    (phi_DSE, phi_wall) = (-2.410640, -2.822029) T_e/e,  current = 8.8e-05

with `misc_output.txt` holding `0.000088, 2.822029, 0.125713, 0.079472, 0.026509, 0.026597` in the
C write order -- net current, `0.5 v_cut^2` (the wall potential's **magnitude**), `Q_e`, `sum Q_i`,
`flux_e`, `sum flux_i`. The electron sheath heat transmission coefficient is 4.742 against the
classical `2 + |phi|` of 4.822.

Two more traps, found by running it rather than reading it:

- **The far field of each profile file is not a profile.** `phi_n_DS.txt` and `phi_n_MP.txt` are
  `x, phi, sum n_i, n_e` (checked in the C, not the post-processor), but one density column is
  written as exactly `0.000000` beyond some index -- `n_e` past `x ~ 67 lambda_D` in the Debye
  sheath, `sum n_i` past `x ~ 14 rho_s` in the presheath. That is an unfilled array, not a
  density. Truncate every comparison where a column reaches exact zero at the far end.
- **Convergence and exit status are different things, and `NaN` appears in both.** Every case,
  including the one that exits cleanly, prints `NaN` for some intermediate diagnostics
  (`momfluxinf`, `n_inf`, `dndphi`). Some parameter sets converge, print the wall potential, write
  every output file, and then abort with SIGABRT -- `M=900, gamma=0.12` does. Gate the reference
  set on the exit status, not on the last line of the log.

#### The matched case has to move, and the arithmetic says where (W8)

A full-orbit PIC run matched to figure 6 needs the whole magnetic presheath, `25 rho_S` deep with
`rho_S/lambda_D = sqrt(M(1+tau)) gamma = 25.5`, plus the `83 lambda_D` Debye sheath: a box of 719
Debye lengths. Ions enter it at `c_s sin(2.5 deg)`, and `Omega_e dt <= 0.25` caps the step at
`omega_pe dt = 0.075`, so one ion transit is 9.3e6 steps. At 0.5 Debye lengths a cell and 100
particles a cell that is **75 hours a transit** on this laptop, and the comparison needs two or
three. Figure 6 is out of reach for full-orbit PIC, and saying so with the arithmetic is part of
the benchmark rather than a failure of it.

GYRAZE costs 11 s a case, so the reference moves instead. Scanning it for where it still converges
cleanly, at `tau = 1` and `gammaflag = 0`:

| `M` | `gamma` | 3 deg | 4 deg | 5 deg |
|---|---|---|---|---|
| 400 | 0.12 | converges, aborts | no convergence | converges, aborts |
| 400 | 0.20 | crash in `densfinorb` | non-monotonic | **-1.5771, -1.6789** |
| 400 | 0.30 | crash | non-monotonic | non-monotonic |
| 900 | 0.12 | converges, aborts | converges, aborts | converges, aborts |
| 900 | 0.20 | **-1.9585, -2.0724** | **-1.8830, -2.0837** | **-1.8060, -2.0940** |
| 900 | 0.30 | **-1.9572, -2.0970** | **-1.8833, -2.1137** | **-1.8065, -2.1281** |

(`(phi_DSE, phi_wall)` in `T_e/e` where it converges and exits.) The critical angle grows with
`gamma` exactly as the README says, and `M=400` is mostly below it.

**The matched case is `M=900, tau=1, gamma=0.2, alpha=4 deg`**, with `alpha = 3` and `5 deg` as the
scan inside the reference's range: 295 Debye lengths, 590 cells, `omega_pe dt = 0.05`, 1.8e6 steps a
transit. `M=400, alpha=5 deg` is the cheaper rehearsal, at the edge of the 5-8 degree range the
README calls inaccurate, so it is a rehearsal and not a result.

**And the marker floor is what actually sets the cost, not the particles per cell.** A species
carries `emit` times its residence in steps, and `emit` cannot go below one marker per step. At
`M=400, alpha=5 deg` an ion's residence is 465000 steps, so the pool holds **465000 ions whatever
the particles-per-cell setting asks for** -- 100 a cell over 324 cells is 32000, and the run carries
fourteen times that. Three transits is then 7e11 particle-steps, about 33 hours on this laptop and
8 to 15 on one A4000. The matched case is worse by the same argument: a residence of 1.15e6 steps
and 4.6e6 steps of run is 7e12 particle-steps, which is a hundred hours of GPU and was stopped
rather than left running. The arithmetic above priced the box and the step count and missed this,
which is the difference between a four-hour job and a four-day one.

What that cost the benchmark: the rehearsal at `M=400, alpha=5 deg` was the one case that finished,
and it sits at the edge of the reference's own accuracy range. A matched comparison inside that
range needed either a source that emits a marker every `k` steps, or a machine-week. **It now rests
on the first.**

#### A source that emits every `k` steps (W8, built)

`Source(every=k)`, static, default 1. The source emits its `emit` markers on the steps with
`(step + 1) % k == 0` -- the absolute step, so a continued run keeps the schedule -- as the window
of the `k` steps that end there, each carrying `Gamma k dt / emit`: the emitted charge per unit time
is exactly `Gamma` and stays differentiable in the reservoir's leaves. The marker that crossed the
plane at the fraction `s_j = (j + 1/2)/emit` of the window is put where its orbit has taken it since,
by the construction W2's single-step entry already uses (S07): position to second order in the flight
`(1 - s_j) k dt`, velocity by the pusher over `((1 - s_j)k - 1/2) dt`, both in the field at the plane.
So the stream is spread over `v k dt` rather than stacked on the plane. An idle step is a `lax.cond`
that leaves the arrays and the ledger alone and skips the draw and the partial sort; `every = 1` takes
the old path, operation for operation. A marker whose flight passes the far wall is left there on
purpose: the collector holds its whole cloud as surface charge and the wall law of the same step
collects it, so the ledger records the impact on the step it was emitted and the charge closes.

The price is a ramp: a particle is in the box only from the emission after it crossed, so within
`v k dt` of the plane the time-averaged density rises from zero to its value. The example therefore
takes `k = residence/markers` with `emit = 1`, which makes `v k dt = dx/markers_per_cell` at the
entrance speed -- the marker spacing, 0.01 dx at 100 a cell -- and caps the window's gyro-angle at the
step's own `Omega k dt <= 0.25`. That keeps the electrons at `k = 1` (`Omega_e dt` is already 0.25)
and turns the ions through under 0.01 rad a window.

Evidence, `tests/test_sources.py`, each shown to fail on a broken variant: `every = 1` against a
verbatim copy of the 7a6cfb4 injection, bit for bit (fails on a reordered weight); the window's
positions and velocities against the uniform-field orbit to 1e-13 (fails when stacked on the plane);
the ledger moves by exactly `Gamma k dt` on emitting steps and by zero between (fails unscaled, and
when the ledger counts idle steps); a drifting reservoir's streaming density at `k = 20` equal to
`k = 1` within four standard errors measured from eight seeds, every cell but the plane's, where the
ramp takes 7 % at `v k dt = 0.25 dx`; the live pool divided by 20.0 at `k = 20` (fails when the
schedule is ignored); `gauss_residual`, `charge_balance` and the floating collector's
`J + eps_0 dE/dt` at round-off with `k = 2` and `5`; the far-wall case; and the gradient of the wall
potential in the ion reservoir density finite and equal to a central difference to 1e-4 (fails with
the weight's gradient stopped).

**The cost, repriced** from the example's own geometry (its matched box is `15 rho_s + 3 rho_s +
60 lambda_D = 213 lambda_D`, 425 cells; the 295 above was a `25 rho_s` box), three entrance-speed
transits, 100 markers a cell, at the per-particle-step rates that priced the rows above: 170 ns on
this laptop, 40-80 ns on one A4000.

| case | ion residence | ion `k` | pool, `k=1` -> `k` | steps | particle-steps | laptop | one A4000 |
|---|---|---|---|---|---|---|---|
| rehearsal, `M=400, 5 deg` | 4.65e5 | 14 | 4.95e5 -> 6.3e4 | 1.40e6 | 6.9e11 -> 8.8e10 | 33 h -> 4.1 h | 8-15 h -> 1-2 h |
| matched, `M=900, 4 deg` | 1.15e6 | 27 | 1.20e6 -> 9.1e4 | 3.44e6 | 4.1e12 -> 3.1e11 | 194 h -> 15 h | 46-91 h -> 3.5-7 h |

The pool after is 32 400 ions and 29 600 electrons in the rehearsal and 42 500 and 48 700 in the
matched case: **the electrons are now half of it**, held at `k = 1` by their gyro-angle, so the next
factor of two is theirs and not the ions'. The A4000 column scales a per-particle rate that was
measured on a full device; at 6-9e4 markers the device is far from full, and the step count, 1.4e6 and
3.4e6, may make the per-step launch overhead the real cost. That overhead has not been measured.

The quick preset runs end to end with it, and once with `--every=1` as the control (on this laptop
under load; the preset is not grazing and measures nothing about the benchmark):

| quick preset | ions live | wall potential | net current | ion flow at wall | ion fluence | mean impact energy | wall time |
|---|---|---|---|---|---|---|---|
| `k = 4` (chosen) | 4 625 | -0.987 T_e/e | +0.13 % | 0.60 c_s | 3.704e13 m^-2 | 3.56 eV | 168 s |
| `k = 1` (control) | 18 825 | -1.067 T_e/e | -0.00 % | 0.62 c_s | 3.699e13 m^-2 | 3.62 eV | 283 s |

The fluence agrees to 0.1 %, as the flux must. The wall potentials differ by 0.08 T_e/e, which is not
resolved: the run keeps no error bar, `k = 4` carries a quarter of the ion markers, and the
unmagnetised quick sheath's own standard error was 0.06-0.07. The control that decides it is the
rehearsal-size one below, with an error bar.

#### The references, generated (W8)

Built at the pinned commit in `~/local/GYRAZE-ref` against MacPorts GSL 2.8:
`make -B PROFILE=release CFLAGS="-Wall -O3 -I/opt/local/include" LDFLAGS="-O3 -L/opt/local/lib -lgsl
-lgslcblas -lm"`. Figure 6 reproduces to the printed digit: `(-2.410640, -2.822029)`, current
`0.000088`, 43 iterations, and the same six `misc_output.txt` numbers as above. Every case runs in its own
directory under `~/local/gyraze-runs/<case>/` through `run_case.py` there (writes the two input files,
runs the binary, lifts the final outputs out of GYRAZE's nested `OUTPUT/` tree and deletes the iteration
histories; clean means exit status 0 **and** "MP+DS combined iteration converged"), and
`make_manifest.py` writes `manifest.json` beside the outputs, refusing a run that did not exit cleanly.
All of it stays outside the repository; the four clean cases are copied to `office:~/gyraze-runs/`.

| case (`gammaflag=0`, `gamma_DS=0.2`, `tau=1`, floating wall) | `(phi_DSE, phi_wall)` | plan's table | `n_e,DSE` | entrance `gamma` that matches | `epsilon = lambda_D,DSE/rho_B` |
|---|---|---|---|---|---|
| `rehearsal-M400-a5-gDS0.2` | -1.577109, -1.678860 | -1.5771, -1.6789 | 0.1384 | 0.5377 | 0.25 |
| `matched-M900-a4-gDS0.2` | -1.882981, -2.083706 | -1.8830, -2.0837 | 0.1108 | 0.6009 | 0.167 |
| `scan-M900-a3-gDS0.2` | -1.958526, -2.072392 | -1.9585, -2.0724 | 0.0946 | 0.6502 | 0.167 |
| `scan-M900-a5-gDS0.2` | -1.806034, -2.094028 | -1.8060, -2.0940 | 0.1264 | 0.5626 | 0.167 |

Each manifest holds: the commit, date, paper version and build; the licence and provenance; the case
and the numerical-parameter file's hash; exit status, iterations, `(phi_DSE, phi_wall)` and
`misc_output.txt` under the C's names; the definition of `gamma` above; both orientations and potential
references; every normalisation; the conversion to this code's units; both entrance distributions,
checked against the closed forms rather than assumed (ions `0.2535-0.2541` against `2/(2 pi sqrt(pi/2))
= 0.2540` times `U exp(-U-mu)` over the bulk, electrons `0.0632-0.0641` against `(2 pi)^(-3/2)`); the
Debye reference density; each file's zero-padding row; `epsilon` and `alpha`; and each file's sha256.

**Two more normalisation traps, from the source, and the example had both.** (1) `phi_n_MP.txt`'s `x`
is in the **Bohm** gyroradius `rho_B = sqrt(Z T_e m_i)/(ZeB) = rho_S/sqrt(1+tau)`, not `rho_S`: the
orbit code writes `chi = (x - xbar)^2/2 + Z phi/tau` in the thermal `rho_i` and is handed
`x_grid/sqrt(tau)`. (2) `phi_n_DS.txt`'s `x` is in `rho_e = sqrt(T_e m_e)/(eB)`, not `lambda_D`: the
sheath Poisson equation is solved as `phi''/gamma_DS^2 = n_e - n_i`. Both are fixed by `B` alone, so in
this code `rho_e = gamma lambda_D` and `rho_B = sqrt(M) gamma lambda_D` whatever the density. So the
figure-6 presheath above is `25 rho_B = 17.7 rho_S` and its Debye sheath `83 rho_e = 25 lambda_D,DS`;
the 719-Debye-length arithmetic overstates that box, and its conclusion stands.

**The example as written is not the reference's problem, and it cannot be.** Its `gamma = 0.2` is at the
entrance plane, so its Debye sheath has `gamma_DS = 0.2 sqrt(n_e,DSE) = 0.067-0.074` and
`epsilon = 0.5-0.67`: no separation of scales at all. GYRAZE cannot supply that case: below its
`SMALLGAMMA = 0.19999` it replaces the Debye-sheath solve by a one-number model (a one-line
`phi_n_DS.txt`) and then aborts with SIGABRT in all four cases (kept as `aborted-*-gup0.2`, not
references; wall potentials 0.02-0.03 `T_e/e` above the table's), and forcing the solve at
`SMALLGAMMA = 0.05` was killed by the OS within a minute. The matched run therefore takes the entrance
`gamma` the manifest names, through the example's new `--gamma=G`, which is what makes it `epsilon =
0.25` and `0.167` -- and costs more, because `rho_s/lambda_D` grows with it.

**The comparison, and its predeclared tolerances** (`grazing_sheath.py`, `--reference=DIR`). The two
reference layers are joined the matched-asymptotic way (potentials add, densities multiply, the sheath
layer is 0 and 1 past its padded far end) at this run's distance from the wall; the example stops before
the run if the manifest is incomplete or GYRAZE did not exit cleanly, and says so when the case differs
from the run's. A number passes when `|run - reference| <= 3 SE + (max(epsilon, alpha) + (dx/l)^2) scale`:
`SE` is the standard error of four block means of the late window; `max(epsilon, alpha)` is the first
order the reference drops, coefficient one because no coefficient is computed; `(dx/l)^2` is the
second-order deposit and field solve on the layer's scale (`lambda_D,DSE` in the sheath, `rho_B` in the
presheath); `scale` is the layer's own drop. Compared: the wall potential and the mean ion impact energy
(`scale = |phi_wall|`), the Debye-sheath drop from the wall to where the reference sheath has done 90 %
of its drop, the largest potential and ion and electron density differences over the presheath beyond
that point and out of the source's run-up (`scale` = the reference's drop and density range there), and
the ion flux to the wall (`sum_flux_i n_0 sqrt(T_e/m_e) sin alpha`; the entrance distribution fixes it in
both, so only `3 SE`). The mean impact energy is `sum_Q_i/sum_flux_i + |phi_wall|` -- each ion's energy
is conserved in static fields, and `sum_Q_i/sum_flux_i = 2.99` is the entrance flux's `3 tau`. The
energy-angle distribution is **not** compared: `Fi_W.txt` holds the wall orbits' invariants, and making a
distribution of them is GYRAZE's post-processing, which would have to be re-derived. At the rehearsal's
`epsilon = 0.25` the wall-potential allowance is about 0.48 `T_e/e` plus `3 SE`, over a quarter of it: a
weak test by construction, and the recorded difference, not the verdict, is the result.

The quick preset against the rehearsal reference ran end to end in 159 s on this laptop, saying first
that it is not the same problem (`M` 100 against 400, 15 against 5 degrees, `gamma` 0.2 against 0.5377):
seven rows, the reference dashed on the potential and density panels, every number in `run.json`. Its
ion flux, 0.0412, is its own `1.5958 sqrt(m_e/m_i) sin(15 deg) = 0.0413`, not the reference's 0.0070,
so that row fails as it should. A timing slice that ends before any ion reaches the wall now says so and
records `null`, where it used to divide by zero.

**Measured cost, and the three GPU runs.** The office A4000 ran the 13 960-step rehearsal slice
(`--transits=0.03`, `gamma = 0.2`, pools 46 540 ions at `k = 14` and 41 480 electrons) in 73 s including
compilation, 5.2 ms a step at most: the rehearsal as written is 2.0 h there and `--matched` 5-7 h. The
matched-`gamma` runs are larger. The range below scales that 73 s by steps alone (the launch-bound
floor) and by steps times pool (the particle-bound ceiling):

| run | cells | steps | pool capacities (ions, electrons) | one A4000 |
|---|---|---|---|---|
| rehearsal, `--gamma=0.5377` | 668 | 2.88e6 | 96 000, 85 600 (`k_i = 14`) | 4.2-8.6 h |
| control, `--gamma=0.5377 --every=7` | 668 | 2.88e6 | 192 000, 85 600 | 4.2-13 h |
| matched, `--matched --gamma=0.6009` | 1038 | 8.39e6 | 145 000, 166 000 (`k_i = 27`) | 12-43 h |

On office, with `REPO` the checkout of this branch; each run writes `grazing_sheath/` (or
`grazing_sheath_matched/`) under its working directory, so each gets its own, and they go one after
another on one GPU:

    mkdir -p ~/w8/rehearsal && cd ~/w8/rehearsal && python $REPO/examples/3_advanced/grazing_sheath.py --gamma=0.5377 --reference=$HOME/gyraze-runs/rehearsal-M400-a5-gDS0.2
    mkdir -p ~/w8/control && cd ~/w8/control && python $REPO/examples/3_advanced/grazing_sheath.py --gamma=0.5377 --every=7 --reference=$HOME/gyraze-runs/rehearsal-M400-a5-gDS0.2
    mkdir -p ~/w8/matched && cd ~/w8/matched && python $REPO/examples/3_advanced/grazing_sheath.py --matched --gamma=0.6009 --reference=$HOME/gyraze-runs/matched-M900-a4-gDS0.2

The control halves `k` rather than setting `k = 1`: at `k = 1` the ion pool is the residence, 1.3e6
markers, fourteen times the run it checks. The result is `k`-independent if the two wall potentials
differ by less than `3 sqrt(SE_1^2 + SE_2^2)` from the two `run.json`s. `--markers=50` halves the
particle-bound end of the matched run if 43 h is too long.
Next after these: the 3 and 5 degree scan against `scan-M900-a{3,5}-gDS0.2` with `--gamma=0.6502` and
`0.5626`, which needs an `--angle` knob the example does not have yet.

### 8.2 Electron-field instability (W10)

Reference, verified 2026-09-18: L. P. Beving, M. M. Hopkins and S. D. Baalrud, *Electron-field
instability: excitation of electron plasma waves by an electric field*, Phys. Plasmas **30**(11),
112105 (2023), doi 10.1063/5.0156041. The publisher page is paywalled; an open full text is at
<https://www.osti.gov/pages/servlets/purl/2311492>. The instability excites electron plasma waves
of wavelength `>~ 30 lambda_De` at a growth rate proportional to the field and, notably,
**does not require a relative drift between electrons** -- which is precisely why the control in
the next paragraph matters. The phrase does not appear in arXiv full text, so the paper is the
only source; do not expect a preprint.

The imposed field is **static and uniform**, so `Simulation(external_E=...)` already supports the
driver and no time-dependent field hook is needed for this benchmark. Reported case:
helium, `n=3e14 m^-3`, `T_e=3 eV`, `T_i=0.026 eV`, `E_0=-800 V/m`, `L=1200 lambda_D`, five cells per
Debye length, 400 particles per cell per species, sixteen realisations. Verify every number and the
thermal-speed convention against the publisher PDF before freezing the benchmark.

Represent the imposed field as an external uniform E, not a sawtooth potential differenced on the
grid. Record external work. Support an exact uniform background, mobile helium ions, and frozen ion
macro-particles as three distinct controls. Resolve the accelerated drift over the **whole** run,
not only at `t=0`.

The indispensable control: for collisionless electrostatic Vlasov-Poisson with an exactly uniform
immobile background and uniform acceleration `a = q_e E_0/m_e`, the change of variables
`x' = x - a t^2/2`, `v' = v - a t` removes the acceleration. A driven and an undriven run
transformed into that frame must converge to the same fluctuation dynamics. Frozen noisy ion
macro-particles, mobile ions, collisions and boundaries each break it; isolate them one at a time.
Do not force a positive growth assertion in a limiting control where the continuum equations imply
equivalence to the undriven case. If the published growth is not reproduced under a matched
configuration, preserve the evidence and report it rather than relabelling numerical heating.

## 9. W11: algorithms

Audit charge continuity, the Gauss law, energy and momentum **separately** for each model, boundary,
filter, gather/current pair, collision model and integrator. Publish the actual discrete invariant or
residual, not an unconditional "conserves energy, charge and momentum".

- **Mandatory: source-free implicit electrostatic.** The implicit path currently rejects
  `model="electrostatic"`. Add it by suppressing transverse Maxwell evolution while keeping the
  compatible longitudinal current, discrete-gradient force and nonlinear solve. Do not compute an
  energy-preserving update and then overwrite `E_x` with a Poisson projection. Keep or explicitly
  choose the periodic mean-field convention. Report the final residual.
- **Mandatory: collision time-centering.** A pair-conserving binary scatter can still heat a
  leapfrog PIC run when inserted at the wrong time. Derive the correct split for this code's
  half-position, integer-velocity convention. Two half-duration Boris rotations do not compose to
  the full-step rotation; the angle is nonlinear in `dt`. Test on an isolated oscillator, then
  homogeneous thermal plasmas, then a relaxation case, comparing secular drift against the
  collisionless baseline.
- **Optional, only with measured gain**: ECSIM-type schemes, Ricketson-Hu explicit
  energy-conserving, Higuera-Cary, Darwin. Each needs independent discrete identities, both AD modes
  and an end-to-end benefit. Keep any prototype on a separate commit with a decision note.
- **Out of scope**: higher dimensions, AMR, a kernel DSL, a general circuit or chemistry framework.

## 10. Work order

Reviewable commits, in this order. W9 and the source-free parts of W11 may run in parallel after W1
and W3. W8 may not use invalid sources or occupancy spectra. W10 may not use misleading energy
plots. Re-run dependent benchmarks after any underlying correction.

- [x] **W0** baseline: reproduce the register, record hardware and versions, port what is wanted from PR #43.
- [x] **W1** parameter contracts, validation, coordinates, absolute step and time, supported combinations.
- [x] **W2** sources and boundary physics: sampling, safe pools, charge/current/energy exchange, true impacts. *(S04's remaining half is the magnetised example's own entrance condition, a physics choice that belongs to W7.)*
- [x] **W3** diagnostics and statistics: weighted moments, independent balances, correct windows, uncertainty. *(G04 and G09 stay open: they are the derivative support matrix of section 4.3, which is a claim to keep rather than a defect to fix. S20 was settled in W7.)*
- [x] **W4** orchestration and progress: pure kernels, host-owned meter, one snapshot schedule, exact restart.
- [x] **W5** TOML/CLI and persistence: one resolver, native archives, openPMD round trip. *(U09's axis label is W6, with the rest of the plotting.)*
- [x] **W6** plots and movies: evolving weighted populations, diagnostic histories, bounded memory, headless tests.
- [x] **W7** repair the four existing sheath and optimisation examples and their documentation. *(S15, S16, S04's second half, G05, U12 and S20 closed; every number on the four pages is from a run of the preset the page names, and the convergence table is a script.)*
- [ ] **W8** grazing-incidence benchmark, then a controlled finite-ordering extension. *(GYRAZE pinned and reproduced; matched case moved to `M=900, gamma=0.2, alpha=4 deg`; `Source(every=k)` built and tested; references for the rehearsal, the matched case and the 3 and 5 degree scan generated with manifests in `~/local/gyraze-runs` and on office; the example compares seven quantities against predeclared tolerances; two source-read traps found -- GYRAZE's `gamma` is at the Debye-sheath entrance and its axes are `rho_B` and `rho_e` -- so the matched runs need `--gamma=0.5377` and `0.6009`; next: the three A4000 runs in 8.1, then the scan.)*
- [x] **W9** model-comparison and Weibel examples on the existing kernels and shared theory. *(E01, E02, U13 and U14 closed; `parameters_and_sampling.py` and `output_and_restart.py` added; every example writes its settings, results and provenance through `jaxincell.save_run`. Section 16, W9.)*
- [x] **W10** electron-field instability with its limiting controls. *(section 16, W10: the paper's case reproduces growth of the paper's size only with discrete ions; the exactly uniform background, where the change of frame removes the field, shows none.)*
- [x] **W11** algorithm audit; source-free implicit electrostatic; collision time-centering. *(section 16, W11. The optional algorithms of section 9 are deferred, not evaluated.)*
- [x] **W13** 3-D external fields ported from `ds/3D_external_fields`: `(cells, ny, nz, 3)` external E and B gathered at x, y, z, and `magnetic_moment`. *(section 16, W13: grad-B drift, mirror bounce and the uniform limit against guiding-centre theory; flat path unchanged.)*
- [ ] **W12** convergence, performance, documentation, review packet. *(Documentation part started: every figure in one style, set in the package as `jaxincell.style()`/`figure()`; README benchmarks grouped as 1D1V, 1D2V and 1D3V with the agreement against each reference; example and user-guide pages led by their figure and a measured-against-reference table, prose kept to the numerics pages; movies written for the web by `docs/scripts/movies.py`, 0.1-0.5 MB each, to be embedded once uploaded as PR attachments, since GitHub plays no video stored in the repository. Open: convergence and device-coverage evidence.)*

## 11. Acceptance checklist

- [x] Work is on `research-release`; `rj/additions-to-pr` untouched; no main writes, merges, force pushes or releases. *(holds through W8; re-check at the end.)*
- [ ] Every S/G/U row reproduced or marked resolved with evidence, then fixed with a test that fails on the old code. *(39 of 45; G04, G09, U13, U14, E01 and E02 remain, and belong to W9 and W12.)* *(W9 closed U13, U14, E01 and E02: 43 of 45, with G04 and G09 left to W12.)*
- [x] Component-wise and drifting or field-aligned source sampling; supported-source contracts complete. *(W2, and W8's `model="sampled"` for a reservoir that is none of the closed forms.)*
- [x] Source, cloud, collector, current and energy/momentum transfers derived and independently verified. *(W2; `charge_balance` is the independent check and found a defect in W2's own overlap accounting.)*
- [x] Event-based impact spectra and event-aware derivative checks; hard-count limitation retained. *(W2's `Impacts` and `_at_impact`; G02 is the derivative check.)*
- [x] Overflow invalidates a run; cutoff error measured; `active=0` safe. *(S09's `validate()`, S10's `Wall.truncated`, and a test that starts from an empty box.)*
- [x] Windows, coordinates, fluence-versus-current labels and statistical uncertainties corrected everywhere. *(S01, U10, and the fluence and scatter reporting of W7's examples.)*
- [x] Pure JAX runner and host-owned progress both work; repeated-run, AD and restart tests pass; verbose equals silent. *(W4.)*
- [ ] One parameter vocabulary with a migration path; README parameter tutorial exists. *(U09 gave the vocabulary and the release notes the migration; the tutorial is W9's `parameters_and_sampling.py`.)* *(W9: the tutorial exists and the README points to it; the box stays open for the `sampling` names of section 7, `low_noise` for `quiet`, which no lane has taken.)*
- [x] TOML/CLI covers every shipped workflow with strict validation. *(W5.)*
- [x] Native archive and exact checkpoint work; openPMD read back independently. *(W5; a restart through a file is bit-identical to one through memory.)*
- [x] Energy, charge and momentum panels restored with truthful open-system residuals; movies weight evolving populations. *(W6.)*
- [x] Existing examples retained and corrected, with saved data and provenance. *(the four sheath and optimisation examples in W7; the rest are W9's.)* *(W9: done, every example ends in `save_run`.)*
- [ ] Matched GYRAZE case with real reference data, uncertainty, and a documented finite-ordering study.
- [x] Weibel linear growth verified mode by mode; PR #43's nonlinear preset ported and preserved. *(W9: 7.2 % mean, 18.0 % worst over 8 of 11 unstable modes.)*
- [x] Explicit/implicit, collisional/collisionless, filtered/unfiltered and relativistic comparisons demonstrated. *(W9's `compare_models.py`.)*
- [x] Independent-target sheath inference with real statistical uncertainty. *(W7's G05: 0.3264 against 0.35 on a residual of 0.0512, and an error bar of +-0.0207 that is a scatter and not a grid.)*
- [x] Electron-field example with verified inputs, limiting controls and an honest interpretation. *(W10, section 16.)*
- [ ] Source-free implicit ES and collision time-centering done; optional algorithms have implement/defer evidence.
- [ ] Fast suite and headless examples pass; scientific claims carry separate convergence evidence.
- [ ] Clean install, CLI and documentation builds pass; precision and supported versions documented.
- [ ] Warm timings, compilation, memory and device coverage reported without fabrication.
- [ ] Pushed to `research-release` with the right author and committer; PR #42 updated and left open.

## 12. Report to the maintainer

Final branch and head; changes grouped by physics, numerics and interface; file and line counts with
reasons for growth; test commands and actual results; the exact command for each figure and movie;
TOML examples; theory provenance; performance and precision; migration notes; and the unresolved
scientific limitations. Distinguish **implemented**, **kernel-tested**, **physically validated** and
**not yet validated**. A benchmark whose script runs is not a completed benchmark.

No requested functionality disappears in a simplification. An example's comments and explicit
construction lines are part of its function as a teaching tool.

## 13. Pause of 2026-09-22: state and next steps

**Done and pushed here.** Every documentation figure in the group's paper style
(`final_fig.ipynb` on `lma/JOSS`), set once as `jaxincell.style()`/`figure()`. README benchmarks
by 1D1V / 1D2V / 1D3V, with relative image paths so they render on any branch. Example and
user-guide pages led by their figure and a measured-against-reference table. `Source(every=k)`.
A 48 % explicit-step regression, which the wall ledger caused in periodic boxes, fixed (41 ns per
particle-step, level with 00f6ce0). GYRAZE rebuilt at the pinned commit outside the repository,
figure 6 reproduced, and references for the rehearsal, matched and scan cases in
`~/local/gyraze-runs` and `office:~/gyraze-runs`. `--gamma` and a seven-quantity comparison in
`grazing_sheath.py`. PR #43 merged here, keeping this tree. CI green at 55bc44d.

**Estimated completion by lane:** W8 ~60 %, W9 ~15 %, W10 ~5 %, W11 ~15 %, W12 ~35 %.

**Paused mid-way. Resume in this order:**

1. **Showcase movies.** `docs/scripts/movies.py` now draws two panels per case in normalised units
   (the phase space or field on the left, the physics quantity revealed in time on the right), at
   1280x720. Two-stream and bump-on-tail are reviewed and good; Weibel (now run to 240 ω_pe to reach
   saturation) and the sheath (velocity axis taken from the data, since the source injects ions near
   Mach 8.6) need one render and a look at their frames. Then post the four MP4s as a PR #42
   comment — GitHub plays only uploaded attachments — and put the resulting links in the README and
   the docs. Posting needs the maintainer's go-ahead.
2. **Docs-only PR to `main`** (branch `docs/benchmarks-style`, pushed, work in progress, no PR yet):
   restyles `main`'s own figures and README with `main`'s code only. Finish it: regenerate
   every figure; add the paper's explicit/implicit 2x2 (Landau and two-stream, with energy error)
   and runtime CPU/GPU plus growth rate against drift for several particle counts; README
   benchmarks by dimensionality; `sphinx -W`; open the PR, get CI green, and merge it — the
   maintainer asked for this merge.
3. **The same paper figures here** (branch `work/longyu-figures`, pushed, work in progress: a
   shared two-stream setup and a `fig_explicit_implicit.py` exist). Finish, review the PNGs, add
   them to the README 1D1V section and the docs, and merge into this branch.
   **Done 2026-09-23:** `explicit_implicit.png` (conservation page) and `runtime_resolution.png`
   (performance page) are in, with the README 1D1V rows. Left: the CPU timings are from the
   laptop at load 57.5 and non-monotonic at 2000; re-time with `fig_runtime.py` on an idle
   machine (the office Xeon with `JAX_PLATFORMS=cpu`, then GPU 1) and redraw.
4. **GYRAZE (W8).** The rehearsal (`--gamma=0.5377 --reference=rehearsal-M400-a5-gDS0.2`) runs on
   office GPU 0 in `~/w8/rehearsal` from code at bfd6f90; read `run.log` when `done` appears. The
   k=7 control was stopped to lend GPU 1 to timings: restart it in `~/w8/control` with the same
   command plus `--every=7`. Then the matched run (`--matched --gamma=0.6009`, 12-43 h on one A4000)
   and the 3 and 5 degree scan, which needs an `--angle` option. Put the figure on the page.
5. **W9, W11, W10, W12** as in section 10. W9's new examples and W11's two mandatory items are
   independent of W8 and can run in parallel.

## 14. State of 2026-09-23

**Finished.**
- Showcase movies: two-stream, bump-on-tail, Weibel (to saturation) and the sheath (electron phase space, with the fastest electron the wall returns drawn over it), 60-500 kB each, from `docs/scripts/movies.py`. They still need uploading as PR attachments and linking from the README; that is a publishing step for the maintainer.
- Documentation and figures on `main` in the paper style, from `main`'s own code (PR #44, merged).
- The paper's explicit/implicit 2x2 and the CPU/GPU runtime and growth-rate-against-drift figures, here, in the README's 1D1V section and the docs. MyST substitutions inside math never rendered on eight pages; fixed.
- W8 rehearsal (`m_i/m_e` 400, 5 degrees, 5.4 h on an A4000): recorded on the grazing-sheath page with its table and figure.

**W8, open: the entrance plane.** The Debye sheath (-0.588 against -0.526 T_e/e) and the impact energy (4.43 against 4.67 T_e) agree. The magnetic presheath does not:
- The ion density at the plane is 0.82 n_0.
- The potential rises 0.8 T_e/e above the plane.
- The ions move along B at about half the reference flow.
- The density piles up to 1.4 n_0.
- The ion flux is 6 % high.

The hypothesis to test first is that the open plane absorbs ions that gyrate back across it within one gyroradius. The test: a buffer of a few rho_i before the compared region, or re-emitting what leaves through the plane. A second test is six transits, to rule out an unfinished transient. The matched run (`--matched --gamma=0.6009`) and the 3 and 5 degree scan wait on this, because they would carry the same defect. W8 is at about 65 %.

**Branches.**
- `rj/additions-to-pr`, `fix/warn-on-ignored-parameters` (#41), `collsion` (#35), `ds/OpenPMD` (#36), `rishi/mixed_BCs` (#34) and `rj/full_EM_2` (#31) are all contained here; each PR closes as merged when #42 merges.
- #41 is also resolved in substance: `external_E`/`external_B` arrays and the `[external]` TOML table are applied in every push, and tested for gyration (tests/test_physics.py).
- Not contained, and a lane of their own:
  - **W13, 3-D external fields** (`ds/3D_external_fields`): external E and B on an (x, y, z) grid interpolated at the particle's y and z as well as x, plus a magnetic-moment diagnostic. It is written against `main`'s old API, so it needs a port onto `Simulation(external_E=..., external_B=...)` with a grid shape `(cx, cy, cz, 3)` and extents, not a merge. **Done**, §16 W13.
  - `ds/source_particles`: superseded by `Source` and the wall ledger.
  - Research branches outside the scope of #42: `rj/gr` (a general-relativistic Boris push), `rj/momentum`, `rj/full_EM`, `rj/full_EM_PIC`, `woolford_comparisons` (PyPIC3D comparison), `multifidelity`, `bump_on_tail`, `merge_exact_conservation_magnetic2`, and the paper branches `JOSS` and `lma/JOSS`. Stale: `development`, `rj/fix_ghost`, `rj/fft_position_velocity`, `lma/enegy_conservation_to_main`, `XYJeff23-patch-1`.

**Left for later, in order:**
1. W8 entrance treatment, then the matched run and the scan.
2. Re-time the runtime figures on an idle office (CPU and GPU 1). Office was unreachable on 2026-09-23 and this Mac was at load 60-190, so the CPU points are inflated.
3. W9 (model-comparison and Weibel examples, ~15 %).
4. W11 (algorithm audit, source-free implicit electrostatic, collision time-centering, ~15 %).
5. W10 (electron-field instability, ~5 %).
6. W12 (convergence and device coverage, then the review packet, ~45 %).
7. ~~W13 (the 3-D external-field port).~~ Done 2026-09-27, §16 W13.

## 15. Integration of main (#44–#50)

`main` gained six merged PRs (#44–#48, #50) and one open one (#49) after this branch forked at
`1d9f280`. Each was read (body, review comments, commits) and its finding checked against this
branch's rewritten code, with a reproduction where the claim is testable. The reproductions ran
a 64-cell periodic two-stream (4000 + 4000 particles, 400 steps) and a cold beam through
immobile ions (50 steps). `main` was then merged with a merge commit, resolving every conflict in
favour of this branch's tree, so that its contributors' commits stay in the history and #42
contains #44–#50.

| PR | what it found on `main` | state here | action and evidence |
|---|---|---|---|
| #44 docs restyle | one paper figure style; explicit/implicit 2x2; drift scan (CPU/GPU runtime, rate against drift); README benchmarks by dimensionality | already here: `jaxincell.style()`/`figure()`, `fig_explicit_implicit.py`, `fig_runtime.py` (runtime and rate against drift at three particle counts), `two_stream_scan.png`, README 1D1V/1D2V/1D3V sections (section 14) | nothing to port; `main`'s `fig_energy_conservation.py`, `landau.py` and old-API scripts are dropped in the merge |
| #45 relativistic two-stream | page and figure: beams at ±0.8c, relativistic Boris on and off; each pusher reproduces its own cold rate and conserves its own energy | missing | ported: `docs/scripts/fig_relativistic.py` on `Species`/`Domain`/`Solver`, `docs/examples/relativistic_two_stream.md`, toctree and table row, link from `numerics/diagnostics.md`, README row and figure, `make_all.py`. Here: rates 0.8 % (relativistic, mode 1) and 1.1 % (non-relativistic, mode 2) from cold theory; E_rel to 6.6e-4 with the relativistic pusher, E_N to 3.3e-3 with the other; no electron reaches c with the relativistic pusher, 24 % do without. Both species on one lattice of positions (random positions gave a noise floor of 10 J/m^2 that hid the linear phase) |
| #46 charge-conserving deposit | the half field updates used the currents of x^{n-1/2}->x^{n+1/2} and x^{n+1/2}->x^{n+3/2}, so Gauss's law drifted (2.9e-2 filtered, 0.87 unfiltered); O(NG) deposit; no Gauss/momentum diagnostics in plot | already handled | each half step takes `current_from_continuity` of the density it covers (x^n->x^{n+1/2}, x^{n+1/2}->x^{n+1}), the filter applied to rho before the continuity solve. Reproduction, max Gauss residual: periodic 3.3e-15 (filter and none), reflective 3.6e-15, `field_solver="gauss"` 1.8e-15, electrostatic 1.9e-15, implicit 6.9e-16; absorbing walls hold it too via the wall ledger (`test_gauss_law_holds_at_every_wall_with_and_without_filtering`). Deposit is a 3-point scatter-add (`_core.deposit`). `gauss_residual`, `momentum_error` and `charge_balance` are in `diagnostics()` and the conservation panel of `plot()` |
| #47 mean of the t=0 field | `E_from_Gauss_1D_Cartesian` fixed E at the left edge, leaving a uniform E in a periodic box; a kick at every restart | already handled | `_integrate_from_walls` removes the mean source and returns F - mean(F) for (0, 0), at t=0 and in every Gauss solve. Reproduction: initial `<E_x>/max|E_x|` = 3e-17 |
| #48 electrostatic Boris step | (1) the Gauss solve used x^n, not x^{n+1}; (2) it deposited on the faces, leaking charge across the wrap; (3) the push saw the Ampere half-step field including the mean current, decelerating a current-carrying plasma | (1), (2) already handled: `model="electrostatic"` and `field_solver="gauss"` deposit the density of the positions the half step ends on, on the centres, and solve on the faces. (3) **present** with `field_solver="gauss"` (not in `model="electrostatic"`, which does no Ampere step): a cold beam at 0.05c through immobile ions lost 5.9 % of its velocity in 50 steps at omega_pe dt = 0.05, the (omega_pe dt)^2/2 per step #48 predicts | ported: `_advance_fields` removes the mean of J_x before the Ampere half steps when `field_solver="gauss"` in a periodic box; the half-step field is then exactly the Gauss field of the half-step density. Beam velocity change after: 8e-14. New test `test_a_current_carrying_beam_is_not_slowed_by_a_mean_field` (gauss and electrostatic), fails before the change; `numerics/field_solvers.md` explains it. `field_solver="ampere"` is unchanged: there the mean current drives the uniform plasma oscillation, which is physics. No documentation figure uses `field_solver="gauss"`, so no number quoted in the docs moves |
| #49 (open) rename | `grid_points_per_Debye_length` is dx/lambda_D, the inverse of its name; docs contradicted each other | not applicable | this branch has no such parameter: `Species` takes `density` and `vth` directly, and `Simulation.debye_length()` reports lambda_D. Nothing renamed; the merge takes none of the old name. #49 stays open |
| #50 quiet velocities | random velocities put N^{-1/2} noise into E_k and hid linear Landau damping | already handled | `Species(sampling="quiet")` places velocities at `vth * erfinv(2q - 1)` with q from van der Corput in bases 2, 3, 5 per axis (the same load), with the `plus_minus` pairing fix; `test_quiet_start_samples_the_maxwellian_and_fills_the_box`. `main`'s `quiet_velocities_{x,y,z}` flags are not added: the three-state `sampling` replaced such booleans (its docstring in `_config.py`) |

Conflicts in the merge were resolved to this branch's files; `main`'s files that this branch had
removed (the old `_algorithms.py`, `_boundary_conditions.py` and the like, their tests and
old-API docs scripts) stay removed.

## 16. Lanes after the integration

### W11: algorithms (2026-09-27)

Three commits on `research-release`, each with its test; numbers from office CPU, double precision.

**Source-free implicit electrostatic (mandatory, done).** `Solver(algorithm="implicit",
model="electrostatic")` is no longer refused. The Picard loop advances E_x by Ampere's law with
the same continuity current and discrete-gradient force as the electromagnetic scheme; E_y, E_z
and B are not evolved (external fields still act). No Poisson projection follows. Periodic
mean-field convention chosen explicitly: the mean of J_x is left out in a periodic box, so
<E_x> = 0 at every step, as in the explicit electrostatic model; between walls nothing is
subtracted. Energy balance unchanged because sum_i E_x,i <J_x> = 0. Final residuals on the
tests' two-stream (2000+2000, 64 cells, 150 steps, c dt/dx = 4.5): energy error 2.8e-11 at 4
Picard iterations, 2.3e-16 at 8 and 12; Gauss residual <= 2e-15; |<E_x>|/max|E_x| <= 6e-17;
same at reflecting walls. `field_solver="gauss"` stays refused with the implicit scheme.
Test: `test_implicit_electrostatic_scheme_conserves_energy_and_charge_without_a_projection`
(periodic, reflective). Docs: numerics/implicit.md, "The electrostatic model".

**Collision time-centring (mandatory, done).** The explicit step kicks u^n -> u^{n+1} at
x^{n+1/2}; collisions used to act right after the kick at x^{n+1/2}, and the next drift then
moved the integer-time position with the scattered velocity, so every collision changed the
energy by (dt/2) Delta p_a . (q_a E_a/m_a - q_b E_b/m_b). They now act at
x^{n+1} = x^{n+1/2} + dt v_kicked/2, between the two half drifts, and the drift continues with the
scattered velocity (Strang-type split around the unsplit kick; splitting the Boris kick is
rejected because two dt/2 rotations do not compose to one dt rotation). Evidence:
- isolated oscillator (electrons and a 25 m_e species of equal charge in one harmonic well,
  all in one cell so every collision is exact, nu/omega ~ 0.1, 16 periods), max relative
  energy error at omega dt = 0.4 / 0.2 / 0.1: collisionless 1.07e-3 / 2.45e-4 / 4.54e-5; new
  placement 1.01e-3 / 1.96e-4 / 3.86e-5 (second order, collisionless level); old placement
  22.1 / 6.83 / 1.53. Test `test_collisions_at_the_integer_time_keep_the_leapfrog_energy_error`
  fails on the old code.
- homogeneous thermal plasma (e + 25 m_e ions, 20000 each, 64 cells, ln Lambda = 1e4,
  200/omega_pe): self-collisions only, total energy drift 3e-5 (omega_pe dt 0.4) and 1e-5 (0.1)
  against 1e-5 collisionless, both placements alike.
- with electron-ion collisions, 6.7e-3 / 2.5e-3 (new) and 5.9e-3 / 2.3e-3 (old); relaxation
  T_i = 4 T_e: 9.0e-4 / 2.8e-4 (new), 9.3e-4 / 1.8e-4 (old). This drift is the between-species
  operator's own (a particle of the shorter list collides several times from the same start
  velocities, second order in angle, so first order in dt per unit time), not the splitting's.
  Open: an exact multi-pass between-species pairing would remove it; not attempted.
- implicit scheme: collisions act after the step at (x^{n+1}, u^{n+1}), both at t^{n+1}; energy
  still exact there (audit table), splitting first order.
Docs: numerics/collisions.md, "Where in the step".

**Audit (done).** `docs/scripts/audit_invariants.py` (in make_all.py) measures, separately, max
energy error, max momentum error and max Gauss residual for explicit EM, explicit EM + filter,
explicit gauss, explicit ES, explicit ES + self-collisions, implicit EM, implicit ES, implicit
ES + self-collisions, periodic and reflecting, on the two-stream setup; values in
measurements.json, table on numerics/verification.md, "Discrete invariants, model by model".
Summary: Gauss residual <= 3e-15 everywhere; energy round-off (2-4e-16) only for the implicit
scheme, explicit 2.3e-5 (4e-6 with the filter) over 150 steps; periodic momentum round-off only
for the explicit scheme (implicit 1e-5); between walls momentum is exchanged with the walls
(4e-3 explicit, 1.5e-2 implicit) and is not an error.

**Optional algorithms: deferred.** ECSIM, Ricketson-Hu, Higuera-Cary and Darwin were not
prototyped: none has a measured end-to-end benefit on a case this project runs, and each would
need its own discrete identities and both AD modes. No code added.

Checks: flake8 0; focused tests (test_collisions, test_boundaries_and_loop, the implicit tests of
test_physics) pass; `sphinx-build -W` clean apart from Python 3.10's own `dataclasses.replace`
docstring on office (CI uses 3.12); full suite left to CI.

### W9: model comparison and Weibel (2026-09-27)

**Done.**
- **Shared theory is public: `jaxincell/theory.py`.** The dispersion functions moved out of
  `docs/scripts/dispersion.py` (now a re-export), plus `populations(simulation)`,
  `two_stream_rate(simulation, mode)` and `weibel_rate(k, wp, vthx, anisotropy)`. NumPy, with
  SciPy imported on first use; SciPy joins the `dev` extra so the tests cover it
  (`tests/test_theory.py`: the Landau root, Z' and Z'', the Weibel rates the physics test
  hard-codes, the cold two-stream limit through `populations`, and the Newton rejections).
  No `sys.path` hack in any example.
- **Weibel (U13, E02).** `2_intermediate/weibel.py` keeps the four-wavelength threshold box
  and adds PR #43's twelve-wavelength run (120000 particles, 284 cells, 12000 steps to
  t omega_pe = 325, through saturation), with every mode fitted over one linear window
  (t omega_pe from 20 to the time the total |B|^2 reaches 5 % of its maximum, here 20 to 70),
  against `weibel_rate`. Modes with R^2 < 0.8 are drawn open and not compared (declared
  before the comparison). Result, double precision on an A4000: **7.2 % mean, 18.0 % worst
  over 8 of 11 unstable modes** (k/k_c 0.17-0.83); single precision gives the same numbers.
  Four-wavelength gains: 13.3 smallest below the cutoff, 4.77 largest above. History: fields
  every 20 steps, no particle history. `--quick` (six wavelengths, 24000 particles, 3000
  steps) is ~2 min on a CPU and says it is a smoke preset. Writes `run.json` with provenance,
  `modes.npz` and the figure. `fig_weibel.py` gains panel (c) by running the example and
  reading its `run.json`, so the page and the example are one run; `weibel_wide_*` keys in
  `measurements.json`.
- **Model comparison (U14, first part).** `2_intermediate/compare_models.py`: one seeded
  two-stream setup (20000 electrons, omega_pe dt = 0.087, to t omega_pe = 40) run with one
  switch changed at a time, each against `two_stream_rate` of the same populations.
  Reference, electrostatic, Gauss and filtered: 0.2805 against 0.2899 (-3.3 %, identical to
  four digits), energy 2e-4. Implicit: 0.2808 (-3.1 %), energy 3e-16. Relativistic: 0.2683
  against 0.2713 (the kinetic root times the cold gamma0^-3/2 reduction at the same k),
  -1.1 %. Collisional (electrostatic): identical to electrostatic, as nu/omega_pe ~ 1e-7
  requires. `fig_compare_models.py` runs it for the page `docs/examples/compare_models.md`.
- **Defect found and fixed: collisions above the light-wave Courant limit.** The first
  collisional run blew up (|v| ~ 1e103 by step 58) at c dt/dx = 4.5 in the explicit
  electromagnetic model: collisions scatter a longitudinal beam into y and z, which seeds
  the light wave, and `_check_courant` only looked at the initial transverse velocity. It
  now also warns when `collisions` is set (`test_courant_warning_fires_only_when_a_light_wave_can_be_seeded`
  extended). The example runs its collisional case electrostatic and says why.
- README 1D1V row and figure, 1D2V Weibel row; examples index and toctree; API page gains
  `jaxincell.theory`; CI Examples job runs `weibel --quick` and `compare_models --quick`
  (with scipy).

**Left in W9 (not ticked):**
1. E01: Landau floor measured where the amplitude stops falling; re-measure example, figure
   and test together at 500 and 800 steps.
2. `1_basic/parameters_and_sampling.py` (the README parameter tutorial of section 11).
3. `2_intermediate/output_and_restart.py` (rest of U14).
4. The remaining existing examples' saved data and provenance (section 11), and the
   filtered/unfiltered pair at a wavelength the filter does act on (mode 1 of 64 cells is
   untouched by two passes, which the page says).

#### W9, second pass: the four open items (2026-09-27)

1. **E01 closed.** `jaxincell.theory.damped_mode(t, amplitude, above=5)`: the floor is the
   median of the maxima from the first one that is not below its predecessor, and only the
   maxima before it and above five times it are fitted; a run that never reaches the floor is
   refused. On the example's setup (150000 particles): 500 steps is refused (the maxima never
   stop falling), 800 and 1200 steps both give **gamma -0.1531, omega 1.4120** against
   -0.1534 and 1.4157 (0.2 % and 0.3 %), floor 45.6 V/m, 7 maxima. The example, the figure and
   `test_landau_damping_matches_the_kinetic_root` all run 800 steps through the same function
   (the test now fails on the old fixed-fraction floor at 800 steps, which gave -0.1433).
   Found on the way: `landau_root` could return the backward twin -conj(omega) at some k
   (the figure's kinetic curve dipped to -1.1 at k lambda_D 0.25 and 0.31); it now returns the
   forward root, with a test.
2. **`1_basic/parameters_and_sampling.py`**: T_e = 10 eV, n = 1e18 m^-3, derived v_th,
   omega_pe, lambda_D checked against `Simulation.plasma_frequency()`/`debye_length()`;
   the same plasma loaded `random`, `lattice`, `quiet`. Density spread per cell 0.108 /
   2.5e-3 / 4.4e-3 against the Poisson 1/sqrt(100) = 0.100 for random; temperature error
   -1.2e-2 / -1.2e-2 / -1.1e-3 against +-1.8e-2 for random velocities; field noise at the end
   3.4e-5 / 3.3e-6 / 9.9e-7 J/m^2. ~30 s on a CPU; in CI.
3. **`2_intermediate/output_and_restart.py`**: 400 two-stream steps in one go against 200 +
   `save_state` + `load_state` + 200: `t`, `E`, `B`, `x`, `v` bit-identical. openPMD series
   written and read back with openpmd-api: field difference 0, coordinates from
   offset + (i + position) dx within 1e-18 m of `out.faces`; skipped with a message when
   openpmd-api is absent. ~20 s; in CI.
4. **Provenance for every example.** `jaxincell.save_run(folder, example, settings, results,
   figure=None, **arrays)` writes `run.json` (settings, results, `provenance()`), `data.npz`
   and `figure.png`; tested in `tests/test_config_and_outputs.py`. Every example that lacked
   a record now ends in it (landau_damping, langmuir_wave, two_stream, bump_on_tail,
   collisions, wall_reflection, conservation, optimize_two_stream, weibel, compare_models,
   and the two new ones); their figures moved to `jaxincell.figure()`. Hard-coded theory
   numbers replaced by `jaxincell.theory`: langmuir's kinetic roots, bump-on-tail's 0.1463
   (Newton on the populations, same value; measured 0.1370, -6.3 %), and
   optimize_two_stream's "about 0.70", now the kinetic optimum of the warm beams,
   k v0/omega_pe = 0.708 (ascent 0.699, -1.2 %).
5. **Filtered/unfiltered where the filter acts.** `compare_models.py` adds a cold
   oscillation at k dx = pi/2 with 0 and 2 passes: frequency ratio **0.7097 against
   sqrt(G) = 0.7071 (+0.37 %)**, G the compensated binomial transfer function; panel (d).
   The mode-1 pair stays, as the check that the filter leaves long waves alone.

Docs: `parameters_and_sampling.md`, `output_and_restart.md`, the filter pair on
`compare_models.md`, E01 prose on `landau_damping.md`; `docs/scripts/common.run_example`
runs an example in a scratch folder and hands back its `run.json` and figure, used by
`fig_compare_models.py`, `fig_parameters_and_sampling.py` and `fig_output_and_restart.py`.
README 1D1V rows for both new examples and the filter ratio; CI Examples job runs both new
scripts.

Left for other lanes: G04 and G09 (W12); the `sampling` rename of section 7.

### W8: the entrance plane, found and fixed (2026-09-27)

**Cause: the cold initial load, not the open plane.** The example filled a quarter of each pool
at `t = 0` with particles at rest (`Species` with no `vth`). Ions with `v_par = 0` are what the
Chodura condition excludes from the entrance; in the presheath's weak field they leave far slower
than a transit. After three transits they were the extra 0.4 `n_0` of the pile-up, and the
potential rose to confine electrons around them.

Evidence, two runs on office GPU 0 at the rehearsal's physical parameters (`--gamma=0.5377
--markers=20 --transits=0.6`), identical but for the start. Scratch diagnostics in
`office:~/w8/diag/{cold,reservoir}/run.log`:

| 0.6 transits | cold start | reservoir start |
|---|---|---|
| ion flux / reference | 0.0023 / 0.0070 (fail) | 0.0070 / 0.0070 (pass) |
| ion, electron density max diff | 0.70, 0.69 (fail) | 0.055, 0.052 (pass) |
| mean impact energy | 4.05 | 4.60 (ref 4.67) |
| injected ion weight leaving back through the plane | 77 % | 72 % |
| electrons leaving back through the plane | 98.6 % | 98.3 % |

The plane re-absorbs about three quarters of the ions it emits in **both** runs, and the
reservoir start still meets every tolerance. So the re-absorption is the half-space reservoir
working as intended, not a defect: a gyrating ion that leaves is replaced by the fresh crossings
drawn from the same distribution. No buffer, re-emission or `phi' = 0` change is needed, and no
package code changed.

**Fix** (`examples/3_advanced/grazing_sheath.py`, commit `80fb5ec`; the runs used the same change as `77a80cc` before a rebase): the initial slots take
their velocities from the same reservoirs the sources emit from (`Species(v=...)`).

**Rehearsal at full preset after the fix** (`office:~/w8/rehearsal-77a80cc`, 5.6 h, GPU 0):

| quantity | before (cold) | after | GYRAZE | tolerance | after |
|---|---|---|---|---|---|
| wall potential | -1.034 | -1.601 | -1.679 | 1.75 | pass |
| Debye-sheath drop | -0.588 | -0.647 | -0.526 | 0.23 | pass |
| presheath potential max diff | 0.89 | 0.87 | -- | 1.80 | pass |
| ion density max diff | 0.44 | 0.039 | -- | 0.21 | pass |
| electron density max diff | 0.44 | 0.036 | -- | 0.21 | pass |
| mean impact energy | 4.43 | 4.83 | 4.67 | 0.51 | pass |
| ion flux | 0.0074 | 0.00684 | 0.00695 | 0.000106 | fail (diff 0.000111, 3.2 SE) |

Flow at the plane 0.098 `c_s`, the entrance value, with `n u_x` constant through the presheath.
Open: the flux is 1.6 % low at 3.2 SE against a 3 SE allowance with no model term. Candidates are
the first cell's 0.82 `n_0` deficit at the plane (a deposit edge, outside the compared region)
and the `O(alpha)` difference between GYRAZE's flux along B times `sin alpha` and a full-orbit
normal flux; neither is tested. The presheath potential's standard error, 0.5 `T_e/e`, makes that
row weak.

**Matched run launched** after this, judged close enough to proceed since it tests the same
physics inside the reference's range: `office:~/w8/matched-77a80cc`, PID 979146, started
2026-09-27 16:24 on GPU 0 from `~/w8/code-77a80cc`, `--matched --gamma=0.6009
--reference=~/gyraze-runs/matched-M900-a4-gDS0.2`, 8.39e6 steps, 12-43 h. Collect: `done` appears
in that directory when it exits; read the tail of `run.log` and `grazing_sheath_matched/run.json`,
then add its table and figure (PIL, ~1600 px, 128 colours) to `grazing_sheath.md`. Next: the 3 and
5 degree scan, which needs `--angle`.

### W10: electron-field instability (2026-09-27)

**Inputs verified** against the open full text (OSTI 2311492, the accepted manuscript of
Phys. Plasmas 30, 112105): helium, n = 3e14 m^-3, T_e = 3 eV, T_i = 0.026 eV, E_0 = -800 V/m,
lambda_De = 7.43e-4 m (reproduced: 7.434e-4), kappa_e lambda_De = 0.198, 1200 lambda_De, five
cells per lambda_De, 400 particles per cell per species, random positions, sixteen
realisations, periodic, electrostatic; fit window t omega_pe 35-170 on <E> (eq. 2, E minus its
time mean at each x). Their thermal speed is v_Te = sqrt(T_e/m_e) in the text and in eq. 7's
sqrt(2(1 + i kappa/k)); the Maxwellian printed after eq. 6 uses exp(-v^2/v_Te^2), which is
the sqrt(2T/m) convention, an inconsistency of the paper. They quote gamma_fit = 7.9e-3
against 1.5e-2 from eq. 10 at the mean drift of the window (20 v_Te).

**Theory** (`jaxincell.theory`): `field_epsilon` (eq. 7, with its derivative), `field_rate`
(eq. 10 with the eq. 8 step), `field_root` (Newton from Bohm-Gross). Tests: kappa = 0 is the
Maxwellian dielectric; the derivative; eq. 10 is the small-field limit of the eq. 7 root to 3 %
at kappa lambda_De = 1/500; 3 k kappa/2 = 0.015 at the paper's numbers. Found: in eq. 7 the
drift enters only as omega - k u, so it Doppler-shifts the root and never changes its growth;
the paper's cutoff k < 1/u (eq. 11) and k* = 1/(kappa t) (eq. 12) come from the eq. 8 heuristic,
where omega_r = 0 is a wave standing still in the ion frame.

**Example** `examples/3_advanced/electron_field.py`, five cases, each of four realisations
(seeds 0-3), omega_pe dt = 0.005 (40 v_Te crosses one cell per step, the paper's condition),
40000 steps to t omega_pe = 200; field by `external_E`; <E> averaged over one plasma period
before fitting (it beats at 2 omega_pe). Office GPU 1 (A4000), double precision, 3.2 h.

| case | d ln<E>/dt, t omega_pe 35-170 | gain |
|---|---|---|
| uniform background, E_0 | -2.2e-5 | 0.999 |
| uniform background, no field | -5.0e-6 | 1.00 |
| frozen ions (1e9 m_p macro-particles), E_0 | 1.56e-2 | 8.5 |
| frozen ions, no field | -1.1e-4 | 0.99 |
| mobile He+, E_0 (the paper's case) | 2.09e-2 | 19 |
| eq. 10, 2 gamma at k* = 0.049 | 2.92e-2 | |
| the paper's fit | 7.9e-3 | |

Change of frame: the driven uniform run moved by a t^2/2 against the undriven one (same
particles) differs by 5.0e-3 over k lambda_De <= 0.3 in the fit window, 0.31 over all
wavelengths (the grid stays in the lab frame while the electrons cross it at up to 40 v_Te).
External work E_0 * integral of J against kinetic plus field energy gained: 2.5e-5 relative.
The spectrum's peak sits at 1.14 (helium) and 1.12 (frozen) times eq. 12.

**Interpretation.** The continuum limit of the setup (uniform background) shows no growth,
as the change of frame requires; growth needs the field and discrete ions, and frozen
macro-particles already give three quarters of the helium rate. k* = omega_pe/(a t) is the
wave standing still in the ion frame: the Cherenkov wake of the ions' charge noise swept
through resonance by the accelerating electrons. The growth reproduced here is of the
paper's size, but it is not evidence for the Fried continuum instability.

**Left open:** the ion macro-particle-count scaling of the helium growth (the decisive
test of the wake picture); sixteen realisations instead of four; collisions and walls, which
also break the change of frame. Docs: `docs/examples/electron_field.md` from
`docs/scripts/fig_electron_field.py` (in make_all.py; runs the example); README 1D1V row;
CI Examples job runs `electron_field --quick`.

### W13: external fields on an (x, y, z) grid (2026-09-27)

Ported from `ds/3D_external_fields` (six commits on old `main`), not merged; the contributor is
credited with a `Co-authored-by` trailer. Numbers from office CPU, double precision.

**API.** `external_E` and `external_B` take either the existing `(cells, 3)` array or
`(cells, ny, nz, 3)`. No new class and no extents argument: `y` and `z` are already periodic over
`Domain.length_y`/`length_z` (`wrap_positions`), so the grid spans those, `ny`, `nz` centres each,
and its `x` cells are the domain's. Both E and B on a grid sit on the centres (the flat E stays on
the faces). `_core.gather_xyz` is the S2 gather as a tensor product along x, y, z, with the `x`
ghosts of `with_ghosts`; `ny = nz = 1` reproduces `gather` to round-off. Flat and grid fields can
be mixed. The shape is checked at construction. `Simulation.external_fields_at(x)` returns the
external E, B anywhere. The `[external]` TOML table stays uniform.

**Magnetic moment.** Not in the step (the branch added a per-step `mus` output, a cost on every
run): `magnetic_moment(out, simulation)` post-processes stored particles, `p_perp^2/(2 m B)` against
the external B at each particle, zero where B = 0.

**Cost.** The flat path lowers to the same StableHLO as before (md5 of the explicit step equal to
5b11b02's with no external field and with a `(cells, 3)` B), so bit-identical and the same cost.
Wall clock A/B/A/B at N = 1e5 on 4 pinned cores, office at load ~10 from other lanes: no field
154 → 151 ns, flat B 159 → 162 ns per particle-step (medians of 3; noise). A `(cells, 4, 4, 3)` B:
367 ns (27-point gather). The plan's 41 ns was an idle machine; re-time with the runtime figures.

**Tests** (`tests/test_external_xyz.py`, 10, ~30 s): grad-B drift at L = 20 rho within 0.3 % of
`v_perp rho/2L`, the remainder falling 4.0x when L doubles (FLR, `(rho/L)^2`); z-varying field →
uniform linearly in the amplitude (ratio 10.0 for 1e-2 vs 1e-3), `(cells,1,1,3)` equals flat to
1e-12 rho; mirror at 45 deg turns at y = L to 1e-3, mu spread < 1e-4 while B doubles; E on a grid
in both schemes; relativistic and zero-field mu; shape errors.

**Example** `2_intermediate/external_fields_3d.py` (40 s, `--quick` 20 s, in CI's quick list),
page `docs/examples/external_fields_3d.md`, README row. Full preset: drift +1.15/+0.28/+0.07 % at
L/rho = 10/20/40; mirror turning points 1.7323/1.0001/0.5774 against cot(theta)
1.7321/1.0000/0.5774; mu spread 7e-6 to 2e-5.

**Left.** Self-consistent fields remain 1-D in x (by design). A time-dependent external field is
still out of scope (user guide). The branch's other edits (energy of the external field in the
diagnostics, grid bookkeeping) have no counterpart to port.
