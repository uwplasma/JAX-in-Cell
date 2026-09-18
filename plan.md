# JAX-in-Cell: correctness, teaching interface, and kinetic benchmarks

**Working plan and live checklist.** It supersedes the sheath-only handoff that PR #42 carried
in its description. It specifies work to do; it is not a report that the work is done. Tick a
box only when a commit, a test and a reproducible number exist for it.

## 0. Where the work happens

| | |
|---|---|
| **Development branch** | **`research-release`**, PR #42 -> `main`, head `433d401` at the time of writing. All changes in this plan are made here. |
| Reference only | `rj/additions-to-pr`, PR #43 -> `research-release`, head `ffa5d6f`. Read it for the example changes and the progress-meter attempt, and port what is worth keeping onto `research-release` deliberately. **Do not commit to it, merge it, or rebase it.** |

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
* **Magnetised-sheath quick settings: not ported.** Three angles, 128 cells, 1.6 transits and three
  times the pool would take the CI example job from 23 s to about four and a half minutes, to improve
  the statistics of a quantity S02 shows is not an impact spectrum at all. S16 already rules that
  quick is a smoke preset; the third angle and the longer run are in the full preset already.
  Revisit at W7, once the measurement is real.
* **The progress meter** is not a port. Section 5 and W4.

### 0.1 What completion means

Correct code and inputs, the requested teaching workflows, independent numerical evidence,
figures regenerated with provenance, clean package and documentation builds, and a review
report. A ticked box, a green coverage badge, a successful trace, a pretty plot, or two copies
of the same formula agreeing is not evidence.

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
| S04 | confirmed by inspection; **half done (W2)** | Magnetic initial particles have transverse spread and a normal ion drift while the source is at rest and isotropic (via S03), and the source does not inherit the species drift. | Every distribution is now explicit: each reservoir states its three spreads, and with S05 a source can be given the drift its species has. What is still inconsistent is the magnetised example itself -- initial ions are a beam at `c_s` along `x` with no transverse spread, while their reservoir is an isotropic Maxwellian at rest -- and choosing between a field-aligned sonic entrance and an isotropic one is a physics decision for W7, not a mechanical fix. Do not claim field-aligned sonic entrance from the present setup. |
| S05 | **fixed (W2)** | Only an at-rest Maxwellian and a cold beam are supported; the drifting crossing distribution is refused. | `Source.model` is `"beam"`, `"maxwellian"` or `"drifting"`, settled at construction because it selects a branch, and the third inverts `F(v,u)=p` by bisection with a `custom_jvp` that differentiates the equation rather than the comparisons. Checked against quadrature of `p` itself at `u/sigma` of -1, 0.5 and 2, against the Rayleigh closed form at `u=0` to 1e-9, and its derivative against a central difference of the bisection to 1e-6. The sign of `u` is carried: a reservoir drifting away sends the reduced flux, and an outward cold beam is refused. The fast paths are kept -- the drifting branch costs 115 us against 60 us for 120 particles. |
| S06 | **fixed for the electrode closure (W2)** | Cloud charge is truncated outside a wall before the centre crosses, while surface charge is credited at centre crossing. A unit charge walked through an absorbing wall in 0.1 dx steps has volume-plus-surface charge between 0.500 and 1.405: half of it has vanished when the centre is at the wall, and 40 % too much exists one tenth of a cell later. | The part of a live cloud that reaches past the collector is a third category, reversible and on the surface, and `field_bc=('open','absorbing')` closes on the collected charge **plus** it. Deposited plus exterior is then one at every sub-cell offset, and a sheet crossing the wall leaves the field at an interior face unchanged to 1e-12 where it used to jump by 5.20e10 V/m against a half-particle's 5.65e10. In a ten-Debye-length sheath the truncated part was 0.4 % of the electron charge, 0.6 % of the ion charge and 0.29 % of what the collector held; the quick preset's wall potential moves from -0.8107 to -0.8197 against a standard error of 0.059, so the shift is real but not resolved at that size. **The other absorbing closures still truncate**: `(2,2)` and `(2,1)` fix their constant from a potential difference rather than a surface charge and would need the closure rederived. |
| S07 | **fixed (W2)** | Injected particles free-stream for a residual step with no field interaction; source correctness is inferred from the Gauss solve alone. | The staggering is written down in `inject`: a particle enters at `t^n + (s_k - 1/2) dt` and is put in the arrays at the position it reaches by `t^{n+1/2}` and with the velocity it would have had at `t^n`, both from the field at its entry plane -- the position to second order in the flight, the velocity from the same pusher run over the signed interval `(1/2 - s_k) dt`, which is exact for a uniform field and is what keeps a magnetised entry's gyro-phase. Tested against two closed forms rather than against the Gauss solve: a cold beam falling through a prescribed uniform E, where the worst error falls from 1.1e-3 to 2.0e-5, and one turning in a prescribed uniform B at `Omega dt = 0.079`, where the mean falls from 4.0e-2 to 9.4e-4. |
| S08 | **fixed (W2)** | With an open plane the continuity current is anchored at zero, now documented as "the internal transport measured from the source plane". Honest, but not an absolute current. | The closure is the collector's own conduction current, `sigma_w_dot`, taken as a difference over the same interval the density change spans. Ampere's law makes the total current uniform in one dimension, so with a floating electrode `J + eps_0 dE/dt` must vanish **at every face**, and it does, to 9e-15 of the current's own size; anchored at zero the residual was 0.62 of it. The three are now distinguishable in the output: `Output.J` is the conduction current, `eps_0 dE/dt` the displacement current, and their sum the circuit current, which is zero while the collector floats. The implicit scheme carries no surface charge and refuses the open plane rather than anchoring on nothing. |
| S09 | **fixed (W2)** | `wall.overflow` exists and `sheath_unmagnetized.py` warns on it. `sheath_magnetized.py` and `sheath_optimization.py` ignore it, and the library never invalidates a run. | `Output.problems` is the reasons a run is unusable and `Output.validate()` raises on them, so a script writes `run(...).validate()` and gets nothing instead of numbers it should not use. `Output.overflow` is the running maximum, so a run cannot look valid because it recovered. The two sheath examples validate; the optimisation checks the most reflective end of its admissible interval once, on the host, which is the longest electron lifetime any trial can ask for. The per-injection check was already there -- `overflow = max(w[slots])` -- and is what these read. |
| S10 | **fixed (W2)** | The reflection weight cutoff truncates the last part of an orbit and adds a branch. | `Wall.truncated` is the weight a wall kept only because `min_weight` stopped the orbit, which its reflection law would otherwise have returned: the cost of the cutoff, in the ledger's own units. A test holds it to fall by more than ten with the cutoff from 1e-1 to 1e-3, so a result can be refined until the budget is below what it is being compared against. The branch's effect on sensitivities stays with G09 and W3's derivative matrix. |
| S11 | **fixed (W2)** | `_record` ran before `_thermalise`, so `energy_out` was the specular energy of the bounce, not the energy of the redrawn particle. At a thermal wall with restitution 1 the ledger reported `energy_in - energy_out` identically zero, which is not a measurement: a run whose walls returned 1.1159e-08 J/m^2 against 1.0716e-08 received was reported as exchanging nothing. | The record is taken after the wall's law has finished acting, including the redraw, in the explicit step, the implicit sub-step and the initial half step -- which did not thermalise at all, though its own comment said it had to meet a wall the way every later step would. `Wall` carries `energy_injected`, `momentum` and `momentum_injected`. The thermal wall's returned energy is checked against the closed form `m(2 sigma_x^2 + sigma_y^2 + sigma_z^2)/2` per unit weight, and the injected energy and normal momentum against theirs. |
| S12 | **fixed (W1, W2)** | Wall energies use `m v^2/2` on relativistic paths, and nonrelativistic collisions can be combined with a relativistic pusher. | The ledger takes energy from the carried momentum as `K = m|u|^2/(gamma+1)`, which is `m v^2/2` when gamma is one and does not subtract two large numbers when it is not; at 0.9 c it is 3.19 times the Newtonian value, which is what the test asserts. Takizuka-Abe beside the relativistic pusher is refused (W1). |
| S13 | **fixed (W1)** | `active=0` reaches `jnp.arange(n) % s.active` and `L / s.active`. Called directly it is a `ZeroDivisionError`; inside `jit` the infinity lands in the untaken branch of the weight's `where`, the forward run survives, and `jax.grad` returns **NaN** -- on the one configuration a source-driven optimisation would start from. Both divisions now use `max(active, 1)`; `test_an_empty_start_is_safe_under_jit_grad_and_vmap` fails on the old code. `Domain` rejects a non-positive length or step, `Species` a non-positive mass or negative density, `Source` a negative density or a `min_weight` outside [0, 1], and `check_sources` rejects `emit > n`. | Done, except that the capacity check at each injection is S09's and stays with W2. |
| S14 | **confirmed** | `gauss_residual` skips cell 0 because that cell's equation defines the missing boundary field. Documented, but therefore not an independent full check. | Report an all-cell residual with stored boundary data, plus independent surface and global ledgers. |
| S15 | confirmed by inspection | `sheath_reflection.py` is still an unmaintained thermal-wall run on the default electromagnetic model with four filter passes and a bare `argmax` edge fallback. | Explicit model selection, maintained-source comparison, measured `R_eff`, the validated edge helper; keep a labelled transient variant. |
| S16 | **confirmed** | "Normal incidence is the field-free case" is printed from the normal-field run itself; no `B=0` control is executed. | Run matched `B=0` and normal-field controls at the same seed and resolution. Quick is a smoke preset, not evidence. |
| S17 | **confirmed** | `floating_potential` says its left side is `1/2` at `phi=0`; it is `1`. The guard therefore rejects `v0` in `[0.3989, 0.7979)`, which have valid roots: `v0=0.5 -> -0.134117`, `v0=0.7 -> -0.012507`. This already forced a benchmark parameter change during the previous round. | Correct the endpoint and the bound to `sqrt(2/pi)`. Distinguish algebraic root existence from sheath admissibility. |
| S18 | **confirmed** | `sheath_unmagnetized.py` hands `np.minimum(phi_profile, 0)` to the reference: in the quick preset **34 of 48** cell centres are positive, up to +0.0505 T_e/e, and every one of them is silently moved to zero. The same call passes the analytic `phi_wall = -0.79926` with a measured profile whose own wall value is -0.73439, and `phi_profile[0]` is a face value used as a centre. | State each reference's domain, use correct coordinates, quantify rather than clip. Distinguish `mean[n(phi)]` from `n(mean[phi])`. |

### 2.2 Derivatives, optimisation, tests that cannot fail

| ID | Status | Finding | Required correction |
|---|---|---|---|
| G01 | **fixed (W3)** | `pytest.approx(x, rel=...)` keeps its default `abs=1e-12`, and `np.allclose` its default `atol=1e-8`. Three assertions in `tests/test_gradients.py` compare SI quantities of order `1e-19` to `1e-25` and therefore pass with **zero and with the wrong sign**. | The suite was audited by wrapping both functions and logging every comparison whose expected value the default absolute tolerance swallows: **24 sites, of which 19 were vacuous** -- times of 1e-10 s, masses of 7e-27 kg, a time step of 2e-13 s compared at `rel=1e-15`, wall energies of 1e-19 J. Each now says what absolute tolerance it means, and the wrapper stays in `conftest.py` as a guard, so a silent default on an SI-scale quantity is refused rather than found again later. `test_gradients.discriminating` asserts in so many words that zero and the wrong sign are rejected. |
| G02 | **fixed (W3)**, and it found a defect in the ledger | In `test_the_impact_energy_...` the particle never reaches the wall: `wall.arrived = 0.0`, `energy_in = 0.0`. The run is 0.80 ns against a 2.26 ns transit. It passes only because of G01. The "analytic impact control" reported in PR #42 tests nothing. | The run is long enough (2800 steps against a 2546-step transit), the impact is asserted before any energy is looked at, and every quantity is in the natural units of the problem. With the control actually running it showed that **the ledger recorded a wall impact at the end of the step the particle overshot to**: the value was off by 1e-4, but the derivative by 32 %, and the 32 % did not fall with the time step, because it was taken at a fixed step index rather than at the crossing -- the missing `dtau/dtheta` term. `Simulation._at_impact` runs the same pusher backwards over the part of the step that follows the impact; the value error falls to 5e-9 and the derivative error to 2e-5, both converging with the step. See S19. |
| G03 | **fixed (W3)** | The charge-sheet invariant compares a `1.35e-23` V/m field using `np.allclose` with default `atol=1e-8`. Zero and the wrong sign both pass. Sparse storage can also step over the cloud overlap. | The field is in units of the sheet's own, every step is stored, and the start is swept across a cell in five sub-cell offsets so the crossing happens at every phase. The bound is 1e-11 of the sheet's field, and the test says in an assertion that zero and the wrong sign do not meet it. That the cloud term is needed at all is S06's test, which fails by 5.20e10 V/m without it. |
| G04 | open | Forward/reverse/small-h agreement verifies one realised map, not an ensemble, continuum or stationary response. | Keep the derivative support matrix of section 4.3 and demonstrate each advertised response separately. |
| G05 | **confirmed** | The optimisation builds one target on the training seeds and another on the held-out seeds, so both are exactly zero at the reference by construction. That is a plumbing self-test, not out-of-sample prediction. | Keep it, labelled a paired-realisation self-test. Add one fixed, independent, refined target used by both training and validation. |
| G06 | **confirmed** | `np.linspace(0.02, 0.50, 25)` has spacing 0.02 and does not contain `r_reference = 0.35`; its nearest point is 0.34. The reported "uncertainty 0.01" is the distance to a grid point. | Separate scan discretisation from statistical uncertainty; include the known control in the grid; refine the minimum independently. |
| G07 | **confirmed** | The loop can leave the final accepted point out of `history`, and "the step is below the uncertainty of the control" is a hard-coded threshold reported as convergence. | Record and evaluate every accepted point, return the best accepted point, separate stalled from converged, use projected-gradient/step/objective criteria with named tolerances. |
| G08 | **confirmed** | The response window is `25 * 0.15 = 3.75` inverse electron plasma frequencies, about 0.60 oscillations. The documentation calls it "about four electron plasma periods". | Report inverse-frequency time and cycles separately. Keep the preparation fixed and do not call the result a stationary derivative. |
| G09 | open | Source clipping, slot selection, sorting, accept/reject collisions and the weight floor all add derivative discontinuities. | Test and document each. No blanket claim that ensemble-averaged branchwise AD repairs missing event terms. |

### 2.3 Interface, progress, plots, export

| ID | Status | Finding | Required correction |
|---|---|---|---|
| U01 | confirmed by inspection | `load_toml` passes a few run keys only; sources, external fields, output, movies, restart, scans and optimisation are not reachable from TOML. | One normalised configuration path; never accept an ignored key silently. |
| U02 | **confirmed** | PR #43 builds `tqdm` inside the jitted `_run` and captures it in `jax.debug.callback`. The bar is trace-time state: a cached program reuses a closed bar on the second identical call. | Own progress on the host, per invocation; kernels stay callback-free. See section 5. |
| U03 | **confirmed** | `from tqdm import tqdm` is unconditional and `tqdm` is absent from `pyproject.toml`, so a clean install cannot import the package. The bar also prints during library calls inside tests. | No new required dependency; lazy optional `tqdm`; cadence independent of the snapshot schedule; silent under tracing. |
| U04 | **confirmed** | `_plot.py` uses `out.weight[-1] > 0` to choose the particles drawn in **every** frame. Particles collected earlier vanish from all frames; refilled slots appear from the start. | Species membership is fixed; activity and weight are per frame. Test losses, injections, unequal weights and reused slots. |
| U05 | **confirmed** | `_hist` clips outliers into the edge bins and the phase-space array adds `+1.0` to weighted counts for log display. | Track overflow or widen documented ranges; mask positive weighted values for log display; state normalisation and units. |
| U06 | **confirmed; half fixed (W1)** | Face quantities (`E_x`, potential) are drawn on `out.grid`, which is cell centres. | `Domain.faces`, `Output.faces` and `Output.walls` are published and documented, and the five places that rebuilt `grid + dx/2` by hand now ask for them. The plot still draws face quantities on the centres: that is a behaviour change and belongs with W6. |
| U07 | confirmed by inspection | Energy, momentum, charge and balance histories are not in the general plot. | Restore configurable diagnostic panels separating closed invariants, open budgets and bare changes. |
| U08 | **confirmed** | `plot` precomputes every frame before drawing any. On a 400-step, 256-cell, 25000-particle run it grew resident memory by **1.1 GB** while the histograms it keeps are 79 MB; the documented cap `_MAX_ELEMENTS = 2e8` allows 0.8 GB per copy. The time axis is an `imshow` extent built from `(t[-1] - t[0])/(S - 1)`, so an irregular store schedule is drawn as if it were even, and `save` and `show` are exclusive: passing `save` gives a still figure on screen, never an animation. | Bounded streaming frames, irregular schedules, independent save and show, keep the blitting. |
| U09 | confirmed by inspection | `omega` always labels the axis `omega_pe`; `quiet` is ambiguous. | Explicit reference-frequency labels and `sampling='low_noise'`, with migration aliases that do not change physics. The label is W6. The rename is **deferred to W5**, deliberately: `quiet` and `random_positions` appear at 102 call sites, and renaming them before the TOML vocabulary is settled would rewrite all of them twice. `quiet`, `random_positions` and the third state they cannot express (`quiet and random_positions` silently means quiet) get one name each, once. |
| U10 | **confirmed** | The standard is `x_i = (gridGlobalOffset + (i + position)*gridSpacing)*gridUnitSI` with `position` in `[0,1)`, `0.0` at the lower corner of the element; openPMD-viewer, WarpX and PIConGPU all agree. `openpmd.py` sets `grid_global_offset=-L/2` with `position=0.0` for centres and `0.5` for faces -- **exactly backwards**. Centres belong at `0.5`. The stored faces are the *right* faces, `-L/2+(i+1)dx`, which is `position=1.0` and outside the allowed range. | Centres get `position=0.5`. For the right faces either shift that record's `grid_global_offset` to `-L/2+dx` with `position=0.0`, or re-index. Test with an independent reader. Note WarpX's default openPMD output writes `0.5` on every component because it cell-centres before writing, so it is **not** a usable reference for Yee staggering; PIConGPU is. |
| U11 | confirmed by inspection | openPMD output is not a restart state and carries no source or wall context or readback example. | Native versioned restart and analysis archives plus an honest openPMD round trip. |
| U12 | confirmed by inspection | Examples do not systematically save configuration, data, figures and provenance; documentation quotes numbers from different presets. | Provenance and controlled saves in every teaching template; regenerate documentation from the exact named preset. |
| U13 | confirmed by inspection | PR #43's Weibel sets float32 while the comment says float64, stores large histories, and widens the unstable spectrum with no fitted linear benchmark. | Keep the engaging run; fix the precision contract and memory policy; add verified linear-mode measurements in the same script. |
| U14 | open | The requested numerical-comparison and output/restart examples do not exist. | Implement the inventory of section 6 without four copies of the PIC setup. |

### 2.4 Found while the work was under way

| ID | Status | Finding | Required correction |
|---|---|---|---|
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

- **`gammaflag` is documented after all.** `1` defines `gamma` at the Debye-sheath entrance, `0`
  at the magnetic-presheath entrance, and the README says `0` is "ALWAYS the appropriate choice
  for matching to a code outside of the magnetic presheath". Use `0`, and record both the flag
  and the resulting definition in the manifest.
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
- [ ] **W3** diagnostics and statistics: weighted moments, independent balances, correct windows, uncertainty.
- [ ] **W4** orchestration and progress: pure kernels, host-owned meter, one snapshot schedule, exact restart.
- [ ] **W5** TOML/CLI and persistence: one resolver, native archives, openPMD round trip.
- [ ] **W6** plots and movies: evolving weighted populations, diagnostic histories, bounded memory, headless tests.
- [ ] **W7** repair the four existing sheath and optimisation examples and their documentation.
- [ ] **W8** grazing-incidence benchmark, then a controlled finite-ordering extension.
- [ ] **W9** model-comparison and Weibel examples on the existing kernels and shared theory.
- [ ] **W10** electron-field instability with its limiting controls.
- [ ] **W11** algorithm audit; source-free implicit electrostatic; collision time-centering.
- [ ] **W12** convergence, performance, documentation, review packet.

## 11. Acceptance checklist

- [ ] Work is on `research-release`; `rj/additions-to-pr` untouched; no main writes, merges, force pushes or releases.
- [ ] Every S/G/U row reproduced or marked resolved with evidence, then fixed with a test that fails on the old code.
- [ ] Component-wise and drifting or field-aligned source sampling; supported-source contracts complete.
- [ ] Source, cloud, collector, current and energy/momentum transfers derived and independently verified.
- [ ] Event-based impact spectra and event-aware derivative checks; hard-count limitation retained.
- [ ] Overflow invalidates a run; cutoff error measured; `active=0` safe.
- [ ] Windows, coordinates, fluence-versus-current labels and statistical uncertainties corrected everywhere.
- [ ] Pure JAX runner and host-owned progress both work; repeated-run, AD and restart tests pass; verbose equals silent.
- [ ] One parameter vocabulary with a migration path; README parameter tutorial exists.
- [ ] TOML/CLI covers every shipped workflow with strict validation.
- [ ] Native archive and exact checkpoint work; openPMD read back independently.
- [ ] Energy, charge and momentum panels restored with truthful open-system residuals; movies weight evolving populations.
- [ ] Existing examples retained and corrected, with saved data and provenance.
- [ ] Matched GYRAZE case with real reference data, uncertainty, and a documented finite-ordering study.
- [ ] Weibel linear growth verified mode by mode; PR #43's nonlinear preset ported and preserved.
- [ ] Explicit/implicit, collisional/collisionless, filtered/unfiltered and relativistic comparisons demonstrated.
- [ ] Independent-target sheath inference with real statistical uncertainty.
- [ ] Electron-field example with verified inputs, limiting controls and an honest interpretation.
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
