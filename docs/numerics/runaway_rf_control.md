# Runaway tail control with RF: a bounded study

`examples/3_advanced/runaway_rf_minimum.py` asks how much supplied RF energy it takes to keep a
dilute runaway tail below 10 MeV at the end of a fixed hold, with the DC drive on throughout. It is a
reduced-model study. The collision and radiation strengths are those of
{doc}`collisions` and {doc}`radiation` and are never changed. The tail carries $10^{-5}$ of the density.

## The handoff scenario has no minimum-RF problem

At $B=2$ T, $n=10^{18}$ m$^{-3}$, $T_e=100$ eV, $\ln\Lambda=15$ and $E=5$ mV/m, the reduced test-particle
ODE of `runaway_baseline.py` has a synchrotron-limited fixed point at $\gamma=13.13$ (6.2 MeV). Every
suppressed start with $\gamma\le 20$ stays below 10 MeV for all time (peak $\gamma$ 13.13, or 20.00
for the start at 20), and the 20 MeV seed itself falls below 10 MeV after 6.8 s with no RF. The minimum RF
is zero, so the problem is degenerate there. This is a negative result, not a failure of the search.

The fixed point does not disappear at any $E$: it moves smoothly to higher energy. It crosses 10 MeV at
$E=6.26$ mV/m. The study therefore declares one alternative, $E=15$ mV/m ($E/E_c=19.6$,
$E/E_{\rm threshold}=2.4$), whose fixed point is at 58 MeV, 5.6 times the threshold in $\gamma$. This is a
numerical study choice, not a prediction for a device. Without RF the tail grows through 10 MeV;
a start at $\gamma=5$ takes 2.70 s, and that measured regrowth time is the RF-off hold. The
RF-on window is 0.3 s. The seed is $\gamma\in[5,25]$ and pitch $\in[0.05,0.5]$ rad uniform, 22 % above
10 MeV at the start and 100 % at the end without RF.

## Slow RF response

The wave is a counter-propagating cold-plasma whistler (R wave, electrons and ions in the
dispersion), resonant through $\omega+|k|v\cos\alpha=\Omega_e/\gamma$. Each line scatters pitch with
$\dot\alpha=A\sin\chi$, $A=(\Omega_e/\gamma)\,b\,(1+v_{\rm ph}\cos\alpha/v)$, which the example checks against
the resolved push to 0.7 %. The quasilinear operator, with Gaussian-smoothed resonances, diffuses
along the wave-frame curve $E+|v_{\rm ph}|p_\parallel=\text{const}$, so pitch scattering changes the energy;
the Ito drift comes from automatic differentiation of the flux form.

It is compared with a resolved relativistic Boris push (gyro-orbit resolved, every particle with its own
random-phase realisation) on 14 windows. The comparison metric is the growth of the variance between half and
full window, which removes the onset transient, as a ratio resolved over quasilinear. All cases except the
first were held out from the choice of smoothing width (the line spacing) and the overlap rule
$2\omega_b/\Delta_{\rm sp}=3$.

| window | pitch-variance growth ratio | energy-variance growth ratio |
|---|---|---|
| nine in-domain cases (amplitude $\times$0.5, $\times$2, other phases, detuning, spread, other $\gamma$ and pitch) | 0.92 to 1.67, mean 1.3 | 1.0 to 2.5 |
| mean energy shift, same cases | resolved / quasilinear 1.1 to 2.0, sign always the same | |
| band $0.6$ widths off resonance | no scattering in either model (absolute growth below 0.3 % of the on-resonance value) | |
| sparse lines ($2\omega_b/\Delta_{\rm sp}=1$) | 0.84 | energy shift changes sign |
| dense lines (6) | 0.47 (large error) | 0.79 |
| one coherent line, three lines | no diffusion; trapped, quasilinear invalid | |

**Validity domain.** A noise-like band of many lines whose spacing satisfies the overlap rule, with the
particle inside the band, amplitude per line low enough that $D_{\alpha\alpha}\,\Delta t\le 0.01$. Within it the
quasilinear model underestimates the resolved pitch scattering by about 30 % (range 0.92 to 1.67) and the
energy exchange by a similar factor. **Fall-back rule:** for a coherent single mode or a few lines, or
sparse or very dense lines, the quasilinear operator is not used; the resolved push is. A resolved wave
and its quasilinear operator are never applied together.

## Source energy

The control prescribes the total wave field through a vector potential, so Faraday's law holds during the
ramps. The explicit drive current is $\boldsymbol J_{\rm drive}=\nabla\times\boldsymbol B/\mu_0-\epsilon_0\partial_t
\boldsymbol E-\boldsymbol J_{\rm pl}$, where $\boldsymbol J_{\rm pl}$ is the response of cold electrons and ions
with the Braginskii friction. The source power is $P_{\rm src}=-\boldsymbol J_{\rm drive}\cdot\boldsymbol E$. An
ideal nonrecovering port supplies $Q_+=\int\max(P_{\rm src},0)\,dt$, integrated with the zero crossing of the
piecewise-linear power resolved. Returned energy $Q_-$ and net energy are reported separately.
For a circular wave the spatially averaged steady power is constant, so there is no cycle-level reactive
sloshing to dump; the ramps are the only reactive part. The steady power matches the closed-form collisional
absorption within 3 %. The ramp supplies the stored energy plus dissipation (closure $10^{-6}$) and returns
it on the way down; the supplied energy changes by less than $10^{-6}$ under refinement of the step.
The tail exchange $-n_{\rm tail}\,\langle dK/dt\rangle_{\rm QL}$ is added to the steady power.

## Optimization and result

A feasible control keeps the 95 % Clopper-Pearson upper bound on the fraction of the tail above 10 MeV at the end
of the hold at or below 0.02. Three strategies use 240 slow-model evaluations each: a scan of a single band
(centre, amplitude and width), a gradient-free random search over three bands, and an evolution strategy
over three bands. Each best-found control is rechecked on independent seeds and, when it passes, at
half the time step, twice the quadrature nodes and $\pm30$ % amplitude. Selection used the search seeds, so
candidates are taken in order of cost and the first that passes on independent seeds is kept.

| strategy | feasible in search | verified $Q_+$ (J/m$^3$) | tail fraction above 10 MeV | $Q_+$ over the checks |
|---|---|---|---|---|
| no RF | none | none | 1.00 | none |
| single band | 2 of 240 | 8.2 | 0.24 % | 6.9 to 9.6 |
| three bands, random search | 2 of 240 | 9.0 | 0.24 % | 7.3 to 10.7 |
| three bands, evolution strategy | 107 of 239 | 5.1 | 1.6 % | 4.2 to 6.1 |

The evolution-strategy control is the best found, at about 5 J/m$^3$ over 0.3 s, against roughly
1 kJ/m$^3$ of bulk Ohmic work by the DC field over the 3.0 s horizon (Spitzer scaling of the baseline
measurement). Most of $Q_+$ is the energy the resonant tail takes from the wave (17 W/m$^3$ for 0.3 s);
bulk collisional absorption is 0.2 W/m$^3$, and the ramps 0.3 mJ/m$^3$, with $Q_-$ about the same. If the
quasilinear model underestimates the tail exchange by the measured 1.0 to 1.7, $Q_+$ lies between 5 and 9 J/m$^3$.
One check at 30 % lower amplitude fails for that control (2.6 %, bound 2.9 %), so the margin is thin.
The controls are noise-like bands: about $10^4$ lines per band satisfy the overlap rule.

**The constraint is satisfied only at the declared end of the hold.** The controls park the tail at the
synchrotron fixed point of the scattered pitch, at 9 MeV mean. Continuing the frozen controls with RF off,
the fraction above 10 MeV reaches 34 to 73 % after 0.5 s and 100 % after 1 s. With 2 s of margin added to the hold
none of the 720 evaluations, three strategies included, comes within a few percent of feasible
(smallest search-seed fraction 99.8 %): a 0.3 s RF window with at most three bands and $b\le 10^{-5}$ cannot hold
the tail for 4.7 s. This is a bounded negative result for that design space, and it
says nothing about longer windows or sustained RF.

**Not modeled:** avalanche (adds about 3 % per second to the tail), bremsstrahlung, spatial transport,
antenna loading and wall-plug efficiency, mode conversion, finite box, and heating of the bulk by the wave.
The ideal port is a lower bound on a real supply.
