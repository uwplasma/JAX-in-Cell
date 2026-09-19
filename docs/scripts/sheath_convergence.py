"""How the maintained sheath's wall potential moves when one thing at a time changes.

This is the convergence table of ``docs/examples/sheath_unmagnetized.md``. It is not in
``make_all.py``'s default list because it is twelve full runs and takes about half an hour,
where every other script there takes minutes; run it on its own when the numerics change::

    python docs/scripts/make_all.py sheath_convergence.py

The pool is scaled wherever a change makes particles live longer, because a source that
overwrites live particles is not a refinement of anything, and every run validates.
"""
import numpy as np
from common import record

from jaxincell import (Domain, Simulation, Solver, Source, Species, epsilon_0, mass_electron, potential,
                       elementary_charge as e_charge)
from jaxincell.sheath import floating_potential, source_density

T_E, DENSITY, MASS_RATIO, BEAM = 1.0, 1e16, 1836.0, 0.2
SIGMA = np.sqrt(T_E * e_charge / mass_electron)
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
DEBYE = SIGMA / OMEGA_PE
PHI_WALL = float(floating_potential(BEAM))
AMPLITUDE = float(source_density(PHI_WALL))


def wall_potential(cells=120, per_period=10.0, capacity=120000, emit=120, transits=6.0, boxes=10.0, seed=0):
    """The mean wall potential over the second half of the run, in T_e/e, with its standard error."""
    length, dt, stored = boxes * DEBYE, 1.0 / (per_period * OMEGA_PE), 60
    steps = stored * max(int(transits * length / (BEAM * SIGMA) / dt) // stored, 1)
    domain = Domain(length=length, cells=cells, time_step=dt,
                    particle_bc="absorbing", field_bc=("open", "absorbing"))
    electrons = Species("electrons", capacity, -1.0, mass_electron, DENSITY, (np.sqrt(2) * SIGMA, 0, 0),
                        active=capacity // 4, sampling="quiet",
                        source=Source(density=AMPLITUDE * DENSITY, vth=(np.sqrt(2) * SIGMA,) * 3, emit=emit))
    ions = Species("ions", capacity, 1.0, MASS_RATIO * mass_electron, DENSITY, 0.0, (BEAM * SIGMA, 0, 0),
                   active=capacity // 4, sampling="quiet",
                   source=Source(density=DENSITY, vth=0.0, drift=(BEAM * SIGMA, 0, 0), emit=emit))
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(
        steps, seed=seed, store_every=steps // stored, store_particles=False).validate()
    late = np.asarray(potential(out))[stored // 2:, -1] / T_E
    return float(late.mean()), float(late.std() / np.sqrt(len(late)))


VARIATIONS = [("baseline", {}),
              ("dx/lambda_D 0.167 (60 cells)", dict(cells=60)),
              ("dx/lambda_D 0.042 (240 cells)", dict(cells=240)),
              ("omega_pe dt 0.05, emit halved", dict(per_period=20.0, emit=60)),
              ("omega_pe dt 0.025, emit quartered", dict(per_period=40.0, emit=30)),
              ("a quarter of the particles", dict(capacity=30000, emit=30)),
              ("four times the particles", dict(capacity=480000, emit=480)),
              ("3 transits", dict(transits=3.0)),
              ("12 transits", dict(transits=12.0)),
              ("box 20 lambda_D, pool doubled", dict(boxes=20.0, transits=3.0, capacity=240000)),
              ("seed 1", dict(seed=1)),
              ("seed 2", dict(seed=2))]

if __name__ == "__main__":
    print(f"reference phi_wall = {PHI_WALL:.5f} T_e/e\n")
    print("%-38s %9s %7s %7s" % ("variation", "phi_wall", "s.e.", "% off"))
    results = {}
    for name, changes in VARIATIONS:
        mean, error = wall_potential(**changes)
        results[name] = mean
        print("%-38s %9.4f %7.4f %7.2f" % (name, mean, error, abs(mean / PHI_WALL - 1) * 100), flush=True)
    seeds = [results[name] for name in ("baseline", "seed 1", "seed 2")]
    # first order in the cell: two refinements, each halving the distance to the reference, so the
    # value at dx -> 0 is the Richardson extrapolation of the finest pair
    extrapolated = 2 * results["dx/lambda_D 0.042 (240 cells)"] - results["baseline"]
    record(source_sheath_convergence_baseline=round(results["baseline"], 4),
           source_sheath_convergence_finest=round(results["dx/lambda_D 0.042 (240 cells)"], 4),
           source_sheath_convergence_extrapolated=round(float(extrapolated), 4),
           source_sheath_convergence_seed_scatter=round(float(np.std(seeds, ddof=1)), 4))
