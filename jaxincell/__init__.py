"""JAX-in-Cell: a differentiable 1D3V particle-in-cell code."""
import sys

from ._config import (Collisions, Domain, Impacts, Solver, Source, Species, boltzmann_constant,
                      elementary_charge, epsilon_0, mass_electron, mass_proton, mu_0, speed_of_light)
from ._diagnostics import (bohm_edge, charge_balance, diagnostics, dominant_frequency, energies,
                           gauss_residual, potential, temperatures)
from ._simulation import Output, Simulation, load_toml, quiet_start

try:
    from .version import __version__
except ImportError:  # a source tree that was never installed: version.py is written at build time
    __version__ = "unknown"

__all__ = ["Simulation", "Output", "Domain", "Species", "Solver", "Source", "Collisions", "Impacts",
           "load_toml",
           "quiet_start",
           "diagnostics", "energies", "gauss_residual", "charge_balance", "potential", "temperatures",
           "bohm_edge",
           "dominant_frequency", "plot",
           "epsilon_0", "mu_0", "speed_of_light", "elementary_charge", "mass_electron",
           "mass_proton", "boltzmann_constant"]


def __getattr__(name):
    """``plot`` imports matplotlib on first use, which would otherwise be a third of the import time."""
    if name == "plot":
        from ._plot import plot
        return plot
    raise AttributeError(f"module 'jaxincell' has no attribute {name!r}")


def main(argv=None):
    """The ``jaxincell input.toml`` command: run the file, print the energy balance, show the animation."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: jaxincell input.toml")
        return 1
    sim, run = load_toml(argv[0])
    out = sim.run(int(run.get("steps", 500)), seed=int(run.get("seed", 0)),
                  store_every=int(run.get("store_every", 1)))
    d = diagnostics(out)
    total = d["total"]
    print(f"steps {out.t.shape[0]}  final time {float(out.t[-1]):.3e} s  "
          f"energy drift {float(abs(total[-1] / total[0] - 1)):.2e}  "
          f"gauss residual {float(d['gauss_residual'][-1]):.2e}")
    if run.get("plot", True):
        from ._plot import plot
        plot(out)
    return 0
