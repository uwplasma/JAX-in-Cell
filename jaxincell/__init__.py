"""JAX-in-Cell: a differentiable 1D3V particle-in-cell code."""
from ._config import Collisions, Domain, Solver, Species
from ._constants import (boltzmann_constant, elementary_charge, epsilon_0, mass_electron,
                         mass_proton, mu_0, speed_of_light)
from ._diagnostics import (diagnostics, dominant_frequency, energies, gauss_residual, potential,
                           temperatures)
from ._plot import plot
from ._simulation import Output, Simulation, load_toml, quiet_start

try:
    from .version import __version__
except ImportError:  # a source tree that was never installed: version.py is written at build time
    __version__ = "unknown"

__all__ = ["Simulation", "Output", "Domain", "Species", "Solver", "Collisions", "load_toml", "quiet_start",
           "diagnostics", "energies", "gauss_residual", "potential", "temperatures",
           "dominant_frequency", "plot",
           "epsilon_0", "mu_0", "speed_of_light", "elementary_charge", "mass_electron",
           "mass_proton", "boltzmann_constant"]
