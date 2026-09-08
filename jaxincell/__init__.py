"""JAX-in-Cell: a differentiable 1D3V particle-in-cell code."""
from ._config import Collisions, Domain, Solver, Species
from ._constants import (boltzmann_constant, elementary_charge, epsilon_0, mass_electron,
                         mass_proton, mu_0, speed_of_light)
from ._diagnostics import diagnostics, dominant_frequency, energies, gauss_residual, temperatures
from ._simulation import Output, Simulation, load_toml, quiet_start

__all__ = ["Simulation", "Output", "Domain", "Species", "Solver", "Collisions", "load_toml", "quiet_start",
           "diagnostics", "energies", "gauss_residual", "temperatures", "dominant_frequency",
           "epsilon_0", "mu_0", "speed_of_light", "elementary_charge", "mass_electron",
           "mass_proton", "boltzmann_constant"]

try:
    from ._plot import plot  # noqa: F401
    __all__.append("plot")
except ImportError:  # matplotlib not installed
    pass
