"""Optional postprocessing: pip install jaxincell[openpmd]."""
from pathlib import Path

from jax import block_until_ready
from jaxincell import Simulation, load_parameters
from jaxincell.openpmd import write_openpmd

here = Path(__file__).resolve().parent
parameters = load_parameters(str(here / "input.toml"))
parameters["domain_parameters"]["total_steps"] = 12
parameters["solver_parameters"]["print_info"] = False
for species in parameters["species_parameters"].values():
    for population in species.values():
        population["number_pseudoparticles"] = 128
output = block_until_ready(Simulation(parameters).run())
print(write_openpmd(output, openpmd_filename=str(here / "openpmd_output" / "run.json"),
                   openpmd_iteration_encoding="fileBased", openpmd_iteration_stride=4))
