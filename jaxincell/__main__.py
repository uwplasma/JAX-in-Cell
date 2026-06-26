"""Main command line interface to JAX-in-Cell."""
import sys
from ._plot import plot
from ._simulation import Simulation, load_parameters
from ._diagnostics import diagnostics

def main(cl_args=sys.argv[1:]):
    """Run the main JAX-in-Cell code from the command line.

    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting simulation.

    """
    if len(cl_args) == 0:
        print("Using standard input parameters instead of an input TOML file.")
        sim = Simulation()
    else:
        parameters = load_parameters(cl_args[0])
        sim = Simulation(parameters)
    output = sim.run()
    diagnostics(output)
    plot(output)

if __name__ == "__main__":
    main(sys.argv[1:])