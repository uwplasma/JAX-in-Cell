"""Main command line interface to JAX-in-Cell."""
import sys
from ._plot import plot
from ._simulation import simulation, load_parameters

def main(cl_args=sys.argv[1:]):
    """Run the main JAX-in-Cell code from the command line.

    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting simulation.

    """
    if len(cl_args) == 0:
        print("Using standard input parameters instead of an input TOML file.")
        output = simulation()
    else:
        input_parameters, solver_parameters = load_parameters('input.toml')
        output = simulation(input_parameters, **solver_parameters)
    plot(output)

if __name__ == "__main__":
    main(sys.argv[1:])