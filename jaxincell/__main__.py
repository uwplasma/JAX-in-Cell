"""Command line: ``jaxincell input.toml`` runs the file, prints the energy
balance and shows the animation."""
import sys

from ._diagnostics import diagnostics
from ._simulation import load_toml


def main(argv=None):
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


if __name__ == "__main__":
    sys.exit(main())
