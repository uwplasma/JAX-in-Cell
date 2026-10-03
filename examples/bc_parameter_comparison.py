"""Compare periodic, elastic, collecting and fractional near-ballistic walls."""
import matplotlib.pyplot as plt
from mixed_bc import run_case

plt.figure(figsize=(8, 4.5))
for fraction, restitution, code in ((1., 1., 0), (1., 1., 1), (0., 1., 2),
                                    (.25, 1., 3), (.5, 1., 3), (.75, 1., 3),
                                    (.5, .5, 3), (.5, 1., 4)):
    time, energy = run_case(fraction, restitution, code)
    label = {0: "periodic", 1: "elastic", 2: "absorbing", 4: "BC4, speed scale=1.2e8 m/s"}.get(
        code, f"BC3, R={fraction}, e={restitution}")
    plt.plot(time, energy/energy[0], label=label)
plt.xlabel("time (s)")
plt.ylabel("electron kinetic energy / first snapshot")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("bc_parameter_comparison.png", dpi=140)
