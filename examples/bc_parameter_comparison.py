"""Compare return fractions and restitution in a near-ballistic control."""
import matplotlib.pyplot as plt
from mixed_bc import run_case

for fraction, restitution, code in ((.25, 1., 3), (.5, 1., 3), (.75, 1., 3), (.5, .5, 3), (.5, 1., 4)):
    time, energy = run_case(fraction, restitution, code)
    plt.plot(time, energy/energy[0], label=f"BC{code}, R={fraction}, e={restitution}")
plt.xlabel("time (s)")
plt.ylabel("electron kinetic energy / first snapshot")
plt.legend()
plt.tight_layout()
plt.savefig("bc_parameter_comparison.png", dpi=140)
