"""Relativistic two-stream growth and energy: python examples/relativistic_two_stream.py."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from jax import block_until_ready
from jaxincell import Simulation, diagnostics, epsilon_0, elementary_charge, mass_electron, speed_of_light

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cells", type=int, default=64)
parser.add_argument("--particles", type=int, default=4096, help="markers per species")
parser.add_argument("--output", help="save the figure instead of opening a window")
args = parser.parse_args()
if args.cells < 32 or args.particles < 128 or args.particles % 2:
    parser.error("use at least 32 cells and an even number of particles >= 128")
c, density, drift = speed_of_light, 1e18, .8
wpe = np.sqrt(density*elementary_charge**2/(epsilon_0*mass_electron))
gamma0 = 1/np.sqrt(1-drift**2)
wb = wpe/np.sqrt(2)/gamma0**1.5
k = np.sqrt(3)*wb/(2*drift*c)
length, growth = 2*np.pi/k, wb/(2*wpe)
dt = .9*length/(args.cells*c)
beam = {"number_pseudoparticles": args.particles, "weight": density*length/args.particles,
        "vth_over_c_x": .001, "drift_speed_x": drift*c, "velocity_plus_minus_x": True,
        "perturbation_amplitude_x": 1e-6*length, "perturbation_wavenumber_x": 1}
ions = {"number_pseudoparticles": args.particles, "weight": density*length/args.particles,
        "vth_over_c_x": 1e-6}
parameters = {"domain_parameters": {"length": length, "number_grid_points": args.cells,
                                    "timestep_over_spatialstep_times_c": .9,
                                    "total_steps": int(np.ceil(90/(wpe*dt)))},
              "species_parameters": {"electrons": {"beams": beam}, "ions": {"background": ions}},
              "solver_parameters": {"relativistic": True, "filter_passes": 0, "print_info": False}}
output = block_until_ready(Simulation(parameters).run())
diagnostics(output)
t = np.asarray(output["time_array"])*wpe
mode = abs(np.fft.rfft(np.asarray(output["electric_field"])[..., 0], axis=1)[:, 1])**2
window = (t >=20) & (t <=35)
fit = np.polyfit(t[window], np.log(mode[window]), 1)
measured = fit[0]/2
energy = np.asarray(output["total_energy"])
error = energy/energy[0]-1
speed = np.linalg.norm(np.asarray(output["velocities"]), axis=-1)/c
assert np.isfinite(energy).all() and speed.max() <1
assert abs(measured/growth-1) <.05, "growth disagrees with cold theory; refine resolution"
print(f"growth/omega_pe: measured={measured:.5f}, theory={growth:.5f}")
print(f"max speed/c={speed.max():.5f}; max relative energy change={abs(error).max():.3e}")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
axes[0].semilogy(t, mode/mode[0], label="mode1")
axes[0].semilogy(t[window], np.exp(fit[1]+2*growth*t[window])/mode[0], "--", label="cold theory")
axes[0].set(xlabel=r"$t\omega_{pe}$", ylabel="mode energy / initial")
axes[0].legend()
axes[1].plot(t, error)
axes[1].set(xlabel=r"$t\omega_{pe}$", ylabel="relative total energy change")
axes[2].scatter(np.asarray(output["position_electrons"])[-1, :, 0]/length,
                np.asarray(output["velocity_electrons"])[-1, :, 0]/c, s=1)
axes[2].set(xlabel="x/L", ylabel="vx/c", title="final electron phase space", ylim=(-1, 1))
if args.output:
    fig.savefig(args.output, dpi=150)
else:
    plt.show()
