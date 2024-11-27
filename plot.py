import matplotlib.pyplot as plt

def plot_results(output):
    plt.figure()
    plt.imshow(output["E_field"][:, :, 0], aspect='auto', cmap='RdBu', origin='lower', extent=[output["grid"][0], output["grid"][-1], 0, output["total_steps"] * output["dt"]])
    plt.colorbar(label='Electric field (V/m)')
    plt.xlabel('Position (m)')
    plt.ylabel('Time (s)')

    plt.figure()
    plt.imshow(output["charge_density"], aspect='auto', cmap='RdBu', origin='lower', extent=[output["grid"][0], output["grid"][-1], 0, output["total_steps"] * output["dt"]])
    plt.colorbar(label='Charge density (C/m^3)')
    plt.xlabel('Position (m)')
    plt.ylabel('Time (s)')

    plt.figure()
    plt.plot(output["time_array"], output["E_energy"])
    plt.xlabel('Time (s)')
    plt.ylabel('Electric field energy (J)')
    plt.show()