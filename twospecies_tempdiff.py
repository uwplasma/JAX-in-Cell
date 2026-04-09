import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from jax import block_until_ready
from jaxincell import simulation, load_parameters, diagnostics

# -----------------------------------------------------------------------------
# 1. Setup & Model Functions
# -----------------------------------------------------------------------------
def setup_plot_style(ax):
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.minorticks_on()
    ax.set_xlabel(r"Time ($\omega_{pe}^{-1}$)", fontsize=14)

def temp_diff_model(t, Delta_T_0, gamma):
    """
    Standard relaxation model for the difference: 
    d(Te - Tp)/dt = -gamma * (Te - Tp)
    """
    return Delta_T_0 * np.exp(-gamma * t)

# -----------------------------------------------------------------------------
# 2. Run Simulation
# -----------------------------------------------------------------------------
norm_factor = 25.1646059279994
current_directory = os.path.dirname(os.path.abspath(__file__))

input_file = 'twospecies_temp.toml'
input_path = os.path.join(current_directory, input_file)

print(f"--- Running Relaxation Benchmark: {input_file} ---")
input_params, solver_params = load_parameters(input_path)

start_time = time.time()
output = block_until_ready(simulation(input_params, **solver_params))
diagnostics(output)
print(f"Simulation complete. ({time.time() - start_time:.2f}s)")

# -----------------------------------------------------------------------------
# 3. Extract and Normalize Data
# -----------------------------------------------------------------------------
time_ts = np.arange(len(output["electric_field_energy"]))
time_norm = time_ts / norm_factor

# Weighting factor to get actual Temperature (K or eV)
weight = 4370228600000
temp_e = np.array(output["temperature_electrons"][:, 0]) / weight
temp_p = np.array(output["temperature_ions"][:, 0]) / weight

# Calculate actual theoretical equilibrium from initial data
t_eq = (temp_e[0] + temp_p[0]) / 2.0
delta_T_data = temp_e - temp_p

# -----------------------------------------------------------------------------
# 4. Direct Unified Fit
# -----------------------------------------------------------------------------
print(f"\nSystem Equilibrium T: {t_eq:.4e}")

try:
    # Fit the difference directly
    popt, pcov = curve_fit(
        temp_diff_model,
        time_norm,
        delta_T_data,
        p0=[delta_T_data[0], 0.05] # Guess initial delta and a small gamma
    )
    
    fit_delta_0, fit_gamma = popt
    perr = np.sqrt(np.diag(pcov))
    
    print(f"Unified Fit Result:")
    print(f"  Gamma (Collision Rate): {fit_gamma:.6f} +/- {perr[1]:.6f}")
    print(f"  Initial Delta T:       {fit_delta_0:.4e}")

    # Generate the symmetric fits for plotting
    # For equal masses: Te = Teq + 0.5*DeltaT; Tp = Teq - 0.5*DeltaT
    temp_e_fit = t_eq + 0.5 * temp_diff_model(time_norm, *popt)
    temp_p_fit = t_eq - 0.5 * temp_diff_model(time_norm, *popt)

except Exception as e:
    print("Unified fitting failed:", e)
    temp_e_fit = np.full_like(time_norm, t_eq)
    temp_p_fit = np.full_like(time_norm, t_eq)
    fit_gamma = 0.0

# -----------------------------------------------------------------------------
# 5. Plotting
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Linear Plot (Evolution) ---
ax1.plot(time_norm, temp_e, label="Electrons (Data)", color='royalblue', alpha=0.6)
ax1.plot(time_norm, temp_p, label="Positrons (Data)", color='crimson', alpha=0.6)

ax1.plot(time_norm, temp_e_fit, 'k--', lw=2, label=f"Unified Fit ($\gamma$={fit_gamma:.4f})")
ax1.plot(time_norm, temp_p_fit, 'k--', lw=2)

ax1.axhline(y=t_eq, color='gray', linestyle=':', label="Theoretical Eq")

setup_plot_style(ax1)
ax1.set_ylabel("Temperature (K)", fontsize=14)
ax1.set_title("Coupled Temperature Relaxation")
ax1.legend()

# --- Right: Semi-Log Plot (Verification) ---
# Plot the absolute value of the temperature difference
ax2.semilogy(time_norm, np.abs(delta_T_data), label=r"$|T_e - T_p|$ Data", color='purple', alpha=0.7)
ax2.semilogy(time_norm, np.abs(temp_diff_model(time_norm, *popt)), 
             'k--', lw=2.5, label="Exponential Fit")

setup_plot_style(ax2)
ax2.set_ylabel(r"$|\Delta T|$ (Log Scale)", fontsize=14)
ax2.set_title("Relaxation Rate Verification")
ax2.legend()

plt.tight_layout()
plt.show()