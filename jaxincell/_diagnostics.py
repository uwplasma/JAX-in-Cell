from jax import lax
import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
from ._constants import epsilon_0, mu_0

__all__ = ['diagnostics']

def diagnostics(output, *, jittable: bool = False):
    """
    If jittable=True: avoid any boolean-mask indexing that changes shapes.
    Compute diagnostics via mask-weighted reductions and keep arrays intact.
    If jittable=False: (old behavior) you may split arrays into electrons/ions.
    """
    # ---------- locals ----------
    E_t = output['electric_field']      # (T, Ng, 3)
    B_t = output['magnetic_field']      # (T, Ng, 3)
    grid = output['grid']               # (Ng,)
    dt   = output['dt']                 # scalar
    T    = output['total_steps']        # int
    dx   = output['dx']                 # scalar
    N    = output['number_pseudoelectrons']  # electrons per default pop

    # ---------- FFT-based dominant frequency ----------
    arr = E_t[:, len(grid)//2, 0]
    arr = (arr - jnp.mean(arr)) / jnp.max(arr)
    fft_vals = lax.slice(fft(arr), (0,), (T//2,))
    freqs = fftfreq(T, d=dt)[:T//2] * 2*jnp.pi
    mag = jnp.abs(fft_vals)
    idx = jnp.argmax(mag)
    dom_omega = jnp.abs(freqs[idx])

    # ---------- trapz integrate ----------
    def integrate_trap(y, dx):
        # y: (..., Ng)
        return 0.5 * (jnp.asarray(dx) * (y[..., 1:] + y[..., :-1])).sum(-1)

    # ---------- field energies ----------
    absE2 = jnp.sum(E_t**2, axis=-1)  # (T, Ng)
    absB2 = jnp.sum(B_t**2, axis=-1)  # (T, Ng)

    # external fields are static in time; make (T, Ng) for energy time series
    absE2_ext = jnp.sum(output['external_electric_field']**2, axis=-1)      # (Ng,)
    absB2_ext = jnp.sum(output['external_magnetic_field']**2, axis=-1)      # (Ng,)
    absE2_ext_T = jnp.broadcast_to(absE2_ext, (absE2.shape[0], absE2.shape[1]))
    absB2_ext_T = jnp.broadcast_to(absB2_ext, (absB2.shape[0], absB2.shape[1]))

    intE2     = integrate_trap(absE2,     dx)
    intB2     = integrate_trap(absB2,     dx)
    intE2_ext = integrate_trap(absE2_ext_T, dx)
    intB2_ext = integrate_trap(absB2_ext_T, dx)

    # ---------- kinetic energies via mask-weighted reductions ----------
    # unified arrays
    pos = output['positions']    # (T, Ntot, 3)
    vel = output['velocities']   # (T, Ntot, 3)
    m   = output['masses'][...,0]   # (Ntot,)
    q   = output['charges'][...,0]  # (Ntot,)

    # masks (elementwise use only; no slicing by mask)
    is_e = (q < 0)  # (Ntot,)
    is_i = ~is_e
    me = is_e.astype(m.dtype)
    mi = is_i.astype(m.dtype)

    v2 = jnp.sum(vel**2, axis=-1)     # (T, Ntot)
    KE_particle = 0.5 * v2 * m[None,:]  # (T, Ntot)
    KE_e = jnp.sum(KE_particle * me[None,:], axis=-1)  # (T,)
    KE_i = jnp.sum(KE_particle * mi[None,:], axis=-1)  # (T,)
    KE   = KE_e + KE_i

    # ---------- pack scalars/time series ----------
    output.update({
        'electric_field_energy_density': (epsilon_0/2) * absE2,          # (T, Ng)
        'electric_field_energy':         (epsilon_0/2) * intE2,           # (T,)
        'magnetic_field_energy_density': 1/(2*mu_0) * absB2,             # (T, Ng)
        'magnetic_field_energy':         1/(2*mu_0) * intB2,              # (T,)
        'external_electric_field_energy_density': (epsilon_0/2) * absE2_ext,  # (Ng,)
        'external_electric_field_energy':         (epsilon_0/2) * intE2_ext,  # (T,)
        'external_magnetic_field_energy_density': 1/(2*mu_0) * absB2_ext,     # (Ng,)
        'external_magnetic_field_energy':         1/(2*mu_0) * intB2_ext,     # (T,)
        'dominant_frequency': dom_omega,
        'kinetic_energy': KE,                      # (T,)
        'kinetic_energy_electrons': KE_e,          # (T,)
        'kinetic_energy_ions': KE_i,               # (T,)
    })

    # ---------- JIT-safe default species split via static slices ----------
    # Your particle ordering is: [electrons (N), ions (N), then any extra species...]
    # We only split the *default* two populations for plotting.
    e_slice = slice(0, N)
    i_slice = slice(N, 2*N)

    output.update({
        "position_electrons": pos[:, e_slice, :],
        "velocity_electrons": vel[:, e_slice, :],
        "mass_electrons":     output["masses"][e_slice, :],
        "charge_electrons":   output["charges"][e_slice, :],
        "species_id_electrons": output["species_ids"][e_slice, :],

        "position_ions":      pos[:, i_slice, :],
        "velocity_ions":      vel[:, i_slice, :],
        "mass_ions":          output["masses"][i_slice, :],
        "charge_ions":        output["charges"][i_slice, :],
        "species_id_ions":    output["species_ids"][i_slice, :],
    })

    # weights for histograms (default pops → 1.0 is fine; keeps plot working)
    # If later you want real per-species weight ratios, emit a parallel array
    # from initialize and slice here similarly.
    ones_N = jnp.ones((N,1), dtype=output["charges"].dtype)
    output.update({
        "weight_electrons": ones_N,
        "weight_ions":      ones_N,
    })

    # ---------- total energy ----------
    total_energy = (output["electric_field_energy"]
                    + output["external_electric_field_energy"]
                    + output["magnetic_field_energy"]
                    + output["external_magnetic_field_energy"]
                    + output["kinetic_energy"])
    output.update({'total_energy': total_energy})

    # In jittable=True, keep unified arrays; do NOT delete anything.
    if not jittable:
        # (optional non-JIT path: you could delete unified arrays here)
        pass