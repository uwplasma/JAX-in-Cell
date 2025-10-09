# solve_dispersion_jax_norm.py
# Electron-only dispersion in normalized vars:
#   khat = k * lambda_D,  omegahat = omega / omega_pe
# Includes Einstein coupling kappa = 8*pi*G/c^4 and prints Lambda (drops at linear order).
#
# Usage:
#   python solve_dispersion_jax_norm.py --khat 0.3 --n0 1e18 --Te_eV 10 --G 6.67430e-11 --Lambda 0.0
#   python solve_dispersion_jax_norm.py --scan 0.05 1.0 80 --n0 1e18 --Te_eV 10
#
# Optional (double precision):
#   export JAX_ENABLE_X64=True

import argparse, math
import numpy as onp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

# ----------------------------
# Physical constants (SI)
# ----------------------------
e   = 1.602176634e-19
me  = 9.1093837015e-31
eps0= 8.8541878128e-12
c   = 299792458.0
pi  = math.pi

# ============================
# Faddeeva / Plasma Z (Weideman)
# ============================
def _weideman_coeffs(M=64, L=6.0):
    m = jnp.arange(-M, M+1)
    t = (m * jnp.pi) / L
    ccoefs = (1.0 / L) * jnp.exp(-t**2)
    return t, ccoefs

T_nodes, C_coefs = _weideman_coeffs(M=64, L=6.0)

@jit
def wofz_weideman(z):
    frac = C_coefs / (z - T_nodes)
    s = jnp.sum(frac)
    return 1j / jnp.sqrt(jnp.pi) * s

@jit
def Z_plasma(z):
    return 1j * jnp.sqrt(jnp.pi) * wofz_weideman(z)

# ----------------------------
# Plasma params
# ----------------------------
def plasma_params(n0, Te_J):
    omega_pe = jnp.sqrt(n0 * e**2 / (eps0 * me))
    vth      = jnp.sqrt(2.0 * Te_J / me)
    lambda_D = jnp.sqrt(eps0 * Te_J / (n0 * e**2))
    return omega_pe, vth, lambda_D

# ----------------------------
# I_e (dimensional)
# ----------------------------
@jit
def I_e_dim(omega, k, n0, vth):
    zeta = omega / (k * vth)
    return n0 / (k * vth**2) * (1.0 + zeta * Z_plasma(zeta))

# ----------------------------
# H and G (dimensional)
# ----------------------------
@jit
def H_dim(omega, k):
    return 2.0 * k**2 / c**2 + 2.0 * (omega**2) / c**4

@jit
def G_dim(omega, k):
    return k - 2.0 * (omega**2) / (k * c**2)

# ----------------------------
# Dispersion residual (dimensional, electron-only) WITH kappa
# F = H + kappa * G * [ me*Ie + ( (qe^2 * Ie^2)/(eps0*k) ) / (1 - (qe^2/(me*eps0*k))*Ie ) ]
# Note: Lambda does not enter linear perturbations.
# ----------------------------
@jit
def dispersion_residual_dim(omega, k, n0, vth, Ggrav):
    kappa = 8.0 * jnp.pi * Ggrav / (c**4)
    qe = -e
    Ie = I_e_dim(omega, k, n0, vth)
    H  = H_dim(omega, k)
    Gg = G_dim(omega, k)
    Den = 1.0 - (qe**2 / (me * eps0 * k)) * Ie
    matter_bracket = me * Ie + ((qe**2) * (Ie**2)) / (eps0 * k * Den)
    return H + kappa * Gg * matter_bracket

# ----------------------------
# Normalized residual on R^2:
# khat = k*lambda_D,  omegahat = omega/omega_pe
# ----------------------------
def residual_R2_norm(what_re_im, khat, n0, Te_J, Ggrav):
    omega_pe, vth, lambda_D = plasma_params(n0, Te_J)
    k = khat / lambda_D
    omega = (what_re_im[0] + 1j * what_re_im[1]) * omega_pe
    F = dispersion_residual_dim(omega, k, n0, vth, Ggrav)
    # scale to keep magnitudes controlled
    scale = (omega_pe**2) / (c**2)
    Fhat = F / scale
    return jnp.array([jnp.real(Fhat), jnp.imag(Fhat)])

residual_R2_norm_jit = jit(residual_R2_norm)
jac_R2_norm = jit(jacfwd(residual_R2_norm, argnums=0))

# ----------------------------
# Initial guess (normalized)
# ----------------------------
def initial_guess_norm(khat):
    omega_r_hat = jnp.sqrt(1.0 + 1.5 * khat**2)
    omega_i_hat = -1e-3 * omega_r_hat
    return jnp.array([omega_r_hat, omega_i_hat])

# ----------------------------
# Manual 2x2 solve
# ----------------------------
@jit
def solve2x2(J, b):
    a, b1 = J[0,0], J[0,1]
    c1, d = J[1,0], J[1,1]
    det = a*d - b1*c1
    invJ = jnp.array([[ d, -b1],
                      [-c1,  a]]) / det
    return invJ @ b

# ----------------------------
# Newton (not JIT; residual/Jacobian are JIT)
# ----------------------------
def newton_norm(w0_hat, khat, n0, Te_J, Ggrav, max_iter=60, tol=1e-11):
    w = w0_hat
    for _ in range(max_iter):
        r = residual_R2_norm_jit(w, khat, n0, Te_J, Ggrav)
        J = jac_R2_norm(w, khat, n0, Te_J, Ggrav)
        delta = -solve2x2(J, r)
        w = w + delta
        if float(jnp.linalg.norm(r, ord=jnp.inf)) < tol:
            return w, True
    return w, False

def solve_for_khat(khat, n0, Te_J, Ggrav):
    w0 = initial_guess_norm(khat)
    w_sol, ok = newton_norm(w0, khat, n0, Te_J, Ggrav)
    omegahat = complex(float(w_sol[0]), float(w_sol[1]))
    return omegahat, ok

# ----------------------------
# Scan & plot Im(omegahat)
# ----------------------------
def scan_and_plot(kmin, kmax, npts, n0, Te_J, Ggrav):
    import matplotlib.pyplot as plt
    ks = onp.linspace(kmin, kmax, npts)
    wr = onp.zeros_like(ks, dtype=onp.float64)
    wi = onp.zeros_like(ks, dtype=onp.float64)

    wseed = None
    for i, khat in enumerate(ks):
        w0 = initial_guess_norm(khat) if wseed is None else jnp.array([wseed.real, wseed.imag])
        w_sol, ok = newton_norm(w0, khat, n0, Te_J, Ggrav)
        wseed = complex(float(w_sol[0]), float(w_sol[1]))
        wr[i], wi[i] = wseed.real, wseed.imag

    plt.figure(figsize=(6,4))
    plt.plot(ks, wi, lw=2)
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel(r'$\hat k = k\lambda_D$')
    plt.ylabel(r'$\Im(\hat\omega) = \gamma/\omega_{pe}$')
    plt.title('Growth/damping vs normalized wavenumber')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Electron-only dispersion (JAX, normalized, with GR coupling).")
    parser.add_argument("--khat", type=float, default=1.0, help="Dimensionless wavenumber khat = k * lambda_D")
    parser.add_argument("--n0", type=float, default=1e18, help="Electron density [m^-3]")
    parser.add_argument("--Te_eV", type=float, default=10.0, help="Electron temperature [eV]")
    parser.add_argument("--G", type=float, default=6.67430e-11, help="Newton's gravitational constant [SI]")
    parser.add_argument("--Lambda", type=float, default=0.0, help="Cosmological constant [1/m^2] (drops at linear order)")
    parser.add_argument("--print-params", action="store_true", help="Print derived dimensional parameters")
    parser.add_argument("--scan", nargs=3, metavar=('KMIN','KMAX','NPTS'),
                        help="Scan khat from KMIN to KMAX with NPTS and plot Im(omegahat)")
    args = parser.parse_args()

    n0   = float(args.n0)
    Te_J = float(args.Te_eV) * e
    Ggrav= float(args.G)
    Lambda = float(args.Lambda)  # not used at linear order

    if args.scan:
        kmin, kmax, npts = float(args.scan[0]), float(args.scan[1]), int(args.scan[2])
        scan_and_plot(kmin, kmax, npts, n0, Te_J, Ggrav)
        return

    khat = float(args.khat)
    omega_pe, vth, lambda_D = plasma_params(n0, Te_J)
    if args.print_params:
        print("=== Derived dimensional parameters ===")
        print(f"omega_pe  = {float(omega_pe):.6e} rad/s")
        print(f"v_th      = {float(vth):.6e} m/s")
        print(f"lambda_D  = {float(lambda_D):.6e} m")
        print(f"k         = {khat/lambda_D:.6e} 1/m")
        print(f"G         = {Ggrav:.6e} SI  -> kappa = {8*math.pi*Ggrav/c**4:.6e} 1/(Pa·m^2)")
        print(f"Lambda    = {Lambda:.6e} 1/m^2 (drops at linear order)")
        print("=====================================")

    what, ok = solve_for_khat(khat, n0, Te_J, Ggrav)
    print(f"Converged: {ok}")
    print(f"Re(omegahat) = {what.real:.10e}")
    print(f"Im(omegahat) = {what.imag:.10e}")

    # Also print zeta = omega/(k v_th)
    zeta = (what * omega_pe) / ((khat / lambda_D) * vth)
    print(f"zeta = omega/(k v_th) = {zeta.real:.6f} + {zeta.imag:.6f}i")

if __name__ == "__main__":
    main()
