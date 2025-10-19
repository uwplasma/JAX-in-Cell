# _metric_tensor.py
import jax.numpy as jnp
from jax import jit, vmap, lax

@jit
def decay_env(t, tau):
    """
    Exponential envelope for 'turning off' metric drive:
        E(t; tau) = exp(-t/tau)  if tau > 0
                  = 1            otherwise (no decay)

    Using this multiplicative factor on the *deviation from a0* ensures:
      - short times (t << tau): original behavior (linear/oscillatory/etc.)
      - long times (t >> tau):  the deviation → 0 and a_i(t) → a0i
    """
    # JAX-friendly branch: tau<=0 => E=1
    return jnp.where(tau > 0.0, jnp.exp(-t / tau), 1.0)

@jit
def _christoffel_from_metric_tx(g, dg_dt, dg_dx):
    """
    Γ^μ_{αβ} = 1/2 g^{μσ} (∂_α g_{σβ} + ∂_β g_{σα} - ∂_σ g_{αβ})
    Only α in {0,1} may be nonzero here: 0≡t, 1≡x.
    Inputs:
      g:     (4,4)
      dg_dt: (4,4)
      dg_dx: (4,4)
    """
    g_inv = jnp.linalg.inv(g)

    # D[α,σ,β] = ∂_α g_{σβ}
    D = jnp.zeros((4,4,4))
    D = D.at[0,:,:].set(dg_dt)
    D = D.at[1,:,:].set(dg_dx)

    term = D + jnp.transpose(D,(0,2,1)) - jnp.transpose(D,(1,0,2))  # (α,σ,β)

    def gamma_mu_alpha_beta(mu, alpha, beta):
        return 0.5 * jnp.tensordot(g_inv[mu], term[alpha, :, beta], axes=1)

    idx = jnp.arange(4)
    return vmap(lambda mu: vmap(lambda a: vmap(lambda b: gamma_mu_alpha_beta(mu,a,b))(idx))(idx))(idx)

# ---- metric “catalog” (x -> g_μν) ----
@jit
def minkowski_metric_tx(x, c):
    g = jnp.diag(jnp.array([-c*c, 1.0, 1.0, 1.0]))
    return g, jnp.zeros((4,4)), jnp.zeros((4,4))

@jit
def rindler_metric_tx(x, g0, c):
    f = 1.0 + (g0 * x) / (c*c)
    g = jnp.diag(jnp.array([-(f*f)*(c*c), 1.0, 1.0, 1.0]))
    dg00_dx = -2.0*c*c*f*(g0/(c*c))
    dg_dx = jnp.zeros((4,4)).at[0,0].set(dg00_dx)
    return g, jnp.zeros((4,4)), dg_dx

@jit
def schwarzschild_isotropic_1D_tx(x, rs, c):
    eps = 1e-9
    phi = rs / (4.0*jnp.abs(x) + eps)
    onep = 1.0 + phi
    onem = 1.0 - phi
    A = (onem/onep)**2
    S = onep**4
    g = jnp.diag(jnp.array([-A*c*c, S, S, S]))
    dphi_dx = -rs * jnp.sign(x) / (4.0*(jnp.abs(x)+eps)**2)
    dA_dx   = (-4.0*onem/(onep**3)) * dphi_dx
    dS_dx   = 4.0 * (onep**3) * dphi_dx
    dg_dx = jnp.zeros((4,4)).at[0,0].set(-c*c*dA_dx).at[1,1].set(dS_dx).at[2,2].set(dS_dx).at[3,3].set(dS_dx)
    return g, jnp.zeros((4,4)), dg_dx

@jit
def flrw_a_powerlaw(t, a0, H0):
    # simple linear-in-time toy: a(t)=a0(1+H0 t), da/dt = a0 H0
    a   = a0 * (1.0 + H0 * t)
    adot = a0 * H0 * jnp.ones_like(t)
    return a, adot

@jit
def flrw_a_exponential(t, a0, H):
    # de Sitter toy: a(t)=a0 * exp(H t), da/dt = H a(t)
    a = a0 * jnp.exp(H * t)
    adot = H * a
    return a, adot

@jit
def flrw_tx(t, x, a, adot, c):
    """
    Flat FLRW (true): ds^2 = -c^2 dt^2 + a(t)^2 (dx^2+dy^2+dz^2)
    """
    g = jnp.diag(jnp.array([-c*c, a*a, a*a, a*a]))
    dg_dt = jnp.diag(jnp.array([0.0, 2*a*adot, 2*a*adot, 2*a*adot]))
    dg_dx = jnp.zeros((4,4))
    return g, dg_dt, dg_dx

@jit
def bianchi_i_a_linear(t, a0x, a0y, a0z, Hx, Hy, Hz, tau_x=0.0, tau_y=0.0, tau_z=0.0):
    r"""
    Decaying linear Bianchi I (diagonal, zero-shift):
        a_i(t) = a0i * [ 1 + H_i t * E_i(t) ],
      where E_i(t) = exp(-t/τ_i) if τ_i>0 else 1.

    Derivatives:
        d/dt [ H_i t E_i ] = H_i [ E_i + t * dE_i/dt ],
        dE_i/dt = -E_i/τ_i (if τ_i>0; else 0).

      Thus:
        a_i(t) = a0i [ 1 + H_i t E_i ],
        ȧ_i(t) = a0i H_i [ E_i - (t/τ_i) E_i ]  (if τ_i>0),
               = a0i H_i                        (if τ_i<=0).

    As t→∞ with τ_i>0, E_i→0 and a_i→a0i.
    """
    Ex = decay_env(t, tau_x)
    Ey = decay_env(t, tau_y)
    Ez = decay_env(t, tau_z)

    ax = a0x * (1.0 + Hx * t * Ex)
    ay = a0y * (1.0 + Hy * t * Ey)
    az = a0z * (1.0 + Hz * t * Ez)

    # ȧ: use d/dt[t E] = E + t dE/dt, with dE/dt = -E/τ (τ>0) else 0
    dEx_dt = jnp.where(tau_x > 0.0, -Ex / tau_x, 0.0)
    dEy_dt = jnp.where(tau_y > 0.0, -Ey / tau_y, 0.0)
    dEz_dt = jnp.where(tau_z > 0.0, -Ez / tau_z, 0.0)

    adx = a0x * Hx * (Ex + t * dEx_dt)
    ady = a0y * Hy * (Ey + t * dEy_dt)
    adz = a0z * Hz * (Ez + t * dEz_dt)
    return (ax, ay, az), (adx, ady, adz)

@jit
def bianchi_i_tx(t, x, ax, ay, az, adx, ady, adz, c):
    """
    Bianchi I (diagonal, zero shift): ds^2 = -c^2 dt^2 + ax(t)^2 dx^2 + ay(t)^2 dy^2 + az(t)^2 dz^2
    """
    g = jnp.diag(jnp.array([-c*c, ax*ax, ay*ay, az*az]))
    dg_dt = jnp.diag(jnp.array([0.0, 2*ax*adx, 2*ay*ady, 2*az*adz]))
    dg_dx = jnp.zeros((4,4))
    return g, dg_dt, dg_dx

@jit
def bianchi_i_a_cosine(t, a0x, a0y, a0z,
                       Ax, Ay, Az,
                       Omegax, Omegay, Omegaz,
                       phix, phiy, phiz,
                       tau_x=0.0, tau_y=0.0, tau_z=0.0):
    r"""
    Decaying oscillatory Bianchi I:
        a_i(t) = a0i [ 1 + A_i cos(Ω_i t + φ_i) E_i(t) ],
        E_i(t) = exp(-t/τ_i) if τ_i>0 else 1.

    Derivatives:
        ȧ_i = a0i * d/dt[ A_i cos(…) E_i ]
            = a0i * A_i [ -Ω_i sin(…) E_i + cos(…) dE_i/dt ],
        dE_i/dt = -E_i/τ_i (if τ_i>0).

    As t→∞ (τ_i>0), E_i→0 and a_i→a0i.
    """
    cx = jnp.cos(Omegax * t + phix); sx = jnp.sin(Omegax * t + phix)
    cy = jnp.cos(Omegay * t + phiy); sy = jnp.sin(Omegay * t + phiy)
    cz = jnp.cos(Omegaz * t + phiz); sz = jnp.sin(Omegaz * t + phiz)

    Ex = decay_env(t, tau_x); dEx_dt = jnp.where(tau_x > 0.0, -Ex / tau_x, 0.0)
    Ey = decay_env(t, tau_y); dEy_dt = jnp.where(tau_y > 0.0, -Ey / tau_y, 0.0)
    Ez = decay_env(t, tau_z); dEz_dt = jnp.where(tau_z > 0.0, -Ez / tau_z, 0.0)

    ax = a0x * (1.0 + Ax * cx * Ex)
    ay = a0y * (1.0 + Ay * cy * Ey)
    az = a0z * (1.0 + Az * cz * Ez)

    adx = a0x * (Ax * (-Omegax * sx * Ex + cx * dEx_dt))
    ady = a0y * (Ay * (-Omegay * sy * Ey + cy * dEy_dt))
    adz = a0z * (Az * (-Omegaz * sz * Ez + cz * dEz_dt))

    # keep tiny positive floor (optional)
    eps = 1e-15
    ax = jnp.clip(ax, eps, jnp.inf); ay = jnp.clip(ay, eps, jnp.inf); az = jnp.clip(az, eps, jnp.inf)
    return (ax, ay, az), (adx, ady, adz)

@jit
def bianchi_i_a_lin_cos(t, a0x, a0y, a0z,
                        Ax, Ay, Az,
                        Bx, By, Bz,
                        Omegax, Omegay, Omegaz,
                        phix, phiy, phiz,
                        tau_x=0.0, tau_y=0.0, tau_z=0.0):
    r"""
    Decaying linear×oscillatory Bianchi I:
        a_i(t) = a0i [ 1 + A_i t (1 + B_i cos(Ω_i t + φ_i)) E_i(t) ],
        E_i(t) = exp(-t/τ_i) if τ_i>0 else 1.

    Let g_i(t) := A_i t (1 + B_i cos(…)).
    Then:
        a_i = a0i [1 + g_i E_i],
        ȧ_i = a0i [ ġ_i E_i + g_i dE_i/dt ].

    With:
        ġ_i = A_i [ (1 + B_i cos(…)) + t (-B_i Ω_i sin(…)) ],
        dE_i/dt = -E_i/τ_i (if τ_i>0).

    As t→∞ (τ_i>0), E_i→0 ⇒ a_i→a0i.
    """
    cx = jnp.cos(Omegax * t + phix); sx = jnp.sin(Omegax * t + phix)
    cy = jnp.cos(Omegay * t + phiy); sy = jnp.sin(Omegay * t + phiy)
    cz = jnp.cos(Omegaz * t + phiz); sz = jnp.sin(Omegaz * t + phiz)

    Ex = decay_env(t, tau_x); dEx_dt = jnp.where(tau_x > 0.0, -Ex / tau_x, 0.0)
    Ey = decay_env(t, tau_y); dEy_dt = jnp.where(tau_y > 0.0, -Ey / tau_y, 0.0)
    Ez = decay_env(t, tau_z); dEz_dt = jnp.where(tau_z > 0.0, -Ez / tau_z, 0.0)

    gx = Ax * t * (1.0 + Bx * cx)
    gy = Ay * t * (1.0 + By * cy)
    gz = Az * t * (1.0 + Bz * cz)

    gdx = Ax * ((1.0 + Bx * cx) + t * (-Bx * Omegax * sx))
    gdy = Ay * ((1.0 + By * cy) + t * (-By * Omegay * sy))
    gdz = Az * ((1.0 + Bz * cz) + t * (-Bz * Omegaz * sz))

    ax = a0x * (1.0 + gx * Ex)
    ay = a0y * (1.0 + gy * Ey)
    az = a0z * (1.0 + gz * Ez)

    adx = a0x * (gdx * Ex + gx * dEx_dt)
    ady = a0y * (gdy * Ey + gy * dEy_dt)
    adz = a0z * (gdz * Ez + gz * dEz_dt)

    eps = 1e-15
    ax = jnp.clip(ax, eps, jnp.inf); ay = jnp.clip(ay, eps, jnp.inf); az = jnp.clip(az, eps, jnp.inf)
    return (ax, ay, az), (adx, ady, adz)

@jit
def bianchi_i_a_volpres_cosine(t, a0x, a0y, a0z, eps, Omega, phi, tau_eps=0.0):
    r"""
    Volume-preserving oscillatory Bianchi I (diagonal, zero shift) with decaying drive:

        a_x(t) = a0x,
        a_y(t) = a0y * den(t),
        a_z(t) = a0z / den(t),

      where
        den(t) = 1 + ε_eff(t) cos(Ω t + φ),
        ε_eff(t) = ε * exp(-t/τ_ε)  if τ_ε > 0,  else ε.

    Derivatives:
        C := cos(Ω t + φ),  S := sin(Ω t + φ)
        ε_eff' = -ε_eff / τ_ε  (if τ_ε>0, else 0)
        den'   = ε_eff' C + ε_eff * (-Ω S)
        ȧ_y    = a0y * den'
        ȧ_z    = a0z * (-den') / den^2

    As t→∞ with τ_ε>0, ε_eff→0 ⇒ den→1 ⇒ a_y→a0y, a_z→a0z (and a_y a_z→a0y a0z).
    """
    C = jnp.cos(Omega * t + phi)
    S = jnp.sin(Omega * t + phi)

    eps_eff = jnp.where(tau_eps > 0.0, eps * jnp.exp(-t / tau_eps), eps)
    deps_dt = jnp.where(tau_eps > 0.0, -eps_eff / tau_eps, 0.0)

    den = 1.0 + eps_eff * C
    den = jnp.clip(den, 1e-15, jnp.inf)   # robustness

    ax = a0x * jnp.ones_like(t)
    ay = a0y * den
    az = a0z / den

    dden_dt = deps_dt * C + eps_eff * (-Omega * S)

    adx = jnp.zeros_like(t)
    ady = a0y * dden_dt
    adz = a0z * (-dden_dt) / (den * den)

    return (ax, ay, az), (adx, ady, adz)


@jit
def metric_bundle(t, x, metric_kind, c, **kwargs):
    """
    metric_kind:
      0=minkowski, 1=rindler, 2=schwarzschild_iso,
      3=flrw_x (static-in-x; keep if you like),
      4=flrw_powerlaw a(t)=a0(1+H0 t),
      5=flrw_exp      a(t)=a0*exp(H t)
      6=bianchi_i_linear
      7=bianchi_i_cosine    # (periodic/oscillatory Bianchi I)
      8=bianchi_i_lin_cos   #  a_i(t) = a0i [1 + A_i t (1 + B_i cos)]
      9=bianchi_i_volpres_cosine  # volume-preserving oscillatory (diag, zero shift)

      
      times are in units of plasma frequency ωₚ⁻¹
    """
    def minkowski_case(_):
        return minkowski_metric_tx(x, c)
    def rindler_case(_):
        g0 = kwargs.get("g0", 9.81)
        return rindler_metric_tx(x, g0, c)
    def schwarzschild_case(_):
        rs = kwargs.get("rs", 0.0)
        return schwarzschild_isotropic_1D_tx(x, rs, c)
    def flrw_powerlaw_case(_):
        a0 = kwargs.get("a0", 1.0); H0 = kwargs.get("H0", 0.0)
        a, adot = flrw_a_powerlaw(t, a0, H0)
        return flrw_tx(t, x, a, adot, c)
    def flrw_exp_case(_):
        a0 = kwargs.get("a0", 1.0); H = kwargs.get("H", 0.0)
        a, adot = flrw_a_exponential(t, a0, H)
        return flrw_tx(t, x, a, adot, c)
    def bianchi_i_linear_case(_):
        a0x = kwargs.get("a0x", 1.0); a0y = kwargs.get("a0y", 1.0); a0z = kwargs.get("a0z", 1.0)
        Hx  = kwargs.get("Hx",  0.0); Hy  = kwargs.get("Hy",  0.0); Hz  = kwargs.get("Hz",  0.0)
        tau_x = kwargs.get("tau_x", 0.0); tau_y = kwargs.get("tau_y", 0.0); tau_z = kwargs.get("tau_z", 0.0)
        (ax, ay, az), (adx, ady, adz) = bianchi_i_a_linear(t, a0x, a0y, a0z, Hx, Hy, Hz, tau_x, tau_y, tau_z)
        return bianchi_i_tx(t, x, ax, ay, az, adx, ady, adz, c)

    def bianchi_i_cosine_case(_):
        a0x = kwargs.get("a0x", 1.0); a0y = kwargs.get("a0y", 1.0); a0z = kwargs.get("a0z", 1.0)
        Ax  = kwargs.get("Ax",  0.0); Ay  = kwargs.get("Ay",  0.0); Az  = kwargs.get("Az",  0.0)
        Omegax = kwargs.get("Omegax", 0.0); Omegay = kwargs.get("Omegay", 0.0); Omegaz = kwargs.get("Omegaz", 0.0)
        phix   = kwargs.get("phix",   0.0); phiy   = kwargs.get("phiy",   0.0); phiz   = kwargs.get("phiz",   0.0)
        tau_x = kwargs.get("tau_x", 0.0); tau_y = kwargs.get("tau_y", 0.0); tau_z = kwargs.get("tau_z", 0.0)
        (ax, ay, az), (adx, ady, adz) = bianchi_i_a_cosine(
            t, a0x, a0y, a0z, Ax, Ay, Az, Omegax, Omegay, Omegaz, phix, phiy, phiz, tau_x, tau_y, tau_z
        )
        return bianchi_i_tx(t, x, ax, ay, az, adx, ady, adz, c)

    def bianchi_i_lin_cos_case(_):
        a0x = kwargs.get("a0x", 1.0); a0y = kwargs.get("a0y", 1.0); a0z = kwargs.get("a0z", 1.0)
        Ax  = kwargs.get("Ax",  0.0); Ay  = kwargs.get("Ay",  0.0); Az  = kwargs.get("Az",  0.0)
        Bx  = kwargs.get("Bx",  0.0); By  = kwargs.get("By",  0.0); Bz  = kwargs.get("Bz",  0.0)
        Omegax = kwargs.get("Omegax", 0.0); Omegay = kwargs.get("Omegay", 0.0); Omegaz = kwargs.get("Omegaz", 0.0)
        phix   = kwargs.get("phix",   0.0); phiy   = kwargs.get("phiy",   0.0); phiz   = kwargs.get("phiz",   0.0)
        tau_x = kwargs.get("tau_x", 0.0); tau_y = kwargs.get("tau_y", 0.0); tau_z = kwargs.get("tau_z", 0.0)
        (ax, ay, az), (adx, ady, adz) = bianchi_i_a_lin_cos(
            t, a0x, a0y, a0z, Ax, Ay, Az, Bx, By, Bz, Omegax, Omegay, Omegaz, phix, phiy, phiz, tau_x, tau_y, tau_z
        )
        return bianchi_i_tx(t, x, ax, ay, az, adx, ady, adz, c)

    def bianchi_i_volpres_cosine_case(_):
        a0x = kwargs.get("a0x", 1.0); a0y = kwargs.get("a0y", 1.0); a0z = kwargs.get("a0z", 1.0)
        eps = kwargs.get("eps", 0.1); Omega = kwargs.get("Omega", 5.0); phi = kwargs.get("phi", 0.0)
        tau_eps = kwargs.get("tau_eps", 0.0)
        (ax, ay, az), (adx, ady, adz) = bianchi_i_a_volpres_cosine(t, a0x, a0y, a0z, eps, Omega, phi, tau_eps)
        return bianchi_i_tx(t, x, ax, ay, az, adx, ady, adz, c)

    g, dg_dt, dg_dx = lax.switch(
        metric_kind,
        [
            minkowski_case,         # 0
            rindler_case,           # 1
            schwarzschild_case,     # 2
            minkowski_case,         # 3 (placeholder)
            flrw_powerlaw_case,     # 4
            flrw_exp_case,          # 5
            bianchi_i_linear_case,  # 6 
            bianchi_i_cosine_case,  # 7
            bianchi_i_lin_cos_case,  # 8
            bianchi_i_volpres_cosine_case,  # 9
        ],
        operand=None,
    )
    g_inv  = jnp.linalg.inv(g)
    Gamma  = _christoffel_from_metric_tx(g, dg_dt, dg_dx)
    sqrtm  = jnp.sqrt(-jnp.linalg.det(g))
    return {"g": g, "g_inv": g_inv, "sqrt_minus_g": sqrtm, "Gamma": Gamma}