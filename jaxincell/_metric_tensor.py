# _metric_tensor.py
import jax.numpy as jnp
from jax import jit, vmap, lax

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
def bianchi_i_a_linear(t, a0x, a0y, a0z, Hx, Hy, Hz):
    ax   = a0x * (1 + Hx * t)
    ay   = a0y * (1 + Hy * t)
    az   = a0z * (1 + Hz * t)
    adx  = Hx * a0x * jnp.ones_like(t)
    ady  = Hy * a0y * jnp.ones_like(t)
    adz  = Hz * a0z * jnp.ones_like(t)
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
                       phix, phiy, phiz):
    """
    Bianchi I with periodic scale factors:
      a_x(t) = a0x * (1 + Ax * cos(Omegax * t + phix))
      a_y(t) = a0y * (1 + Ay * cos(Omegay * t + phiy))
      a_z(t) = a0z * (1 + Az * cos(Omegaz * t + phiz))

    Notes:
    - Keep |A?| < 1 to ensure a_i(t) > 0. (We do a tiny floor for safety.)
    - t is in your code’s time units (ω_p^{-1}).
    """
    ax = a0x * (1.0 + Ax * jnp.cos(Omegax * t + phix))
    ay = a0y * (1.0 + Ay * jnp.cos(Omegay * t + phiy))
    az = a0z * (1.0 + Az * jnp.cos(Omegaz * t + phiz))

    # time derivatives
    adx = a0x * (-Ax * Omegax * jnp.sin(Omegax * t + phix))
    ady = a0y * (-Ay * Omegay * jnp.sin(Omegay * t + phiy))
    adz = a0z * (-Az * Omegaz * jnp.sin(Omegaz * t + phiz))

    # tiny positive floor to avoid exact zeros (keeps metric well-defined)
    eps = 1e-15
    ax = jnp.clip(ax, eps, jnp.inf)
    ay = jnp.clip(ay, eps, jnp.inf)
    az = jnp.clip(az, eps, jnp.inf)
    return (ax, ay, az), (adx, ady, adz)


@jit
def bianchi_i_cos_tx(t, x,
                     a0x, a0y, a0z,
                     Ax, Ay, Az,
                     Omegax, Omegay, Omegaz,
                     phix, phiy, phiz,
                     c):
    """
    Wrapper that builds (g, ∂_t g, ∂_x g) for oscillatory Bianchi I
    using the existing bianchi_i_tx.
    """
    (ax, ay, az), (adx, ady, adz) = bianchi_i_a_cosine(
        t, a0x, a0y, a0z, Ax, Ay, Az, Omegax, Omegay, Omegaz, phix, phiy, phiz
    )
    return bianchi_i_tx(t, x, ax, ay, az, adx, ady, adz, c)

@jit
def bianchi_i_a_lin_cos(t, a0x, a0y, a0z,
                        Ax, Ay, Az,          # linear slopes A_?
                        Bx, By, Bz,          # oscillation amplitudes B_?
                        Omegax, Omegay, Omegaz,
                        phix, phiy, phiz):
    """
    Bianchi I with linear×oscillatory scale factors:

      a_i(t) = a0i * [ 1 + A_i * t * ( 1 + B_i * cos(Ω_i t + φ_i) ) ]

    Derivative:
      da_i/dt = a0i * [ A_i * ( 1 + B_i * cos(Ω_i t + φ_i) )
                        + A_i * t * ( -B_i * Ω_i * sin(Ω_i t + φ_i) ) ]

    Keep |1 + A_i t (1 + B_i cos)| > 0; we clip very near zero for safety.
    """
    cx = jnp.cos(Omegax * t + phix); sx = jnp.sin(Omegax * t + phix)
    cy = jnp.cos(Omegay * t + phiy); sy = jnp.sin(Omegay * t + phiy)
    cz = jnp.cos(Omegaz * t + phiz); sz = jnp.sin(Omegaz * t + phiz)

    # scale factors
    ax = a0x * (1.0 + Ax * t * (1.0 + Bx * cx))
    ay = a0y * (1.0 + Ay * t * (1.0 + By * cy))
    az = a0z * (1.0 + Az * t * (1.0 + Bz * cz))

    # time derivatives
    dax = a0x * ( Ax * (1.0 + Bx * cx) + Ax * t * (-Bx * Omegax * sx) )
    day = a0y * ( Ay * (1.0 + By * cy) + Ay * t * (-By * Omegay * sy) )
    daz = a0z * ( Az * (1.0 + Bz * cz) + Az * t * (-Bz * Omegaz * sz) )

    # tiny positive floor to avoid singular metric when bracket ~ 0
    eps = 1e-15
    ax = jnp.clip(ax, eps, jnp.inf)
    ay = jnp.clip(ay, eps, jnp.inf)
    az = jnp.clip(az, eps, jnp.inf)

    return (ax, ay, az), (dax, day, daz)


@jit
def bianchi_i_lin_cos_tx(t, x,
                         a0x, a0y, a0z,
                         Ax, Ay, Az,
                         Bx, By, Bz,
                         Omegax, Omegay, Omegaz,
                         phix, phiy, phiz,
                         c):
    """
    Build (g, ∂_t g, ∂_x g) for linear×oscillatory Bianchi I
    using the existing bianchi_i_tx.
    """
    (ax, ay, az), (adx, ady, adz) = bianchi_i_a_lin_cos(
        t, a0x, a0y, a0z,
        Ax, Ay, Az, Bx, By, Bz,
        Omegax, Omegay, Omegaz, phix, phiy, phiz
    )
    return bianchi_i_tx(t, x, ax, ay, az, adx, ady, adz, c)

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
        # defaults: unit scale factors at t=0, choose your H's
        a0x = kwargs.get("a0x", 1.0); a0y = kwargs.get("a0y", 1.0); a0z = kwargs.get("a0z", 1.0)
        Hx  = kwargs.get("Hx",  0.0); Hy  = kwargs.get("Hy",  0.0); Hz  = kwargs.get("Hz",  0.0)
        (ax, ay, az), (adx, ady, adz) = bianchi_i_a_linear(t, a0x, a0y, a0z, Hx, Hy, Hz)
        return bianchi_i_tx(t, x, ax, ay, az, adx, ady, adz, c)
    def bianchi_i_cosine_case(_):
        # defaults chosen so that with Ax=Ay=Az=0 you recover Minkowski space (a_i=a0i)
        a0x = kwargs.get("a0x", 1.0); a0y = kwargs.get("a0y", 1.0); a0z = kwargs.get("a0z", 1.0)
        Ax  = kwargs.get("Ax",  0.0); Ay  = kwargs.get("Ay",  0.0); Az  = kwargs.get("Az",  0.0)
        Omegax = kwargs.get("Omegax", 0.0); Omegay = kwargs.get("Omegay", 0.0); Omegaz = kwargs.get("Omegaz", 0.0)
        phix   = kwargs.get("phix",   0.0); phiy   = kwargs.get("phiy",   0.0); phiz   = kwargs.get("phiz",   0.0)
        return bianchi_i_cos_tx(t, x, a0x, a0y, a0z,
            Ax, Ay, Az, Omegax, Omegay, Omegaz, phix, phiy, phiz, c)
    def bianchi_i_lin_cos_case(_):
        # Defaults recover Minkowski when A?=0.
        a0x = kwargs.get("a0x", 1.0); a0y = kwargs.get("a0y", 1.0); a0z = kwargs.get("a0z", 1.0)
        Ax  = kwargs.get("Ax",  0.0); Ay  = kwargs.get("Ay",  0.0); Az  = kwargs.get("Az",  0.0)
        Bx  = kwargs.get("Bx",  0.0); By  = kwargs.get("By",  0.0); Bz  = kwargs.get("Bz",  0.0)
        Omegax = kwargs.get("Omegax", 0.0); Omegay = kwargs.get("Omegay", 0.0); Omegaz = kwargs.get("Omegaz", 0.0)
        phix   = kwargs.get("phix",   0.0); phiy   = kwargs.get("phiy",   0.0); phiz   = kwargs.get("phiz",   0.0)
        return bianchi_i_lin_cos_tx( t, x, a0x, a0y, a0z, Ax, Ay, Az, Bx, By, Bz,
            Omegax, Omegay, Omegaz, phix, phiy, phiz,c)

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
        ],
        operand=None,
    )
    g_inv  = jnp.linalg.inv(g)
    Gamma  = _christoffel_from_metric_tx(g, dg_dt, dg_dx)
    sqrtm  = jnp.sqrt(-jnp.linalg.det(g))
    return {"g": g, "g_inv": g_inv, "sqrt_minus_g": sqrtm, "Gamma": Gamma}