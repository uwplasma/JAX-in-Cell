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
def metric_bundle(t, x, metric_kind, c, **kwargs):
    """
    metric_kind:
      0=minkowski, 1=rindler, 2=schwarzschild_iso,
      3=flrw_x (static-in-x; keep if you like),
      4=flrw_powerlaw a(t)=a0(1+H0 t),
      5=flrw_exp      a(t)=a0*exp(H t)

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

    g, dg_dt, dg_dx = lax.switch(
        metric_kind,
        [minkowski_case, rindler_case, schwarzschild_case,
         minkowski_case,  # (placeholder for old #3 if you keep it)
         flrw_powerlaw_case, flrw_exp_case],
        operand=None,
    )
    g_inv  = jnp.linalg.inv(g)
    Gamma  = _christoffel_from_metric_tx(g, dg_dt, dg_dx)
    sqrtm  = jnp.sqrt(-jnp.linalg.det(g))
    return {"g": g, "g_inv": g_inv, "sqrt_minus_g": sqrtm, "Gamma": Gamma}