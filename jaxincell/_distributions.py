import jax.numpy as jnp
from jax import lax
from jax.random import PRNGKey, uniform, normal, split

from ._constants import speed_of_light
from ._particles import v_from_p, p_from_v

def sample_maxwell_boltzmann(key, N, vth_xyz, drift_xyz, flip_x, flip_y, flip_z):
    """
    Draw nonrelativistic velocities with anisotropic thermal speeds and drifts,
    then (optionally) flip the *entire* velocity by alternating sign per particle
    on selected axes (x/y/z), matching the original jnp.where(... v * (-1)**i ...).

    Args:
      key: PRNGKey
      N:   number of particles
      vth_xyz: (3,) thermal speeds [m/s] per axis
      drift_xyz: (3,) drift speeds [m/s] per axis
      flip_x, flip_y, flip_z: booleans (can be traced) indicating ± beams per axis

    Returns:
      V: (N, 3) velocities
    """
    vth_xyz   = jnp.asarray(vth_xyz)
    drift_xyz = jnp.asarray(drift_xyz)

    # Independent normals per axis
    kx, ky, kz = split(key, 3)
    vx = vth_xyz[0] / jnp.sqrt(2.0) * normal(kx, (N,))
    vy = vth_xyz[1] / jnp.sqrt(2.0) * normal(ky, (N,))
    vz = vth_xyz[2] / jnp.sqrt(2.0) * normal(kz, (N,))

    # Add drifts first (important!)
    V = jnp.stack([vx, vy, vz], axis=1) + drift_xyz[None, :]

    # Alternating sign pattern: +1, -1, +1, -1, ...
    # Use % instead of ** to avoid subtle dtype issues
    alt = jnp.where((jnp.arange(N) % 2) == 0, 1.0, -1.0)  # (N,)

    # Axis-wise flip masks (JAX-safe: jnp.where on traced booleans is fine)
    sx = jnp.where(jnp.asarray(flip_x), alt, 1.0)
    sy = jnp.where(jnp.asarray(flip_y), alt, 1.0)
    sz = jnp.where(jnp.asarray(flip_z), alt, 1.0)

    S = jnp.stack([sx, sy, sz], axis=1)  # (N,3)

    # Apply flips to the *entire* velocity (thermal + drift)
    V = V * S
    return V


# theta = kT / (m c^2)
def _normalize_with_eps(v, eps=1e-20):
    """Return unit vector and norm with a safe epsilon to avoid NaNs at v=0."""
    n = jnp.linalg.norm(v, axis=-1, keepdims=True)
    n_safe = jnp.maximum(n, eps)
    return v / n_safe, n[..., 0]  # (..,3) unit, (..) norm

def _broadcast_1d(x, N):
    """Broadcast x (scalar, (N,), or (N,1)) to shape (N,)."""
    x = jnp.asarray(x)
    if x.ndim == 0:
        x = jnp.broadcast_to(x, (N,))
    elif x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    else:
        x = jnp.broadcast_to(x, (N,))
    return x

def _unit_vectors(key, N: int):
    """Sample N isotropic unit vectors in R^3."""
    k1, k2 = split(key)
    u = uniform(k1, (N,), minval=-1.0, maxval=1.0)     # cos(theta)
    phi = 2.0 * jnp.pi * uniform(k2, (N,))
    s = jnp.sqrt(jnp.clip(1.0 - u * u, 0.0, 1.0))
    return jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), u], axis=-1)  # (N,3)

def sample_maxwell_juttner_p(key, N, m, theta, max_rounds: int = 16):
    """
    Maxwell–Jüttner sampler (isotropic) that returns momenta P ∈ R^{N×3}.

    Target (in p-space): f(p) ∝ p^2 exp(-γ/θ), where γ = sqrt(1 + (p/(m c))^2),
    θ = kT/(m c^2). We do rejection sampling on the magnitude p with a proposal
    Gamma(k=3, scale=a) (matching the p^2 factor), where a ≈ m c θ, and then
    draw an isotropic direction n̂ on the unit sphere and set P = p n̂.

    Args:
      key: JAX PRNGKey
      N:   number of particles
      m:   mass; scalar, (N,), or (N,1)
      theta: dimensionless temperature θ; scalar or (N,) or (N,1)
      max_rounds: number of proposal rounds (vectorized); 8–32 is usually fine

    Returns:
      P: momenta array, shape (N, 3)
    """
    # Broadcast m, theta to shape (N,) to simplify vectorized ops
    m     = _broadcast_1d(m, N)
    theta = _broadcast_1d(theta, N)

    # Split RNG: one stream for magnitudes, one for directions
    key_mag, key_dir = split(key)

    # Proposal scale a_i ~ m_i c θ_i. Avoid exact zeros for numerical safety.
    a = m * speed_of_light * jnp.maximum(theta, 1e-12)  # (N,)

    def one_round(carry, _):
        k, p_keep, accepted_mask, last_prop = carry
        k, k1, k2, k3, kU = split(k, 5)

        # Gamma(k=3) proposal for p via sum of 3 Exp(1) variates, then scale by a.
        u1 = uniform(k1, (N,))
        u2 = uniform(k2, (N,))
        u3 = uniform(k3, (N,))
        p_prop = a * (-jnp.log(u1) - jnp.log(u2) - jnp.log(u3))  # (N,)

        # γ(p) = sqrt(1 + (p/(m c))^2)
        gamma_prop = jnp.sqrt(1.0 + (p_prop / (m * speed_of_light)) ** 2)  # (N,)

        # Acceptance ratio (up to const): exp[-γ/θ + p/a]
        log_r = -gamma_prop / jnp.maximum(theta, 1e-12) + (p_prop / jnp.maximum(a, 1e-38))
        r     = jnp.exp(jnp.clip(log_r, a_min=-60.0, a_max=60.0))  # (N,)

        # Decide which proposals are accepted for the not-yet-accepted samples
        acc   = uniform(kU, (N,)) < r
        take  = (~accepted_mask) & acc

        p_keep      = jnp.where(take, p_prop, p_keep)
        accepted_mask = accepted_mask | take
        last_prop   = p_prop  # keep the latest proposal as fallback

        return (k, p_keep, accepted_mask, last_prop), None

    # Vectorized rejection: try up to `max_rounds` times and fill what we can
    init = (key_mag, jnp.zeros((N,)), jnp.zeros((N,), dtype=bool), jnp.zeros((N,)))
    (key_mag, p_mag, got, last_prop), _ = lax.scan(one_round, init, xs=None, length=max_rounds)

    # Fallback for any particles not accepted within max_rounds: use last proposal
    p_mag = jnp.where(got, p_mag, last_prop)  # (N,)

    # Draw isotropic directions and build the 3D momenta
    n_hat = _unit_vectors(key_dir, N)         # (N,3)
    P     = p_mag[:, None] * n_hat            # (N,3)
    return P

def sample_boosted_maxwell_juttner_p(key, N, m, theta, beta, max_rounds: int = 16):
    """
    Sample momenta from a Maxwell–Jüttner distribution in the fluid rest frame
    and Lorentz-boost them to the lab frame with drift velocity β = u/c.

    Args:
      key: PRNGKey
      N: number of particles
      m: particle mass (scalar or (N,) or (N,1)) *of the pseudoparticle*
      theta: rest-frame dimensionless temperature θ = kT/(m c^2) (scalar or (N,))
      beta: drift velocity / c. Either shape (3,) or per-particle (N,3).
      max_rounds: proposal rounds for the MJ sampler

    Returns:
      P_lab: momenta in lab frame, shape (N,3)
    """
    # Broadcast shapes
    m     = _broadcast_1d(m, N)                 # (N,)
    theta = _broadcast_1d(theta, N)             # (N,)
    beta  = jnp.asarray(beta)
    beta  = jnp.where(beta.ndim == 1, jnp.broadcast_to(beta, (N, 3)), beta)  # (N,3)

    # 1) Sample *rest-frame* isotropic momenta
    key_rf, = split(key, 1)
    P_rf = sample_maxwell_juttner_p(key_rf, N, m, theta, max_rounds)  # (N,3)

    # 2) Boost to lab along β direction (parallel/perpendicular decomposition)
    #    p_∥' = (p'·n)n, p_⊥' = p' - p_∥'
    n_hat, beta_mag = _normalize_with_eps(beta)          # (N,3), (N,)
    ppar_scalar = jnp.sum(P_rf * n_hat, axis=1)          # (N,)
    ppar_vec    = ppar_scalar[:, None] * n_hat           # (N,3)
    pperp_vec   = P_rf - ppar_vec                        # (N,3)

    # Rest-frame energy E' = sqrt((mc^2)^2 + (p' c)^2)
    mc2   = m * speed_of_light**2                        # (N,)
    Ec2   = jnp.sqrt(mc2**2 + (jnp.linalg.norm(P_rf, axis=1) * speed_of_light)**2)  # (N,)

    # γ = 1/sqrt(1-β^2); boost only the parallel component:
    # p_∥ = γ (p_∥' + β E'/c),   p_⊥ = p_⊥'
    gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta_mag**2, 1e-30))  # (N,)
    ppar_boost_scalar = gamma * (ppar_scalar + beta_mag * (Ec2 / speed_of_light))    # (N,)
    P_lab = pperp_vec + ppar_boost_scalar[:, None] * n_hat                           # (N,3)

    return P_lab

def _relativistic_distribution(params, Np: int):
    """
    Relativistic initialization: Maxwell–Jüttner in rest frame, then Lorentz-boost
    to impose near-c drift. Also honors velocity_plus_minus_* toggles to create
    counter-streaming beams per axis.
    """
    parameters, number_pseudoelectrons, masses, mass_electrons, mass_ions, weight, seed, theta_e, theta_i = params

    # Rest-frame temperatures (θ = kT/(m c^2))
    theta_e_eff = jnp.maximum(theta_e, 1e-10)
    theta_i_eff = jnp.maximum(theta_i, 1e-12)

    idx = jnp.arange(Np)
    alt = jnp.where((idx % 2) == 0, 1.0, -1.0)

    def signed_beta(beta_xyz, flip_x, flip_y, flip_z):
        # beta_xyz: (3,)
        sx = jnp.where(flip_x, alt, 1.0)
        sy = jnp.where(flip_y, alt, 1.0)
        sz = jnp.where(flip_z, alt, 1.0)
        S  = jnp.stack([sx, sy, sz], axis=1)  # (Np,3)
        return S * beta_xyz[None, :]

    beta_e = jnp.array([
        parameters["electron_drift_speed_x"] / speed_of_light,
        parameters["electron_drift_speed_y"] / speed_of_light,
        parameters["electron_drift_speed_z"] / speed_of_light,
    ])
    beta_i = jnp.array([
        parameters["ion_drift_speed_x"] / speed_of_light,
        parameters["ion_drift_speed_y"] / speed_of_light,
        parameters["ion_drift_speed_z"] / speed_of_light,
    ])

    beta_e_signed = signed_beta(beta_e,
                                parameters["velocity_plus_minus_electrons_x"],
                                parameters["velocity_plus_minus_electrons_y"],
                                parameters["velocity_plus_minus_electrons_z"])
    beta_i_signed = signed_beta(beta_i,
                                parameters["velocity_plus_minus_ions_x"],
                                parameters["velocity_plus_minus_ions_y"],
                                parameters["velocity_plus_minus_ions_z"])

    # Sample boosted MJ momenta
    pe = sample_boosted_maxwell_juttner_p(
        PRNGKey(seed + 20),
        Np,
        mass_electrons * weight,
        theta_e_eff,
        beta_e_signed,
        max_rounds=16,
    )
    pi = sample_boosted_maxwell_juttner_p(
        PRNGKey(seed + 21),
        Np,
        mass_ions * weight,
        theta_i_eff,
        beta_i_signed,
        max_rounds=16,
    )

    # Combine and convert to velocities
    momenta    = jnp.concatenate([pe, pi], axis=0)  # (2Np,3)
    velocities = v_from_p(momenta, masses)          # needed elsewhere

    return velocities, momenta

def _nonrelativistic_distribution(params, Np: int):
    parameters, number_pseudoelectrons, masses, mass_electrons, mass_ions, _, seed, _, _ = params

    # --- electrons ---
    e_vth_xyz = jnp.array([
        parameters["vth_electrons_over_c_x"] * speed_of_light,
        parameters["vth_electrons_over_c_y"] * speed_of_light,
        parameters["vth_electrons_over_c_z"] * speed_of_light,
    ])
    e_drift_xyz = jnp.array([
        parameters["electron_drift_speed_x"],
        parameters["electron_drift_speed_y"],
        parameters["electron_drift_speed_z"],
    ])

    electron_velocities = sample_maxwell_boltzmann(
        key=PRNGKey(seed + 7),  # one base key; helper splits internally
        N=Np,
        vth_xyz=e_vth_xyz,
        drift_xyz=e_drift_xyz,
        flip_x=parameters["velocity_plus_minus_electrons_x"],
        flip_y=parameters["velocity_plus_minus_electrons_y"],
        flip_z=parameters["velocity_plus_minus_electrons_z"],
    )

    # --- ions ---
    vth_ions_x = jnp.sqrt(jnp.abs(parameters["ion_temperature_over_electron_temperature_x"])) \
                * parameters["vth_electrons_over_c_x"] * speed_of_light \
                * jnp.sqrt(jnp.abs(mass_electrons / mass_ions))
    vth_ions_y = jnp.sqrt(jnp.abs(parameters["ion_temperature_over_electron_temperature_y"])) \
                * parameters["vth_electrons_over_c_y"] * speed_of_light \
                * jnp.sqrt(jnp.abs(mass_electrons / mass_ions))
    vth_ions_z = jnp.sqrt(jnp.abs(parameters["ion_temperature_over_electron_temperature_z"])) \
                * parameters["vth_electrons_over_c_z"] * speed_of_light \
                * jnp.sqrt(jnp.abs(mass_electrons / mass_ions))

    i_vth_xyz = jnp.array([vth_ions_x, vth_ions_y, vth_ions_z])
    i_drift_xyz = jnp.array([
        parameters["ion_drift_speed_x"],
        parameters["ion_drift_speed_y"],
        parameters["ion_drift_speed_z"],
    ])

    ion_velocities = sample_maxwell_boltzmann(
        key=PRNGKey(seed + 10),
        N=Np,
        vth_xyz=i_vth_xyz,
        drift_xyz=i_drift_xyz,
        flip_x=parameters["velocity_plus_minus_ions_x"],
        flip_y=parameters["velocity_plus_minus_ions_y"],
        flip_z=parameters["velocity_plus_minus_ions_z"],
    )

    # Combine and cap
    velocities = jnp.concatenate((electron_velocities, ion_velocities), axis=0)
    speed_limit = 0.99 * speed_of_light
    velocities = jnp.where(jnp.abs(velocities) >= speed_limit,
                        jnp.sign(velocities) * speed_limit,
                        velocities)
    momenta = p_from_v(velocities, masses)
    return velocities, momenta