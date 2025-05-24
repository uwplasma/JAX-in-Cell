#!/usr/bin/env python3
import os
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import minimize
from contextlib import contextmanager
from jax.debug import print as jprint
from jax import grad, block_until_ready, value_and_grad, jit
from jaxincell import simulation, epsilon_0, load_parameters

input_file = os.path.join(os.path.dirname(__file__), "input.toml")
input_params, solver_params = load_parameters(input_file)
input_params['print_info'] = False

HIGH, LOW = 1500, 500
skip = 250
TR_delta_max, TR_max_iter = 0.30, 25
TR_gamma1, TR_gamma2, TR_sens = 0.15, 6.5, 1.0
TR_tol, TR_target = 1e-6, 1e-2
z0 = jnp.array([3.0], dtype=jnp.float32)

@contextmanager
def tic(msg: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    jprint("{msg}: {dt:.4f}s", msg=msg, dt=dt)
    
def energy_from_out(out):
    absE2 = jnp.sum(out["electric_field"] ** 2, axis=-1)
    intE2 = jnp.trapezoid(absE2, dx=out["dx"], axis=-1)
    return 0.5 * epsilon_0 * intE2

@partial(jit, static_argnames=["steps", "number_grid_points", "number_pseudoelectrons", "field_solver"])
def fidelity_obj_static_steps(ti, steps=1500, number_grid_points=100, number_pseudoelectrons=300, field_solver = 0):
    ti = jnp.squeeze(jnp.asarray(ti))
    jprint("[fidelity_obj] steps={} Ti/Te={}", steps, ti)
    params = input_params.copy()
    params["ion_temperature_over_electron_temperature"] = ti
    with tic("[fidelity_obj] Doing simulation_dynamic_ratio"):
        out = simulation(params,
            number_grid_points=number_grid_points,
            number_pseudoelectrons=number_pseudoelectrons,
            total_steps=steps,
            field_solver=field_solver)
        e = energy_from_out(out)
        mean_e = jnp.mean(e[-(steps - skip):])
        jprint("→ Mean energy (steps={steps}): {energy}", steps=steps, energy=mean_e)
        return mean_e

low_fid = partial(fidelity_obj_static_steps, steps=LOW)
high_fid = partial(fidelity_obj_static_steps, steps=HIGH)
value_and_grad_hi = jit(value_and_grad(high_fid))

def make_step(z, f_lo_z, f_hi_z, grad_hi_z):
    jprint("    Creating step function at z = {}, f_hi = {}, f_lo = {}", z, f_hi_z, f_lo_z)
    def loss(s):
        s_j = jnp.asarray(s)
        return low_fid(z + s_j) + (f_hi_z - f_lo_z) + jnp.dot(grad_hi_z, s_j)
    grad_loss = jit(grad(loss))
    def step_loss(s_np):
        val = loss(jnp.asarray(s_np)).block_until_ready()
        return float(val)
    def step_grad(s_np):
        g = grad_loss(jnp.asarray(s_np)).block_until_ready()
        return np.asarray(g, dtype=float)
    return step_loss, step_grad

def trust_region(z0):
    z = z0.copy()
    delta = 0.5 * TR_delta_max
    obj_vals, delta_hist = [], [delta]
    for k in range(TR_max_iter):
        jprint("\n[TR] Iter {k}  Δ={delta:.4f}", k=k, delta=delta)
        with tic("[TR] HF Value and grad eval"):
            f_hi, g_hi = block_until_ready(value_and_grad_hi(z))
            jprint("[TR] f_hi = {f}, grad = {g}", f=f_hi, g=g_hi)
        with tic("[TR] LF Value eval"):
            f_lo = low_fid(z)
            jprint("[TR] f_hi = {f}, f_lo = {l}", f=f_hi, l=f_lo)

        with tic("[TR] Step creation and L-BFGS-B"):
            step_loss, step_grad = make_step(z, f_lo, f_hi, g_hi)
            with tic("[TR] L-BFGS-B inner opt"):
                res = minimize(
                    step_loss,
                    np.zeros_like(z),
                    jac=step_grad,
                    bounds=[(-delta, delta)],
                    method="L-BFGS-B",
                    options={"maxiter": 3, "maxfun": 50, "ftol": 1e-9, "gtol": 1e-8},
                )
        s_new = res.x
        jprint("[TR] Step s_new = {s}", s=s_new)

        with tic("[TR] Evaluate f_hi at z + s"):
            f_new = float(high_fid(z + s_new).block_until_ready())

        act_red = f_hi - f_new
        pred_red = f_hi - step_loss(s_new)
        gamma = act_red / pred_red if pred_red > 0 else 0.0
        jprint("[TR] Actual Δf: {a:.4e}, Predicted: {p:.4e}, γ = {g:.4f}", a=act_red, p=pred_red, g=gamma)

        # Update delta
        if TR_gamma1 <= gamma <= TR_gamma2:
            delta = min(TR_delta_max, delta * (1 + TR_sens * (1 - abs(gamma - 1))))
        else:
            delta = max(TR_tol, delta * (1 - TR_sens * min(abs(gamma - TR_gamma1), abs(gamma - TR_gamma2))))
        delta_hist.append(delta)

        if act_red > 0:
            z = z + s_new
            jprint("[TR] Step accepted → z = {z}", z=z)
        else:
            jprint("[TR] Step rejected")

        obj_vals.append(f_hi)
        if f_new < TR_target:
            jprint("[TR] Target loss reached — stopping early.")
            break

    return z, obj_vals, delta_hist

if __name__ == "__main__":
    jprint("Starting Trust-Region Optimization")
    z_opt, obj_vals, delta_hist = trust_region(z0)
    jprint("\nOptimal Ti/Te {}  final loss {}", z_opt[0], obj_vals[-1])

    jprint("Plotting results...")
    plt.figure()
    plt.semilogy(obj_vals)
    plt.title("HF loss")
    plt.xlabel("outer iter")
    plt.grid(True)

    plt.figure()
    plt.plot(delta_hist)
    plt.title("Trust-region radius")
    plt.grid(True)
    plt.show()
