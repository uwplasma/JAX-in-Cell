"""Minimize supplied RF energy for a cold-plasma energy-deposition benchmark.

A prescribed vector potential has an explicit external Maxwell current. Its
source work uses the total electric field and includes field cross-energy.
The ideal actuator supplies positive instantaneous work and dumps all returned
energy. This is not a runaway-suppression or wall-plug-power result.

Run with the JAX-in-Cell development environment. Optional JIC_RF_OUTPUT names
an output directory outside the checkout; otherwise only the figure is shown.
All physical and numerical settings are below. No production code is changed.
"""
import gc
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e,
                       epsilon_0 as eps, figure, mass_electron as m, mass_proton as mi,
                       speed_of_light as c)

jax.config.update("jax_enable_x64", True)

# Fixed target, actuator convention and coefficient bounds precede optimization.
design_steps, reference_steps, verification_steps = 512, (128, 256, 512, 1024), (512, 1024, 2048)
coefficient_max, target_margin = 2.5, 1.002
output_directory = Path(os.environ["JIC_RF_OUTPUT"]) if os.environ.get("JIC_RF_OUTPUT") else None
frequency, cycles, length, field_scale = 500e6, 4, 1., 1e3
omega, duration = 2 * np.pi * frequency, cycles / frequency
plasma_frequency = .8 * omega
density = eps * plasma_frequency**2 / (e**2 * (1 / m + 1 / mi))


def envelope(t, a):
    """A cubic Bernstein shape with a sin^4 ramp; return g, g' and g''."""
    s = jnp.clip(t / duration, 0., 1.)
    r, sn, cs = 1 - s, jnp.sin(jnp.pi * s), jnp.cos(jnp.pi * s)
    p = a[0] * r**3 + 3 * a[1] * s * r**2 + 3 * a[2] * s**2 * r + a[3] * s**3
    dp = 3 * ((a[1] - a[0]) * r**2 + 2 * (a[2] - a[1]) * s * r + (a[3] - a[2]) * s**2)
    ddp = 6 * ((a[2] - 2 * a[1] + a[0]) * r + (a[3] - 2 * a[2] + a[1]) * s)
    g, dg, ddg = sn**4, 4 * jnp.pi * sn**3 * cs, 4 * jnp.pi**2 * (3 * sn**2 * cs**2 - sn**4)
    inside = (t > 0) & (t < duration)
    return tuple(jnp.where(inside, value, 0.) for value in
                 (g * p, (dg * p + g * dp) / duration,
                  (ddg * p + 2 * dg * dp + g * ddp) / duration**2))


def mode(x, t, a, k=0.):
    """E, B and J_ext for A_y=(E_scale/omega) g(t) sin(k*x-omega*t).

    J_ext drives the prescribed field alone. The PIC plasma current already
    drives the self fields and must not be subtracted from this source again.
    A periodic nonzero k must equal 2*pi*integer/length.
    """
    g, dg, ddg = envelope(t, a)
    phase = k * x[:, 0] - omega * t
    si, co, zero = jnp.sin(phase), jnp.cos(phase), jnp.zeros_like(phase)
    E = field_scale * jnp.stack((zero, g * co - dg / omega * si, zero), axis=-1)
    B = field_scale * k / omega * jnp.stack((zero, zero, g * co), axis=-1)
    J = eps * field_scale / omega * jnp.stack(
        (zero, ((c**2 * k**2 - omega**2) * g + ddg) * si - 2 * omega * dg * co, zero), axis=-1)
    return E, B, J


def prescribed_fields(x, t, a):
    return mode(x, t, a)[:2]


def simulate(a, steps=design_steps, relativistic=False):
    """Uniform cold electron-proton response, with a charge-neutral quiet start."""
    n, cells = 16, 4
    x = jnp.zeros((n, 3)).at[:, 0].set(length * ((jnp.arange(n) + .5) / n - .5))
    zero = jnp.zeros_like(x)
    particles = (Species.electrons(n, density, x=x, v=zero),
                 Species.ions(n, density, vth=0., x=x, v=zero))
    sim = Simulation(Domain(length=length, cells=cells, time_step=duration / steps), particles,
                     Solver(relativistic=relativistic), prescribed_fields=prescribed_fields,
                     field_parameters=a)
    initial = sim.initial_state(jax.random.key(0))[0]
    return sim.run(steps, state=initial, store_particles=False, checkpoint=True), initial


def positive_linear(a, b, dt):
    """Integrate the positive part of linearly interpolated endpoint powers."""
    crossing = ((a < 0) & (b > 0)) | ((a > 0) & (b < 0))
    denominator = jnp.where(crossing, jnp.abs(b - a), 1.)
    crossing_area = (jnp.maximum(a, 0.)**2 + jnp.maximum(b, 0.)**2) / denominator
    return .5 * dt * jnp.where(crossing, crossing_area, jnp.maximum(a + b, 0.))


def source_ledger(out, initial, a):
    """Endpoint source quadrature for an explicitly nonrecovering current actuator.

    Requires every step of this complete four-cycle pulse, including its known
    initial state. E/J are sampled on faces, B on centers. This is an external
    diagnostic with an O(dt^2) quadrature error, not a new exact discrete PIC law.
    The piecewise-linear power integrates zero crossings before rectification.
    Positive instantaneous source work is supplied; returned energy is dumped.
    Cycle-averaged costs are reported separately to expose their window dependence.
    """
    t = jnp.concatenate((initial.time[None], out.t))
    Es = jnp.concatenate((initial.E[None], out.E))
    Bs = jnp.concatenate((initial.B[None], out.B))
    face = jnp.zeros((out.grid.size, 3)).at[:, 0].set(out.faces)
    center = face.at[:, 0].set(out.grid)
    Ep, _, Jp = jax.vmap(lambda time: mode(face, time, a))(t)
    Bp = jax.vmap(lambda time: mode(center, time, a)[1])(t)
    tm, dt = .5 * (t[:-1] + t[1:]), jnp.diff(t)
    Em = jax.vmap(lambda time: mode(face, time, a)[0])(tm)
    power = -out.dx * jnp.sum(Jp * (Es + Ep), axis=(1, 2))
    left, right = power[:-1], power[1:]
    increments = .5 * (left + right) * dt
    cycle_energy = jnp.sum(increments.reshape(cycles, -1), axis=1)
    self_energy = .5 * eps * out.dx * jnp.sum(Es**2 + c**2 * Bs**2, axis=(1, 2))
    prescribed_energy = .5 * eps * out.dx * jnp.sum(Ep**2 + c**2 * Bp**2, axis=(1, 2))
    cross_energy = eps * out.dx * jnp.sum(Es * Ep + c**2 * Bs * Bp, axis=(1, 2))
    kinetic = jnp.concatenate((jnp.sum(initial.work.kinetic)[None], jnp.sum(out.work.kinetic, axis=1)))
    source = jnp.concatenate((jnp.zeros(1), jnp.cumsum(increments)))
    total = kinetic + self_energy + prescribed_energy + cross_energy
    mechanical = jnp.sum(out.work.rr_work + out.work.collision_work, axis=1)
    mechanical = jnp.concatenate((jnp.sum(initial.work.rr_work + initial.work.collision_work)[None], mechanical))
    return dict(t=t, power=power, source=source, cycle_energy=cycle_energy, kinetic=kinetic,
                self_energy=self_energy, prescribed_energy=prescribed_energy, cross_energy=cross_energy,
                residual=total - total[0] - source - mechanical + mechanical[0], net=source[-1],
                supplied=jnp.sum(positive_linear(left, right, dt)),
                returned=jnp.sum(positive_linear(-left, -right, dt)),
                cycle_supplied=jnp.sum(jnp.maximum(cycle_energy, 0.)),
                cycle_returned=jnp.sum(jnp.maximum(-cycle_energy, 0.)),
                final_energy=kinetic[-1] + self_energy[-1],
                peak_prescribed_E=jnp.max(jnp.linalg.norm(Ep, axis=-1)),
                peak_drive_J=jnp.max(jnp.linalg.norm(Jp, axis=-1)),
                peak_source_power=jnp.max(power), min_source_power=jnp.min(power),
                exposure=jnp.sum(jnp.mean(jnp.sum(Em**2, axis=2), axis=1) * dt))


def metrics(a, steps=design_steps):
    out, initial = simulate(a, steps)
    budget = source_ledger(out, initial, a)
    return jnp.array([budget['supplied'], budget['final_energy'], budget['exposure'], budget['cycle_supplied']])


def cold_reference(t, scale=1.):
    """Independent five-frequency linear solution for a=scale*ones(4)."""
    t = np.asarray(t)
    offsets = np.arange(-2, 3)
    frequencies = omega + 2 * np.pi / duration * offsets
    amplitudes = field_scale * scale * np.array([1/16, -1/4, 3/8, -1/4, 1/16]) * frequencies / omega
    factor = -plasma_frequency**2 * amplitudes / (plasma_frequency**2 - frequencies**2)
    phase = t[..., None] * frequencies
    plasma_phase = plasma_frequency * t[..., None]
    Es = np.sum(factor * (np.cos(phase) - np.cos(plasma_phase)), axis=-1)
    dEs = np.sum(factor * (-frequencies * np.sin(phase) + plasma_frequency * np.sin(plasma_phase)), axis=-1)
    kinetic = .5 * eps / plasma_frequency**2 * dEs**2 * length
    return Es, kinetic + .5 * eps * Es**2 * length


assert jax.config.read('jax_enable_x64'), 'The reference gates require float64.'
start_time = time.perf_counter()
source_file = Path(__file__).resolve()
source_hashes = {source_file.name: hashlib.sha256(source_file.read_bytes()).hexdigest()}
if output_directory is not None:
    output_directory.mkdir(parents=True, exist_ok=True)
report = {'scope': 'Uniform cold-plasma source-accounting/control benchmark; not runaway cooling.',
          'source_hashes': source_hashes, 'jax': jax.__version__, 'devices': [str(x) for x in jax.devices()],
          'density_m-3': float(density), 'frequency_Hz': frequency, 'pulse_s': duration,
          'length_m': length, 'field_scale_V_m': field_scale, 'plasma_frequency_rad_s': plasma_frequency,
          'source_quadrature': 'Exact positive/negative area of piecewise-linear endpoint power.',
          'field_exposure_units': 'V^2 s/m^2',
          'cycle_average_s': 1 / frequency, 'particles': 32, 'cells': 4}


def save():
    report['elapsed_s'] = time.perf_counter() - start_time
    if output_directory is not None:
        (output_directory / 'rf_source_results.json').write_text(json.dumps(report, indent=2) + '\n')


def curl(gradient):
    return jnp.array([gradient[2, 1] - gradient[1, 2], gradient[0, 2] - gradient[2, 0],
                      gradient[1, 0] - gradient[0, 1]])


# Independent triangle areas and the isolated-zero endpoint sensitivity.
segments = jnp.array([[2., 4.], [-2., -4.], [-1., 3.], [3., -1.], [0., 3.], [0., 0.]])
areas = jax.vmap(lambda pair: positive_linear(pair[0], pair[1], 1.))(segments)
np.testing.assert_allclose(areas, [3., 0., 9/8, 9/8, 1.5, 0.], rtol=1e-14, atol=1e-14)
endpoint_gradient = jax.jacfwd(lambda pair: positive_linear(pair[0], pair[1], 1.))(segments[4])
np.testing.assert_allclose(endpoint_gradient, [.5, .5], rtol=1e-14, atol=1e-14)
report['positive_segment_check'] = {'areas': np.asarray(areas).tolist(),
                                    'isolated_zero_gradient': np.asarray(endpoint_gradient).tolist()}


# Independent differentiation of the fields checks the analytic source and curl signs.
a, x, t, k = jnp.array([.7, 1.3, .9, 1.1]), jnp.array([.23 * length, 0., 0.]), .37 * duration, 2 * np.pi / length


def point(position, time):
    return mode(position[None], time, a, k)


E, B, J = (v[0] for v in point(x, t))
dE, dB = (jax.jacfwd(lambda time: point(x, time)[i][0])(t) for i in range(2))
gradE, gradB = (jax.jacfwd(lambda position: point(position, t)[i][0])(x) for i in range(2))
faraday = np.linalg.norm(dB + curl(gradE)) / (field_scale * (abs(k) + omega / c))
ampere = np.linalg.norm(eps * (c**2 * curl(gradB) - dE) - J) / (eps * field_scale * omega)


def energy(time):
    return .5 * eps * sum(jnp.sum(v[0]**2) * f for v, f in zip(point(x, time)[:2], (1., c**2)))


def flux(position):
    return eps * c**2 * jnp.cross(*[v[0] for v in point(position, t)[:2]])


poynting = float(abs(jax.grad(energy)(t) + jnp.trace(jax.jacfwd(flux)(x)) + jnp.dot(J, E))
                 / (eps * field_scale**2 * omega))
assert max(faraday, ampere, poynting) < 2e-13
report['maxwell_residuals'] = {'faraday': float(faraday), 'ampere': float(ampere), 'poynting': poynting}
print('Analytic Maxwell source and local Poynting identities passed.', flush=True)

report['refinement'] = []
for count in reference_steps:
    out, initial = simulate(jnp.ones(4), count)
    out.validate()
    budget = source_ledger(out, initial, jnp.ones(4))
    reference_E, reference_energy = cold_reference(out.t)
    measured_E = np.asarray(out.E[:, :, 1]).mean(axis=1)
    final = float(budget['final_energy'])
    record = {'steps': count,
              'field_relative_error': float(np.max(abs(measured_E - reference_E)) / np.max(abs(reference_E))),
              'final_energy_relative_error': float(final / reference_energy[-1] - 1),
              'max_budget_residual_over_final_energy': float(jnp.max(jnp.abs(budget['residual'])) / final),
              'wrong_budget_without_cross_over_final_energy':
                  float(jnp.max(jnp.abs(budget['residual'] - budget['cross_energy'])) / final),
              'net_vs_prescribed_work_relative': float(budget['net'] / jnp.sum(out.work.prescribed_work[-1]) - 1),
              **{key: np.asarray(budget[key]).tolist() for key in
                 ('net', 'supplied', 'returned', 'cycle_supplied', 'cycle_returned',
                  'final_energy', 'exposure', 'cycle_energy')}}
    report['refinement'].append(record)
    print('Refinement', record, flush=True)
    save()
    del out, initial, budget
    jax.clear_caches()
    gc.collect()
errors = [r['field_relative_error'] for r in report['refinement']]
ratios = np.array(errors[:-1]) / errors[1:]
assert np.all((ratios > 3.8) & (ratios < 4.2)), ratios
assert report['refinement'][-1]['max_budget_residual_over_final_energy'] < 2e-4
report['field_refinement_ratios'] = ratios.tolist()

# Differentiate the actual PIC loop and the source ledger at fixed time resolution.
steps = design_steps


def value_with_aux(controls):
    value = metrics(controls, steps)
    return value, value


derivative = jax.jit(jax.jacfwd(value_with_aux, has_aux=True))
forward = jax.jit(lambda controls: metrics(controls, steps))


@lru_cache(maxsize=2)
def measured(controls):
    gradient, value = derivative(jnp.asarray(controls))
    return np.asarray(value), np.asarray(gradient)


point_a = np.array([.8, 1.1, .9, 1.2])
value, gradient = measured(tuple(point_a))
direction, h = np.array([1., -.5, .25, -.75]), 1e-4
signs = []
for controls in (point_a - h * direction, point_a + h * direction):
    out, initial = simulate(jnp.asarray(controls), steps)
    signs.append(np.sign(np.asarray(source_ledger(out, initial, jnp.asarray(controls))['power'])))
assert np.array_equal(*signs), 'Finite difference crosses a discrete positive-part kink; reduce its step.'
fd = np.asarray((forward(point_a + h * direction) - forward(point_a - h * direction)) / (2 * h))
ad = gradient @ direction
relative = np.abs(ad - fd) / np.maximum(np.abs(ad), np.abs(value) * 1e-5)
homogeneity = np.abs(gradient @ point_a - 2 * value) / np.abs(value)
assert max(relative) < 2e-7 and max(homogeneity) < 2e-12, (relative, homogeneity)
report['derivatives'] = {'ad': ad.tolist(), 'finite_difference': fd.tolist(), 'finite_difference_step': h,
                         'discrete_power_signs_unchanged': True,
                         'relative_errors': relative.tolist(), 'quadratic_homogeneity': homogeneity.tolist()}
print('PIC/source-ledger derivatives passed.', report['derivatives'], flush=True)
save()

target = float(cold_reference(duration)[1])
design_target = target_margin * target
base_value, _ = measured((1., 1., 1., 1.))
baseline = np.ones(4) * np.sqrt(design_target / base_value[1])
baseline_value, _ = measured(tuple(baseline))
normalizers = baseline_value.copy()


def objective(controls, index):
    v, g = measured(tuple(controls))
    return v[index] / normalizers[index], g[index] / normalizers[index]


def constraint(controls):
    v, g = measured(tuple(controls))
    return v[1] / design_target - 1, g[1] / design_target


report['optimization'] = {'target_J_m-2': target, 'design_target_J_m-2': design_target,
                          'bound': f'0 <= each Bernstein coefficient <= {coefficient_max}', 'steps': steps,
                          'cost': 'Instantaneous positive source work; all returned energy is dumped.',
                          'comparison_meter': 'Positive part after integrating each fixed carrier cycle, origin t=0.',
                          'peak_limits': 'No additional field/current bound; both demands are reported.'}
controls_by_name = {'flat_baseline': baseline}
for name, index in [('nonrecovering_source', 0), ('field_exposure', 2), ('cycle_meter', 3)]:
    result = minimize(lambda controls: objective(controls, index), baseline, jac=True, method='SLSQP',
                      bounds=[(0., coefficient_max)] * 4,
                      constraints={'type': 'ineq', 'fun': lambda controls: constraint(controls)[0],
                                   'jac': lambda controls: constraint(controls)[1]},
                      options={'maxiter': 60, 'ftol': 1e-10})
    controls_by_name[name] = result.x
    report['optimization'][name] = {'success': bool(result.success), 'message': result.message,
                                    'iterations': result.nit, 'controls': result.x.tolist()}
    print('Optimizer', name, report['optimization'][name], flush=True)
    assert result.success, result.message

report['frozen_controls'] = {}
for name, controls in controls_by_name.items():
    checked = []
    for count in verification_steps:
        out, initial = simulate(jnp.asarray(controls), count)
        out.validate()
        b = source_ledger(out, initial, jnp.asarray(controls))
        row = {'steps': count, **{key: np.asarray(b[key]).tolist() for key in
               ('supplied', 'returned', 'net', 'final_energy', 'exposure', 'cycle_supplied',
                'cycle_returned', 'cycle_energy',
                'peak_prescribed_E', 'peak_drive_J', 'peak_source_power', 'min_source_power')},
               'max_budget_residual_over_target': float(jnp.max(jnp.abs(b['residual'])) / target)}
        if count == verification_steps[-1]:
            cumulative = np.asarray(b['source'])
            stride = count // cycles
            shifted = {}
            for fraction in (0., .25, .5, .75):
                boundaries = sorted(set([0, count, *range(int(fraction * stride), count, stride)]))
                shifted[str(fraction)] = float(np.maximum(np.diff(cumulative[boundaries]), 0.).sum())
            row['one_cycle_supplied_by_origin_fraction'] = shifted
            row['two_cycle_supplied'] = float(np.maximum(np.diff(cumulative[::2 * stride]), 0.).sum())
            row['supplied_over_endpoint_lower_bound'] = row['supplied'] / target
        checked.append(row)
        assert row['final_energy'] >= target, (name, row)
        del out, initial, b
    report['frozen_controls'][name] = {'controls': controls.tolist(), 'verification': checked}
    print('Frozen control', name, checked[-1], flush=True)
    save()
fine = {name: data['verification'][-1] for name, data in report['frozen_controls'].items()}
report['improvements'] = {}
for name, key in [('nonrecovering_source', 'supplied'), ('field_exposure', 'exposure')]:
    gain = fine['flat_baseline'][key] - fine[name][key]
    refinement = sum(abs(data['verification'][-1][key] - data['verification'][-2][key])
                     for data in (report['frozen_controls']['flat_baseline'], report['frozen_controls'][name]))
    assert gain > 10 * refinement, (name, gain, refinement)
    report['improvements'][name] = {'relative_gain': gain / fine['flat_baseline'][key],
                                    'absolute_gain': gain, 'last_refinement_change_sum': refinement}
assert hashlib.sha256(source_file.read_bytes()).hexdigest() == source_hashes[source_file.name]
report['status'] = 'passed'
save()
print('All source-accounting/control checks passed; this benchmark deposits energy into a cold plasma.', flush=True)


fig, axes = figure(2, aspect=.75)
plot_t = jnp.linspace(0., duration, 400)
labels = {'flat_baseline': 'Flat envelope', 'nonrecovering_source': 'Source-energy optimum',
          'field_exposure': 'Exposure optimum'}
for name, label in labels.items():
    axes[0].plot(np.asarray(plot_t) * 1e9, np.asarray(envelope(plot_t, controls_by_name[name])[0]), label=label)
axes[0].set(xlabel='Time (ns)', ylabel='Envelope g', title='Four-coefficient prescribed pulse')
axes[0].legend(fontsize=8)
centers = np.arange(len(labels))
for offset, (key, label) in enumerate([('supplied', 'Supplied'), ('returned', 'Dumped'), ('net', 'Net')]):
    values = [fine[name][key] * 1e6 for name in labels]
    axes[1].bar(centers + .24 * (offset - 1), values, width=.24, label=label)
axes[1].set(xticks=centers, xticklabels=['Flat', 'Source cost', 'Exposure'],
            ylabel='Energy (microjoule/m²)', title='Same terminal-energy constraint')
axes[1].legend(fontsize=8)
fig.tight_layout()
if output_directory is not None:
    fig.savefig(output_directory / 'rf_source_control.png', dpi=150, pil_kwargs={'optimize': True})
plt.show()
