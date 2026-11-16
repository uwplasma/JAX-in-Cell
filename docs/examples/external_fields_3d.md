# External fields across the box

An external field given on an $(x, y, z)$ grid, gathered at each particle's $x$, $y$ and
$z$, checked against guiding-centre theory with one tenuous electron: the grad-$B$ drift
at three gradient lengths, and a magnetic mirror at three pitch angles.

```{figure} ../_static/figures/external_fields_3d.png
:width: 100%
:alt: Grad-B drift of the guiding centre, mirror bounce orbits, and the magnetic moment through the bounce

The example's own output, written to `external_fields_3d/figure.png` beside `run.json` and
`data.npz` (`docs/scripts/fig_external_fields.py` runs it for this page). Left: the relative
error of the guiding-centre drift in $\mathbf B = B_0(1 + y/L)\hat{\mathbf x}$ against
$v_\perp\rho/2L$, falling as $(\rho/L)^2$ (dashed), the finite-Larmor-radius order. Middle:
the position along a mirror, turning at the analytic $\pm L\cot\theta$ (dashed). Right: the
magnetic moment, constant to $2\times10^{-5}$ while the field at the electron changes by up
to a factor four (dotted, right axis).
```

## What is measured against what

$B_0 = 10$ mT, $v = 10^5$ m/s, $\Omega\Delta t = 0.1$. The grad-B drift is the slope of the
guiding centre $\mathbf R = \mathbf x + m\,\mathbf v\times\mathbf B/(qB^2)$ over 48 gyro-periods,
against $v_d = m v_\perp^2 |\mathbf B\times\nabla B|/(2|q|B^3) = v_\perp\rho/2L$:

| $L/\rho$ | measured (m/s) | analytic (m/s) | deviation | $(\rho/L)^2$ |
|---|---|---|---|---|
| 10 | 5057.5 | 5000.0 | +1.15 % | 1.00 % |
| 20 | 2507.0 | 2500.0 | +0.28 % | 0.25 % |
| 40 | 1250.9 | 1250.0 | +0.07 % | 0.06 % |

The deviation is the finite-Larmor-radius correction, second order in $\rho/L$: it falls by
four each time $L$ doubles, and does not change when $\Delta t$ is halved.

The mirror is $B_y = B_0(1 + y^2/L^2)$ with $L = 50\rho$, closed by the radial field
$-(r/2)\,\partial_y B_y$ that $\nabla\cdot\mathbf B = 0$ asks for. An electron of pitch angle
$\theta$ at the midplane turns where $B/B_0 = 1/\sin^2\theta$, at $y = L\cot\theta$:

| pitch angle | turning point $y/L$ | analytic $\cot\theta$ | $B_{\max}/B_0$ | spread of $\mu$ |
|---|---|---|---|---|
| $30^\circ$ | 1.7323 | 1.7321 | 4.00 | $7\times10^{-6}$ |
| $45^\circ$ | 1.0001 | 1.0000 | 2.00 | $1.5\times10^{-5}$ |
| $60^\circ$ | 0.5774 | 0.5774 | 1.33 | $2\times10^{-5}$ |

$\mu$ is {func}`~jaxincell.magnetic_moment`, $p_\perp^2/(2mB)$ relative to the external field
at the particle. The same three checks, at a smaller scale, are in
`tests/test_external_xyz.py`, with a fourth: a field that varies along $z$ goes over to the
uniform one linearly in the amplitude of the variation.

## Running it

```bash
python examples/2_intermediate/external_fields_3d.py            # ~40 s on one CPU core
python examples/2_intermediate/external_fields_3d.py --quick    # one length, one angle
```

The self-consistent fields still depend on $x$ alone; the grid in $y$ and $z$ is for what
is imposed ({doc}`../user_guide/external_fields`).
