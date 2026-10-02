# Inferring a parameter from the dynamics

`examples/inference_two_stream.py` treats the two-stream instability as an inverse
problem: given the growth rate of the electrostatic energy, what was the drift speed?
The script defines a differentiable growth-rate estimator, checks it against a scan,
and then recovers the drift speed with a damped Newton iteration in $\log_{10}v_d$
driven by forward-mode derivatives.

The core of the script is the estimator, which turns one simulation into a single
differentiable number:

```{literalinclude} ../../examples/inference_two_stream.py
:language: python
:lines: 105-155
```

Three choices make the inverse problem well behaved:

* the fit window is a fixed fraction of the time series and does not depend on the
  drift speed, so that the estimator is a smooth function of its input;
* the derivative $d\hat\gamma/dv_d$ comes from `jax.jvp`, one forward pass per
  parameter, which needs no storage of the trajectory;
* steps in $\log_{10}v_d$ are clipped to keep the iteration in the range where the
  instability exists.

The script produces a figure of the energy history for the true, initial and
recovered drift speeds and prints the iteration history.
