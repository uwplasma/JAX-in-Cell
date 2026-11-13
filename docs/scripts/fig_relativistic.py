"""Relativistic two-stream instability with the relativistic Boris pusher on and off:
examples/2_intermediate/relativistic_two_stream.py, run here and read back, so that the page's
figure and numbers are the example's own run."""
from common import record, run_example, savefig

run, fig = run_example("2_intermediate/relativistic_two_stream")
savefig(fig, "relativistic_two_stream")
record(**run["results"]["measurements"])
