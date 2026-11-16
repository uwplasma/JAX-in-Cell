"""A restart from disk against the uninterrupted run, and an openPMD round trip:
examples/2_intermediate/output_and_restart.py, run here and read back for its page."""
from common import record, run_example, savefig

run, fig = run_example("2_intermediate/output_and_restart")
savefig(fig, "output_and_restart")
r = run["results"]
record(restart_bit_identical="yes" if all(r["bit_identical"].values()) else "no",
       restart_openpmd_field=f"{r['openpmd']['field_difference']:.0e}",
       restart_openpmd_coordinates=f"{r['openpmd']['coordinate_difference']:.0e}")
