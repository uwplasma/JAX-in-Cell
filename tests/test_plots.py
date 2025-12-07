# tests/test_plot.py

import pytest
import matplotlib

# Use non-interactive backend suitable for tests
matplotlib.use("Agg")

from jaxincell._simulation import simulation
from jaxincell._diagnostics import diagnostics
import jaxincell._plot as plot_mod


def _small_output_for_plot(direction="x", total_steps=20):
    """
    Run a small simulation and diagnostics to feed into the plotting function.
    direction == "x" or "xz" both work.
    """
    input_parameters = {
        "length": 0.01,
        "grid_points_per_Debye_length": 1.0,
        "print_info": False,
    }

    # For 2D direction ("xz") we make sure there is some temperature in z
    if "z" in direction:
        input_parameters.update(
            {
                "vth_electrons_over_c_z": 0.05,
                "ion_temperature_over_electron_temperature_z": 1.0,
            }
        )

    output = simulation(
        input_parameters,
        number_grid_points=16,
        number_pseudoelectrons=50,
        total_steps=total_steps,
        field_solver=0,
    )

    diagnostics(output)
    return output


def test_plot_single_direction_x(monkeypatch):
    """
    Ensure plot(output, 'x') runs without error and doesn't try to open a GUI.
    """
    output = _small_output_for_plot(direction="x", total_steps=20)

    # Avoid opening any windows
    monkeypatch.setattr(plot_mod.plt, "show", lambda: None)

    # Just check that this runs to completion
    plot_mod.plot(output, direction="x")


def test_plot_two_directions_xz(monkeypatch):
    """
    Ensure plot(output, 'xz') runs without error and exercises the
    second-direction phase-space and positions/B-field panel.
    """
    output = _small_output_for_plot(direction="xz", total_steps=15)

    # Avoid opening any windows
    monkeypatch.setattr(plot_mod.plt, "show", lambda: None)

    # This should exercise:
    # - electron/ion phase space in x
    # - electron/ion phase space in xz (velocity-velocity)
    # - positions overlaid on B-field density
    plot_mod.plot(output, direction="xz")
