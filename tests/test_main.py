# tests/test_main.py

import pytest
from unittest.mock import patch, MagicMock

from jaxincell.__main__ import main


@pytest.fixture
def mock_simulation_class():
    with patch("jaxincell.__main__.Simulation") as mock_simulation:
        # Minimal dummy output; diagnostics is mocked so structure doesn't matter
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"dummy": "output"}
        mock_simulation.return_value = mock_instance
        yield mock_simulation


@pytest.fixture
def mock_load_parameters():
    with patch("jaxincell.__main__.load_parameters") as mock_load:
        mock_load.return_value = {
            "domain_parameters": {"length": 0.01, "number_grid_points": 16},
            "species_parameters": {
                "electrons": {"electrons0": {"number_pseudoparticles": 32}},
            },
        }
        yield mock_load


@pytest.fixture
def mock_plot():
    with patch("jaxincell.__main__.plot") as mock_plot:
        yield mock_plot


@pytest.fixture
def mock_diagnostics():
    with patch("jaxincell.__main__.diagnostics") as mock_diag:
        yield mock_diag


def test_main_no_args_uses_default_parameters(
    mock_simulation_class, mock_plot, mock_diagnostics, capsys
):
    """
    When no command-line arguments are provided, main() should:
      - print the default-parameters message
      - construct Simulation() with no arguments
      - call run()
      - call diagnostics(output) and plot(output)
    """
    # Call main with an explicit empty list instead of playing with sys.argv
    main([])

    mock_simulation_class.assert_called_once_with()
    mock_simulation_class.return_value.run.assert_called_once_with()

    # diagnostics and plot must be called on the output of simulation()
    output = mock_simulation_class.return_value.run.return_value
    mock_diagnostics.assert_called_once_with(output)
    mock_plot.assert_called_once_with(output)

    # Check that the default-parameters message was printed
    captured = capsys.readouterr()
    assert "Using standard input parameters instead of an input TOML file." in captured.out


@pytest.mark.skip(reason="scaffold only")
def test_main_ignores_extra_arguments_after_toml_path():
    """Test jaxincell.__main__.main argument handling.

    Cases to implement:
    - when more than one CLI argument is supplied, only cl_args[0] is passed to load_parameters.
    - Simulation, diagnostics, and plot still run exactly once.
    - no default-parameters message is printed in this branch.
    """


@pytest.mark.skip(reason="scaffold only")
def test_main_propagates_simulation_or_plot_errors():
    """Test jaxincell.__main__.main side-effect orchestration.

    Cases to implement:
    - Simulation construction errors are not swallowed.
    - run errors are not swallowed and diagnostics/plot are not called afterward.
    - plot errors are not swallowed after diagnostics has been called.
    """


def test_main_with_toml_argument(
    mock_simulation_class, mock_load_parameters, mock_plot, mock_diagnostics
):
    """
    When a TOML file path is provided, main() should:
      - call load_parameters(toml_path)
      - construct Simulation(parameters)
      - call run()
      - call diagnostics(output) and plot(output)
    """
    toml_path = "path/to/input.toml"

    # Call main with a single argument (the TOML path)
    main([toml_path])

    # Ensure load_parameters gets the correct path
    mock_load_parameters.assert_called_once_with(toml_path)

    mock_simulation_class.assert_called_once_with(mock_load_parameters.return_value)
    mock_simulation_class.return_value.run.assert_called_once_with()

    # diagnostics and plot must be called on the output of simulation()
    output = mock_simulation_class.return_value.run.return_value
    mock_diagnostics.assert_called_once_with(output)
    mock_plot.assert_called_once_with(output)
