# tests/test_main.py

import pytest
from unittest.mock import patch, MagicMock

from jaxincell.__main__ import main


@pytest.fixture
def mock_simulation():
    with patch("jaxincell.__main__.simulation") as mock_sim:
        # Minimal dummy output; diagnostics is mocked so structure doesn't matter
        mock_sim.return_value = {"dummy": "output"}
        yield mock_sim


@pytest.fixture
def mock_load_parameters():
    with patch("jaxincell.__main__.load_parameters") as mock_load:
        input_params = {"length": 0.01}
        solver_params = {"number_grid_points": 16, "number_pseudoelectrons": 32}
        mock_load.return_value = (input_params, solver_params)
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
    mock_simulation, mock_plot, mock_diagnostics, capsys
):
    """
    When no command-line arguments are provided, main() should:
      - print the default-parameters message
      - call simulation() with no arguments
      - call diagnostics(output) and plot(output)
    """
    # Call main with an explicit empty list instead of playing with sys.argv
    main([])

    # simulation should have been called once with no arguments
    mock_simulation.assert_called_once_with()

    # diagnostics and plot must be called on the output of simulation()
    mock_diagnostics.assert_called_once_with(mock_simulation.return_value)
    mock_plot.assert_called_once_with(mock_simulation.return_value)

    # Check that the default-parameters message was printed
    captured = capsys.readouterr()
    assert "Using standard input parameters instead of an input TOML file." in captured.out


def test_main_with_toml_argument(
    mock_simulation, mock_load_parameters, mock_plot, mock_diagnostics
):
    """
    When a TOML file path is provided, main() should:
      - call load_parameters(toml_path)
      - call simulation(input_parameters, **solver_parameters)
      - call diagnostics(output) and plot(output)
    """
    toml_path = "path/to/input.toml"

    # Call main with a single argument (the TOML path)
    main([toml_path])

    # Ensure load_parameters gets the correct path
    mock_load_parameters.assert_called_once_with(toml_path)

    # Unpack what our fixture made load_parameters return
    input_params, solver_params = mock_load_parameters.return_value

    # simulation should be called with those unpacked solver parameters
    mock_simulation.assert_called_once_with(input_params, **solver_params)

    # diagnostics and plot must be called on the output of simulation()
    mock_diagnostics.assert_called_once_with(mock_simulation.return_value)
    mock_plot.assert_called_once_with(mock_simulation.return_value)
