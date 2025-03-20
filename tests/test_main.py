import pytest
import sys
from unittest.mock import patch, MagicMock
from jaxincell.__main__ import main

@pytest.fixture
def mock_simulation():
    with patch('jaxincell.__main__.simulation') as mock_sim:
        mock_sim.return_value = {
            "position_electrons": MagicMock(),
            "velocity_electrons": MagicMock(),
            "electric_field": MagicMock(),
            "magnetic_field": MagicMock()
        }
        yield mock_sim

@pytest.fixture
def mock_load_parameters():
    with patch('jaxincell.__main__.load_parameters') as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())
        yield mock_load

@pytest.fixture
def mock_plot():
    with patch('jaxincell.__main__.plot') as mock_plot:
        yield mock_plot

def test_main_no_args(mock_simulation, mock_plot):
    """Test main function with no command line arguments."""
    test_args = ["__main__.py"]
    with patch.object(sys, 'argv', test_args):
        main()
        mock_simulation.assert_called_once()
        mock_plot.assert_called_once()

def test_main_with_args(mock_simulation, mock_load_parameters, mock_plot):
    """Test main function with command line arguments."""
    test_args = ["__main__.py", "input.toml"]
    with patch.object(sys, 'argv', test_args):
        main()
        mock_load_parameters.assert_called_once_with('input.toml')
        mock_simulation.assert_called_once()
        mock_plot.assert_called_once()

if __name__ == "__main__":
    pytest.main()