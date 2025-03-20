import os
import sys
import pytest
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
        main(sys.argv[1:])
        mock_simulation.assert_called_once()
        mock_plot.assert_called_once()

def test_main_with_args(mock_simulation, mock_load_parameters, mock_plot):
    """Test main function with command line arguments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_toml_path = str(os.path.join(current_dir, '..', 'examples', 'input.toml'))
    test_args = [input_toml_path]
    with patch.object(sys, 'argv', test_args):
        main(sys.argv)
        mock_load_parameters.assert_called_once_with(input_toml_path)
        mock_simulation.assert_called_once()
        mock_plot.assert_called_once()

if __name__ == "__main__":
    pytest.main()