import pytest
import numpy as np
from alltheplots import plot
from pathlib import Path


@pytest.fixture
def tensor_1d():
    return np.random.randn(1000)


@pytest.mark.parametrize("filename", [None, "test_plot.png"])
@pytest.mark.parametrize("style", ["darkgrid", "whitegrid"])
def test_plot_1d(tensor_1d, filename, style, tmp_path):
    # Redirect output to a temporary directory if filename is specified
    if filename:
        filename = tmp_path / filename

    try:
        plot(tensor_1d, filename=filename, dpi=100, style=style)
    except Exception as e:
        pytest.fail(f"1D plotting failed with error: {e}")

    # If filename is set, ensure file is created
    if filename:
        assert Path(filename).exists(), f"Output file {filename} not created"
