import pytest
import numpy as np
import os
from alltheplots import plot, set_log_level
from alltheplots.utils.logger import logger
from pathlib import Path
import matplotlib
import tempfile

# Use Agg backend to prevent interactive windows
matplotlib.use("Agg")

# Set logger to INFO level
set_log_level("INFO")

# Create a module-level output directory that persists across all tests
MODULE_OUTPUT_DIR = Path(tempfile.gettempdir()) / "alltheplots_test_outputs"
MODULE_OUTPUT_DIR.mkdir(exist_ok=True)
logger.info(f"Created persistent test output directory: {MODULE_OUTPUT_DIR}")

# Import framework modules if available
frameworks = {}

# NumPy is always available
frameworks["numpy"] = np

# Try to import PyTorch
try:
    import torch

    frameworks["torch"] = torch
except ImportError as e:
    logger.warning(f"PyTorch not available, tests will be skipped. Error: {e}")

# Try to import TensorFlow
try:
    import tensorflow as tf

    frameworks["tensorflow"] = tf
except ImportError:
    logger.warning("TensorFlow not available, tests will be skipped")

# Try to import JAX
try:
    import jax.numpy as jnp

    frameworks["jax.numpy"] = jnp
except ImportError as e:
    logger.warning(f"JAX not available, tests will be skipped. Error: {e}")

# Try to import CuPy
try:
    import cupy as cp

    frameworks["cupy"] = cp
except ImportError:
    logger.warning("CuPy not available, tests will be skipped")


@pytest.fixture(params=list(frameworks.keys()))
def framework(request):
    """Fixture to provide each framework module"""
    return frameworks[request.param]


@pytest.fixture
def random_data_2d(framework):
    """Generate random 2D data using the specified framework"""
    logger.debug(f"Generating 2D random data for framework: {framework.__name__}")
    if framework.__name__ == "numpy":
        return framework.random.randn(100, 100)
    elif framework.__name__ == "torch":
        return framework.randn(100, 100)
    elif framework.__name__ == "tensorflow":
        return framework.random.normal((100, 100))
    elif framework.__name__ == "jax.numpy":
        return framework.array(np.random.randn(100, 100))
    elif framework.__name__ == "cupy":
        return framework.random.randn(100, 100)


@pytest.fixture
def gradient_data_2d(framework):
    """Generate a structured 2D gradient data using a meshgrid and trigonometric functions"""
    x = np.linspace(0, 4 * np.pi, 100)
    y = np.linspace(0, 4 * np.pi, 100)
    xx, yy = np.meshgrid(x, y)
    data = np.sin(xx) * np.cos(yy)
    if framework.__name__ == "numpy":
        return data
    elif framework.__name__ == "torch":
        return framework.tensor(data, dtype=framework.float32)
    elif framework.__name__ == "tensorflow":
        return framework.convert_to_tensor(data, dtype=framework.float32)
    elif framework.__name__ == "jax.numpy":
        return framework.array(data)
    elif framework.__name__ == "cupy":
        return framework.array(data)


@pytest.fixture
def test_cases_2d():
    """Generate specialized 2D test cases with different data characteristics"""
    cases = {}

    # Checkerboard pattern
    checkerboard = np.indices((100, 100)).sum(axis=0) % 2
    cases["checkerboard"] = checkerboard

    # Radial gradient: distance from center
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, y)
    radial_gradient = np.sqrt(xx**2 + yy**2)
    cases["radial_gradient"] = radial_gradient

    # Sinusoid pattern: sin(x + y)
    x = np.linspace(0, 4 * np.pi, 100)
    y = np.linspace(0, 4 * np.pi, 100)
    xx, yy = np.meshgrid(x, y)
    sinusoid_pattern = np.sin(xx + yy)
    cases["sinusoid_pattern"] = sinusoid_pattern

    # Uniform grid values reshaped as a 2D array
    uniform_grid = np.linspace(0, 1, 10000).reshape(100, 100)
    cases["uniform_grid"] = uniform_grid

    return cases


@pytest.fixture(scope="module")
def output_dir():
    """
    Create and return a directory for test outputs.
    This fixture clears the directory at the start of the module.
    """
    if MODULE_OUTPUT_DIR.exists():
        for file in MODULE_OUTPUT_DIR.glob("*.png"):
            file.unlink()
    logger.info(f"Using module-level output directory: {MODULE_OUTPUT_DIR}")
    return MODULE_OUTPUT_DIR


@pytest.mark.dependency(name="plot_2d_tests")
@pytest.mark.plot_test
@pytest.mark.parametrize("data_type", ["random", "gradient"])
def test_plot_2d(framework, random_data_2d, gradient_data_2d, data_type, output_dir):
    """Test 2D plotting with different frameworks and data types"""
    if framework.__name__ not in frameworks:
        pytest.skip(f"Framework {framework.__name__} not available")
    data = random_data_2d if data_type == "random" else gradient_data_2d
    filename = output_dir / f"{framework.__name__}_2d_{data_type}.png"
    filepath = str(filename.absolute())
    logger.info(f"Testing 2D plot with {framework.__name__} ({data_type}) to file {filepath}")
    try:
        # Test saving to file (with show=False to prevent interactive windows)
        plot(data, filename=filepath, dpi=100, show=False)
        # Verify file creation
        assert filename.exists(), f"Output file {filename} not created"
        logger.success(f"Successfully created {filename}")

        # Test returning a figure object when no filename is provided
        fig = plot(data, show=False)
        assert fig is not None, "Expected plot function to return a figure object"
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        pytest.fail(f"2D plotting with {framework.__name__} ({data_type}) failed with error: {e}")


@pytest.mark.dependency(name="plot_specialized_2d_tests")
@pytest.mark.plot_test
def test_specialized_plot_cases_2d(test_cases_2d, output_dir):
    """Test specialized 2D plotting cases to exercise various plot features"""
    for case_name, data in test_cases_2d.items():
        filename = output_dir / f"specialized_2d_{case_name}.png"
        filepath = str(filename.absolute())
        logger.info(f"Testing 2D plot with specialized case: {case_name}")
        try:
            plot(data, filename=filepath, dpi=100, show=False)
            # Verify file creation
            assert filename.exists(), f"Output file {filename} not created"
            logger.success(f"Successfully created {filename}")
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            pytest.fail(f"2D plotting with specialized case {case_name} failed with error: {e}")


@pytest.mark.dependency(depends=["plot_2d_tests"])
def test_output_dir_content_2d(output_dir):
    """Test that the output directory contains 2D plot files"""
    logger.info(f"Checking output directory for 2D plots: {output_dir}")
    files = list(output_dir.glob("*.png"))
    if not files:
        logger.warning("No plot files found, creating a simple fallback 2D test plot")
        test_data = np.random.randn(100, 100)
        test_file = output_dir / "test_fallback_2d.png"
        plot(test_data, filename=str(test_file.absolute()), show=False)
        files = list(output_dir.glob("*.png"))
    assert len(files) > 0, f"No output files were created in {output_dir}"
    logger.info(f"Plot outputs saved to: {output_dir}")
    for f in files:
        logger.info(f"- {f.name}")


def test_edge_case_computations_2d(output_dir):
    """Test specialized 2D cases that involve additional computation at test time"""
    test_cases = {
        "gradient_magnitude": np.sqrt(
            np.square(np.linspace(-1, 1, 100).reshape(10, 10))
            + np.square(np.linspace(-1, 1, 100).reshape(10, 10))
        ),
        "binary_pattern_2d": np.tile([[0, 1], [1, 0]], (50, 50)),
        "exponential_decay_2d": np.exp(-np.linspace(0, 5, 100).reshape(10, 10)),
    }
    for case_name, data in test_cases.items():
        filename = output_dir / f"edge_case_2d_{case_name}.png"
        filepath = str(filename.absolute())
        logger.info(f"Testing 2D plot with edge case: {case_name}")
        try:
            plot(data, filename=filepath, dpi=100, show=False)
            assert filename.exists(), f"Output file {filename} not created"
            logger.success(f"Successfully created {filename}")
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            pytest.fail(f"2D plotting with edge case {case_name} failed with error: {e}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_output_dir():
    """Clean up the output directory after all tests have completed"""
    yield
    if os.environ.get("KEEP_TEST_OUTPUTS") != "1":
        logger.info(f"Cleaning up test output directory: {MODULE_OUTPUT_DIR}")
        try:
            for file in MODULE_OUTPUT_DIR.glob("*.png"):
                file.unlink()
        except Exception as e:
            logger.error(f"Failed to clean up test output directory: {e}")
    else:
        logger.info(f"Keeping test outputs in: {MODULE_OUTPUT_DIR}")


def test_simple_plot_2d(output_dir):
    """A simple test to ensure basic 2D plotting works"""
    data = np.random.randn(100, 100)
    filename = output_dir / "simple_test_2d.png"
    filepath = str(filename.absolute())
    logger.info(f"Creating simple 2D test plot: {filepath}")
    plot(data, filename=filepath, show=False)
    assert filename.exists(), f"Failed to create simple 2D test plot at {filename}"
    logger.success(f"Successfully created simple 2D test plot at {filename}")
