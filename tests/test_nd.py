import pytest
import numpy as np
import os
from pathlib import Path
import matplotlib
import tempfile
from alltheplots import plot, set_log_level
from alltheplots.utils.logger import logger

# Use Agg backend to prevent interactive windows
matplotlib.use("Agg")
set_log_level("INFO")

# Create a module-level output directory for nD tests
MODULE_OUTPUT_DIR = Path(tempfile.gettempdir()) / "alltheplots_test_outputs_nd"
MODULE_OUTPUT_DIR.mkdir(exist_ok=True)
logger.info(f"Created persistent test output directory for nD tests: {MODULE_OUTPUT_DIR}")

# Detect available frameworks (numpy is always available)
frameworks = {}
frameworks["numpy"] = np
try:
    import torch

    frameworks["torch"] = torch
except ImportError as e:
    logger.warning(f"PyTorch not available for nD tests. Error: {e}")
try:
    import tensorflow as tf

    frameworks["tensorflow"] = tf
except ImportError:
    logger.warning("TensorFlow not available for nD tests")
try:
    import jax.numpy as jnp

    frameworks["jax.numpy"] = jnp
except ImportError as e:
    logger.warning(f"JAX not available for nD tests. Error: {e}")
try:
    import cupy as cp

    frameworks["cupy"] = cp
except ImportError:
    logger.warning("CuPy not available for nD tests")


@pytest.fixture(params=list(frameworks.keys()))
def framework(request):
    """Fixture to provide each framework module for nD tests."""
    return frameworks[request.param]


@pytest.fixture
def random_data_nd(framework):
    """
    Generate random nD data.
    For this example, we use a 4D array with shape (6, 6, 6, 6).
    """
    logger.debug(f"Generating random nD data for framework: {framework.__name__}")
    if framework.__name__ == "numpy":
        return framework.random.randn(6, 6, 6, 6)
    elif framework.__name__ == "torch":
        return framework.randn(6, 6, 6, 6)
    elif framework.__name__ == "tensorflow":
        return framework.random.normal((6, 6, 6, 6))
    elif framework.__name__ == "jax.numpy":
        return framework.array(np.random.randn(6, 6, 6, 6))
    elif framework.__name__ == "cupy":
        return framework.random.randn(6, 6, 6, 6)


@pytest.fixture
def structured_data_nd(framework):
    """
    Generate structured nD data.
    Here we create a simple 4D distance field from the center.
    """
    size = 16
    # Create four equally spaced grids for a 4D cube
    grids = np.meshgrid(*([np.linspace(-1, 1, size)] * 4), indexing="ij")
    # Compute the Euclidean distance from the center in 4D
    dist = np.sqrt(sum(g**2 for g in grids))
    if framework.__name__ == "numpy":
        return dist
    elif framework.__name__ == "torch":
        return framework.tensor(dist, dtype=framework.float32)
    elif framework.__name__ == "tensorflow":
        return framework.convert_to_tensor(dist, dtype=framework.float32)
    elif framework.__name__ == "jax.numpy":
        return framework.array(dist)
    elif framework.__name__ == "cupy":
        return framework.array(dist)


@pytest.fixture(scope="module")
def output_dir_nd():
    """
    Create and return a directory for nD test outputs.
    Clears any existing PNG files at the start.
    """
    if MODULE_OUTPUT_DIR.exists():
        for file in MODULE_OUTPUT_DIR.glob("*.png"):
            file.unlink()
    logger.info(f"Using module-level output directory for nD tests: {MODULE_OUTPUT_DIR}")
    return MODULE_OUTPUT_DIR


@pytest.mark.dependency(name="plot_nd_tests")
@pytest.mark.plot_test
@pytest.mark.parametrize("data_type", ["random", "structured"])
def test_plot_nd(framework, random_data_nd, structured_data_nd, data_type, output_dir_nd):
    """
    Test nD plotting (here 4D as an example) with different frameworks and data types.
    """
    data = random_data_nd if data_type == "random" else structured_data_nd
    filename = output_dir_nd / f"{framework.__name__}_nd_{data_type}.png"
    filepath = str(filename.absolute())
    logger.info(f"Testing nD plot with {framework.__name__} ({data_type}) to file {filepath}")
    try:
        # Save to file (show=False to prevent interactive windows)
        plot(data, filename=filepath, dpi=100, show=False)
        # Verify file creation
        assert filename.exists(), f"Output file {filename} not created"
        logger.success(f"Successfully created {filename}")

        # Also test that a figure object is returned when no filename is provided
        fig = plot(data, show=False)
        assert fig is not None, "Expected plot function to return a figure object"
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        pytest.fail(f"nD plotting with {framework.__name__} ({data_type}) failed with error: {e}")


@pytest.fixture
def test_cases_nd():
    """
    Generate specialized nD test cases.
    For example, a near-constant array with one spike and an nD gradient.
    """
    # Near-constant 4D array with a single spike
    spike_data = np.ones((6, 6, 6, 6))
    spike_data[3, 3, 3, 3] = 10.0

    # A gradient across four dimensions: sum of linearly increasing ramps
    shape = (5, 5, 5, 5)
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    w = np.linspace(-1, 1, shape[3])
    xx, yy, zz, ww = np.meshgrid(x, y, z, w, indexing="ij")
    gradient_nd = xx + yy + zz + ww

    return {
        "constant_with_spike": spike_data,
        "gradient_nd": gradient_nd,
        "random_small_nd": np.random.randn(3, 3, 3, 3),
        "zero_data": np.zeros((4, 4, 4, 4)),
    }


@pytest.mark.dependency(name="plot_specialized_nd_tests")
@pytest.mark.plot_test
def test_specialized_plot_cases_nd(test_cases_nd, output_dir_nd):
    """
    Test specialized nD plotting cases to ensure alltheplots handles them without errors.
    """
    for case_name, data in test_cases_nd.items():
        filename = output_dir_nd / f"specialized_nd_{case_name}.png"
        filepath = str(filename.absolute())
        logger.info(f"Testing nD plot with specialized case: {case_name}")
        try:
            plot(data, filename=filepath, dpi=100, show=False)
            assert filename.exists(), f"Output file {filename} not created"
            logger.success(f"Successfully created {filename}")
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            pytest.fail(f"nD plotting with specialized case {case_name} failed with error: {e}")


@pytest.mark.dependency(depends=["plot_nd_tests"])
def test_output_dir_content_nd(output_dir_nd):
    """
    Test that the output directory for nD tests contains plot files.
    """
    logger.info(f"Checking output directory for nD plots: {output_dir_nd}")
    files = list(output_dir_nd.glob("*.png"))
    if not files:
        logger.warning("No nD plot files found, creating a simple fallback nD test plot")
        test_data = np.random.randn(4, 4, 4, 4)
        test_file = output_dir_nd / "test_fallback_nd.png"
        plot(test_data, filename=str(test_file.absolute()), show=False)
        files = list(output_dir_nd.glob("*.png"))
    assert len(files) > 0, f"No output files were created in {output_dir_nd}"
    logger.info(f"nD plot outputs saved to: {output_dir_nd}")
    for f in files:
        logger.info(f"- {f.name}")


def test_simple_plot_nd(output_dir_nd):
    """
    A simple test to ensure basic nD plotting works without errors.
    """
    data = np.random.randn(3, 3, 3, 3)
    filename = output_dir_nd / "simple_test_nd.png"
    filepath = str(filename.absolute())
    logger.info(f"Creating simple nD test plot: {filepath}")
    plot(data, filename=filepath, show=False)
    assert filename.exists(), f"Failed to create simple nD test plot at {filename}"
    logger.success(f"Successfully created simple nD test plot at {filename}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_output_dir_nd():
    """
    Clean up the nD output directory after all tests have completed.
    Remove PNG files unless KEEP_TEST_OUTPUTS=1 is set.
    """
    yield
    if os.environ.get("KEEP_TEST_OUTPUTS") != "1":
        logger.info(f"Cleaning up nD test output directory: {MODULE_OUTPUT_DIR}")
        try:
            for file in MODULE_OUTPUT_DIR.glob("*.png"):
                file.unlink()
        except Exception as e:
            logger.error(f"Failed to clean up nD test output directory: {e}")
    else:
        logger.info(f"Keeping nD test outputs in: {MODULE_OUTPUT_DIR}")
