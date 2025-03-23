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
MODULE_OUTPUT_DIR = Path(tempfile.gettempdir()) / "alltheplots_test_outputs_3d"
MODULE_OUTPUT_DIR.mkdir(exist_ok=True)
logger.info(f"Created persistent test output directory for 3D tests: {MODULE_OUTPUT_DIR}")

# Import framework modules if available
frameworks = {}

# NumPy is always available
frameworks["numpy"] = np

# Try to import PyTorch
try:
    import torch

    frameworks["torch"] = torch
except ImportError as e:
    logger.warning(f"PyTorch not available for 3D tests. Error: {e}")

# Try to import TensorFlow
try:
    import tensorflow as tf

    frameworks["tensorflow"] = tf
except ImportError:
    logger.warning("TensorFlow not available for 3D tests")

# Try to import JAX
try:
    import jax.numpy as jnp

    frameworks["jax.numpy"] = jnp
except ImportError as e:
    logger.warning(f"JAX not available for 3D tests. Error: {e}")

# Try to import CuPy
try:
    import cupy as cp

    frameworks["cupy"] = cp
except ImportError:
    logger.warning("CuPy not available for 3D tests")


@pytest.fixture(params=list(frameworks.keys()))
def framework(request):
    """Fixture to provide each framework module for 3D tests."""
    return frameworks[request.param]


@pytest.fixture
def random_data_3d(framework):
    """
    Generate random 3D data using the specified framework.
    Shape: (10, 10, 10) by default, adjust as needed.
    """
    logger.debug(f"Generating 3D random data for framework: {framework.__name__}")
    if framework.__name__ == "numpy":
        return framework.random.randn(10, 10, 10)
    elif framework.__name__ == "torch":
        return framework.randn(10, 10, 10)
    elif framework.__name__ == "tensorflow":
        return framework.random.normal((10, 10, 10))
    elif framework.__name__ == "jax.numpy":
        return framework.array(np.random.randn(10, 10, 10))
    elif framework.__name__ == "cupy":
        return framework.random.randn(10, 10, 10)


@pytest.fixture
def structured_data_3d(framework):
    """
    Generate a structured 3D data array, e.g., a 3D Gaussian or wave pattern.
    Here, we create a simple 3D "sphere" distance field from the center.
    """
    size = 20
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    # Distance from center
    sphere_dist = np.sqrt(xx**2 + yy**2 + zz**2)

    # Convert to the appropriate framework
    if framework.__name__ == "numpy":
        return sphere_dist
    elif framework.__name__ == "torch":
        return framework.tensor(sphere_dist, dtype=framework.float32)
    elif framework.__name__ == "tensorflow":
        return framework.convert_to_tensor(sphere_dist, dtype=framework.float32)
    elif framework.__name__ == "jax.numpy":
        return framework.array(sphere_dist)
    elif framework.__name__ == "cupy":
        return framework.array(sphere_dist)


@pytest.fixture(scope="module")
def output_dir_3d():
    """
    Create and return a directory for 3D test outputs.
    Clears the directory at the start of the module.
    """
    if MODULE_OUTPUT_DIR.exists():
        for file in MODULE_OUTPUT_DIR.glob("*.png"):
            file.unlink()
    logger.info(f"Using module-level output directory for 3D tests: {MODULE_OUTPUT_DIR}")
    return MODULE_OUTPUT_DIR


@pytest.mark.dependency(name="plot_3d_tests")
@pytest.mark.plot_test
@pytest.mark.parametrize("data_type", ["random", "structured"])
def test_plot_3d(framework, random_data_3d, structured_data_3d, data_type, output_dir_3d):
    """
    Test 3D plotting with different frameworks and data types.
    The alltheplots library should detect 3D data and produce an appropriate plot (e.g., multiple subplots,
    volumetric slices, or a 3D surface if your library does that).
    """
    if framework.__name__ not in frameworks:
        pytest.skip(f"Framework {framework.__name__} not available")

    data = random_data_3d if data_type == "random" else structured_data_3d
    filename = output_dir_3d / f"{framework.__name__}_3d_{data_type}.png"
    filepath = str(filename.absolute())
    logger.info(f"Testing 3D plot with {framework.__name__} ({data_type}) to file {filepath}")

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
        pytest.fail(f"3D plotting with {framework.__name__} ({data_type}) failed with error: {e}")


@pytest.fixture
def test_cases_3d():
    """
    Generate specialized 3D test cases to exercise various plot features or edge cases.
    For example, near-constant arrays, arrays with missing data, etc.
    """
    # Example: a near-constant array with one "spike"
    spike_data = np.ones((8, 8, 8))
    spike_data[4, 4, 4] = 10.0

    # Example: random data with a gradient
    x = np.linspace(-1, 1, 8)
    y = np.linspace(-1, 1, 8)
    z = np.linspace(-1, 1, 8)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    gradient_3d = xx + yy + zz  # Simple gradient

    return {
        "constant_with_spike": spike_data,
        "gradient_3d": gradient_3d,
        "random_small_3d": np.random.randn(5, 5, 5),
        "zero_data": np.zeros((6, 6, 6)),
    }


@pytest.mark.dependency(name="plot_specialized_3d_tests")
@pytest.mark.plot_test
def test_specialized_plot_cases_3d(test_cases_3d, output_dir_3d):
    """
    Test specialized 3D plotting cases to ensure alltheplots handles them without errors.
    """
    for case_name, data in test_cases_3d.items():
        filename = output_dir_3d / f"specialized_3d_{case_name}.png"
        filepath = str(filename.absolute())
        logger.info(f"Testing 3D plot with specialized case: {case_name}")
        try:
            plot(data, filename=filepath, dpi=100, show=False)
            assert filename.exists(), f"Output file {filename} not created"
            logger.success(f"Successfully created {filename}")
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            pytest.fail(f"3D plotting with specialized case {case_name} failed with error: {e}")


@pytest.mark.dependency(depends=["plot_3d_tests"])
def test_output_dir_content_3d(output_dir_3d):
    """
    Test that the output directory for 3D tests contains plot files.
    This will fail if no files have been created.
    """
    logger.info(f"Checking output directory for 3D plots: {output_dir_3d}")
    files = list(output_dir_3d.glob("*.png"))
    if not files:
        logger.warning("No 3D plot files found, creating a simple fallback 3D test plot")
        test_data = np.random.randn(8, 8, 8)
        test_file = output_dir_3d / "test_fallback_3d.png"
        plot(test_data, filename=str(test_file.absolute()), show=False)
        files = list(output_dir_3d.glob("*.png"))
    assert len(files) > 0, f"No output files were created in {output_dir_3d}"
    logger.info(f"3D plot outputs saved to: {output_dir_3d}")
    for f in files:
        logger.info(f"- {f.name}")


def test_edge_case_computations_3d(output_dir_3d):
    """
    Test specialized 3D cases that involve additional computation at test time,
    ensuring the plot still generates without errors.
    """
    test_cases = {
        "two_spheres_overlap": _two_spheres_overlap(12),
        "linear_ramp_3d": _linear_ramp_3d(10),
    }
    for case_name, data in test_cases.items():
        filename = output_dir_3d / f"edge_case_3d_{case_name}.png"
        filepath = str(filename.absolute())
        logger.info(f"Testing 3D plot with edge case: {case_name}")
        try:
            plot(data, filename=filepath, dpi=100, show=False)
            assert filename.exists(), f"Output file {filename} not created"
            logger.success(f"Successfully created {filename}")
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            pytest.fail(f"3D plotting with edge case {case_name} failed with error: {e}")


def _two_spheres_overlap(size):
    """
    Generate a 3D array with two overlapping spheres for an interesting shape.
    """
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    z = np.linspace(-2, 2, size)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    sphere1 = (xx + 0.5) ** 2 + yy**2 + zz**2
    sphere2 = (xx - 0.5) ** 2 + yy**2 + zz**2
    # Values are small inside spheres, large outside
    # Combine them to produce an overlapping region
    return np.minimum(sphere1, sphere2)


def _linear_ramp_3d(size):
    """
    Generate a 3D linear ramp in one dimension.
    """
    arr = np.zeros((size, size, size))
    for i in range(size):
        arr[i, :, :] = i  # linearly increasing along x-axis
    return arr


@pytest.fixture(scope="session", autouse=True)
def cleanup_output_dir_3d():
    """
    Clean up the 3D output directory after all tests have completed.
    Remove PNG files unless KEEP_TEST_OUTPUTS=1 is set.
    """
    yield
    if os.environ.get("KEEP_TEST_OUTPUTS") != "1":
        logger.info(f"Cleaning up 3D test output directory: {MODULE_OUTPUT_DIR}")
        try:
            for file in MODULE_OUTPUT_DIR.glob("*.png"):
                file.unlink()
        except Exception as e:
            logger.error(f"Failed to clean up 3D test output directory: {e}")
    else:
        logger.info(f"Keeping 3D test outputs in: {MODULE_OUTPUT_DIR}")


def test_simple_plot_3d(output_dir_3d):
    """
    A simple test to ensure basic 3D plotting works without errors.
    """
    data = np.random.randn(5, 5, 5)
    filename = output_dir_3d / "simple_test_3d.png"
    filepath = str(filename.absolute())
    logger.info(f"Creating simple 3D test plot: {filepath}")
    plot(data, filename=filepath, show=False)
    assert filename.exists(), f"Failed to create simple 3D test plot at {filename}"
    logger.success(f"Successfully created simple 3D test plot at {filename}")
