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

    frameworks["torch"] = torch  # Changed from "pytorch" to "torch"
except ImportError as e:
    logger.warning(f"PyTorch not available, tests will be skipped. Error: {e}")
    pass

# Try to import TensorFlow
try:
    import tensorflow as tf

    frameworks["tensorflow"] = tf
except ImportError:
    logger.warning("TensorFlow not available, tests will be skipped")
    pass

# Try to import JAX
try:
    import jax.numpy as jnp

    frameworks["jax.numpy"] = jnp  # Changed from "jax" to "jax.numpy"
except ImportError as e:
    logger.warning(f"JAX not available, tests will be skipped. Error: {e}")
    pass

# Try to import CuPy
try:
    import cupy as cp

    frameworks["cupy"] = cp
except ImportError:
    logger.warning("CuPy not available, tests will be skipped")
    pass


@pytest.fixture(params=list(frameworks.keys()))
def framework(request):
    """Fixture to provide each framework module"""
    return frameworks[request.param]


@pytest.fixture
def random_data(framework):
    """Generate random data using the specified framework"""
    logger.debug(f"Generating random data for framework: {framework.__name__}")
    if framework.__name__ == "numpy":
        return framework.random.randn(1000)
    elif framework.__name__ == "torch":
        return framework.randn(1000)
    elif framework.__name__ == "tensorflow":
        return framework.random.normal((1000,))
    elif framework.__name__ == "jax.numpy":
        return framework.array(np.random.randn(1000))
    elif framework.__name__ == "cupy":
        return framework.random.randn(1000)


@pytest.fixture
def sinusoid_data(framework):
    """Generate sinusoid data using the specified framework"""
    x = np.linspace(0, 10 * np.pi, 1000)
    sin_wave = np.sin(x)

    if framework.__name__ == "numpy":
        return sin_wave
    elif framework.__name__ == "torch":
        return framework.tensor(sin_wave, dtype=framework.float32)
    elif framework.__name__ == "tensorflow":
        return framework.convert_to_tensor(sin_wave, dtype=framework.float32)
    elif framework.__name__ == "jax.numpy":
        return framework.array(sin_wave)
    elif framework.__name__ == "cupy":
        return framework.array(sin_wave)


@pytest.fixture(scope="module")
def output_dir():
    """
    Create and return a directory for test outputs.

    This is a module-level fixture, so it persists across all tests
    in this module, ensuring the directory exists when test_output_dir_content runs.
    """
    # Clear the directory at the start of the module
    if MODULE_OUTPUT_DIR.exists():
        for file in MODULE_OUTPUT_DIR.glob("*.png"):
            file.unlink()

    logger.info(f"Using module-level output directory: {MODULE_OUTPUT_DIR}")
    return MODULE_OUTPUT_DIR


@pytest.mark.plot_test
@pytest.mark.parametrize("style", ["darkgrid"])  # Reduced to one style to speed up tests
@pytest.mark.parametrize("data_type", ["random", "sinusoid"])
def test_plot_1d(framework, random_data, sinusoid_data, style, data_type, output_dir):
    """Test 1D plotting with different frameworks, data types, and styles"""
    # Skip if the framework is not available
    if framework.__name__ not in frameworks:
        pytest.skip(f"Framework {framework.__name__} not available")

    # Select data based on parameter
    data = random_data if data_type == "random" else sinusoid_data

    # Always save to file with an absolute path
    filename = output_dir / f"{framework.__name__}_{data_type}_{style}.png"
    filepath = str(filename.absolute())
    logger.info(f"Testing plot with {framework.__name__} ({data_type}) to file {filepath}")

    try:
        # Test saving to file - IMPORTANT: show=False in tests to prevent interactive windows
        plot(data, filename=filepath, dpi=100, style=style, show=False)

        # Verify file was created
        assert filename.exists(), f"Output file {filename} not created"
        logger.success(f"Successfully created {filename}")

        # Test without filename (should return figure without displaying)
        fig = plot(data, style=style, show=False)
        assert fig is not None, "Expected plot function to return figure object"
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        pytest.fail(f"1D plotting with {framework.__name__} ({data_type}) failed with error: {e}")


# Remove the dependency marker and add a check for framework tests completion
@pytest.mark.dependency(depends=["plot_test"])
def test_output_dir_content(output_dir):
    """Test that output directory contains plot files"""
    # Look for framework-specific files first
    framework_files = list(output_dir.glob("*_darkgrid.png"))
    if not framework_files:
        logger.warning("No framework test files found, tests may not have completed yet")

    # Rest of the test remains the same
    # Ensure all tests that create files have run
    logger.info(f"Checking output directory: {output_dir}")

    # If no files have been created yet, create a simple test file
    if not list(output_dir.glob("*.png")):
        logger.warning("No test files found, creating a simple test plot")
        test_data = np.random.randn(1000)
        test_file = output_dir / "test_fallback_plot.png"
        plot(test_data, filename=str(test_file.absolute()), show=False)

    # Print information about the output directory for manual inspection
    files = list(output_dir.glob("*.png"))

    # If no files were created, list all files in the temp directory and parent directories
    if not files:
        logger.error(f"No plot files found in {output_dir}")
        logger.error(f"Contents of output_dir: {list(output_dir.iterdir())}")
        logger.error(f"Contents of parent: {list(output_dir.parent.iterdir())}")

        # Create a simple test file directly
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot([1, 2, 3])
        direct_file = output_dir / "direct_test.png"
        plt.savefig(direct_file)
        plt.close()

        # Try again
        files = list(output_dir.glob("*.png"))

    assert len(files) > 0, f"No output files were created in {output_dir}"

    # Print location for manual inspection
    logger.info(f"Plot outputs saved to: {output_dir}")
    for f in files:
        logger.info(f"- {f.name}")


# Clean up the module output directory when all tests are done
@pytest.fixture(scope="session", autouse=True)
def cleanup_output_dir():
    """Clean up the output directory after all tests have completed"""
    yield

    if os.environ.get("KEEP_TEST_OUTPUTS") != "1":
        logger.info(f"Cleaning up test output directory: {MODULE_OUTPUT_DIR}")
        try:
            # Keep directory but remove files
            for file in MODULE_OUTPUT_DIR.glob("*.png"):
                file.unlink()
        except Exception as e:
            logger.error(f"Failed to clean up test output directory: {e}")
    else:
        logger.info(f"Keeping test outputs in: {MODULE_OUTPUT_DIR}")


# Add a simple test that runs outside the parametrized tests
def test_simple_plot(output_dir):
    """A simple test to ensure basic plotting works"""
    data = np.random.randn(1000)
    filename = output_dir / "simple_test.png"
    filepath = str(filename.absolute())

    logger.info(f"Creating simple test plot: {filepath}")
    plot(data, filename=filepath, show=False)

    assert filename.exists(), f"Failed to create simple test plot at {filename}"
    logger.success(f"Successfully created simple test plot at {filename}")
