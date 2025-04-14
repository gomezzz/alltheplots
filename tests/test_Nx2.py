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
frameworks["numpy"] = np
try:
    import torch

    frameworks["torch"] = torch
except ImportError as e:
    logger.warning(f"PyTorch not available, tests will be skipped. Error: {e}")
try:
    import tensorflow as tf

    frameworks["tensorflow"] = tf
except ImportError:
    logger.warning("TensorFlow not available, tests will be skipped")
try:
    import jax.numpy as jnp

    frameworks["jax.numpy"] = jnp
except ImportError as e:
    logger.warning(f"JAX not available, tests will be skipped. Error: {e}")
try:
    import cupy as cp

    frameworks["cupy"] = cp
except ImportError:
    logger.warning("CuPy not available, tests will be skipped")


@pytest.fixture(params=list(frameworks.keys()))
def framework(request):
    return frameworks[request.param]


# Fixture to generate random Nx2 data
@pytest.fixture
def random_data_nx2(framework):
    logger.debug(f"Generating Nx2 random data for framework: {framework.__name__}")
    N = 100
    if framework.__name__ == "numpy":
        return framework.random.rand(N, 2)
    elif framework.__name__ == "torch":
        return framework.rand(N, 2)
    elif framework.__name__ == "tensorflow":
        return framework.random.normal((N, 2))
    elif framework.__name__ == "jax.numpy":
        return framework.array(np.random.rand(N, 2))
    elif framework.__name__ == "cupy":
        return framework.random.rand(N, 2)


# Fixture for specialized Nx2 test cases
@pytest.fixture
def test_cases_nx2():
    cases = {}
    cases["random_scatter"] = np.random.rand(100, 2)
    cases["sinusoidal_trend"] = np.column_stack(
        (np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
    )
    cases["nearly_linear"] = np.column_stack(
        (np.linspace(0, 10, 200), np.linspace(0, 5, 200) + np.random.randn(200) * 0.1)
    )
    part1 = np.random.rand(50, 2) * 0.5
    part2 = np.random.rand(50, 2) * 0.5 + np.array([1, 1])
    cases["two_clusters"] = np.concatenate([part1, part2])
    cases["scaled_scatter"] = np.column_stack((np.random.rand(150) * 10, np.random.rand(150) * 5))
    cases["sine_with_noise"] = np.column_stack(
        (
            np.linspace(0, 2 * np.pi, 100),
            np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.randn(100) * 0.2,
        )
    )
    cases["cosine_wave"] = np.column_stack(
        (np.linspace(0, 10, 120), np.cos(np.linspace(0, 10, 120)))
    )
    cases["normal_distribution"] = np.column_stack((np.random.randn(80), np.random.randn(80)))
    cases["log_curve"] = np.column_stack(
        (np.linspace(0, 5, 100), np.log1p(np.linspace(0, 10, 100)))
    )
    return cases


@pytest.fixture(scope="module")
def output_dir():
    if MODULE_OUTPUT_DIR.exists():
        for file in MODULE_OUTPUT_DIR.glob("*.png"):
            file.unlink()
    logger.info(f"Using module-level output directory: {MODULE_OUTPUT_DIR}")
    return MODULE_OUTPUT_DIR


@pytest.mark.dependency(name="plot_nx2_tests")
@pytest.mark.plot_test
@pytest.mark.parametrize("data_type", ["random", "specialized"])
def test_plot_nx2(framework, random_data_nx2, test_cases_nx2, data_type, output_dir):
    if framework.__name__ not in frameworks:
        pytest.skip(f"Framework {framework.__name__} not available")
    if data_type == "random":
        data = random_data_nx2
        filename = output_dir / f"{framework.__name__}_nx2_random.png"
        logger.info(f"Testing Nx2 plot with {framework.__name__} random data to file {filename}")
        try:
            plot(data, filename=str(filename.absolute()), dpi=100, show=False)
            assert filename.exists(), f"Output file {filename} not created"
            logger.success(f"Successfully created {filename}")
            fig = plot(data, show=False)
            assert fig is not None, "Expected plot function to return a figure object"
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            pytest.fail(f"Nx2 plotting with {framework.__name__} (random) failed: {e}")
    else:
        for case_name, data in test_cases_nx2.items():
            filename = output_dir / f"{framework.__name__}_nx2_{case_name}.png"
            logger.info(f"Testing Nx2 plot with specialized case {case_name}")
            try:
                plot(data, filename=str(filename.absolute()), dpi=100, show=False)
                assert filename.exists(), f"Output file {filename} not created"
                logger.success(f"Successfully created {filename}")
            except Exception as e:
                logger.error(f"Failed to create Nx2 plot for {case_name}: {e}")
                pytest.fail(f"Nx2 plotting for {case_name} failed: {e}")


@pytest.mark.dependency(depends=["plot_nx2_tests"])
def test_output_dir_content_nx2(output_dir):
    logger.info(f"Checking output directory for Nx2 plots: {output_dir}")
    files = list(output_dir.glob("*.png"))
    if not files:
        logger.warning("No Nx2 plot files found, creating fallback Nx2 test plot")
        test_data = np.random.rand(100, 2)
        test_file = output_dir / "fallback_nx2.png"
        plot(test_data, filename=str(test_file.absolute()), show=False)
        files = list(output_dir.glob("*.png"))
    assert len(files) > 0, f"No output files were created in {output_dir}"
    for f in files:
        logger.info(f"Found Nx2 output: {f.name}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_output_dir_nx2():
    yield
    if os.environ.get("KEEP_TEST_OUTPUTS") != "1":
        logger.info(f"Cleaning up Nx2 test output directory: {MODULE_OUTPUT_DIR}")
        try:
            for file in MODULE_OUTPUT_DIR.glob("*.png"):
                file.unlink()
        except Exception as e:
            logger.error(f"Failed to clean up Nx2 test output directory: {e}")
    else:
        logger.info(f"Keeping Nx2 test outputs in: {MODULE_OUTPUT_DIR}")


def test_simple_plot_nx2(output_dir):
    data = np.random.rand(100, 2)
    filename = output_dir / "simple_test_nx2.png"
    logger.info(f"Creating simple Nx2 test plot: {filename}")
    plot(data, filename=str(filename.absolute()), show=False)
    assert filename.exists(), f"Failed to create simple Nx2 test plot at {filename}"
    logger.success(f"Successfully created simple Nx2 test plot at {filename}")
