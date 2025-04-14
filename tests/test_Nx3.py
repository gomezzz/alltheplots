import pytest
import numpy as np
import os
from alltheplots import plot, set_log_level
from alltheplots.utils.logger import logger
from pathlib import Path
import matplotlib
import tempfile
from IPython.display import Image

# Use Agg backend to prevent interactive windows
matplotlib.use("Agg")

# Set logger level to INFO
set_log_level("INFO")

# Create a module-level output directory for tests
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


# Fixture to generate random Nx3 data (each array has shape (N, 3))
@pytest.fixture
def random_data_nx3(framework):
    logger.debug(f"Generating Nx3 random data for framework: {framework.__name__}")
    N = 100
    if framework.__name__ == "numpy":
        return framework.random.rand(N, 3)
    elif framework.__name__ == "torch":
        return framework.rand(N, 3)
    elif framework.__name__ == "tensorflow":
        return framework.random.normal((N, 3))
    elif framework.__name__ == "jax.numpy":
        return framework.array(np.random.rand(N, 3))
    elif framework.__name__ == "cupy":
        return framework.random.rand(N, 3)


# Fixture for specialized Nx3 test cases
@pytest.fixture
def test_cases_nx3():
    cases = {}

    # 1. Basic random scatter (baseline)
    cases["random_scatter"] = np.random.rand(100, 3)

    # 2. 3D trajectory with smooth curve (sin & cos)
    cases["trajectory"] = np.column_stack(
        (np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100)))
    )

    # 3. Two or three clusters with multivariate normals
    cases["three_clusters"] = np.concatenate(
        [
            np.random.multivariate_normal(
                mean=[0, 0, 0], cov=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], size=50
            ),
            np.random.multivariate_normal(
                mean=[2, 2, 2], cov=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]], size=50
            ),
            np.random.multivariate_normal(
                mean=[-2, 2, -2], cov=[[0.15, 0, 0], [0, 0.15, 0], [0, 0, 0.15]], size=50
            ),
        ]
    )

    # 4. 3D spiral (helix)
    cases["spiral"] = np.column_stack(
        (
            np.linspace(0, 4 * np.pi, 200) * np.cos(np.linspace(0, 4 * np.pi, 200)),
            np.linspace(0, 4 * np.pi, 200) * np.sin(np.linspace(0, 4 * np.pi, 200)),
            np.linspace(0, 4 * np.pi, 200),
        )
    )

    # 5. Oblique noisy plane: z = 0.5 * x + 0.2 * y plus noise
    cases["noisy_plane"] = (
        lambda x, y: np.column_stack((x, y, 0.5 * x + 0.2 * y + np.random.randn(len(x)) * 0.5))
    )(np.random.rand(120) * 10, np.random.rand(120) * 10)

    # 6. Torus-like structure with two angles
    cases["torus"] = (
        lambda t, s: np.column_stack(
            ((3 + np.cos(s)) * np.cos(t), (3 + np.cos(s)) * np.sin(t), np.sin(s))
        )
    )(np.random.rand(200) * 2 * np.pi, np.random.rand(200) * 2 * np.pi)

    # 7. Uniformly distributed points in a cube (scaled)
    cases["cube"] = np.random.rand(150, 3) * 10

    # 8. Perfect diagonal line in 3D
    cases["diagonal"] = np.column_stack(
        (np.linspace(0, 10, 100), np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    )

    # 9. Normally distributed 3D points
    cases["normal_distribution"] = np.column_stack(
        (np.random.randn(100), np.random.randn(100), np.random.randn(100))
    )

    return cases


@pytest.fixture(scope="module")
def output_dir():
    if MODULE_OUTPUT_DIR.exists():
        for file in MODULE_OUTPUT_DIR.glob("*.png"):
            file.unlink()
    logger.info(f"Using module-level output directory: {MODULE_OUTPUT_DIR}")
    return MODULE_OUTPUT_DIR


@pytest.mark.dependency(name="plot_nx3_tests")
@pytest.mark.plot_test
@pytest.mark.parametrize("data_type", ["random", "specialized"])
def test_plot_nx3(framework, random_data_nx3, test_cases_nx3, data_type, output_dir):
    if framework.__name__ not in frameworks:
        pytest.skip(f"Framework {framework.__name__} not available")
    if data_type == "random":
        data = random_data_nx3
        filename = output_dir / f"{framework.__name__}_nx3_random.png"
        logger.info(f"Testing Nx3 plot with {framework.__name__} random data to file {filename}")
        try:
            plot(data, filename=str(filename.absolute()), dpi=100, show=False)
            assert filename.exists(), f"Output file {filename} not created"
            logger.success(f"Successfully created {filename}")
            fig = plot(data, show=False)
            assert fig is not None, "Expected plot function to return a figure object"
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            pytest.fail(f"Nx3 plotting with {framework.__name__} (random) failed: {e}")
    else:
        for case_name, data in test_cases_nx3.items():
            filename = output_dir / f"{framework.__name__}_nx3_{case_name}.png"
            logger.info(f"Testing Nx3 plot with specialized case {case_name}")
            try:
                plot(data, filename=str(filename.absolute()), dpi=100, show=False)
                assert filename.exists(), f"Output file {filename} not created"
                logger.success(f"Successfully created {filename}")
            except Exception as e:
                logger.error(f"Failed to create Nx3 plot for {case_name}: {e}")
                pytest.fail(f"Nx3 plotting for {case_name} failed: {e}")


@pytest.mark.dependency(depends=["plot_nx3_tests"])
def test_output_dir_content_nx3(output_dir):
    logger.info(f"Checking output directory for Nx3 plots: {output_dir}")
    files = list(output_dir.glob("*.png"))
    if not files:
        logger.warning("No Nx3 plot files found, creating fallback Nx3 test plot")
        test_data = np.random.rand(100, 3)
        test_file = output_dir / "fallback_nx3.png"
        plot(test_data, filename=str(test_file.absolute()), show=False)
        files = list(output_dir.glob("*.png"))
    assert len(files) > 0, f"No output files were created in {output_dir}"
    for f in files:
        logger.info(f"Found Nx3 output: {f.name}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_output_dir_nx3():
    yield
    if os.environ.get("KEEP_TEST_OUTPUTS") != "1":
        logger.info(f"Cleaning up Nx3 test output directory: {MODULE_OUTPUT_DIR}")
        try:
            for file in MODULE_OUTPUT_DIR.glob("*.png"):
                file.unlink()
        except Exception as e:
            logger.error(f"Failed to clean up Nx3 test output directory: {e}")
    else:
        logger.info(f"Keeping Nx3 test outputs in: {MODULE_OUTPUT_DIR}")


def test_simple_plot_nx3(output_dir):
    data = np.random.rand(100, 3)
    filename = output_dir / "simple_test_nx3.png"
    logger.info(f"Creating simple Nx3 test plot: {filename}")
    plot(data, filename=str(filename.absolute()), show=False)
    assert filename.exists(), f"Failed to create simple Nx3 test plot at {filename}"
    logger.success(f"Successfully created simple Nx3 test plot at {filename}")
