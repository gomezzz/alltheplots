from .plots_1d import plot_1d
from .plots_2d import plot_2d
from .plots_3d import plot_3d
from .plots_nd import plot_nd
from ..utils.type_handling import to_numpy
from ..utils.logger import logger


def plot(tensor, filename=None, dpi=100, show=True):
    """
    Plot a tensor based on its dimensionality using appropriate visualization methods.

    Currently supports:
    - 1D: Time-domain, Frequency-domain, Histogram plots in a 3×3 grid
    - 2D: Heatmap, Contour, 3D Surface, Distribution, and Cross-section plots in a 3×3 grid
    - 3D: Slice views, Projections, and Distribution analysis in a 3×3 grid
    - nD: Dimension reduction, Projections, and Distribution analysis in a 3×3 grid

    Parameters:
        tensor (array-like): The input tensor to plot.
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        show (bool): Whether to display the plot interactively. Default is True.

    Returns:
        matplotlib.figure.Figure: The figure object if filename is None and show=False,
                                  otherwise None
    """
    logger.info("Plotting tensor")

    # Convert to numpy array to determine dimensionality, using our robust conversion utility
    try:
        tensor_np = to_numpy(tensor)
        logger.debug(f"Converted tensor to numpy array of shape {tensor_np.shape}")
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise

    # Get the dimensionality (excluding dimensions of size 1)
    effective_dims = [dim for dim in tensor_np.shape if dim > 1]
    logger.debug(f"Effective dimensions: {effective_dims}")

    # Route to the appropriate plotting function based on dimensionality
    if (
        len(effective_dims) <= 1
    ):  # Handle 1D case (including scalars and arrays with singleton dimensions)
        logger.info("Detected 1D tensor, routing to plot_1d")
        return plot_1d(tensor_np, filename=filename, dpi=dpi, show=show)
    elif len(effective_dims) == 2:  # Handle 2D case
        logger.info("Detected 2D tensor, routing to plot_2d")
        return plot_2d(tensor_np, filename=filename, dpi=dpi, show=show)
    elif len(effective_dims) == 3:  # Handle 3D case
        logger.info("Detected 3D tensor, routing to plot_3d")
        return plot_3d(tensor_np, filename=filename, dpi=dpi, show=show)
    else:  # Handle nD case (4D and higher)
        logger.info(f"Detected {len(effective_dims)}D tensor, routing to plot_nd")
        return plot_nd(tensor_np, filename=filename, dpi=dpi, show=show)
