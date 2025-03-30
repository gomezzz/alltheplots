import matplotlib.pyplot as plt
from .plots_1d import plot_1d
from ..utils.type_handling import to_numpy
from ..utils.logger import logger


def plot(tensor, filename=None, dpi=100, show=True):
    """
    Plot a tensor based on its dimensionality using appropriate visualization methods.

    Currently supports:
    - 1D: Time-domain, Frequency-domain, Histogram, and Histogram with log scale

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
    else:
        logger.info(f"Detected {len(effective_dims)}D tensor, using basic plotting")
        # For now, just use the basic plotting for higher dimensions
        # Create a figure and axis
        fig, ax = plt.subplots()
        logger.debug("Created basic plot")

        # Flatten and plot the tensor
        ax.plot(tensor_np.flatten())
        logger.debug("Plotted flattened tensor")

        # Set labels and title
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title(f"{len(tensor_np.shape)}D Tensor Plot (Flattened)")

        # Save or return the plot
        if filename:
            logger.info(f"Saving plot to file: {filename}")
            try:
                plt.savefig(filename, dpi=dpi)
                logger.success(f"Plot saved to {filename}")
            except Exception as e:
                logger.error(f"Failed to save plot to {filename}: {e}")
                raise
            finally:
                plt.close(fig)
            return None
        elif show:
            logger.debug("Displaying plot interactively")
            plt.show()
            return None
        else:
            logger.debug("Returning figure without displaying")
            plt.close(fig)
            return fig
