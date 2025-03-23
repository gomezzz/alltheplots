import seaborn as sns
import matplotlib.pyplot as plt
from ..utils.type_handling import to_numpy
from ..utils.logger import logger

# Import functions from the individual plot modules
from .one_dim.line_plot import create_line_plot
from .one_dim.fft_plot import create_fft_plot
from .one_dim.hist_kde_plot import create_hist_kde_plot


def plot_1d(tensor, filename=None, dpi=100, show=True, remove_dc=True):
    """
    Generate a comprehensive visualization of a 1D tensor with four plots:
    1. Time-domain line plot (or scatter plot for high-frequency data)
    2. Frequency-domain (FFT magnitude) plot (with optional DC component removal)
    3. Histogram with KDE overlay (normal scale)
    4. Histogram with KDE overlay (logarithmic scale)

    Parameters:
        tensor (array-like): The input 1D tensor to plot
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        show (bool): Whether to display the plot interactively (True) or just return the figure (False).
                   Default is True except in test environments.
        remove_dc (bool): Whether to remove the DC component from the FFT plot. Default is True.

    Returns:
        matplotlib.figure.Figure: The figure containing the plots, or None if displayed
    """
    logger.info("Creating 1D plot")

    # Convert to numpy array using our robust conversion utility
    try:
        tensor_np = to_numpy(tensor).flatten()
        logger.debug(f"Converted tensor to numpy array of shape {tensor_np.shape}")
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    logger.debug("Created 2x2 subplot grid")
    fig.suptitle("1D Tensor Analysis", fontsize=16)

    # 1. Time-domain line/scatter plot (top-left)
    create_line_plot(tensor_np, ax=axs[0, 0])

    # 2. Frequency-domain FFT plot (top-right)
    create_fft_plot(tensor_np, ax=axs[0, 1], remove_dc=remove_dc)

    # 3. Histogram with KDE overlay (normal scale) (bottom-left)
    create_hist_kde_plot(tensor_np, ax=axs[1, 0], log_scale=False)

    # 4. Histogram with KDE overlay (logarithmic scale) (bottom-right)
    create_hist_kde_plot(tensor_np, ax=axs[1, 1], log_scale=True)

    # Adjust layout for better spacing
    plt.tight_layout()
    logger.debug("Adjusted layout")

    # Save or display the plot
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
