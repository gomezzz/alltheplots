import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import fft
from ..utils.type_handling import to_numpy
from ..utils.logger import logger


def plot_1d(tensor, filename=None, dpi=100, style="darkgrid", show=True):
    """
    Generate a comprehensive visualization of a 1D tensor with four plots:
    1. Time-domain line plot
    2. Frequency-domain (FFT magnitude) plot
    3. Histogram with KDE overlay (normal scale)
    4. Histogram with KDE overlay (logarithmic scale)

    Parameters:
        tensor (array-like): The input 1D tensor to plot
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        style (str): The style of the plot. Default is "darkgrid".
        show (bool): Whether to display the plot interactively (True) or just return the figure (False).
                   Default is True except in test environments.

    Returns:
        matplotlib.figure.Figure: The figure containing the plots, or None if displayed
    """
    logger.info(f"Creating 1D plot with style '{style}'")

    # Convert to numpy array using our robust conversion utility
    try:
        tensor_np = to_numpy(tensor).flatten()
        logger.debug(f"Converted tensor to numpy array of shape {tensor_np.shape}")
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise

    # Set the style
    sns.set(style=style)
    logger.debug(f"Set seaborn style to '{style}'")

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    logger.debug("Created 2x2 subplot grid")
    fig.suptitle("1D Tensor Analysis", fontsize=16)

    # 1. Time-domain line plot (top-left)
    logger.debug("Creating time-domain line plot")
    axs[0, 0].plot(tensor_np, linewidth=1)
    axs[0, 0].set_title("Time Domain")
    axs[0, 0].set_xlabel("Index")
    axs[0, 0].set_ylabel("Value")

    # 2. Frequency-domain FFT plot (top-right)
    logger.debug("Creating FFT magnitude plot")
    # Compute FFT and frequency axis
    N = len(tensor_np)
    try:
        fft_values = fft.fft(tensor_np)
        fft_magnitudes = np.abs(fft_values[: N // 2]) / N  # Only take half (positive frequencies)
        freq = fft.fftfreq(N)[: N // 2]

        axs[0, 1].plot(freq, fft_magnitudes, linewidth=1)
        axs[0, 1].set_title("Frequency Domain (FFT Magnitude)")
        axs[0, 1].set_xlabel("Frequency")
        axs[0, 1].set_ylabel("Magnitude")
    except Exception as e:
        logger.error(f"Failed to compute or plot FFT: {e}")
        axs[0, 1].text(
            0.5,
            0.5,
            f"FFT Error: {str(e)}",
            ha="center",
            va="center",
            transform=axs[0, 1].transAxes,
        )
        axs[0, 1].set_title("Frequency Domain (Error)")

    # 3. Histogram with KDE overlay (normal scale) (bottom-left)
    logger.debug("Creating histogram with KDE")
    try:
        sns.histplot(tensor_np, kde=True, ax=axs[1, 0])
        axs[1, 0].set_title("Histogram with KDE")
        axs[1, 0].set_xlabel("Value")
        axs[1, 0].set_ylabel("Count")
    except Exception as e:
        logger.error(f"Failed to create histogram: {e}")
        axs[1, 0].text(
            0.5,
            0.5,
            f"Histogram Error: {str(e)}",
            ha="center",
            va="center",
            transform=axs[1, 0].transAxes,
        )
        axs[1, 0].set_title("Histogram (Error)")

    # 4. Histogram with KDE overlay (logarithmic scale) (bottom-right)
    logger.debug("Creating histogram with KDE (log scale)")
    try:
        sns.histplot(tensor_np, kde=True, ax=axs[1, 1])
        axs[1, 1].set_title("Histogram with KDE (Log Scale)")
        axs[1, 1].set_xlabel("Value")
        axs[1, 1].set_ylabel("Count")
        axs[1, 1].set_yscale("log")
    except Exception as e:
        logger.error(f"Failed to create log-scale histogram: {e}")
        axs[1, 1].text(
            0.5,
            0.5,
            f"Log Histogram Error: {str(e)}",
            ha="center",
            va="center",
            transform=axs[1, 1].transAxes,
        )
        axs[1, 1].set_title("Histogram Log Scale (Error)")

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
