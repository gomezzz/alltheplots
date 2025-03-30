import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ....utils.logger import logger


def create_psd_plot(tensor_np, ax=None, is_shared_x=False):
    """
    Create a Power Spectral Density (PSD) plot for 1D data.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to compute PSD for
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        is_shared_x (bool): Whether this plot shares x-axis with other plots.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating Power Spectral Density plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    try:
        # Compute power spectral density using scipy's welch method
        # This provides a more stable estimate than a direct FFT
        freqs, psd = signal.welch(tensor_np, fs=1.0, nperseg=min(256, len(tensor_np)))

        # Check dynamic range to determine if log scale is needed
        # Only consider non-zero values to avoid -inf in log
        non_zero_psd = psd[psd > 0]
        if len(non_zero_psd) > 0:
            min_non_zero = np.min(non_zero_psd)
            max_val = np.max(psd)
            dynamic_range = max_val / min_non_zero if min_non_zero > 0 else 1

            # Use log scale if dynamic range is large
            use_log_scale = dynamic_range > 1000  # Threshold can be adjusted
        else:
            use_log_scale = False

        # Plot PSD with thinner line
        ax.plot(freqs, psd, linewidth=0.8)

        # Apply log scale if needed
        if use_log_scale:
            logger.debug(f"Using log scale for PSD (dynamic range: {dynamic_range:.1e})")
            ax.set_yscale("log")
            ax.set_title("Power Spectral Density \n (Log Scale)")
        else:
            ax.set_title("Power Spectral Density")

        # Set plot labels
        ax.set_xlabel("Frequency" if not is_shared_x else "")
        ax.set_ylabel("Power/Frequency")

    except Exception as e:
        logger.error(f"Failed to compute or plot PSD: {e}")
        ax.text(
            0.5,
            0.5,
            f"PSD Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Power Spectral Density (Error)")

    return ax
