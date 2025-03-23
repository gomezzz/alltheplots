import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_autocorrelation_plot(tensor_np, ax=None, is_shared_x=False, max_lags=None):
    """
    Create an autocorrelation plot for 1D data.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to compute autocorrelation for
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        is_shared_x (bool): Whether this plot shares x-axis with other plots.
        max_lags (int, optional): Maximum number of lags to include. Default is None (uses N/2).

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating autocorrelation plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    try:
        # Center the data by subtracting the mean
        centered_data = tensor_np - np.mean(tensor_np)
        N = len(centered_data)

        # Set default max_lags if not provided
        if max_lags is None:
            max_lags = N // 2
        else:
            max_lags = min(max_lags, N - 1)  # Ensure max_lags doesn't exceed N-1

        # Compute autocorrelation using numpy's correlate function
        # 'same' mode returns the middle part of the correlation with same size as input
        autocorr = np.correlate(centered_data, centered_data, mode="full")

        # Normalize by the autocorrelation at lag 0
        autocorr = autocorr / np.max(autocorr)

        # Extract the positive lags (including zero lag)
        lags = np.arange(-N + 1, N)

        # Keep only lags up to max_lags
        center_idx = len(lags) // 2
        start_idx = center_idx - max_lags
        end_idx = center_idx + max_lags + 1
        plot_lags = lags[start_idx:end_idx]
        plot_autocorr = autocorr[start_idx:end_idx]

        # Plot the autocorrelation
        ax.plot(plot_lags, plot_autocorr, linewidth=0.8)

        # Add a horizontal line at y=0
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.3, linewidth=0.8)

        # Set plot labels
        ax.set_title("Autocorrelation")
        ax.set_xlabel("Lag" if not is_shared_x else "")
        ax.set_ylabel("Correlation")

        # Set limits
        ax.set_xlim(min(plot_lags), max(plot_lags))
        ax.set_ylim(-1.1, 1.1)

    except Exception as e:
        logger.error(f"Failed to compute or plot autocorrelation: {e}")
        ax.text(
            0.5,
            0.5,
            f"Autocorrelation Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Autocorrelation (Error)")

    return ax
