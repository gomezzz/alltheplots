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
        # Handle NaN or Inf values
        if np.any(~np.isfinite(tensor_np)):
            clean_data = tensor_np.copy()
            clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=0.0, neginf=0.0)
            logger.warning(
                "Found NaN or Inf values in data, replaced with zeros for autocorrelation"
            )
        else:
            clean_data = tensor_np

        # Center the data by subtracting the mean
        centered_data = clean_data - np.mean(clean_data)
        N = len(centered_data)

        # Set default max_lags if not provided
        if max_lags is None:
            max_lags = N // 2
        else:
            max_lags = min(max_lags, N - 1)  # Ensure max_lags doesn't exceed N-1

        # Compute autocorrelation using numpy's correlate function
        # 'full' mode returns the correlation at each lag
        autocorr = np.correlate(centered_data, centered_data, mode="full")

        # Handle all-zero or constant input data, which results in zero autocorrelation
        max_autocorr = np.max(np.abs(autocorr))
        if max_autocorr < 1e-10:  # Effectively zero
            logger.warning(
                "Autocorrelation is effectively zero, possibly constant or all-zero input"
            )
            # Set autocorr to zeros except at lag 0 (perfect correlation with itself)
            autocorr = np.zeros_like(autocorr)
            autocorr[len(autocorr) // 2] = 1.0
        else:
            # Normalize by the autocorrelation at lag 0 (maximum value)
            autocorr = autocorr / max_autocorr

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
