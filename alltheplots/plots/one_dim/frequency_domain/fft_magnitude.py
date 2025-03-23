import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from ....utils.logger import logger


def create_fft_magnitude_plot(tensor_np, ax=None, remove_dc=True, is_shared_x=False):
    """
    Create a frequency-domain (FFT magnitude) plot for 1D data with auto log-scale
    for magnitude if there's a large dynamic range.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to compute FFT for
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        remove_dc (bool): Whether to remove the DC component (first frequency bin) from the plot. Default is True.
        is_shared_x (bool): Whether this plot shares x-axis with other plots.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating FFT magnitude plot with potential auto log scale")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    try:
        # Compute FFT and frequency axis
        N = len(tensor_np)
        fft_values = fft.fft(tensor_np)

        # Only take half (positive frequencies)
        fft_magnitudes = np.abs(fft_values[: N // 2]) / N
        freqs = fft.fftfreq(N)[: N // 2]

        # Remove DC component if requested
        start_idx = 1 if remove_dc else 0
        if remove_dc:
            logger.debug("Removing DC component from FFT plot")

        # Check dynamic range to determine if log scale is needed
        # Only consider non-zero magnitudes to avoid -inf in log
        non_zero_mags = fft_magnitudes[fft_magnitudes > 0]
        if len(non_zero_mags) > 0:
            min_non_zero = np.min(non_zero_mags)
            max_val = np.max(fft_magnitudes)
            dynamic_range = max_val / min_non_zero if min_non_zero > 0 else 1

            # Use log scale if dynamic range is large
            use_log_scale = dynamic_range > 1000  # Threshold can be adjusted
        else:
            use_log_scale = False

        # Plot FFT magnitudes
        ax.plot(freqs[start_idx:], fft_magnitudes[start_idx:], linewidth=0.8)

        # Apply log scale if needed
        if use_log_scale:
            logger.debug(f"Using log scale for FFT magnitude (dynamic range: {dynamic_range:.1e})")
            ax.set_yscale("log")
            ax.set_title("FFT Magnitude (Log Scale)")
        else:
            ax.set_title("FFT Magnitude")

        # Set plot labels
        ax.set_xlabel("Frequency" if not is_shared_x else "")
        ax.set_ylabel("Magnitude")

    except Exception as e:
        logger.error(f"Failed to compute or plot FFT: {e}")
        ax.text(
            0.5,
            0.5,
            f"FFT Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("FFT Magnitude (Error)")

    return ax
