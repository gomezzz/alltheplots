import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from ...utils.logger import logger


def create_fft_plot(tensor_np, ax=None, remove_dc=True):
    """
    Create a frequency-domain (FFT magnitude) plot for 1D data.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to compute FFT for
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        remove_dc (bool): Whether to remove the DC component (first frequency bin) from the plot. Default is True.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating FFT magnitude plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

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

        # Plot FFT magnitudes
        ax.plot(freqs[start_idx:], fft_magnitudes[start_idx:], linewidth=1)

        # Set plot labels
        ax.set_title("Frequency Domain (FFT Magnitude)")
        ax.set_xlabel("Frequency")
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
        )
        ax.set_title("Frequency Domain (Error)")

    return ax
