import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from ....utils.logger import logger


def create_fft2d_plot(tensor_np, ax=None, remove_dc=True, add_colorbar=True):
    """
    Create a 2D FFT magnitude plot with automatic log scaling and DC component removal.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to compute FFT for
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        remove_dc (bool): Whether to remove the DC component (center frequency) from the plot. Default is True.
        add_colorbar (bool): Whether to add a colorbar to the plot. Default is True.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 2D FFT magnitude plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Compute 2D FFT
        fft2d = fft.fft2(tensor_np)

        # Shift zero frequency to center
        fft2d_shifted = fft.fftshift(fft2d)

        # Compute magnitude spectrum
        magnitude = np.abs(fft2d_shifted)

        # Remove DC component if requested
        if remove_dc:
            center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            dc_region = 1  # Size of region around DC to remove
            magnitude[
                center_y - dc_region : center_y + dc_region + 1,
                center_x - dc_region : center_x + dc_region + 1,
            ] = np.nan

        # Apply log scaling (add small constant to avoid log(0))
        magnitude_log = np.log1p(magnitude)

        # Create frequency grids for plotting
        freq_y = fft.fftshift(fft.fftfreq(tensor_np.shape[0]))
        freq_x = fft.fftshift(fft.fftfreq(tensor_np.shape[1]))
        freq_x, freq_y = np.meshgrid(freq_x, freq_y)

        # Create the plot with a colormap
        im = ax.pcolormesh(freq_x, freq_y, magnitude_log, shading="auto", cmap="viridis")

        # Add colorbar
        if add_colorbar:
            plt.colorbar(im, ax=ax, label="Log Magnitude")

        # Set plot labels and title
        ax.set_title("2D FFT Magnitude")
        ax.set_xlabel("Frequency X")
        ax.set_ylabel("Frequency Y")

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle="--")

    except Exception as e:
        logger.error(f"Failed to create 2D FFT plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"2D FFT Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("2D FFT (Error)")

    return ax
