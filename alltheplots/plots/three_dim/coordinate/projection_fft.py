import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ....utils.logger import logger


def create_projection_fft_plot(tensor_np, ax=None):
    """
    Create a 2D FFT magnitude plot of a selected projection of Nx3 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx3 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 2D FFT magnitude plot of projection")

    # Create ax if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    try:
        # Extract x, y, and z coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]
        z = tensor_np[:, 2]

        n_points = len(x)

        # Need enough points for a meaningful FFT
        if n_points < 20:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for FFT\n(need at least 20 points)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title("2D FFT (Insufficient Data)")
            return ax

        # Determine which projection to use based on variance
        # We'll choose the two axes with the highest variance
        variances = [np.var(x), np.var(y), np.var(z)]
        top_axes = np.argsort(variances)[-2:]  # Indices of two highest variance axes

        # Map indices to data and labels
        axes_data = [x, y, z]
        axes_labels = ["X", "Y", "Z"]

        # Extract the two dimensions with highest variance
        dim1_idx, dim2_idx = top_axes
        dim1_data = axes_data[dim1_idx]
        dim2_data = axes_data[dim2_idx]
        dim1_label = axes_labels[dim1_idx]
        dim2_label = axes_labels[dim2_idx]

        # Determine grid size for 2D FFT
        # We want a reasonable grid size that's also efficient for FFT
        grid_size = min(128, max(32, int(np.sqrt(n_points) * 2)))

        # Create a 2D histogram (grid) for the chosen projection
        hist_range = [
            [np.min(dim1_data), np.max(dim1_data)],
            [np.min(dim2_data), np.max(dim2_data)],
        ]

        hist, xedges, yedges = np.histogram2d(
            dim1_data, dim2_data, bins=grid_size, range=hist_range
        )

        # Apply window function to reduce spectral leakage
        window2d = np.outer(signal.windows.hann(grid_size), signal.windows.hann(grid_size))
        hist_windowed = hist * window2d

        # Compute 2D FFT
        fft2d = np.fft.fft2(hist_windowed)

        # Shift zero frequency component to center
        fft2d_shifted = np.fft.fftshift(fft2d)

        # Compute magnitude spectrum (log scale for better visualization)
        magnitude = np.abs(fft2d_shifted)
        epsilon = np.finfo(float).eps  # Small value to avoid log(0)
        log_magnitude = np.log10(magnitude + epsilon)

        # Normalize for better visualization
        log_magnitude = (log_magnitude - np.min(log_magnitude)) / (
            np.max(log_magnitude) - np.min(log_magnitude)
        )

        # Create the heatmap
        im = ax.imshow(
            log_magnitude,
            cmap="viridis",
            aspect="equal",
            origin="lower",
            extent=[-0.5, 0.5, -0.5, 0.5],  # Normalized frequency range
        )

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Log Magnitude (normalized)")

        # Find peaks in the FFT magnitude
        from scipy import ndimage

        # Use maximum filter to find local maxima
        neighborhood_size = max(3, grid_size // 20)
        data_max = ndimage.maximum_filter(log_magnitude, neighborhood_size)
        maxima = log_magnitude == data_max

        # Exclude border
        border = np.zeros_like(log_magnitude, dtype=bool)
        border[1:-1, 1:-1] = True
        maxima = maxima & border

        # Threshold for significant peaks (adjust as needed)
        threshold = np.mean(log_magnitude) + np.std(log_magnitude)
        significant = log_magnitude > threshold
        peaks = maxima & significant

        # Get coordinates of peaks
        peak_coords = np.where(peaks)
        n_peaks = len(peak_coords[0])

        # Plot top peaks (limit to avoid clutter)
        max_peak_count = 5
        if n_peaks > 0:
            # Get magnitudes for sorting
            peak_magnitudes = log_magnitude[peak_coords]

            # Sort by magnitude (descending)
            sorted_indices = np.argsort(-peak_magnitudes)

            # Limit to top peaks
            top_indices = sorted_indices[:max_peak_count]

            # Plot markers for top peaks
            peak_y = peak_coords[0][top_indices]
            peak_x = peak_coords[1][top_indices]

            # Convert to normalized frequency
            freq_x = (peak_x / grid_size) - 0.5
            freq_y = (peak_y / grid_size) - 0.5

            ax.plot(freq_x, freq_y, "ro", markersize=5, alpha=0.7)

            # Annotate the top 3 peaks
            for i in range(min(3, len(top_indices))):
                ax.annotate(
                    f"({freq_x[i]:.2f}, {freq_y[i]:.2f})",
                    (freq_x[i], freq_y[i]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=8,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7),
                )

        # Add labels and title
        ax.set_xlabel(f"Frequency ({dim1_label})")
        ax.set_ylabel(f"Frequency ({dim2_label})")
        ax.set_title(f"2D FFT of {dim1_label}-{dim2_label} Projection")

        # Add stats text
        stats_text = (
            f"Projection: {dim1_label}-{dim2_label}\n"
            f"Grid: {grid_size}×{grid_size}\n"
            f"Peaks: {n_peaks}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

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
