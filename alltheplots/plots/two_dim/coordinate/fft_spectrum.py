import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ....utils.logger import logger


def create_fft_spectrum_plot(tensor_np, ax=None):
    """
    Create an FFT magnitude spectrum plot for Nx2 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx2 numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating FFT magnitude spectrum plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Extract x and y coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]

        # Check if data needs to be interpolated to equidistant points
        x_sorted_indices = np.argsort(x)
        x_sorted = x[x_sorted_indices]
        y_sorted = y[x_sorted_indices]

        # Check if x values are approximately equidistant
        x_diffs = np.diff(x_sorted)
        is_equidistant = np.allclose(x_diffs, x_diffs[0], rtol=0.1) if len(x_diffs) > 0 else False

        if not is_equidistant and len(x) > 3:
            # Interpolate to equidistant grid for FFT
            logger.debug("Interpolating non-equidistant data for FFT")
            x_equidistant = np.linspace(x_sorted[0], x_sorted[-1], len(x))
            from scipy.interpolate import interp1d

            try:
                interp_func = interp1d(x_sorted, y_sorted, kind="linear")
                y_equidistant = interp_func(x_equidistant)

                # Use interpolated y values for FFT
                y_for_fft = y_equidistant
                x_interval = x_equidistant[1] - x_equidistant[0]
            except Exception as e:
                logger.warning(f"Interpolation failed: {e}. Using original data.")
                y_for_fft = y_sorted
                x_interval = np.mean(x_diffs)
        else:
            # Use original sorted data for FFT
            y_for_fft = y_sorted
            x_interval = np.mean(x_diffs) if len(x_diffs) > 0 else 1.0

        # Handle very small dataset
        if len(y_for_fft) < 4:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for FFT\n(need at least 4 points)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
        else:
            # Calculate sampling rate from x interval
            fs = 1.0 / x_interval

            # Apply windowing to reduce spectral leakage
            window = signal.windows.hann(len(y_for_fft))
            y_windowed = y_for_fft * window

            # Compute FFT
            n_fft = max(256, 2 ** int(np.ceil(np.log2(len(y_for_fft)))))  # Next power of 2
            fft_result = np.fft.rfft(y_windowed, n=n_fft)
            magnitude = np.abs(fft_result)

            # Compute frequency bins
            freq = np.fft.rfftfreq(n_fft, d=x_interval)

            # Scaling for proper magnitude
            if len(freq) > 0:
                # Plot magnitude spectrum
                ax.plot(freq, magnitude, "b-")

                # Find and highlight dominant frequencies
                if len(magnitude) > 3:
                    # Find peaks in the magnitude spectrum
                    from scipy.signal import find_peaks

                    peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.1, distance=3)

                    if len(peaks) > 0:
                        # Limit number of peaks to avoid cluttering
                        max_peaks = min(5, len(peaks))
                        top_peaks = peaks[np.argsort(magnitude[peaks])[-max_peaks:]]

                        # Highlight dominant frequencies
                        ax.plot(freq[top_peaks], magnitude[top_peaks], "ro", alpha=0.7)

                        # Annotate peak frequencies
                        for i, peak in enumerate(top_peaks):
                            if i < 3:  # limit annotations to top 3
                                ax.annotate(
                                    f"{freq[peak]:.3f} Hz",
                                    xy=(freq[peak], magnitude[peak]),
                                    xytext=(5, 5 + i * 15),
                                    textcoords="offset points",
                                    fontsize=8,
                                    arrowprops=dict(arrowstyle="->", alpha=0.5),
                                )

                # Set x-axis limits to exclude high frequencies with little information
                if len(freq) > 5:
                    # Find frequency where magnitude drops below threshold
                    threshold = np.max(magnitude) * 0.05
                    significant_freqs = np.where(magnitude > threshold)[0]
                    if len(significant_freqs) > 0:
                        max_freq_idx = max(significant_freqs[-1] + 5, len(freq) // 4)
                        max_freq_idx = min(max_freq_idx, len(freq) - 1)
                        ax.set_xlim(0, freq[max_freq_idx])

                # Add magnitude scale
                if np.max(magnitude) > 0:
                    ax.set_ylim(0, np.max(magnitude) * 1.1)

            # Log scale can be helpful for wide dynamic range
            if (
                np.max(magnitude)
                / (np.min(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1)
                > 100
            ):
                ax.set_yscale("log")
                logger.debug("Using log scale for large magnitude range")

        # Set plot title and labels
        ax.set_title("FFT Magnitude Spectrum")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create FFT spectrum plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"FFT Spectrum Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("FFT Spectrum (Error)")

    return ax
