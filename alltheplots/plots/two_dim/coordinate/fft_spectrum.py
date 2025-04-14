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

        # Check if we have enough data for FFT
        if len(y) < 10:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for FFT\n(need at least 10 points)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
        else:
            # First, check if x-data is monotonically increasing
            is_monotonic = np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)

            if is_monotonic:
                logger.debug("X data is monotonic, treating as time series")
                # Ensure x is sorted for time series analysis
                if not np.all(np.diff(x) > 0):
                    sort_idx = np.argsort(x)
                    x = x[sort_idx]
                    y = y[sort_idx]

                # Check if x values are approximately equidistant
                x_diffs = np.diff(x)
                is_equidistant = (
                    np.allclose(x_diffs, x_diffs[0], rtol=0.2) if len(x_diffs) > 0 else False
                )

                if not is_equidistant and len(x) > 10:
                    # Interpolate to equidistant grid for FFT
                    logger.debug("Interpolating non-equidistant data for FFT")
                    x_equidistant = np.linspace(x[0], x[-1], len(x))
                    from scipy.interpolate import interp1d

                    try:
                        interp_func = interp1d(x, y, kind="linear")
                        y_equidistant = interp_func(x_equidistant)

                        # Use interpolated y values for FFT
                        y_for_fft = y_equidistant
                        x_interval = x_equidistant[1] - x_equidistant[0]
                    except Exception as e:
                        logger.warning(f"Interpolation failed: {e}. Using original data.")
                        y_for_fft = y
                        x_interval = np.mean(x_diffs)
                else:
                    # Use original data for FFT
                    y_for_fft = y
                    x_interval = np.mean(x_diffs) if len(x_diffs) > 0 else 1.0

                # Remove mean (DC component) for better frequency analysis
                y_for_fft = y_for_fft - np.mean(y_for_fft)

                # Apply windowing to reduce spectral leakage
                window = signal.windows.hann(len(y_for_fft))
                y_windowed = y_for_fft * window

                # Calculate sampling rate from x interval
                fs = 1.0 / x_interval

                # Compute FFT with zero padding for better frequency resolution
                n_fft = max(
                    256, 2 ** int(np.ceil(np.log2(len(y_for_fft) * 2)))
                )  # Next power of 2, with padding
                fft_result = np.fft.rfft(y_windowed, n=n_fft)
                magnitude = np.abs(fft_result)

                # Normalize magnitude
                magnitude = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude

                # Compute frequency bins
                freq = np.fft.rfftfreq(n_fft, d=x_interval)

                # Skip very low frequencies which often dominate
                skip_idx = max(1, int(len(freq) * 0.01))

                # Plot magnitude spectrum (skip the lowest frequencies which often dominate)
                ax.plot(freq[skip_idx:], magnitude[skip_idx:], "b-")

                # Find and highlight dominant frequencies
                if len(magnitude[skip_idx:]) > 5:
                    # Find peaks in the magnitude spectrum
                    from scipy.signal import find_peaks

                    # More permissive peak detection
                    peaks, _ = find_peaks(magnitude[skip_idx:], height=0.05, distance=2)

                    if len(peaks) > 0:
                        # Adjust peak indices to account for skipped low frequencies
                        peaks = peaks + skip_idx

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
                if len(freq) > 10:
                    # Show first half of spectrum by default as higher frequencies often just have noise
                    max_freq_idx = len(freq) // 2
                    ax.set_xlim(0, freq[max_freq_idx])

                # Set title and labels
                title = "FFT Magnitude Spectrum (Time Series)"
                x_label = "Frequency (Hz)"

            else:
                # Not monotonic - try to interpret as phase plot or arbitrary coordinates
                logger.debug("X data is not monotonic, treating as arbitrary coordinates")

                # Option 1: Try FFT on y values directly (without considering x spacing)
                y_centered = y - np.mean(y)  # Remove mean

                # Apply window
                window = signal.windows.hann(len(y_centered))
                y_windowed = y_centered * window

                # Compute FFT
                n_fft = max(256, 2 ** int(np.ceil(np.log2(len(y_centered) * 2))))
                fft_result = np.fft.rfft(y_windowed, n=n_fft)
                magnitude = np.abs(fft_result)

                # Normalize
                magnitude = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude

                # Frequency bins (using normalized frequency)
                freq = np.arange(len(magnitude)) / len(magnitude)

                # Skip very low frequencies which often dominate
                skip_idx = max(1, int(len(freq) * 0.01))

                # Plot magnitude spectrum
                ax.plot(freq[skip_idx:], magnitude[skip_idx:], "b-")

                # Find and highlight dominant frequencies
                if len(magnitude[skip_idx:]) > 5:
                    from scipy.signal import find_peaks

                    peaks, _ = find_peaks(magnitude[skip_idx:], height=0.05, distance=2)

                    if len(peaks) > 0:
                        # Adjust peak indices
                        peaks = peaks + skip_idx

                        # Limit number of peaks
                        max_peaks = min(5, len(peaks))
                        top_peaks = peaks[np.argsort(magnitude[peaks])[-max_peaks:]]

                        # Highlight dominant frequencies
                        ax.plot(freq[top_peaks], magnitude[top_peaks], "ro", alpha=0.7)

                        # Annotate peak frequencies
                        for i, peak in enumerate(top_peaks):
                            if i < 3:  # limit annotations to top 3
                                ax.annotate(
                                    f"f = {freq[peak]:.3f}",
                                    xy=(freq[peak], magnitude[peak]),
                                    xytext=(5, 5 + i * 15),
                                    textcoords="offset points",
                                    fontsize=8,
                                    arrowprops=dict(arrowstyle="->", alpha=0.5),
                                )

                # Set x-axis limits
                if len(freq) > 10:
                    max_freq_idx = len(freq) // 2
                    ax.set_xlim(0, freq[max_freq_idx])

                # Set title and labels
                title = "FFT Magnitude Spectrum (Sequence)"
                x_label = "Normalized Frequency"

            # Add information about preprocessing
            info_text = f"n = {len(y)}"
            if is_monotonic:
                info_text += f"\nΔt = {x_interval:.3g}"

            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            # Set plot title and labels
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Normalized Magnitude")

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
