import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ....utils.logger import logger


def create_spectral_analysis_plot(tensor_np, ax=None, method="psd"):
    """
    Create a power spectral density (PSD) or autocorrelation plot for Nx2 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx2 numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        method (str): Analysis method, either "psd" (power spectral density) or "autocorr" (autocorrelation).
            Default is "psd".

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug(f"Creating spectral analysis plot using {method} method")

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
            # Interpolate to equidistant grid
            logger.debug("Interpolating non-equidistant data for spectral analysis")
            x_equidistant = np.linspace(x_sorted[0], x_sorted[-1], len(x))
            from scipy.interpolate import interp1d

            try:
                interp_func = interp1d(x_sorted, y_sorted, kind="linear")
                y_equidistant = interp_func(x_equidistant)

                # Use interpolated y values for analysis
                y_for_analysis = y_equidistant
                x_interval = x_equidistant[1] - x_equidistant[0]
            except Exception as e:
                logger.warning(f"Interpolation failed: {e}. Using original data.")
                y_for_analysis = y_sorted
                x_interval = np.mean(x_diffs)
        else:
            # Use original sorted data for analysis
            y_for_analysis = y_sorted
            x_interval = np.mean(x_diffs) if len(x_diffs) > 0 else 1.0

        # Handle very small dataset
        if len(y_for_analysis) < 10:
            ax.text(
                0.5,
                0.5,
                f"Insufficient data for {method.upper()}\n(need at least 10 points)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
        else:
            # Determine which method to use (adaptively choose based on data characteristics)
            if method == "autocorr" or len(y_for_analysis) < 30:
                # Autocorrelation is better for smaller datasets
                method = "autocorr"

                # Calculate and plot autocorrelation
                from scipy.signal import correlate

                # Remove mean for proper autocorrelation
                y_centered = y_for_analysis - np.mean(y_for_analysis)

                # Calculate autocorrelation
                autocorr = correlate(y_centered, y_centered, mode="full")
                autocorr = autocorr[len(autocorr) // 2 :]  # Keep only positive lags

                # Normalize
                autocorr = autocorr / autocorr[0]

                # Generate lag values
                lags = np.arange(len(autocorr)) * x_interval

                # Plot autocorrelation
                ax.plot(lags, autocorr, "b-")

                # Add horizontal lines at 0, 95% confidence bounds
                conf_level = 1.96 / np.sqrt(len(y_for_analysis))
                ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
                ax.axhline(y=conf_level, color="r", linestyle="--", alpha=0.3)
                ax.axhline(y=-conf_level, color="r", linestyle="--", alpha=0.3)

                # Find and annotate peaks (excluding lag 0)
                from scipy.signal import find_peaks

                peak_indices, _ = find_peaks(autocorr[1:], height=max(conf_level, 0.1))
                peak_indices = peak_indices + 1  # Adjust for slicing

                if len(peak_indices) > 0:
                    # Limit to top 3 peaks to avoid cluttering
                    top_peaks = peak_indices[np.argsort(autocorr[peak_indices])[-3:]]
                    ax.plot(lags[top_peaks], autocorr[top_peaks], "ro", ms=5, alpha=0.7)

                    # Annotate with lag time
                    for i, peak in enumerate(top_peaks):
                        ax.annotate(
                            f"Lag: {lags[peak]:.2f}",
                            xy=(lags[peak], autocorr[peak]),
                            xytext=(5, 5 + i * 15),
                            textcoords="offset points",
                            fontsize=8,
                            arrowprops=dict(arrowstyle="->", alpha=0.5),
                        )

                # Set limits and title
                ax.set_xlim(0, min(len(y_for_analysis) // 2 * x_interval, lags[-1]))
                ax.set_ylim(-1, 1.1)
                title = "Autocorrelation Function"
                xlabel = "Lag"
                ylabel = "Correlation"

            else:
                # Use Welch's method for PSD estimation (better for larger datasets)
                method = "psd"

                # Calculate sampling rate
                fs = 1.0 / x_interval

                # Calculate PSD using Welch's method
                # Adjust segment length based on data size
                nperseg = min(256, len(y_for_analysis))

                # Use Welch's method
                freq, Pxx = signal.welch(
                    y_for_analysis, fs=fs, nperseg=nperseg, scaling="density", detrend="constant"
                )

                # Plot PSD
                ax.semilogy(freq, Pxx, "b-")

                # Find and annotate peaks
                from scipy.signal import find_peaks

                # Exclude the lowest frequencies which often dominate
                start_idx = max(1, len(freq) // 20)
                if len(Pxx[start_idx:]) > 0:
                    peak_indices, _ = find_peaks(
                        Pxx[start_idx:], height=np.max(Pxx[start_idx:]) * 0.2
                    )
                    peak_indices = peak_indices + start_idx  # Adjust for slicing

                    if len(peak_indices) > 0:
                        # Limit to top 3 peaks
                        top_peaks = peak_indices[np.argsort(Pxx[peak_indices])[-3:]]
                        ax.plot(freq[top_peaks], Pxx[top_peaks], "ro", ms=5, alpha=0.7)

                        # Annotate with frequency
                        for i, peak in enumerate(top_peaks):
                            ax.annotate(
                                f"{freq[peak]:.3f} Hz",
                                xy=(freq[peak], Pxx[peak]),
                                xytext=(5, 5 + i * 15),
                                textcoords="offset points",
                                fontsize=8,
                                arrowprops=dict(arrowstyle="->", alpha=0.5),
                            )

                # Set limits and labels
                if len(freq) > 5:
                    # Show only the most informative frequency range
                    significant_freqs = np.where(Pxx > np.max(Pxx) * 0.05)[0]
                    if len(significant_freqs) > 0:
                        max_freq_idx = max(significant_freqs[-1] + 5, len(freq) // 4)
                        max_freq_idx = min(max_freq_idx, len(freq) - 1)
                        ax.set_xlim(0, freq[max_freq_idx])

                title = "Power Spectral Density (Welch)"
                xlabel = "Frequency (Hz)"
                ylabel = "PSD (V²/Hz)"

            # Set plot title and labels
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Add method note
            ax.text(
                0.02,
                0.02,
                f"Method: {method}",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            )

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create spectral analysis plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Spectral Analysis Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Spectral Analysis (Error)")

    return ax
