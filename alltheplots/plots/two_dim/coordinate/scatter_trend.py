import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from ....utils.logger import logger


def create_scatter_trend_plot(tensor_np, ax=None):
    """
    Create a scatter plot with trend smoothing for Nx2 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx2 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating scatter plot with trend smoothing")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Extract x and y coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]

        # Check if data is too sparse
        if len(x) < 3:
            # Just plot the scatter points if too few points for smoothing
            ax.scatter(x, y, alpha=0.6, edgecolor="k", linewidth=0.5)
            ax.text(
                0.5,
                0.5,
                "Too few points for trend smoothing",
                ha="center",
                va="center",
                transform=ax.transAxes,
                alpha=0.5,
            )
        else:
            # Plot scatter points
            ax.scatter(x, y, alpha=0.6, edgecolor="k", linewidth=0.5)

            # Sort points by x-coordinate for smoother trend line
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]

            # Determine which smoothing method to use based on data characteristics
            if len(x) <= 10:
                # Linear regression for very small datasets
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                trend_line = poly(x_sorted)
                ax.plot(x_sorted, trend_line, "r-", linewidth=2, alpha=0.7, label="Linear Trend")
            elif len(x) < 30:
                # Polynomial fit for small datasets
                try:
                    # Calculate degree based on number of points (but not more than 3)
                    degree = min(3, max(1, len(x) // 10))
                    coeffs = np.polyfit(x, y, degree)
                    poly = np.poly1d(coeffs)
                    trend_line = poly(x_sorted)
                    ax.plot(
                        x_sorted,
                        trend_line,
                        "r-",
                        linewidth=2,
                        alpha=0.7,
                        label=f"Polynomial (deg={degree})",
                    )
                except Exception as e:
                    logger.warning(f"Polynomial fit failed: {e}. Falling back to linear.")
                    coeffs = np.polyfit(x, y, 1)
                    poly = np.poly1d(coeffs)
                    trend_line = poly(x_sorted)
                    ax.plot(
                        x_sorted, trend_line, "r-", linewidth=2, alpha=0.7, label="Linear Trend"
                    )
            else:
                # For larger datasets, assess data entropy/noise first
                try:
                    # Evaluate entropy/noise level of the data
                    # 1. Calculate residuals from linear fit as baseline
                    linear_coeffs = np.polyfit(x, y, 1)
                    linear_pred = np.poly1d(linear_coeffs)(x)
                    residuals = y - linear_pred

                    # 2. Calculate noise metrics
                    residual_std = np.std(residuals)
                    signal_std = np.std(y)
                    noise_ratio = residual_std / signal_std if signal_std > 0 else 1.0

                    # 3. Check for high noise (random) vs structured pattern
                    # Calculate autocorrelation of residuals to check for structure
                    from scipy.signal import correlate

                    res_auto = correlate(
                        residuals - np.mean(residuals), residuals - np.mean(residuals), mode="full"
                    )[len(residuals) - 1 :]
                    res_auto = res_auto / res_auto[0]  # Normalize

                    # High autocorrelation indicates structure in residuals (not just noise)
                    structured_pattern = np.any(np.abs(res_auto[1:10]) > 0.3)

                    # 4. Choose method based on noise and structure assessment
                    if noise_ratio > 0.8 and not structured_pattern:
                        # High noise, little structure - use more aggressive smoothing
                        logger.debug(
                            f"High noise data (ratio: {noise_ratio:.2f}), using stronger smoothing"
                        )

                        if len(x) > 100:
                            # For larger datasets with high noise, use LOWESS with strong smoothing
                            # Higher frac means more smoothing
                            frac = min(0.7, max(0.3, 30 / len(x)))
                            trend_lowess = lowess(y, x, frac=frac, it=3, return_sorted=False)
                            ax.plot(
                                x_sorted,
                                trend_lowess[sorted_indices],
                                "r-",
                                linewidth=2,
                                alpha=0.7,
                                label=f"LOWESS (strong, frac={frac:.2f})",
                            )
                        else:
                            # For smaller noisy datasets, polynomial is often better
                            degree = min(3, max(1, len(x) // 15))
                            coeffs = np.polyfit(x, y, degree)
                            poly = np.poly1d(coeffs)
                            trend_line = poly(x_sorted)
                            ax.plot(
                                x_sorted,
                                trend_line,
                                "r-",
                                linewidth=2,
                                alpha=0.7,
                                label=f"Polynomial (deg={degree})",
                            )
                    elif noise_ratio > 0.5:
                        # Medium noise - moderate smoothing
                        logger.debug(f"Medium noise data (ratio: {noise_ratio:.2f})")

                        # Try LOWESS with moderate parameters
                        frac = min(0.5, max(0.2, 25 / len(x)))
                        trend_lowess = lowess(y, x, frac=frac, it=2, return_sorted=False)
                        ax.plot(
                            x_sorted,
                            trend_lowess[sorted_indices],
                            "r-",
                            linewidth=2,
                            alpha=0.7,
                            label=f"LOWESS (frac={frac:.2f})",
                        )
                    else:
                        # Low noise or structured residuals - light smoothing to preserve details
                        logger.debug(f"Low noise/structured data (ratio: {noise_ratio:.2f})")

                        # Try LOWESS with light smoothing
                        frac = min(0.3, max(0.1, 15 / len(x)))
                        trend_lowess = lowess(y, x, frac=frac, it=1, return_sorted=False)
                        ax.plot(
                            x_sorted,
                            trend_lowess[sorted_indices],
                            "r-",
                            linewidth=2,
                            alpha=0.7,
                            label=f"LOWESS (light, frac={frac:.2f})",
                        )
                except Exception as e:
                    # Fall back to Savitzky-Golay if LOWESS/analysis fails
                    logger.warning(
                        f"Advanced trend analysis failed: {e}. Falling back to SG filter."
                    )
                    try:
                        # Choose window length based on data size (must be odd)
                        window_length = min(51, max(5, len(x) // 10 * 2 + 1))
                        polyorder = min(
                            3, window_length - 2
                        )  # Order must be less than window length
                        trend_sg = savgol_filter(y_sorted, window_length, polyorder)
                        ax.plot(
                            x_sorted,
                            trend_sg,
                            "r-",
                            linewidth=2,
                            alpha=0.7,
                            label=f"SG Filter (w={window_length})",
                        )
                    except Exception as e2:
                        logger.warning(f"SG filter failed: {e2}. Falling back to linear trend.")
                        coeffs = np.polyfit(x, y, 1)
                        poly = np.poly1d(coeffs)
                        trend_line = poly(x_sorted)
                        ax.plot(
                            x_sorted, trend_line, "r-", linewidth=2, alpha=0.7, label="Linear Trend"
                        )

            # Add legend to identify the trend line
            ax.legend(loc="best", fontsize="small")

        # Set plot title and labels
        ax.set_title("Scatter with Trend Smoothing")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

        # Add stats summary
        stats_text = (
            f"n = {len(x)}\n"
            f"x̄ = {np.mean(x):.2f}, σx = {np.std(x):.2f}\n"
            f"ȳ = {np.mean(y):.2f}, σy = {np.std(y):.2f}"
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
        logger.error(f"Failed to create scatter trend plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Scatter Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Scatter with Trend (Error)")

    return ax
