import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_line_scatter_logy_plot(tensor_np, ax=None, is_shared_x=False):
    """
    Create a time-domain line or scatter plot with logarithmic y-axis scale.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to plot
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        is_shared_x (bool): Whether this plot shares x-axis with other plots.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating time-domain plot with log y-axis")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    try:
        # First handle NaN and infinite values
        if np.any(~np.isfinite(tensor_np)):
            plot_data = np.array(tensor_np, copy=True)
            # Replace non-finite values with NaN to preserve other valid data
            plot_data = np.where(np.isfinite(plot_data), plot_data, np.nan)
            logger.warning("Found non-finite values in data, will be excluded from plot")
        else:
            plot_data = tensor_np

        # Check if data is compatible with log scale (all positive)
        # Only consider finite values for determining offset
        finite_mask = np.isfinite(plot_data)
        if finite_mask.sum() == 0:
            logger.warning("No finite data points for log y-plot, showing empty plot")
            ax.text(
                0.5, 0.5, "No valid data points", ha="center", va="center", transform=ax.transAxes
            )
            plot_data = np.array([1.0])  # Add a dummy point to avoid plotting errors
            use_original = False
        else:
            min_finite = np.min(plot_data[finite_mask]) if np.any(finite_mask) else 0
            if min_finite <= 0:
                # Offset data to make it positive for log scale
                offset = abs(min_finite) + 1e-10
                # Apply offset only to finite values
                offset_data = np.where(finite_mask, plot_data + offset, np.nan)
                logger.info(
                    f"Data contains non-positive values, offsetting by {offset} for log scale"
                )
                use_original = False
            else:
                offset_data = plot_data
                use_original = True

        # Determine if we should use scatter instead of line plot
        n_points = len(tensor_np)
        unique_values = np.unique(tensor_np[np.isfinite(tensor_np)])
        n_unique = len(unique_values)

        # Safely calculate zero crossings on finite data only
        finite_data = tensor_np[np.isfinite(tensor_np)]
        if len(finite_data) > 1:
            finite_mean = np.mean(finite_data)
            zero_crossings = np.where(np.diff(np.signbit(finite_data - finite_mean)))[0]
            n_crossings = len(zero_crossings)
        else:
            n_crossings = 0

        # Use scatter if:
        # 1. Few data points (<= 50)
        # 2. Highly discrete (few unique values relative to total points)
        # 3. Many zero crossings (high frequency data)
        use_scatter = (
            n_points <= 50 or n_unique <= min(20, n_points / 5) or n_crossings > n_points / 10
        )

        # Create the appropriate plot
        if use_scatter:
            logger.debug(
                f"Using scatter plot (log y-axis) (points: {n_points}, unique values: {n_unique})"
            )
            ax.scatter(np.arange(len(offset_data)), offset_data, s=3, alpha=0.7)
        else:
            logger.debug("Using line plot for time-domain visualization (log y-axis)")
            ax.plot(np.arange(len(offset_data)), offset_data, linewidth=0.8)

        # Set log scale for y-axis
        ax.set_yscale("log")

        # Set plot labels
        ax.set_title("Time Domain (Log Y)")
        ax.set_xlabel("Index" if not is_shared_x else "")

        # Add note if data was offset
        if not use_original:
            ax.set_ylabel("Value + offset (Log Scale)")
            # Add annotation about the offset
            if min_finite <= 0:
                ax.annotate(
                    f"Offset: +{offset:.2e}",
                    xy=(0.02, 0.02),
                    xycoords="axes fraction",
                    fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )
        else:
            ax.set_ylabel("Value (Log Scale)")

    except Exception as e:
        logger.error(f"Failed to create time-domain plot with log y-axis: {e}")
        ax.text(
            0.5,
            0.5,
            f"Time Domain (Log Y) Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Time Domain (Log Y) (Error)")

    return ax
