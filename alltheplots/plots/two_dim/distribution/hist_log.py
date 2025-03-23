import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_hist_log_plot(tensor_np, ax=None):
    """
    Create a histogram with logarithmic scale for better visualization of wide-ranging values.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating log-scale histogram")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Flatten the array and handle non-positive values
        flat_data = tensor_np.flatten()

        # Remove any non-finite values
        clean_data = flat_data[np.isfinite(flat_data)]

        if len(clean_data) == 0:
            raise ValueError("No finite values in data")

        # Check if data is constant
        unique_vals = np.unique(clean_data)
        n_unique = len(unique_vals)

        if n_unique <= 1:
            # Handle constant data case
            logger.debug("Constant data detected, showing bar instead of histogram")
            if n_unique == 1:
                val = unique_vals[0]
                # Create a single bar for the constant value
                if val > 0:  # Can use log scale
                    log_val = np.log10(val)
                    ax.bar([log_val], [len(clean_data)], width=0.1, color="blue", alpha=0.5)
                    ax.set_xticks([log_val])
                    ax.set_xticklabels([f"{val:.1e}"])
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"Constant Value: {val}\n(Cannot show on log scale)",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            else:
                ax.text(0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes)
        else:
            # Handle non-positive values for log scale
            min_val = np.min(clean_data)
            if min_val <= 0:
                # Add offset to make all values positive
                offset = abs(min_val) + 1e-10
                log_data = np.log10(clean_data + offset)
                logger.debug(f"Added offset of {offset} to handle non-positive values")
            else:
                log_data = np.log10(clean_data)
                offset = 0

            # Create histogram with automatically determined bins
            n_bins = min(50, max(20, int(np.sqrt(len(log_data)))))
            sns.histplot(data=log_data, ax=ax, bins=n_bins, kde=True, stat="density", alpha=0.6)

            # Set custom x-ticks to show actual values
            log_ticks = ax.get_xticks()
            # Filter out ticks that would result in negative or zero values
            if offset > 0:
                valid_ticks = log_ticks[log_ticks > np.log10(offset)]
                tick_labels = [f"{(10**x - offset):.1e}" for x in valid_ticks]
            else:
                valid_ticks = log_ticks
                tick_labels = [f"{10**x:.1e}" for x in valid_ticks]

            # Set ticks before labels
            ax.set_xticks(valid_ticks)
            ax.set_xticklabels(tick_labels, rotation=45)

            # Add note about offset if used
            if offset > 0:
                ax.text(
                    0.02,
                    0.98,
                    f"Offset: +{offset:.1e}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                    bbox=dict(facecolor="white", alpha=0.8),
                )

        # Set plot labels and title
        ax.set_title("Value Distribution (Log Scale)")
        ax.set_xlabel("Value (log scale)")
        ax.set_ylabel("Density")

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create log-scale histogram: {e}")
        ax.text(
            0.5,
            0.5,
            f"Log Histogram Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Log Histogram (Error)")

    return ax
