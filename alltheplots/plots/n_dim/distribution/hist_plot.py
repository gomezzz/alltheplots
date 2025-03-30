import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_hist_plot_nd(tensor_np, ax=None):
    """
    Create a histogram plot for n-dimensional array values.

    Parameters:
        tensor_np (numpy.ndarray): The n-dimensional numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating histogram plot for n-dimensional data")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Flatten the array for histogram
        flat_data = tensor_np.flatten()
        clean_data = flat_data[np.isfinite(flat_data)]

        if len(clean_data) == 0:
            raise ValueError("No finite values in data")

        # Check if data is discrete or continuous
        unique_vals = np.unique(clean_data)
        n_unique = len(unique_vals)
        is_discrete = np.allclose(clean_data, np.round(clean_data)) or n_unique <= min(
            30, len(clean_data) / 10
        )

        if is_discrete and n_unique <= 50:
            # For discrete data with few unique values, use bar plot
            logger.debug(f"Using discrete histogram with {n_unique} unique values")
            value_counts = {}
            for value in unique_vals:
                value_counts[value] = np.sum(clean_data == value)

            # Create bar plot
            bars = ax.bar(
                list(value_counts.keys()),
                list(value_counts.values()),
                alpha=0.6,
                width=0.8 if n_unique < 10 else 0.6,
            )

            # Add value labels for small number of bars
            if n_unique <= 20:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            # Set x-ticks to actual values if there aren't too many
            if n_unique <= 20:
                ax.set_xticks(list(value_counts.keys()))
                if n_unique > 10:
                    plt.xticks(rotation=45)
        else:
            # For continuous data or discrete with many values, use regular histplot
            logger.debug("Using continuous histogram")
            n_bins = min(50, max(20, int(np.sqrt(len(clean_data)))))
            ax.hist(clean_data, bins=n_bins, alpha=0.6)

        # Set plot labels and title
        ax.set_title(f"Value Distribution\n{tensor_np.shape} Array")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

        # Add summary statistics
        stats_text = (
            f"Mean: {np.mean(clean_data):.2f}\n"
            f"Std: {np.std(clean_data):.2f}\n"
            f"N: {len(clean_data)}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    except Exception as e:
        logger.error(f"Failed to create histogram: {e}")
        ax.text(
            0.5,
            0.5,
            f"Histogram Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Histogram (Error)")

    return ax
