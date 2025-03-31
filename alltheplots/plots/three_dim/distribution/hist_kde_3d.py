import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_hist_kde_plot_3d(tensor_np, ax=None):
    """
    Create a histogram with KDE overlay for 3D array voxel values.

    Parameters:
        tensor_np (numpy.ndarray): The 3D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 3D histogram with KDE")

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

            # Add KDE on separate axis if enough points
            if len(clean_data) > 5:
                ax2 = ax.twinx()
                sns.kdeplot(data=clean_data, ax=ax2, color="r", alpha=0.7, warn_singular=False)
                ax2.set_ylabel("Density", color="r")
                ax2.tick_params(axis="y", labelright=False)
        else:
            # For continuous data or discrete with many values, use regular histplot
            logger.debug("Using continuous histogram with KDE overlay")
            n_bins = min(50, max(20, int(np.sqrt(len(clean_data)))))
            sns.histplot(data=clean_data, ax=ax, bins=n_bins, kde=True, stat="density", alpha=0.6)

        # Set plot labels and title
        ax.set_title("Voxel Value Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count" if is_discrete else "Density")

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
        logger.error(f"Failed to create 3D histogram: {e}")
        ax.text(
            0.5,
            0.5,
            f"Histogram Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("3D Histogram (Error)")

    return ax
