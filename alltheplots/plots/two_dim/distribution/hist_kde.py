import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_hist_kde_plot(tensor_np, ax=None, add_colorbar=True):
    """
    Create a combined histogram and KDE plot for 2D data.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        add_colorbar (bool): Whether to add a colorbar to the plot. Default is True.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating combined histogram and KDE plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Flatten the array for histogram
        flat_data = tensor_np.flatten()

        # Remove any non-finite values
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

            # Count occurrences of each unique value
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

            # Add value labels on top of bars if there aren't too many
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

            # Add KDE on a separate axis if we have enough points
            if len(clean_data) > 5:
                ax2 = ax.twinx()
                sns.kdeplot(data=clean_data, ax=ax2, color="r", alpha=0.7, warn_singular=False)
                ax2.set_ylabel("Density", color="r")
                # Only show y-axis on the left
                ax2.tick_params(axis="y", labelright=False)
        else:
            # For continuous data, use histogram with KDE
            logger.debug("Using continuous histogram with KDE overlay")

            # Calculate optimal number of bins
            n_bins = min(50, max(20, int(np.sqrt(len(clean_data)))))

            # Create the combined plot
            sns.histplot(data=clean_data, ax=ax, bins=n_bins, kde=True, stat="density", alpha=0.6)

        # Set plot labels and title
        ax.set_title("Value Distribution")
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
        logger.error(f"Failed to create histogram plot: {e}")
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
