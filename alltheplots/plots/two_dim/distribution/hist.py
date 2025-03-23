import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_hist_plot(tensor_np, ax=None):
    """
    Create a histogram of all values in a 2D array with automatic bin selection.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating histogram of 2D array values")

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

        # Determine if data is discrete
        is_discrete = np.allclose(clean_data, np.round(clean_data))
        unique_values = np.unique(clean_data)
        n_unique = len(unique_values)

        if is_discrete and n_unique <= 50:
            # For discrete data with few unique values, use bar plot
            logger.debug(f"Using discrete histogram with {n_unique} unique values")

            # Count occurrences of each unique value
            value_counts = {}
            for value in unique_values:
                value_counts[value] = np.sum(clean_data == value)

            # Create bar plot
            ax.bar(list(value_counts.keys()), list(value_counts.values()), alpha=0.7, width=0.8)

            # Set x-ticks to the actual values if there aren't too many
            if n_unique <= 20:
                ax.set_xticks(list(value_counts.keys()))
                if n_unique > 10:
                    plt.xticks(rotation=45)
        else:
            # For continuous data, use seaborn's histplot with automatic bin selection
            logger.debug("Using continuous histogram")

            # Use Freedman-Diaconis rule for bin width
            iqr = np.percentile(clean_data, 75) - np.percentile(clean_data, 25)
            bin_width = 2 * iqr / (len(clean_data) ** (1 / 3)) if iqr > 0 else 1
            n_bins = max(10, min(50, int((np.max(clean_data) - np.min(clean_data)) / bin_width)))

            sns.histplot(data=clean_data, ax=ax, bins=n_bins, kde=True, stat="density", alpha=0.6)

        # Set plot labels and title
        ax.set_title("Value Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density" if not is_discrete else "Count")

        # Add grid for better readability
        ax.grid(True, alpha=0.5)

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
