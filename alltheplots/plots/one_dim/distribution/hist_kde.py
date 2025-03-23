import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_hist_kde_plot(tensor_np, ax=None, is_shared_x=False):
    """
    Create a histogram with KDE overlay plot for 1D data with adaptive features:
    - Auto log-scale if data has large dynamic range
    - Discrete bar histogram if data is integer or has few unique values

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to create histogram for
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        is_shared_x (bool): Whether this plot shares x-axis with other plots.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating adaptive histogram with KDE")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    try:
        # Analyze data characteristics
        n_points = len(tensor_np)
        unique_values = np.unique(tensor_np)
        n_unique = len(unique_values)
        is_discrete = np.allclose(tensor_np, np.round(tensor_np)) or n_unique <= min(
            30, n_points / 10
        )

        # Check if data has large range that might benefit from log scale
        count, _ = np.histogram(tensor_np, bins="auto")
        if np.max(count) / np.mean(count[count > 0]) > 50:  # Large variation in bin counts
            use_log_scale = True
            logger.debug("Using log scale for histogram due to large count variations")
        else:
            use_log_scale = False

        # Create appropriate histogram plot based on data characteristics
        if is_discrete and n_unique <= 50:
            # For discrete data with few unique values, use a bar plot
            logger.debug(f"Using discrete bar histogram (unique values: {n_unique})")

            # Count occurrences of each unique value
            value_counts = {}
            for value in unique_values:
                value_counts[value] = np.sum(tensor_np == value)

            # Sort by values for consistent display
            sorted_items = sorted(value_counts.items())
            values, counts = zip(*sorted_items)

            # Create bar plot with smaller bars
            ax.bar(values, counts, width=0.8 if n_unique < 10 else 0.6, alpha=0.7)

            # Set x-ticks to the actual values if there aren't too many
            if n_unique <= 20:
                ax.set_xticks(values)
                ax.tick_params(axis="x", rotation=45 if n_unique > 10 else 0)

            # Set KDE plot separately if we have enough data points
            if n_points > 5:
                # Create a twin axis for the KDE to avoid scaling issues with discrete data
                ax2 = ax.twinx()
                ax2._get_lines.get_next_color()  # Skip to next color to avoid same color as bars
                try:
                    # Use standard kdeplot without passing linewidth in the kde_kws
                    density = sns.kdeplot(x=tensor_np, ax=ax2, color="r", alpha=0.7)
                    ax2.set_ylabel("Density", fontsize=8)
                    # Hide the right y-axis labels to avoid clutter
                    ax2.tick_params(axis="y", labelright=False)
                except Exception as kde_error:
                    logger.warning(f"Could not create KDE overlay for discrete data: {kde_error}")
        else:
            # For continuous data or discrete with many values, use regular histplot with smaller bins
            logger.debug("Using continuous histogram with KDE overlay")
            bins = min(50, max(10, int(n_unique / 5))) if n_unique > 5 else "auto"
            # Fix: Remove the kde_kws with linewidth parameter
            sns.histplot(x=tensor_np, kde=True, ax=ax, bins=bins, alpha=0.6, edgecolor="none")

        # Apply log scale to y-axis if needed
        if use_log_scale:
            ax.set_yscale("log")
            ax.set_title("Histogram with KDE (Log Scale)")
        else:
            ax.set_title("Histogram with KDE")

        # Set plot labels
        ax.set_xlabel("Value" if not is_shared_x else "")
        ax.set_ylabel("Count")

    except Exception as e:
        error_msg = f"Failed to create histogram: {e}"
        logger.error(error_msg)
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
