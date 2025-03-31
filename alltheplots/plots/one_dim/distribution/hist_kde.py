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
        # Clean data by replacing NaN/Inf values
        if np.any(~np.isfinite(tensor_np)):
            clean_data = np.array(tensor_np, copy=True)
            clean_data = np.nan_to_num(
                clean_data, nan=np.nanmean(clean_data) if np.any(~np.isnan(clean_data)) else 0
            )
            logger.warning("Found NaN or Inf values in histogram data, replaced with mean or zeros")
        else:
            clean_data = tensor_np

        # Analyze data characteristics
        n_points = len(clean_data)
        unique_values = np.unique(clean_data[np.isfinite(clean_data)])
        n_unique = len(unique_values)

        # Check for constant data or zero variance
        has_variance = n_unique > 1
        if not has_variance:
            logger.info("Dataset has zero variance (constant values)")

        is_discrete = np.allclose(clean_data, np.round(clean_data)) or n_unique <= min(
            30, n_points / 10
        )

        # Check if data has large range that might benefit from log scale
        if has_variance:
            count, _ = np.histogram(clean_data, bins="auto")
            if np.max(count) / np.mean(count[count > 0]) > 50:  # Large variation in bin counts
                use_log_scale = True
                logger.debug("Using log scale for histogram due to large count variations")
            else:
                use_log_scale = False
        else:
            use_log_scale = False

        # Create appropriate histogram plot based on data characteristics
        if is_discrete and n_unique <= 50:
            # For discrete data with few unique values, use a bar plot
            logger.debug(f"Using discrete bar histogram (unique values: {n_unique})")

            # Count occurrences of each unique value
            value_counts = {}
            for value in unique_values:
                value_counts[value] = np.sum(clean_data == value)

            # Sort by values for consistent display
            sorted_items = sorted(value_counts.items())
            values, counts = zip(*sorted_items) if sorted_items else ([], [])

            # Create bar plot with smaller bars
            if sorted_items:
                ax.bar(values, counts, width=0.8 if n_unique < 10 else 0.6, alpha=0.7)

                # Set x-ticks to the actual values if there aren't too many
                if n_unique <= 20:
                    ax.set_xticks(values)
                    ax.tick_params(axis="x", rotation=45 if n_unique > 10 else 0)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid data points to plot",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            # Set KDE plot separately if we have enough data points and variance
            if n_points > 5 and has_variance:
                # Create a twin axis for the KDE to avoid scaling issues with discrete data
                ax2 = ax.twinx()
                ax2._get_lines.get_next_color()  # Skip to next color to avoid same color as bars
                try:
                    # Use standard kdeplot only for data with variance, avoiding warning
                    _ = sns.kdeplot(x=clean_data, ax=ax2, color="r", alpha=0.7)
                    ax2.set_ylabel("Density", fontsize=8)
                    # Hide the right y-axis labels to avoid clutter
                    ax2.tick_params(axis="y", labelright=False)
                except Exception as kde_error:
                    logger.warning(f"Could not create KDE overlay: {kde_error}")
        else:
            # For continuous data or discrete with many values, use regular histplot with smaller bins
            logger.debug("Using continuous histogram with KDE overlay")

            if has_variance:
                bins = min(50, max(10, int(n_unique / 5))) if n_unique > 5 else "auto"
                # Only add KDE for data with variance
                sns.histplot(
                    x=clean_data,
                    kde=has_variance,
                    ax=ax,
                    bins=bins,
                    alpha=0.6,
                    edgecolor="none",
                )
            else:
                # Just plot a basic histogram without KDE for constant data
                sns.histplot(
                    x=clean_data, kde=False, ax=ax, bins="auto", alpha=0.6, edgecolor="none"
                )
                # Add a text note about constant data
                ylim = ax.get_ylim()
                ax.text(
                    unique_values[0],
                    ylim[1] * 0.9,
                    "Constant Data",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.4),
                )

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
