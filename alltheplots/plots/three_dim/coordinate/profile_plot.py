import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_profile_plot(tensor_np, ax=None):
    """
    Create a profile plot with mean or median values along one axis for Nx3 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx3 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating profile plot")

    # Create ax if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    try:
        # Extract x, y, and z coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]
        z = tensor_np[:, 2]

        n_points = len(x)

        # Only meaningful with enough points
        if n_points < 10:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for profile plot\n(need at least 10 points)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title("Profile Plot (Insufficient Data)")
            return ax

        # Determine along which axis to create the profile
        # We'll analyze variance and choose the axis with the largest spread
        variances = [np.var(x), np.var(y), np.var(z)]
        primary_axis_idx = np.argmax(variances)

        # Determine the secondary axis with the next highest variance
        secondary_axis_idx = np.argsort(variances)[-2]

        # Map axis indices to variables and labels
        axes_data = [x, y, z]
        axes_labels = ["X", "Y", "Z"]

        primary_data = axes_data[primary_axis_idx]
        primary_label = axes_labels[primary_axis_idx]

        secondary_data = axes_data[secondary_axis_idx]
        secondary_label = axes_labels[secondary_axis_idx]

        # Create bins along the primary axis for aggregation
        n_bins = min(50, max(10, int(np.sqrt(n_points))))
        bins = np.linspace(np.min(primary_data), np.max(primary_data), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute bin indices for each point
        bin_indices = np.digitize(primary_data, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge cases

        # Initialize arrays for mean, median, min, max
        means = np.zeros(n_bins)
        medians = np.zeros(n_bins)
        mins = np.zeros(n_bins)
        maxs = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        # Compute statistics for each bin
        for i in range(n_bins):
            bin_mask = bin_indices == i
            bin_values = secondary_data[bin_mask]

            if len(bin_values) > 0:
                means[i] = np.mean(bin_values)
                medians[i] = np.median(bin_values)
                mins[i] = np.min(bin_values)
                maxs[i] = np.max(bin_values)
                counts[i] = len(bin_values)
            else:
                means[i] = np.nan
                medians[i] = np.nan
                mins[i] = np.nan
                maxs[i] = np.nan
                counts[i] = 0

        # Determine whether to show mean or median based on data skewness
        # If data is highly skewed, median might be better
        skewness = np.mean(np.abs((secondary_data - np.mean(secondary_data)) ** 3)) / (
            np.std(secondary_data) ** 3
        )

        use_median = abs(skewness) > 1.0
        profile_label = "Median" if use_median else "Mean"
        profile_values = medians if use_median else means

        # Remove NaN entries
        valid_mask = ~np.isnan(profile_values)
        bin_centers = bin_centers[valid_mask]
        profile_values = profile_values[valid_mask]
        mins = mins[valid_mask]
        maxs = maxs[valid_mask]

        # Plot the data and profile
        # Scatter plot of original data (downsampled if necessary)
        max_scatter_points = 500
        if n_points > max_scatter_points:
            # Downsample for scatter plot
            idx = np.random.choice(n_points, max_scatter_points, replace=False)
            ax.scatter(
                primary_data[idx],
                secondary_data[idx],
                alpha=0.3,
                s=5,
                color="gray",
                label="Data (sampled)",
            )
        else:
            ax.scatter(primary_data, secondary_data, alpha=0.3, s=5, color="gray", label="Data")

        # Plot the profile line
        ax.plot(bin_centers, profile_values, "r-", linewidth=2, label=f"{profile_label} Profile")

        # Plot min/max range for variation
        ax.fill_between(bin_centers, mins, maxs, alpha=0.2, color="blue", label="Min-Max Range")

        # Add labels and title
        ax.set_xlabel(primary_label)
        ax.set_ylabel(secondary_label)
        ax.set_title(f"Profile Plot ({profile_label})")
        ax.legend(fontsize=8)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

        # Add stats text
        stats_text = (
            f"n = {n_points}\n" f"Profile along: {primary_label}\n" f"Method: {profile_label}"
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
        logger.error(f"Failed to create profile plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Profile Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Profile Plot (Error)")

    return ax
