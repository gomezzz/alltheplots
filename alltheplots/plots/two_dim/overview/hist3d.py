import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_hist3d_plot(tensor_np, ax=None):
    """
    Create a 3D histogram visualization of 2D data values distribution.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib 3D axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 3D value distribution histogram")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))

    try:
        # Get the flattened data and remove non-finite values
        flat_data = tensor_np.flatten()
        clean_data = flat_data[np.isfinite(flat_data)]

        if len(clean_data) == 0:
            raise ValueError("No finite values in data")

        # Determine number of bins based on data characteristics
        n_points = len(clean_data)
        n_unique = len(np.unique(clean_data))

        if n_unique <= 50:  # Discrete data
            bins = n_unique
            logger.debug(f"Using {bins} bins for discrete data")
        else:  # Continuous data
            # Use Freedman-Diaconis rule for bin width
            iqr = np.percentile(clean_data, 75) - np.percentile(clean_data, 25)
            bin_width = 2 * iqr / (n_points ** (1 / 3)) if iqr > 0 else 1
            bins = max(10, min(50, int((np.max(clean_data) - np.min(clean_data)) / bin_width)))
            logger.debug(f"Using {bins} bins for continuous data")

        # Create regular bins for the data range
        min_val = np.min(clean_data)
        max_val = np.max(clean_data)
        if min_val == max_val:
            # Handle constant data
            hist_range = [min_val - 0.5, max_val + 0.5]
            bins = 1
        else:
            hist_range = [min_val, max_val]

        # Compute 1D histogram for the values
        hist, edges = np.histogram(clean_data, bins=bins, range=hist_range, density=True)

        # Get the centers of the bins
        centers = (edges[:-1] + edges[1:]) / 2
        bin_width = edges[1] - edges[0]

        # Create the 3D bar plot
        x_pos = centers
        y_pos = np.zeros_like(centers)

        # Normalize heights for better visualization
        heights = hist / np.max(hist) if np.max(hist) > 0 else hist

        # Create bars with dynamic width based on data range
        width = bin_width * 0.8  # Slightly narrower than bin width
        depth = width  # Make bars square in x-y plane

        # Plot the bars
        ax.bar3d(
            x_pos,
            y_pos,
            np.zeros_like(heights),
            width,
            depth,
            heights,
            shade=True,
            alpha=0.8,
            color=plt.cm.viridis(heights),  # Color by height
        )

        # Add statistical annotations
        stats_text = (
            f"Mean: {np.mean(clean_data):.2f}\n"
            f"Std: {np.std(clean_data):.2f}\n"
            f"N: {len(clean_data)}"
        )
        ax.text2D(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Adjust the view angle for better visibility
        ax.view_init(elev=20, azim=45)

        # Set plot labels and title
        ax.set_title("Value Distribution (3D)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Position")
        ax.set_zlabel("Frequency")

    except Exception as e:
        logger.error(f"Failed to create 3D histogram: {e}")
        ax.text2D(
            0.5,
            0.5,
            f"3D Histogram Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("3D Histogram (Error)")

    return ax
