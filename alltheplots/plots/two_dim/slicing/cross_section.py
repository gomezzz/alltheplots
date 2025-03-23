import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_cross_section_plot(tensor_np, ax=None):
    """
    Create a plot showing the central horizontal and vertical cross-sections of the 2D array.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating cross-section plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Get central indices
        center_row = tensor_np.shape[0] // 2
        center_col = tensor_np.shape[1] // 2

        # Get cross sections
        horizontal_slice = tensor_np[center_row, :]
        vertical_slice = tensor_np[:, center_col]

        # Create x-axes for the slices
        x_horizontal = np.arange(len(horizontal_slice))
        x_vertical = np.arange(len(vertical_slice))

        # Plot horizontal slice
        h_line = ax.plot(
            x_horizontal, horizontal_slice, "b-", label=f"Row {center_row}", linewidth=1.5
        )

        # Create twin axis for vertical slice
        ax2 = ax.twinx()
        v_line = ax2.plot(
            x_vertical, vertical_slice, "r-", label=f"Column {center_col}", linewidth=1.5
        )

        # Add legend combining both plots
        lines = h_line + v_line
        labels = [line.get_label() for line in lines]
        # ax.legend(lines, labels, loc="upper right")

        # Set plot labels and title
        ax.set_title("Central Cross Sections")
        ax.set_xlabel("Index")
        ax.set_ylabel("Row Value (blue)", color="b")
        ax2.set_ylabel("Column Value (red)", color="r")

        # Color the axis labels to match the lines
        ax.tick_params(axis="y", labelcolor="b")
        ax2.tick_params(axis="y", labelcolor="r")

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Add annotation showing the intersection point
        intersection_value = tensor_np[center_row, center_col]
        ax.text(
            0.02,
            0.02,
            f"Intersection: ({center_col}, {center_row})\nValue: {intersection_value:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    except Exception as e:
        logger.error(f"Failed to create cross-section plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Cross Section Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Cross Section (Error)")

    return ax
