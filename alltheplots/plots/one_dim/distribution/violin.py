import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_violin_plot(tensor_np, ax=None, is_shared_x=False):
    """
    Create a violin plot for 1D data.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to create violin plot for
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        is_shared_x (bool): Whether this plot shares x-axis with other plots.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating violin plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    try:
        # Create a DataFrame-like structure for seaborn
        data = {"values": tensor_np}

        # Create the violin plot with more compact style
        sns.violinplot(
            y="values", data=data, ax=ax, inner="quartile", linewidth=0.8, saturation=0.9, width=0.8
        )

        # Remove the x-axis label and ticks since there's only one category
        ax.set_xlabel("")
        ax.set_xticks([])

        # Set plot labels
        ax.set_title("Violin Plot")
        ax.set_ylabel("Value")
        ax.set_xlabel("Frequency of Values")

        # Add horizontal grid lines with less visual weight
        ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.5)

    except Exception as e:
        logger.error(f"Failed to create violin plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Violin Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Violin Plot (Error)")

    return ax
