import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_hexbin_plot(tensor_np, ax=None):
    """
    Create a hexbin plot for Nx2 data to visualize point density.

    Parameters:
        tensor_np (numpy.ndarray): The Nx2 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating hexbin plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Extract x and y coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]

        # Handle very small datasets
        if len(x) < 20:
            # Just show scatter for small datasets
            ax.scatter(x, y, alpha=0.7)
            ax.text(
                0.5,
                0.95,
                "Insufficient data for hexbin plot\nshowing scatter instead",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )
        else:
            # Determine optimal hexbin gridsize based on data size
            # More data points -> finer grid
            gridsize = int(min(50, max(10, np.sqrt(len(x) / 2))))

            # Create the hexbin plot
            # Use log scale for better visualization with clustered data
            use_log = (
                len(x) > 100
                or np.max(np.bincount(np.digitize(x, bins=20) * 100 + np.digitize(y, bins=20))) > 10
            )

            if use_log:
                hb = ax.hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    cmap="viridis",
                    bins="log",  # Log scale for color
                    mincnt=1,  # Minimum count to show a hexagon
                )
                cb_label = "log10(N)"
            else:
                hb = ax.hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    cmap="viridis",
                    mincnt=1,  # Minimum count to show a hexagon
                )
                cb_label = "Count"

            # Add colorbar
            plt.colorbar(hb, ax=ax, label=cb_label)

            # Add count statistics
            total_count = len(x)
            max_count = np.max(hb.get_array())
            stats_text = f"Total: {total_count}\nGridsize: {gridsize}"
            if not use_log:
                stats_text += f"\nMax bin: {max_count}"

            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

        # Set plot title and labels
        ax.set_title("Hexagonal Bin Density Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    except Exception as e:
        logger.error(f"Failed to create hexbin plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Hexbin Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Hexbin Plot (Error)")

    return ax
