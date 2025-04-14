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
        y = tensor_np[:, 1]  # Handle very small datasets
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
            # Check for valid data (finite values)
            valid_mask = np.isfinite(x) & np.isfinite(y)
            if not np.all(valid_mask):
                logger.warning(f"Found {np.sum(~valid_mask)} non-finite values, removing them")
                x = x[valid_mask]
                y = y[valid_mask]

            # Check for constant data (all x or all y values are the same)
            if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
                logger.warning("Data has constant values, showing scatter instead")
                ax.scatter(x, y, alpha=0.7)
                ax.text(
                    0.5,
                    0.95,
                    "Cannot create hexbin with constant values\nshowing scatter instead",
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                )
            else:
                # Add tiny jitter to prevent binning issues when all points fall exactly on grid lines
                if len(np.unique(x)) < 10 or len(np.unique(y)) < 10:
                    jitter_scale_x = (
                        (np.max(x) - np.min(x)) * 0.001 if np.max(x) != np.min(x) else 0.001
                    )
                    jitter_scale_y = (
                        (np.max(y) - np.min(y)) * 0.001 if np.max(y) != np.min(y) else 0.001
                    )
                    x = x + np.random.normal(0, jitter_scale_x, size=len(x))
                    y = y + np.random.normal(0, jitter_scale_y, size=len(y))
                    logger.debug("Added small jitter to discrete data for better binning")

                # Determine optimal hexbin gridsize based on data size and uniqueness
                # More data points and more unique values -> finer grid
                unique_points = min(len(np.unique(x)), len(np.unique(y)))
                gridsize = int(min(50, max(10, min(unique_points, np.sqrt(len(x) / 2)))))

                try:
                    # First try to determine if we should use log scale
                    # Use simpler method to avoid issues with digitize
                    density_ratio = len(x) / (len(np.unique(x)) * len(np.unique(y)))
                    use_log = density_ratio > 3 or len(x) > 200

                    # Create the hexbin plot
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
                    max_count = np.max(hb.get_array()) if len(hb.get_array()) > 0 else 0
                    stats_text = f"Total: {total_count}\nGridsize: {gridsize}"
                    if not use_log and max_count > 0:
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

                except Exception as e:
                    # Fall back to scatter plot if hexbin fails
                    logger.warning(f"Hexbin failed: {e}, falling back to scatter plot")
                    ax.scatter(x, y, alpha=0.6, s=min(20, 500 / len(x)))
                    ax.text(
                        0.5,
                        0.95,
                        f"Hexbin plot failed: {str(e)}\nshowing scatter instead",
                        ha="center",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=9,
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
