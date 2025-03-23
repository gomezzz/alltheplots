import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_contour_plot(tensor_np, ax=None, add_colorbar=True):
    """
    Create a contour plot of 2D data with automatic level selection.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        add_colorbar (bool): Whether to add a colorbar to the plot. Default is True.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating contour plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Create coordinate grids for the contour plot
        Y, X = np.mgrid[: tensor_np.shape[0], : tensor_np.shape[1]]

        # Check if data is constant
        unique_vals = np.unique(tensor_np[np.isfinite(tensor_np)])
        n_unique = len(unique_vals)

        if n_unique <= 1:
            # Handle constant data case
            logger.debug("Constant data detected, showing text instead of contour plot")
            ax.text(
                0.5,
                0.5,
                f"Constant Value: {unique_vals[0] if n_unique > 0 else 'NaN'}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            # Add a subtle background color to indicate the constant value
            if n_unique > 0:
                ax.imshow(
                    [[unique_vals[0]]],
                    extent=[0, tensor_np.shape[1], 0, tensor_np.shape[0]],
                    cmap="viridis",
                    alpha=0.2,
                )
        else:
            # Determine appropriate number of levels based on data characteristics
            if n_unique < 5:
                levels = unique_vals  # Use actual values for discrete data
                logger.debug(f"Using {n_unique} discrete levels for contour plot")
            else:
                # For continuous data, use Freedman-Diaconis rule to estimate bin count
                iqr = np.percentile(tensor_np[np.isfinite(tensor_np)], 75) - np.percentile(
                    tensor_np[np.isfinite(tensor_np)], 25
                )
                bin_width = 2 * iqr / (len(tensor_np.flatten()) ** (1 / 3)) if iqr > 0 else 1
                n_levels = max(5, min(20, int((np.max(tensor_np) - np.min(tensor_np)) / bin_width)))
                levels = n_levels
                logger.debug(f"Using {n_levels} levels for contour plot")

            # Create filled contour plot with colorbar
            contour = ax.contourf(X, Y, tensor_np, levels=levels, cmap="viridis")
            if add_colorbar:
                plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)

            # Add contour lines with labels for better readability
            contour_lines = ax.contour(
                X, Y, tensor_np, levels=levels, colors="white", linewidths=0.5, alpha=0.5
            )
            if n_unique < 10:  # Only add labels if there aren't too many levels
                ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

        # Set plot labels and title
        ax.set_title("Contour Plot")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")

    except Exception as e:
        logger.error(f"Failed to create contour plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Contour Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Contour Plot (Error)")

    return ax
