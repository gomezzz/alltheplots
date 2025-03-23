import matplotlib.pyplot as plt
from ..utils.type_handling import to_numpy
from ..utils.logger import logger

# Import functions from the individual plot modules (will add these as we create them)
from .two_dim.overview import (
    create_heatmap_plot,
    create_contour_plot,
    create_surface_3d_plot,
)
from .two_dim.distribution import (
    create_hist_plot,
    create_kde_plot,
    create_hist_log_plot,
)
from .two_dim.slicing import (
    create_row_mean_plot,
    create_col_mean_plot,
    create_cross_section_plot,
)


def plot_2d(tensor, filename=None, dpi=100, show=True):
    """
    Generate a comprehensive visualization of a 2D tensor with a 3×3 grid of plots:

    Column 1 (Overview):
    - Heatmap visualization
    - Histogram of all values
    - Row mean plot

    Column 2 (Advanced View):
    - Contour plot
    - KDE of pixel intensities
    - Column mean plot

    Column 3 (Additional Analysis):
    - 3D Surface plot
    - Histogram (log scale)
    - Central cross-section view

    Parameters:
        tensor (array-like): The input 2D tensor to plot
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        show (bool): Whether to display the plot interactively (True) or just return the figure (False).

    Returns:
        matplotlib.figure.Figure: The figure containing the plots, or None if displayed
    """
    logger.info("Creating 2D plot with 3×3 grid layout")

    # Convert to numpy array using our robust conversion utility
    try:
        tensor_np = to_numpy(tensor)
        if len(tensor_np.shape) != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tensor_np.shape}")
        logger.debug(f"Converted tensor to numpy array of shape {tensor_np.shape}")
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise

    # Create a 3x3 grid of subplots
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Create all subplots
    axes = []
    for i in range(3):
        row = []
        for j in range(3):
            if i == 2 and j == 2:  # Last plot (cross-section) needs more space
                ax = fig.add_subplot(gs[i, j])
            elif j == 2 and i == 0:  # 3D surface plot
                ax = fig.add_subplot(gs[i, j], projection="3d")
            else:
                ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        axes.append(row)

    try:
        # Column 1: Overview
        create_heatmap_plot(tensor_np, ax=axes[0][0])
        create_hist_plot(tensor_np, ax=axes[1][0])
        create_row_mean_plot(tensor_np, ax=axes[2][0])

        # Column 2: Advanced View
        create_contour_plot(tensor_np, ax=axes[0][1])
        create_kde_plot(tensor_np, ax=axes[1][1])
        create_col_mean_plot(tensor_np, ax=axes[2][1])

        # Column 3: Additional Analysis
        create_surface_3d_plot(tensor_np, ax=axes[0][2])
        create_hist_log_plot(tensor_np, ax=axes[1][2])
        create_cross_section_plot(tensor_np, ax=axes[2][2])

        # Add column headers
        for col, title in enumerate(["Overview", "Advanced View", "Additional Analysis"]):
            fig.text(
                0.15 + col * 0.33,
                0.95,
                title,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

    except Exception as e:
        logger.error(f"Failed to create one or more plots: {e}")
        # Clean up the figure in case of error
        plt.close(fig)
        raise

    # Save or display the plot
    if filename:
        logger.info(f"Saving plot to file: {filename}")
        try:
            plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
            logger.success(f"Plot saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save plot to {filename}: {e}")
            raise
        finally:
            plt.close(fig)
        return None
    elif show:
        logger.debug("Displaying plot interactively")
        plt.show()
        return None
    else:
        logger.debug("Returning figure without displaying")
        plt.close(fig)
        return fig
