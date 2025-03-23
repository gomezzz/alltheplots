import matplotlib.pyplot as plt
from ..utils.type_handling import to_numpy
from ..utils.logger import logger

# Import functions from the individual plot modules
from .two_dim.overview import (
    create_heatmap_plot,
    create_contour_plot,
    create_surface_3d_plot,
)
from .two_dim.distribution import create_hist_kde_plot
from .two_dim.slicing import create_row_mean_plot, create_col_mean_plot
from .two_dim.frequency_domain import create_fft2d_plot


def plot_2d(tensor, filename=None, dpi=100, show=True):
    """
    Generate a comprehensive visualization of a 2D tensor with a 3×3 grid of plots:

    Column 1 (3D Views):
    - 3D Surface plot (front view)
    - 3D Surface plot (side view)
    - 3D Surface plot (top view)

    Column 2 (Distribution Analysis):
    - Value distribution (Histogram + KDE)
    - Row means profile
    - Column means profile

    Column 3 (Shape Analysis):
    - Contour plot
    - 2D FFT Magnitude
    - Heatmap

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
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Create all subplots with appropriate projections
    axes = []
    for i in range(3):
        row = []
        for j in range(3):
            if j == 0:  # All of column 1 need 3D projection
                ax = fig.add_subplot(gs[i, j], projection="3d")
            else:
                ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        axes.append(row)

    try:
        # Column 1: 3D Views
        # Front view
        create_surface_3d_plot(tensor_np, ax=axes[0][0], view_angle=(30, -60))
        axes[0][0].set_title("3D Surface (Front)")

        # Side view
        create_surface_3d_plot(tensor_np, ax=axes[1][0], view_angle=(30, -120))
        axes[1][0].set_title("3D Surface (Side)")

        # Top view
        create_surface_3d_plot(tensor_np, ax=axes[2][0], view_angle=(90, -90))
        axes[2][0].set_title("3D Surface (Top)")

        # Column 2: Distribution Analysis
        create_hist_kde_plot(tensor_np, ax=axes[0][1])
        create_row_mean_plot(tensor_np, ax=axes[1][1])
        create_col_mean_plot(tensor_np, ax=axes[2][1])

        # Column 3: Shape Analysis
        create_contour_plot(tensor_np, ax=axes[0][2])
        create_fft2d_plot(tensor_np, ax=axes[1][2], remove_dc=True)
        create_heatmap_plot(tensor_np, ax=axes[2][2])

        # Add column headers
        for col, title in enumerate(["3D Views", "Distribution Analysis", "Shape Analysis"]):
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
