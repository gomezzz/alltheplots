import matplotlib.pyplot as plt
from ..utils.type_handling import to_numpy
from ..utils.logger import logger

# Import functions from the individual plot modules
from .three_dim.slice_views import (
    create_xy_slice_plot,
    create_xz_slice_plot,
    create_yz_slice_plot,
)
from .three_dim.projections import (
    create_max_intensity_plot,
    create_mean_intensity_plot,
    create_std_intensity_plot,
)
from .three_dim.distribution import (
    create_hist_kde_plot_3d,
    create_cdf_plot_3d,
)


def plot_3d(tensor, filename=None, dpi=100, show=True):
    """
    Generate a comprehensive visualization of a 3D tensor with a 3×3 grid of plots:

    Column 1 (Slice Views):
    - Central XY slice
    - Central XZ slice
    - Central YZ slice

    Column 2 (Projections):
    - Maximum intensity projection (XY)
    - Mean intensity projection (XY)
    - Standard deviation projection (XY)

    Column 3 (Distribution Analysis):
    - Histogram of all voxel values with KDE
    - KDE over flattened array
    - CDF of voxel intensities

    Parameters:
        tensor (array-like): The input 3D tensor to plot
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        show (bool): Whether to display the plot interactively (True) or just return the figure (False).

    Returns:
        matplotlib.figure.Figure: The figure containing the plots, or None if displayed
    """
    logger.info("Creating 3D plot with 3×3 grid layout")

    # Convert to numpy array using our robust conversion utility
    try:
        tensor_np = to_numpy(tensor)
        if len(tensor_np.shape) != 3:
            raise ValueError(f"Expected 3D tensor, got shape {tensor_np.shape}")
        logger.debug(f"Converted tensor to numpy array of shape {tensor_np.shape}")
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise

    # Create a 3x3 grid of subplots
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.75)

    # Create all subplots
    axes = []
    for i in range(3):
        row = []
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        axes.append(row)

    try:
        # Column 1: Slice Views
        _, im1 = create_xy_slice_plot(tensor_np, ax=axes[0][0], add_colorbar=False)
        _, im2 = create_xz_slice_plot(tensor_np, ax=axes[1][0], add_colorbar=False)
        _, im3 = create_yz_slice_plot(tensor_np, ax=axes[2][0], add_colorbar=False)

        # Column 2: Projections
        _, im4 = create_max_intensity_plot(tensor_np, ax=axes[0][1], add_colorbar=False)
        _, im5 = create_mean_intensity_plot(tensor_np, ax=axes[1][1], add_colorbar=False)
        _, im6 = create_std_intensity_plot(tensor_np, ax=axes[2][1], add_colorbar=False)

        # Column 3: Distribution Analysis
        create_hist_kde_plot_3d(tensor_np, ax=axes[0][2])
        create_hist_kde_plot_3d(tensor_np, ax=axes[1][2])  # Will be similar but good for comparison
        create_cdf_plot_3d(tensor_np, ax=axes[2][2])

        # Add shared colorbar on the right for the image plots
        colorbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im1, cax=colorbar_ax, label="Value")

        # Add column headers
        axes[0][0].text(
            0.5,
            1.25,
            "Slice Views",
            ha="center",
            va="center",
            transform=axes[0][0].transAxes,
            fontsize=12,
            fontweight="bold",
        )
        axes[0][1].text(
            0.5,
            1.25,
            "Projections",
            ha="center",
            va="center",
            transform=axes[0][1].transAxes,
            fontsize=12,
            fontweight="bold",
        )
        axes[0][2].text(
            0.5,
            1.25,
            "Distribution",
            ha="center",
            va="center",
            transform=axes[0][2].transAxes,
            fontsize=12,
            fontweight="bold",
        )

    except Exception as e:
        logger.error(f"Failed to create one or more plots: {e}")
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
