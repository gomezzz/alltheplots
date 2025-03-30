import matplotlib.pyplot as plt
import numpy as np
from ..utils.type_handling import to_numpy
from ..utils.logger import logger

# Import functions from the individual plot modules
from .n_dim.dim_reduction import (
    create_pca_projection_plot,
    create_umap_projection_plot,
)
from .n_dim.projection import (
    create_mean_projection_plot,
    create_std_projection_plot,
    create_max_projection_plot,
)
from .n_dim.distribution import (
    create_hist_plot_nd,
    create_kde_plot_nd,
    create_cdf_plot_nd,
)


def plot_nd(tensor, filename=None, dpi=100, show=True):
    """
    Generate a comprehensive visualization of an n-dimensional tensor with a 3×3 grid of plots:

    Column 1 (Dimension Reduction):
    - PCA projection to 2D
    - t-SNE projection to 2D (when available)
    - UMAP projection to 2D (when available)

    Column 2 (Aggregate Projections):
    - Mean projection to 2D
    - Standard deviation projection to 2D
    - Maximum value projection to 2D

    Column 3 (Value Distribution):
    - Histogram of all values
    - Kernel Density Estimation (KDE)
    - Cumulative Distribution Function (CDF)

    Parameters:
        tensor (array-like): The input n-dimensional tensor to plot
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        show (bool): Whether to display the plot interactively (True) or just return the figure (False).

    Returns:
        matplotlib.figure.Figure: The figure containing the plots, or None if displayed
    """
    logger.info("Creating nD plot with 3×3 grid layout")

    # Convert to numpy array using our robust conversion utility
    try:
        tensor_np = to_numpy(tensor)
        if len(tensor_np.shape) < 2:
            raise ValueError(f"Expected at least 2D tensor, got shape {tensor_np.shape}")
        logger.debug(f"Converted tensor to numpy array of shape {tensor_np.shape}")
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise

    # Create figure and gridspec
    fig = plt.figure(figsize=(8, 8))
    # Force each of the 3 columns to have the same width
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1], hspace=0.75, wspace=0.5)

    # Create subplots
    axes = []
    for i in range(3):
        row = []
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        axes.append(row)

    # Image placeholder for the colorbar
    im_for_colorbar = None

    try:
        # --- Column 1: Dimension Reduction ---
        create_pca_projection_plot(tensor_np, ax=axes[0][0])

        # Try to create a t-SNE projection plot
        try:
            # Import dynamically to isolate potential issues
            from .n_dim.dim_reduction import create_tsne_projection_plot

            create_tsne_projection_plot(tensor_np, ax=axes[1][0])
        except Exception as e:
            logger.warning(f"t-SNE projection failed: {e}. Using error placeholder.")
            axes[1][0].text(
                0.5,
                0.5,
                f"t-SNE Error: {str(e)}",
                ha="center",
                va="center",
                transform=axes[1][0].transAxes,
                fontsize=8,
            )
            axes[1][0].set_title("t-SNE Projection (Error)")

        create_umap_projection_plot(tensor_np, ax=axes[2][0])

        # --- Column 2: Aggregate Projections ---
        axes[0][1], im1 = create_mean_projection_plot(tensor_np, ax=axes[0][1], add_colorbar=False)
        axes[1][1], im2 = create_std_projection_plot(tensor_np, ax=axes[1][1], add_colorbar=False)
        axes[2][1], im3 = create_max_projection_plot(tensor_np, ax=axes[2][1], add_colorbar=False)

        # Choose the first valid image for the colorbar
        for im in [im1, im2, im3]:
            if im is not None:
                im_for_colorbar = im
                break

        # --- Column 3: Distribution Analysis ---
        create_hist_plot_nd(tensor_np, ax=axes[0][2])
        create_kde_plot_nd(tensor_np, ax=axes[1][2])
        create_cdf_plot_nd(tensor_np, ax=axes[2][2])

        # Add shared colorbar on the right for the projection plots
        if im_for_colorbar:
            colorbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Cover most of the figure height
            fig.colorbar(im_for_colorbar, cax=colorbar_ax, label="Value")

        # --- Column Headers ---
        axes[0][0].text(
            0.5,
            1.35,
            "Dimension Reduction",
            ha="center",
            va="center",
            transform=axes[0][0].transAxes,
            fontsize=12,
            fontweight="bold",
        )
        axes[0][1].text(
            0.5,
            1.35,
            "Aggregate Projections",
            ha="center",
            va="center",
            transform=axes[0][1].transAxes,
            fontsize=12,
            fontweight="bold",
        )
        axes[0][2].text(
            0.5,
            1.35,
            "Value Distribution",
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

    plt.subplots_adjust(
        left=0.06,  # move subplots slightly further left
        right=0.88,  # give more room for your colorbar on the right
        top=0.92,
        bottom=0.08,
        wspace=0.3,
        hspace=0.4,
    )

    # Save or show
    if filename:
        logger.info(f"Saving plot to file: {filename}")
        try:
            plt.savefig(filename, dpi=dpi)
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
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
