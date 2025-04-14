import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ....utils.logger import logger


def create_orthogonal_projections_plot(tensor_np, ax=None, fig=None):
    """
    Create orthogonal 2D projections (XY, XZ, YZ) for Nx3 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx3 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        fig (matplotlib.figure.Figure, optional): The figure to use. Required when used within a grid.

    Returns:
        matplotlib.axes.Axes: The main axis with the plot
    """
    logger.debug("Creating orthogonal projections plot")

    # Check if we're within a subplot grid
    in_grid = ax is not None and fig is not None

    # Create standalone figure and axes if not in grid
    if not in_grid:
        fig, ax = plt.subplots(figsize=(8, 8))

    try:
        # Extract x, y, and z coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]
        z = tensor_np[:, 2]

        n_points = len(x)

        # Determine marker size based on number of points
        marker_size = max(3, min(20, 300 / n_points))

        # When in a grid, create the projection plots within the given axis
        if in_grid:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            # Main plot area will be XY projection
            ax.scatter(x, y, s=marker_size, alpha=0.6, edgecolor="k", linewidth=0.5)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("XY Projection")

            # Create inset for XZ projection (bottom)
            ax_xz = inset_axes(
                ax,
                width="40%",
                height="30%",
                loc="lower left",
                bbox_to_anchor=(0.05, 0.05, 0.4, 0.3),
                bbox_transform=ax.transAxes,
            )

            # Create inset for YZ projection (right)
            ax_yz = inset_axes(
                ax,
                width="30%",
                height="40%",
                loc="lower right",
                bbox_to_anchor=(0.65, 0.05, 0.3, 0.4),
                bbox_transform=ax.transAxes,
            )

        else:
            # For standalone figure, use gridspec for better layout
            gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.1, hspace=0.1)

            # Create axes
            ax = plt.subplot(gs[1, 0])  # XY projection (main)
            ax_xz = plt.subplot(gs[0, 0], sharex=ax)  # XZ projection (top)
            ax_yz = plt.subplot(gs[1, 1], sharey=ax)  # YZ projection (right)

            # Turn off shared tick labels
            plt.setp(ax_xz.get_xticklabels(), visible=False)
            plt.setp(ax_yz.get_yticklabels(), visible=False)

        # Determine if we should use scatter or density (heatmap) based on number of points
        use_density = n_points > 200

        if use_density:
            # For larger datasets, use 2D histograms for better visualization
            bins = max(20, min(50, int(np.sqrt(n_points / 5))))

            # XY projection
            h_xy = ax.hist2d(x, y, bins=bins, cmap="viridis", alpha=0.8)
            fig.colorbar(h_xy[3], ax=ax, pad=0.01, shrink=0.7)

            # XZ projection
            h_xz = ax_xz.hist2d(x, z, bins=bins, cmap="viridis", alpha=0.8)
            fig.colorbar(h_xz[3], ax=ax_xz, pad=0.01, shrink=0.7)

            # YZ projection
            h_yz = ax_yz.hist2d(z, y, bins=bins, cmap="viridis", alpha=0.8)
            fig.colorbar(h_yz[3], ax=ax_yz, pad=0.01, shrink=0.7)

        else:
            # For smaller datasets, use scatter plots
            # XY projection (main)
            ax.scatter(x, y, s=marker_size, alpha=0.6, edgecolor="k", linewidth=0.5)

            # XZ projection
            ax_xz.scatter(x, z, s=marker_size, alpha=0.6, edgecolor="k", linewidth=0.5)

            # YZ projection
            ax_yz.scatter(z, y, s=marker_size, alpha=0.6, edgecolor="k", linewidth=0.5)

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax_xz.set_ylabel("Z")
        ax_yz.set_xlabel("Z")

        # Set titles
        if in_grid:
            ax.set_title("Orthogonal Projections", fontsize=10)
            ax_xz.set_title("XZ", fontsize=8)
            ax_yz.set_title("YZ", fontsize=8)
        else:
            ax.set_title("XY Projection")
            ax_xz.set_title("XZ Projection")
            ax_yz.set_title("YZ Projection")

        # Add central plot title
        if not in_grid:
            fig.suptitle("Orthogonal 2D Projections", fontsize=14)

        # Add stats text
        stats_text = f"n = {n_points}"
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
        logger.error(f"Failed to create orthogonal projections plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Projections Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Orthogonal Projections (Error)")

    return ax
