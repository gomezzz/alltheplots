import matplotlib.pyplot as plt
from ..utils.type_handling import to_numpy
from ..utils.logger import logger

# Import specialized plot functions for Nx3 data
from .three_dim.coordinate.scatter_3d import create_scatter_3d_plot
from .three_dim.coordinate.density_scatter_3d import create_density_scatter_3d_plot
from .three_dim.coordinate.trajectory_3d import create_trajectory_3d_plot
from .three_dim.coordinate.orthogonal_projections import create_orthogonal_projections_plot
from .three_dim.coordinate.profile_plot import create_profile_plot
from .three_dim.coordinate.projection_fft import create_projection_fft_plot
from .three_dim.coordinate.delaunay_mesh import create_delaunay_mesh_plot
from .three_dim.coordinate.convex_hull_3d import create_convex_hull_3d_plot
from .three_dim.coordinate.cluster_3d import create_cluster_3d_plot


def plot_2d_Nx3(tensor, filename=None, dpi=100, show=True):
    """
    Generate a specialized visualization for Nx3 tensors (3D coordinate data) with a 3×3 grid of plots:

    Column 1 (Direct 3D Visualizations):
    - Basic 3D Scatter Plot: Points in 3D space with transparent markers
    - 3D Scatter with Density Coloring: Color-coded by local density
    - 3D Trajectory or Line Plot: Connected points showing path through space

    Column 2 (Projection and Aggregation):
    - Orthogonal 2D Projections: XY, XZ, and YZ projections
    - Profile Plot (Mean or Median): Average or median values along one axis
    - 2D FFT Magnitude of a Projection: Frequency content of a projection

    Column 3 (Structural and Advanced Analyses):
    - Delaunay Tetrahedral Mesh or Wireframe: 3D connectivity visualization
    - Convex Hull or Alpha Shape Surface: Outline of data extent
    - Cluster Visualization: Points colored by cluster membership

    Parameters:
        tensor (array-like): The input Nx3 tensor to plot
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        show (bool): Whether to display the plot interactively (True) or just return the figure (False).

    Returns:
        matplotlib.figure.Figure: The figure containing the plots, or None if displayed
    """
    logger.info("Creating specialized Nx3 plot with 3×3 grid layout")

    # Convert to numpy array using our robust conversion utility
    try:
        tensor_np = to_numpy(tensor)
        if len(tensor_np.shape) != 2 or tensor_np.shape[1] != 3:
            raise ValueError(f"Expected Nx3 tensor, got shape {tensor_np.shape}")
        logger.debug(f"Converted tensor to numpy array of shape {tensor_np.shape}")
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise

    # Create a 3x3 grid of subplots with appropriate spacing
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.6)

    # Create all subplots
    axes = []
    for i in range(3):
        row = []
        for j in range(3):
            # Create 3D axes for columns 0 and 2; use 2D axes for column 1
            if j == 0 or j == 2:
                ax = fig.add_subplot(gs[i, j], projection="3d")
            else:
                ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        axes.append(row)

    try:
        # Column 1: Direct 3D Visualizations
        create_scatter_3d_plot(tensor_np, ax=axes[0][0])
        create_density_scatter_3d_plot(tensor_np, ax=axes[1][0])
        create_trajectory_3d_plot(tensor_np, ax=axes[2][0])

        # Column 2: Projection and Aggregation
        create_orthogonal_projections_plot(tensor_np, ax=axes[0][1], fig=fig)
        create_profile_plot(tensor_np, ax=axes[1][1])
        create_projection_fft_plot(tensor_np, ax=axes[2][1])

        # Column 3: Structural and Advanced Analyses
        create_delaunay_mesh_plot(tensor_np, ax=axes[0][2])
        create_convex_hull_3d_plot(tensor_np, ax=axes[1][2])
        create_cluster_3d_plot(tensor_np, ax=axes[2][2])

        # Add column headers
        # For 3D axes (columns 0 and 2), use text2D() method
        axes[0][0].text2D(
            0.5,
            1.25,
            "Direct 3D Visualizations",
            ha="center",
            va="center",
            transform=axes[0][0].transAxes,
            fontsize=11,
            fontweight="bold",
        )
        # Column 2 (2D axes) can use regular text call:
        axes[0][1].text(
            0.5,
            1.25,
            "Projection and Aggregation",
            ha="center",
            va="center",
            transform=axes[0][1].transAxes,
            fontsize=11,
            fontweight="bold",
        )
        axes[0][2].text2D(
            0.5,
            1.25,
            "Structural and Advanced Analyses",
            ha="center",
            va="center",
            transform=axes[0][2].transAxes,
            fontsize=11,
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
