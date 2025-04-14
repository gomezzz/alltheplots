import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from ....utils.logger import logger


def create_delaunay_voronoi_plot(tensor_np, ax=None, method="delaunay"):
    """
    Create a Delaunay triangulation or Voronoi diagram overlay on scatter plot for Nx2 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx2 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        method (str): The method to use, either "delaunay" or "voronoi". Default is "delaunay".
            Automatically selects based on data size if not specified.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug(f"Creating {method} plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Extract x and y coordinates
        points = tensor_np

        # Handle small datasets
        if len(points) < 4:
            # Just show scatter for very small datasets (not enough for triangulation)
            ax.scatter(points[:, 0], points[:, 1], alpha=0.7)
            ax.text(
                0.5,
                0.95,
                f"Insufficient data for {method}\nNeed at least 4 points",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )
        else:
            # Detect duplicate points which can cause problems
            # Small amount of jitter to avoid computational issues
            if len(points) != len(np.unique(points.round(decimals=10), axis=0)):
                logger.warning("Duplicate points detected, adding small jitter for computation")
                # Add very small random jitter to avoid computational issues
                jitter_scale = 1e-10
                points = points + np.random.normal(0, jitter_scale, points.shape)

            # Automatically select method based on data size if "auto"
            if method == "auto":
                method = "delaunay" if len(points) < 100 else "voronoi"
                logger.debug(f"Auto-selected {method} based on data size")

            # Generate scatter plot of points
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                alpha=0.7,
                s=min(50, max(10, 500 / len(points))),
                edgecolor="k",
                linewidth=0.5,
            )

            if method == "delaunay":
                # Create Delaunay triangulation
                try:
                    tri = Delaunay(points)

                    # Plot the triangulation
                    ax.triplot(
                        points[:, 0],
                        points[:, 1],
                        tri.simplices,
                        color="red",
                        alpha=0.5,
                        linewidth=0.8,
                    )

                    # Add stats
                    stats_text = f"n = {len(points)}\n" f"triangles = {len(tri.simplices)}"

                    title = "Delaunay Triangulation"
                except Exception as e:
                    logger.warning(f"Delaunay triangulation failed: {e}")
                    ax.text(
                        0.5,
                        0.5,
                        f"Triangulation error: {str(e)}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="red",
                        fontsize=9,
                    )
                    stats_text = f"n = {len(points)}"
                    title = "Delaunay (Failed)"

            else:  # method == "voronoi"
                # Voronoi diagrams sometimes fail with edge cases
                try:
                    # Compute Voronoi diagram
                    vor = Voronoi(points)

                    # Plot the Voronoi diagram
                    voronoi_plot_2d(
                        vor,
                        ax=ax,
                        show_vertices=False,
                        line_colors="green",
                        line_width=0.8,
                        line_alpha=0.6,
                        point_size=0,  # Don't plot points again
                    )

                    # Add stats
                    stats_text = f"n = {len(points)}\n" f"regions = {len(vor.regions)}"

                    title = "Voronoi Diagram"
                except Exception as e:
                    logger.warning(f"Voronoi diagram failed: {e}")
                    ax.text(
                        0.5,
                        0.5,
                        f"Voronoi error: {str(e)}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="red",
                        fontsize=9,
                    )
                    stats_text = f"n = {len(points)}"
                    title = "Voronoi (Failed)"

            # Add stats box
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            # Set title
            ax.set_title(title)

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create {method} plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"{method.capitalize()} Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title(f"{method.capitalize()} (Error)")

    return ax
