import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ....utils.logger import logger


def create_convex_hull_3d_plot(tensor_np, ax=None):
    """
    Create a 3D convex hull visualization for Nx3 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx3 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 3D convex hull plot")

    # Create ax if not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    try:
        # Extract points
        points = tensor_np
        n_points = len(points)  # Need at least 4 non-coplanar points for 3D convex hull
        if n_points < 4:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for convex hull\n(need at least 4 points)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            # Still plot the points
            if n_points > 0:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=30, alpha=0.8, c="blue")
            ax.set_title("Convex Hull (Insufficient Data)")
            return ax

        # Check for coplanarity in small datasets
        if n_points == 4:
            # For exactly 4 points, we can check the volume of the tetrahedron
            # If it's zero or very small, the points are coplanar
            tetra_volume = (
                np.abs(np.linalg.det(np.vstack([points[1:] - points[0], np.ones(3)]))) / 6.0
            )

            is_coplanar = tetra_volume < 1e-10

            if is_coplanar:
                ax.text(
                    0.5,
                    0.5,
                    "Coplanar points detected\nCannot compute 3D convex hull",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                # Still plot the points
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=30, alpha=0.8, c="blue")
                ax.set_title("Convex Hull (Coplanar Data)")
                return ax

        # Create a scatter plot of the points with size based on number of points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=max(5, min(30, 500 / n_points)),
            alpha=0.6,
            c="blue",
            edgecolors="k",
            linewidth=0.3,
            label="Data Points",
        )

        # Compute 3D convex hull
        hull = ConvexHull(points)

        # Get hull faces (simplices)
        simplices = hull.simplices

        # Extract coordinates of hull faces
        faces = []
        for simplex in simplices:
            faces.append([points[simplex[0]], points[simplex[1]], points[simplex[2]]])

        # Create a Poly3DCollection
        poly3d = Poly3DCollection(
            faces, alpha=0.25, facecolor="cyan", edgecolor="k", linewidth=0.5  # Transparent
        )

        # Add the collection to the plot
        ax.add_collection3d(poly3d)

        # Optionally highlight the hull vertices
        hull_points = points[hull.vertices]
        ax.scatter(
            hull_points[:, 0],
            hull_points[:, 1],
            hull_points[:, 2],
            s=30,
            c="red",
            alpha=0.8,
            edgecolors="k",
            linewidth=0.5,
            label="Hull Vertices",
        )

        # Calculate hull volume and surface area
        volume = hull.volume
        area = hull.area

        # Calculate a metric for compactness
        # For a perfect sphere, this ratio is minimized
        # Higher values indicate less spherical shapes
        sphericity = (area**3) / (36 * np.pi * volume**2)

        # Add statistics
        stats_text = (
            f"n = {n_points}\n"
            f"hull vertices = {len(hull.vertices)}\n"
            f"hull faces = {len(simplices)}\n"
            f"volume = {volume:.2f}\n"
            f"surface area = {area:.2f}\n"
            f"sphericity = {sphericity:.3f}"
        )

        ax.text2D(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

        # Add labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Convex Hull")

        # Add legend
        ax.legend(loc="upper right", fontsize=8)

        # Set equal aspect ratio for better perception
        max_range = (
            np.array(
                [
                    points[:, 0].max() - points[:, 0].min(),
                    points[:, 1].max() - points[:, 1].min(),
                    points[:, 2].max() - points[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    except Exception as e:
        logger.error(f"Failed to create 3D convex hull plot: {e}")
        # Make sure points are still shown even if hull computation fails
        if n_points >= 3:
            try:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.7)
            except Exception:
                pass

        ax.text(
            0.5,
            0.5,
            f"Convex Hull Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Convex Hull (Error)")

    return ax
