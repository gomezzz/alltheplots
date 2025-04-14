import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ....utils.logger import logger


def create_delaunay_mesh_plot(tensor_np, ax=None):
    """
    Create a 3D Delaunay tetrahedral mesh visualization for Nx3 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx3 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 3D Delaunay mesh plot")

    # Create ax if not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    # Ensure ax is a 3D axes (should have add_collection3d)
    if ax is None or not hasattr(ax, "add_collection3d"):
        raise ValueError(
            "A 3D Axes (with projection='3d') is required for create_delaunay_mesh_plot"
        )

    try:
        # Extract x, y, and z coordinates
        points = tensor_np
        n_points = len(points)

        # Need at least 4 non-coplanar points for 3D Delaunay triangulation
        if n_points < 4:
            ax.text(
                0.5,
                0.5,
                0,
                "Insufficient data for Delaunay mesh\n(need at least 4 points)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            if n_points > 0:
                # Removed s=marker_size
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.8, c="blue")
            ax.set_title("Delaunay Mesh (Insufficient Data)")
            return ax

        # Check for coplanarity by attempting to fit a plane
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
                    0,
                    "Coplanar points detected\nCannot compute 3D Delaunay",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )  # Still plot the points
                ax.scatter(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    alpha=0.8,
                    c="blue",
                )
                ax.set_title("Delaunay Mesh (Coplanar Data)")
                return ax  # Create a scatter plot of the points
        # Use the helper function for safe marker size calculation

        _ = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            alpha=0.6,
            c="blue",
            edgecolors="k",
            linewidths=0.3,
        )

        # Compute 3D Delaunay triangulation
        try:
            delaunay = Delaunay(points)

            # Get simplices (tetrahedra)
            tetrahedra = delaunay.simplices
            n_tetra = len(tetrahedra)

            # For visualization, we'll either:
            # 1. For smaller datasets: Show the wireframe of all edges
            # 2. For larger datasets: Show triangular surfaces of the convex hull

            # Threshold for switching between modes
            wireframe_threshold = 50  # Number of points

            if n_points <= wireframe_threshold:
                # Edge visualization (wireframe)
                # Extract all edges from the tetrahedra
                edges = set()
                for tetra in tetrahedra:
                    # Each tetrahedron has 6 edges
                    for i in range(4):
                        for j in range(i + 1, 4):
                            # Sort so we don't add the same edge twice
                            edge = tuple(sorted([tetra[i], tetra[j]]))
                            edges.add(edge)

                # Plot each edge
                for i, j in edges:
                    ax.plot(
                        [points[i, 0], points[j, 0]],
                        [points[i, 1], points[j, 1]],
                        [points[i, 2], points[j, 2]],
                        "k-",
                        alpha=0.3,
                        linewidth=0.5,
                    )

                # Add stats
                stats_text = f"n = {n_points}\n" f"edges = {len(edges)}\n" f"tetrahedra = {n_tetra}"

            else:
                # Surface visualization
                # For larger datasets, we'll just show the convex hull triangles
                hull_triangles = set()
                for tetra in tetrahedra:
                    # Each tetrahedron has 4 triangular faces
                    # A face is on the convex hull if it appears only once
                    faces = [
                        tuple(sorted([tetra[0], tetra[1], tetra[2]])),
                        tuple(sorted([tetra[0], tetra[1], tetra[3]])),
                        tuple(sorted([tetra[0], tetra[2], tetra[3]])),
                        tuple(sorted([tetra[1], tetra[2], tetra[3]])),
                    ]

                    for face in faces:
                        if face in hull_triangles:
                            hull_triangles.remove(face)
                        else:
                            hull_triangles.add(face)

                # Convert to list of triangles
                hull_triangles = [list(tri) for tri in hull_triangles]

                # Create a Poly3DCollection
                triangles = [[points[i] for i in triangle] for triangle in hull_triangles]

                # Create the mesh surface with transparency
                mesh = Poly3DCollection(
                    triangles, alpha=0.2, facecolor="cyan", edgecolor="k", linewidth=0.5
                )

                # Add the collection to the plot
                ax.add_collection3d(mesh)

                # Add stats
                stats_text = (
                    f"n = {n_points}\n"
                    f"hull triangles = {len(hull_triangles)}\n"
                    f"tetrahedra = {n_tetra}"
                )  # Add stats text
            ax.text(
                0.02,
                0.98,
                0,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

        except Exception as e:
            logger.warning(f"Delaunay computation failed: {e}. Showing points only.")
            ax.text(
                0.5,
                0.5,
                0,
                f"Delaunay computation failed:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )

        # Add axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set title
        ax.set_title("Delaunay Tetrahedral Mesh")

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
        logger.error(f"Failed to create Delaunay mesh plot: {e}")
        ax.text(
            0.5,
            0.5,
            0,
            f"Delaunay Mesh Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Delaunay Mesh (Error)")

    return ax
