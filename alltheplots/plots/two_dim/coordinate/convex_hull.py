import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from ....utils.logger import logger


def create_convex_hull_plot(tensor_np, ax=None):
    """
    Create a scatter plot with convex hull outline for Nx2 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx2 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating convex hull plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Extract points
        points = tensor_np

        # Handle small datasets
        if len(points) < 4:
            # Just show scatter for very small datasets (need at least 3 points for hull)
            ax.scatter(points[:, 0], points[:, 1], alpha=0.7)
            ax.text(
                0.5,
                0.95,
                "Insufficient data for convex hull\nNeed at least 4 points",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )
        else:
            # Detect collinear points which can cause problems with hull computation
            if len(points) <= 4:
                # For very small datasets, check if all points are collinear
                x, y = points[:, 0], points[:, 1]

                # Check if points are approximately collinear
                if len(points) == 3:
                    # For 3 points, check if the area of the triangle is near zero
                    area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
                    is_collinear = area < 1e-10
                else:
                    # For 4 points, can use determinants of each triplet
                    is_collinear = True
                    for i in range(len(points)):
                        p1, p2, p3 = np.delete(points, i, axis=0)
                        area = 0.5 * abs(
                            (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                        )
                        if area > 1e-10:
                            is_collinear = False
                            break

                if is_collinear:
                    # Just show scatter for collinear points
                    ax.scatter(points[:, 0], points[:, 1], alpha=0.7)
                    ax.text(
                        0.5,
                        0.95,
                        "Collinear points detected\nCannot compute convex hull",
                        ha="center",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    )
                    raise ValueError("Collinear points detected")

            # Create scatter plot
            _ = ax.scatter(
                points[:, 0],
                points[:, 1],
                alpha=0.7,
                s=min(50, max(10, 500 / len(points))),
                edgecolor="k",
                linewidth=0.5,
            )

            # Compute convex hull
            hull = ConvexHull(points)

            # Get hull points and append the first point to close the polygon
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])

            # Plot the hull
            ax.plot(
                hull_points[:, 0],
                hull_points[:, 1],
                "r-",
                alpha=0.8,
                linewidth=2,
                label="Convex Hull",
            )

            # Highlight hull vertices
            ax.plot(hull_points[:, 0], hull_points[:, 1], "ro", alpha=0.8, markersize=6)

            # Calculate hull area and perimeter
            area = hull.volume  # In 2D, volume is actually area

            # Calculate perimeter
            perimeter = 0
            for i in range(len(hull_points) - 1):
                perimeter += np.sqrt(np.sum((hull_points[i + 1] - hull_points[i]) ** 2))

            # Add statistics
            stats_text = (
                f"n = {len(points)}\n"
                f"hull points = {len(hull.vertices)}\n"
                f"area = {area:.2f}\n"
                f"perimeter = {perimeter:.2f}"
            )

            # Calculate the convex hull ratio (compact measure)
            # A perfect circle has ratio = 1, more irregular shapes have lower values
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter**2)
                stats_text += f"\ncompactness = {compactness:.3f}"

            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            # Add legend
            ax.legend(loc="lower right")

        # Set plot title and labels
        ax.set_title("Convex Hull Outline")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create convex hull plot: {e}")
        # Make sure points are still shown even if hull computation fails
        if len(points) >= 3:
            try:
                ax.scatter(points[:, 0], points[:, 1], alpha=0.7)
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
