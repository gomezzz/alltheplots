import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ....utils.logger import logger


def _safe_marker_size(n_points, min_size=5, max_size=30, scale_factor=500):
    """Helper function to calculate a safe marker size based on number of points"""
    # Ensure n_points is at least 1 to avoid division by zero
    safe_n = max(1, n_points)
    # Calculate and bound the marker size
    return max(min_size, min(max_size, scale_factor / safe_n))


def create_cluster_3d_plot(tensor_np, ax=None):
    """
    Create a 3D scatter plot with DBSCAN clustering for Nx3 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx3 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating 3D cluster plot")

    # Create ax if not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    try:
        # Extract points
        points = tensor_np
        n_points = len(points)  # Need enough points for meaningful clustering
        if n_points < 10:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for clustering\n(need at least 10 points)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )  # Still plot the points
            if n_points > 0:
                # Use the helper function for safe marker size calculation
                marker_size = _safe_marker_size(n_points)
                ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2], s=marker_size, alpha=0.8, c="blue"
                )
            ax.set_title("Clustering (Insufficient Data)")
            return ax

        # Normalize the data for better clustering
        try:
            scaled_points = StandardScaler().fit_transform(points)
        except Exception as e:
            logger.warning(f"Failed to scale data for clustering: {e}. Using raw data instead.")
            scaled_points = points

        # Determine appropriate DBSCAN parameters
        # For eps, we'll use a heuristic based on data distribution
        # For min_samples, we'll scale based on dataset size

        # Calculate average distance to k-nearest neighbors to determine eps
        from scipy.spatial import KDTree

        k = min(5, max(2, int(np.sqrt(n_points) / 4)))
        tree = KDTree(scaled_points)
        distances, _ = tree.query(scaled_points, k=k + 1)  # +1 because first point is self
        avg_knn_dist = np.mean(distances[:, 1 : k + 1])  # Skip the first (distance to self)

        # Set eps as a multiple of the average k-nn distance
        eps = avg_knn_dist * 1.2

        # Set min_samples based on dataset size, with some reasonable bounds
        min_samples = max(3, min(20, int(np.log(n_points) * 2)))

        # Run DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(scaled_points)

            # Get unique clusters and count noise points (-1 is noise)
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters[unique_clusters >= 0])
            n_noise = np.sum(clusters == -1)

            # Get a color map that works well for the actual number of clusters
            if n_clusters <= 10:
                cmap = plt.cm.tab10
            elif n_clusters <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.viridis

            # Create a color list
            colors = []
            for i in range(len(clusters)):
                if clusters[i] == -1:
                    # Black for noise points
                    colors.append((0, 0, 0, 1))  # RGBA black
                else:
                    # Colormap for cluster points
                    rgba = cmap(clusters[i] % cmap.N)
                    colors.append(rgba)  # Scatter plot with clusters
            # Ensure marker size is positive and reasonable
            marker_size = max(5, min(30, 500 / max(1, n_points)))
            _ = ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=marker_size,
                c=colors,
                alpha=0.8,
                edgecolors="k",
                linewidth=0.3,
            )

            # Add stats text
            stats_text = (
                f"n = {n_points}\n"
                f"clusters = {n_clusters}\n"
                f"noise = {n_noise} ({n_noise/n_points:.1%})\n"
                f"eps = {eps:.2f}, min_samples = {min_samples}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            # Add a legend for cluster counts
            from matplotlib.lines import Line2D

            legend_elements = []

            # Only show up to 8 largest clusters in the legend to avoid clutter
            if n_clusters > 0:
                # Get the size of each cluster
                cluster_sizes = {}
                for c in unique_clusters:
                    if c >= 0:  # Skip noise cluster
                        cluster_sizes[c] = np.sum(clusters == c)

                # Sort clusters by size (descending)
                sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

                # Keep only the top 8 clusters for the legend
                top_clusters = sorted_clusters[:8]

                # Create legend elements for top clusters
                for cluster_id, size in top_clusters:
                    color = cmap(cluster_id % cmap.N)
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=8,
                            label=f"Cluster {cluster_id}: {size} points",
                        )
                    )

            # Add noise to legend if present
            if n_noise > 0:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="black",
                        markersize=8,
                        label=f"Noise: {n_noise} points",
                    )
                )

            if legend_elements:
                ax.legend(
                    handles=legend_elements,
                    loc="upper right",
                    fontsize=7,
                    bbox_to_anchor=(1, 1),
                    bbox_transform=ax.transAxes,
                )

        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}. Showing unclustered points.")

            # Just show the unclustered points
            # Use the helper function for safe marker size calculation
            marker_size = _safe_marker_size(n_points)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=marker_size,
                alpha=0.7,
                c="blue",
                edgecolors="k",
                linewidth=0.3,
            )

            ax.text(
                0.5,
                0.5,
                f"Clustering failed: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
            )

        # Add labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("DBSCAN Clustering")

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
        logger.error(f"Failed to create cluster plot: {e}")
        # Make sure points are still shown even if clustering fails
        try:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.7)
        except Exception:
            pass

        ax.text(
            0.5,
            0.5,
            f"Cluster Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Clustering (Error)")

    return ax
