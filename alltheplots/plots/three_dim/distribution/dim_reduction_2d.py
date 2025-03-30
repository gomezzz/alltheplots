import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from ....utils.logger import logger


def create_2d_projection_plot(tensor_np, ax=None, method="tsne", sample_limit=5000):
    """
    Create a 2D projection plot from 3D data using dimensionality reduction.

    Parameters:
        tensor_np (numpy.ndarray): The 3D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on
        method (str): The dimensionality reduction method ('tsne' or 'pca')
        sample_limit (int): Maximum number of samples to use (for performance)

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug(f"Creating 2D projection using {method}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Flatten the 3D array into a 2D array where each row is a position
        # and its corresponding voxel value
        positions = []
        values = []

        # Get all coordinates and values
        for i in range(tensor_np.shape[0]):
            for j in range(tensor_np.shape[1]):
                for k in range(tensor_np.shape[2]):
                    if np.isfinite(tensor_np[i, j, k]):
                        positions.append([i, j, k])
                        values.append(tensor_np[i, j, k])

        if not positions:
            raise ValueError("No finite values in data")

        # Convert to numpy arrays
        positions = np.array(positions)
        values = np.array(values)

        # Sample points if there are too many (for performance)
        if len(positions) > sample_limit:
            logger.debug(f"Sampling {sample_limit} points from {len(positions)} total points")
            indices = np.random.choice(len(positions), sample_limit, replace=False)
            positions = positions[indices]
            values = values[indices]

        # Normalize the positions
        scaler = StandardScaler()
        positions_scaled = scaler.fit_transform(positions)

        # Perform dimensionality reduction
        if method.lower() == "tsne":
            projection = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(positions_scaled) - 1)
            )
            embedding = projection.fit_transform(positions_scaled)
        else:  # Default to PCA
            from sklearn.decomposition import PCA

            projection = PCA(n_components=2)
            embedding = projection.fit_transform(positions_scaled)

        # Create a scatter plot with points colored by their values
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=values,
            cmap="viridis",
            alpha=0.7,
            s=10,
            edgecolors="none",
        )

        # Add a colorbar
        plt.colorbar(scatter, ax=ax, label="Value")

        # Set plot labels and title
        ax.set_title(f"2D {method.upper()} Projection")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        # Remove ticks as they have no meaning in the embedded space
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create 2D projection plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"2D projection Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("2D Projection (Error)")

    return ax
