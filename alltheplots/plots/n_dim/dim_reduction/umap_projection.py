import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ....utils.logger import logger
import warnings


def create_umap_projection_plot(
    tensor_np, ax=None, n_neighbors=15, min_dist=0.1, sample_limit=5000
):
    """
    Create a 2D UMAP projection plot from n-dimensional data.

    Parameters:
        tensor_np (numpy.ndarray): The n-dimensional numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on
        n_neighbors (int): UMAP n_neighbors parameter (default: 15)
        min_dist (float): UMAP min_dist parameter (default: 0.1)
        sample_limit (int): Maximum number of samples to use (for performance)

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating UMAP projection")
    # Try to import UMAP - handle gracefully if not installed
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("UMAP is not installed. Install it with 'pip install umap-learn'")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:

        # Reshape to 2D array: (samples x features)
        tensor_shape = tensor_np.shape
        reshaped_data = tensor_np.reshape(
            -1, np.prod(tensor_shape[1:]) if len(tensor_shape) > 1 else 1
        )

        # Keep only finite values
        valid_mask = np.all(np.isfinite(reshaped_data), axis=1)
        clean_data = reshaped_data[valid_mask]

        if len(clean_data) == 0:
            raise ValueError("No finite values in data")

        # For very small datasets, UMAP won't work properly - require minimum number of samples
        if len(clean_data) < 5:
            raise ValueError(
                f"Too few data points for UMAP (got {len(clean_data)}, need at least 5)"
            )

        # When there are very few samples, warn rather than throwing an error
        if len(clean_data) < 20:
            logger.warning(
                f"UMAP works best with larger datasets. Only {len(clean_data)} points available."
            )

        # Sample points if there are too many
        if len(clean_data) > sample_limit:
            logger.debug(f"Sampling {sample_limit} points from {len(clean_data)} total points")
            indices = np.random.choice(len(clean_data), sample_limit, replace=False)
            clean_data = clean_data[indices]

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_data)

        # Create UMAP reducer with reasonable parameters
        # Make sure n_neighbors is at least 2 and less than the number of samples
        # For very small datasets, use an even smaller n_neighbors value
        adjusted_n_neighbors = max(2, min(min(15, n_neighbors), len(scaled_data) - 1))
        logger.debug(
            f"Using n_neighbors={adjusted_n_neighbors} for UMAP (from {len(scaled_data)} samples)"
        )

        # Create a color array based on the position in the original data
        color_values = np.arange(len(scaled_data)) / len(scaled_data)

        # Suppress UMAP warnings that are normal for small datasets or specific configurations
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="n_jobs value 1 overridden to 1 by setting random_state",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="n_neighbors is larger than the dataset size",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="A large number of your vertices were disconnected",
                category=UserWarning,
            )

            # Use reduced random_state for consistency and to avoid warning
            reducer = UMAP(
                n_components=2,
                n_neighbors=adjusted_n_neighbors,
                min_dist=min_dist,
                random_state=42,
                low_memory=True,  # Use less memory for large datasets
                metric="euclidean",  # Stick with simpler metrics for robustness
                verbose=False,  # Suppress additional output
            )

            # For very small datasets or high-dimensional data relative to samples,
            # we need special handling to avoid "zero-size array to reduction operation maximum"
            try:
                projection = reducer.fit_transform(scaled_data)
            except ValueError as e:
                if "zero-size array to reduction operation maximum" in str(e):
                    # This can happen with very few samples or high dimensions
                    # Fall back to a simpler embedding approach
                    logger.warning("UMAP failed due to dataset characteristics, using PCA fallback")
                    from sklearn.decomposition import PCA

                    pca = PCA(n_components=2)
                    projection = pca.fit_transform(scaled_data)
                    ax.set_title("PCA Fallback\n(UMAP failed)")
                else:
                    raise

        # Create a scatter plot with points colored by their position
        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=color_values,
            cmap="viridis",
            alpha=0.7,
            s=10,
            edgecolors="none",
        )

        # Set title and labels
        if "ax.set_title" not in locals():  # Only set if not already set above
            ax.set_title(
                f"UMAP Projection\nn_neighbors={adjusted_n_neighbors}, min_dist={min_dist}"
            )
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        # Remove ticks as they have no meaning in the embedded space
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create UMAP projection: {e}")
        ax.text(
            0.5,
            0.5,
            f"UMAP Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("UMAP Projection (Error)")

    return ax
