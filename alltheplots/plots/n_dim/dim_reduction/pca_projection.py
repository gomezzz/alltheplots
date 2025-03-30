import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from ....utils.logger import logger


def create_pca_projection_plot(tensor_np, ax=None, n_components=2, sample_limit=5000):
    """
    Create a 2D PCA projection plot from n-dimensional data.

    Parameters:
        tensor_np (numpy.ndarray): The n-dimensional numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on
        n_components (int): Number of PCA components to compute (default: 2)
        sample_limit (int): Maximum number of samples to use (for performance)

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug(f"Creating {n_components}D PCA projection")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Suppress scikit-learn deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'force_all_finite' was renamed to 'ensure_all_finite'",
                category=FutureWarning,
            )

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

            # Sample points if there are too many
            if len(clean_data) > sample_limit:
                logger.debug(f"Sampling {sample_limit} points from {len(clean_data)} total points")
                indices = np.random.choice(len(clean_data), sample_limit, replace=False)
                clean_data = clean_data[indices]

            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clean_data)

            # Apply PCA
            pca = PCA(n_components=n_components)
            projection = pca.fit_transform(scaled_data)

            # Create a color array based on the position in the original data
            # Use the first dimension for coloring
            color_values = np.arange(len(projection)) / len(projection)

            ax.scatter(
                projection[:, 0],
                projection[:, 1],
                c=color_values,  # Use color values to avoid warnings
                cmap="viridis",
                alpha=0.7,
                s=10,
                edgecolors="none",
            )

            # Add explained variance information
            explained_variance = pca.explained_variance_ratio_
            total_explained_var = sum(explained_variance[:n_components]) * 100
            ax.set_title(f"PCA Projection\n{total_explained_var:.1f}% variance explained")

            # Label axes with variance explained
            ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
            ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%})")

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create PCA projection: {e}")
        ax.text(
            0.5,
            0.5,
            f"PCA Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("PCA Projection (Error)")

    return ax
