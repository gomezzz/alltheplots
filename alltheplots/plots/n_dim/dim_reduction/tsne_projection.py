import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
from ....utils.logger import logger


def create_tsne_projection_plot(tensor_np, ax=None, perplexity=30, sample_limit=5000):
    """
    Create a 2D t-SNE projection plot from n-dimensional data.

    Parameters:
        tensor_np (numpy.ndarray): The n-dimensional numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on
        perplexity (int): t-SNE perplexity parameter (default: 30)
        sample_limit (int): Maximum number of samples to use (for performance)

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug(f"Creating t-SNE projection")

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

        # Check for constant data (like all zeros) which t-SNE can't handle well
        if np.allclose(clean_data, clean_data[0]):
            raise ValueError("All values are identical - t-SNE requires variation in the data")

        # Sample points if there are too many
        if len(clean_data) > sample_limit:
            logger.debug(f"Sampling {sample_limit} points from {len(clean_data)} total points")
            indices = np.random.choice(len(clean_data), sample_limit, replace=False)
            clean_data = clean_data[indices]

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_data)

        # Verify there's actual variance in the data after scaling
        if np.allclose(scaled_data, 0) or np.allclose(scaled_data, scaled_data[0]):
            raise ValueError("Data has no variance after scaling - t-SNE requires variation")

        # Apply t-SNE with safety measures
        # Cap perplexity to be less than n_samples - 1
        adjusted_perplexity = min(perplexity, len(scaled_data) - 1)

        # Suppress scikit-learn deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'force_all_finite' was renamed to 'ensure_all_finite'",
                category=FutureWarning,
            )

            # Use early_exaggeration=1 for constant data to prevent numerical issues
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=adjusted_perplexity,
                max_iter=250,
                init="random",  # More stable than PCA init for edge cases
            )
            projection = tsne.fit_transform(scaled_data)

        # Create a color array based on the position in the original data
        color_values = np.arange(len(projection)) / len(projection)

        # Create a scatter plot with points colored by their density
        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=color_values,  # Add color values to avoid warning
            cmap="viridis",
            alpha=0.7,
            s=10,
            edgecolors="none",
        )

        # Set title and labels
        ax.set_title(f"t-SNE Projection\nperplexity={adjusted_perplexity}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        # Remove ticks as they have no meaning in the embedded space
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create t-SNE projection: {e}")
        ax.text(
            0.5,
            0.5,
            f"t-SNE Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("t-SNE Projection (Error)")

    return ax
