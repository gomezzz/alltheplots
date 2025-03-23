import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ....utils.logger import logger


def create_kde_plot(tensor_np, ax=None):
    """
    Create a Kernel Density Estimation (KDE) plot of pixel intensities.

    Parameters:
        tensor_np (numpy.ndarray): The 2D numpy array to analyze
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating KDE plot of pixel intensities")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Flatten the array for KDE
        flat_data = tensor_np.flatten()

        # Remove any non-finite values
        clean_data = flat_data[np.isfinite(flat_data)]

        if len(clean_data) == 0:
            raise ValueError("No finite values in data")

        # Check if data is constant
        unique_vals = np.unique(clean_data)
        n_unique = len(unique_vals)

        if n_unique <= 1:
            # Handle constant data case
            logger.debug("Constant data detected, showing bar instead of KDE")
            if n_unique == 1:
                val = unique_vals[0]
                # Create a single bar for the constant value
                ax.bar(
                    [val], [1.0], width=abs(val) / 100 if val != 0 else 0.1, color="blue", alpha=0.5
                )
                ax.text(val, 0.5, f"Constant\nValue: {val}", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes)

            # Set reasonable axis limits
            if n_unique == 1:
                margin = abs(val) * 0.1 if val != 0 else 0.1
                ax.set_xlim(val - margin, val + margin)
            ax.set_ylim(0, 1.1)
        else:
            # Determine if we need to adjust bandwidth based on data characteristics
            data_range = np.ptp(clean_data)

            if n_unique <= 5:
                # For very discrete data, use a smaller bandwidth
                bw = data_range / 20 if data_range > 0 else 0.1
                logger.debug(f"Using small bandwidth ({bw:.3f}) for discrete data")
            else:
                # Let seaborn determine bandwidth automatically
                bw = "scott"
                logger.debug("Using automatic bandwidth selection")

            # Create KDE plot with fill and individual samples marked
            sns.kdeplot(
                data=clean_data,
                ax=ax,
                fill=True,
                alpha=0.5,
                bw_method=bw,
                color="blue",
                linewidth=1,
                warn_singular=False,
            )

            # Add rug plot for actual data points if there aren't too many
            if len(clean_data) <= 1000:
                ax.plot(
                    clean_data,
                    np.zeros_like(clean_data),
                    "|",
                    color="black",
                    alpha=0.5,
                    markersize=10,
                )

            # Add median and quartile lines
            quartiles = np.percentile(clean_data, [25, 50, 75])
            for q, label in zip(quartiles, ["Q1", "Median", "Q3"]):
                ax.axvline(x=q, color="red", linestyle="--", alpha=0.3)
                ax.text(q, ax.get_ylim()[1], label, rotation=90, va="top", fontsize=8)

        # Set plot labels and title
        ax.set_title("Pixel Intensity Distribution")
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Density")

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.error(f"Failed to create KDE plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"KDE Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("KDE Plot (Error)")

    return ax
