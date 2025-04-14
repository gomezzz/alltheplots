import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ....utils.logger import logger


def create_kde_contour_plot(tensor_np, ax=None):
    """
    Create a bivariate KDE contour plot for Nx2 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx2 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating KDE contour plot")

    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    try:
        # Extract x and y coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]

        # Handle very small datasets
        if len(x) < 20:
            # Just show scatter for small datasets
            ax.scatter(x, y, alpha=0.7)
            ax.text(
                0.5,
                0.95,
                "Insufficient data for KDE contour plot\nshowing scatter instead",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )
        else:
            # Calculate kernel bandwidth based on data characteristics
            # Scott's rule adapted to bivariate case
            def scott_factor(n):
                return np.power(n, -1 / 6)

            try:
                # Stack coordinates for KDE calculation
                xy = np.vstack([x, y])

                # Calculate bandwidth factor
                bw_factor = scott_factor(len(x))

                # Create KDE
                kde = gaussian_kde(xy, bw_method=bw_factor)

                # Create a grid to evaluate the KDE
                x_range = np.max(x) - np.min(x)
                y_range = np.max(y) - np.min(y)
                x_margin = x_range * 0.1
                y_margin = y_range * 0.1

                # Determine grid resolution based on data size
                grid_size = min(100, max(40, int(np.sqrt(len(x) * 2))))

                # Create grid for evaluation
                x_grid = np.linspace(np.min(x) - x_margin, np.max(x) + x_margin, grid_size)
                y_grid = np.linspace(np.min(y) - y_margin, np.max(y) + y_margin, grid_size)
                X, Y = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X.ravel(), Y.ravel()])

                # Evaluate KDE on grid
                Z = np.reshape(kde(positions).T, X.shape)

                # Determine contour levels
                # Use percentile-based levels for better visualization
                density_threshold = np.max(Z) * 0.01
                masked_Z = Z[Z > density_threshold]

                if len(masked_Z) > 0:
                    percentiles = [20, 40, 60, 80, 95]
                    levels = [np.percentile(masked_Z, p) for p in percentiles]

                    # Plot filled contours
                    contour_filled = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.7)

                    # Add contour lines
                    contour_lines = ax.contour(
                        X, Y, Z, levels=levels, colors="white", linewidths=0.5, alpha=0.7
                    )

                    # Add contour labels
                    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

                    # Add colorbar
                    plt.colorbar(contour_filled, ax=ax, label="Density")

                    # Overlay scatter for smaller datasets for better visualization
                    if len(x) < 500:
                        alpha = max(0.1, min(0.5, 50 / len(x)))
                        ms = max(3, min(10, 300 / len(x)))
                        ax.scatter(
                            x, y, alpha=alpha, s=ms, color="white", edgecolor="black", linewidth=0.5
                        )

                # Add some stats
                ax.text(
                    0.02,
                    0.98,
                    f"n = {len(x)}\nbw = {bw_factor:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                )

            except Exception as e:
                logger.warning(f"KDE calculation failed: {e}. Falling back to scatter plot.")
                ax.scatter(x, y, alpha=0.7)
                ax.text(
                    0.5,
                    0.95,
                    f"KDE calculation failed: {str(e)}\nshowing scatter instead",
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                )

        # Set plot title and labels
        ax.set_title("Bivariate KDE Contour Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    except Exception as e:
        logger.error(f"Failed to create KDE contour plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"KDE Contour Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("KDE Contour (Error)")

    return ax
