import numpy as np
import matplotlib.pyplot as plt
from ....utils.logger import logger


def create_scatter_marginal_plot(tensor_np, ax=None, fig=None):
    """
    Create a scatter plot with marginal histograms for Nx2 data.

    Parameters:
        tensor_np (numpy.ndarray): The Nx2 numpy array to visualize
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        fig (matplotlib.figure.Figure, optional): The matplotlib figure to use. Required when used within a grid.

    Returns:
        matplotlib.axes.Axes: The main axis with the scatter plot
    """
    logger.debug("Creating scatter plot with marginal histograms")

    # Check if we're within a subplot grid
    in_grid = ax is not None and fig is not None

    # Create standalone figure and axes if not in grid
    if not in_grid:
        fig, ax = plt.subplots(figsize=(7, 7))

    try:
        # Extract x and y coordinates
        x = tensor_np[:, 0]
        y = tensor_np[:, 1]  # Set up axes for marginal distributions
        if in_grid:
            # When in a grid, create the marginal plots within the given axis
            # This is tricky because we need to create "inset" axes
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            # Create inset axes for marginals - make them larger with more padding
            # Top marginal (x distribution) - increased height from 20% to 30%
            ax_top = inset_axes(
                ax,
                width="100%",
                height="30%",  # Increased from 20% to 30%
                loc="upper center",
                bbox_to_anchor=(0, 1.2, 1, 0.3),  # Increased height and moved up
                bbox_transform=ax.transAxes,
            )

            # Right marginal (y distribution) - increased width from 20% to 30%
            ax_right = inset_axes(
                ax,
                width="30%",  # Increased from 20% to 30%
                height="100%",
                loc="right",
                bbox_to_anchor=(1.2, 0, 0.3, 1),  # Increased width and moved right
                bbox_transform=ax.transAxes,
            )
        else:
            # For standalone figure, use gridspec for better layout
            import matplotlib.gridspec as gridspec

            # Create the gridspec with improved ratios
            gs = gridspec.GridSpec(
                2,
                2,
                width_ratios=[
                    3,
                    1,
                ],  # Changed from [4, 1] to [3, 1] to give more space to y-marginal
                height_ratios=[
                    1,
                    3,
                ],  # Changed from [1, 4] to [1, 3] to give more space to x-marginal
                wspace=0.1,  # Increased spacing slightly
                hspace=0.1,  # Increased spacing slightly
            )

            # Create axes
            ax = plt.subplot(gs[1, 0])  # Main scatter plot
            ax_top = plt.subplot(gs[0, 0], sharex=ax)  # Top marginal (x distribution)
            ax_right = plt.subplot(gs[1, 1], sharey=ax)  # Right marginal (y distribution)

            # Turn off tick labels on shared axes
            plt.setp(ax_top.get_xticklabels(), visible=False)
            plt.setp(ax_right.get_yticklabels(), visible=False)

        # Create scatter plot
        _ = ax.scatter(
            x, y, alpha=0.6, s=min(50, max(10, 500 / len(x))), edgecolor="k", linewidth=0.5
        )

        # Determine optimal bin count using Freedman-Diaconis rule
        def freedman_diaconis_bins(data):
            # IQR-based bin width estimation
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            bin_width = 2 * iqr / (len(data) ** (1 / 3)) if iqr > 0 else 1
            # Calculate number of bins based on range and width
            data_range = np.max(data) - np.min(data)
            num_bins = int(np.ceil(data_range / bin_width)) if bin_width > 0 else 10
            return max(5, min(50, num_bins))

        # Determine whether to use histograms or KDE/violin based on data characteristics
        use_kde = len(x) > 100 and len(np.unique(x)) > 20 and len(np.unique(y)) > 20

        if use_kde:
            try:  # Try to import seaborn for KDE
                import seaborn as sns

                # Create KDE plots
                sns.kdeplot(x=x, ax=ax_top, color="blue", fill=True, alpha=0.5)

                # Fix for the deprecation warning - use y parameter instead of vertical=True
                # Also rotate the Y axis so it aligns better with the main scatter plot
                sns.kdeplot(y=y, ax=ax_right, color="blue", fill=True, alpha=0.5)

                # Properly orient the Y axis ticks and labels
                ax_right.tick_params(axis="y", labelleft=False, labelright=True)
                ax_right.yaxis.set_label_position("right")

                logger.debug("Using KDE for marginal distributions")
            except Exception as e:
                logger.warning(f"KDE failed: {e}. Falling back to histograms.")
                use_kde = False

        if not use_kde:
            # Use histograms
            bins_x = freedman_diaconis_bins(x)
            bins_y = freedman_diaconis_bins(y)  # Create histograms
            ax_top.hist(x, bins=bins_x, alpha=0.7, color="blue", edgecolor="k", linewidth=0.5)
            ax_right.hist(
                y,
                bins=bins_y,
                alpha=0.7,
                orientation="horizontal",
                color="blue",
                edgecolor="k",
                linewidth=0.5,
            )

            # Properly orient the Y axis ticks and labels for better readability
            ax_right.tick_params(axis="y", labelleft=False, labelright=True)
            ax_right.yaxis.set_label_position("right")

            logger.debug(f"Using histograms with {bins_x} and {bins_y} bins for marginals")

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Set titles for marginal plots
        if in_grid:
            # For inset axes, we need to be more careful with titles
            ax_top.set_title("X Distribution", fontsize=8)
            ax_right.set_title("Y Distribution", fontsize=8)

            # Make tick labels smaller for insets
            ax_top.tick_params(axis="both", which="major", labelsize=6)
            ax_right.tick_params(axis="both", which="major", labelsize=6)
        else:
            ax_top.set_title("X Distribution")
            ax_right.set_title("Y Distribution")

        # Remove unnecessary spines
        ax_top.spines["right"].set_visible(False)
        ax_top.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.spines["top"].set_visible(False)

        # Add grid to main plot
        ax.grid(True, linestyle="--", alpha=0.3)

        # Set main plot title
        ax.set_title("Scatter with Marginal Distributions")

        # Add correlation information
        if len(x) > 2:
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(
                0.02,
                0.98,
                f"n = {len(x)}\nr = {corr:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

    except Exception as e:
        logger.error(f"Failed to create scatter marginal plot: {e}")
        ax.text(
            0.5,
            0.5,
            f"Marginal Plot Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("Scatter with Marginals (Error)")

    return ax
