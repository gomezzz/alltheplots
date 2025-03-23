import seaborn as sns
import matplotlib.pyplot as plt
from ..utils.type_handling import to_numpy
from ..utils.logger import logger

# Import functions from the individual plot modules
from .one_dim.time_domain import (
    create_line_scatter_plot,
    create_line_scatter_logx_plot,
    create_line_scatter_logy_plot,
)
from .one_dim.frequency_domain import (
    create_fft_magnitude_plot,
    create_autocorrelation_plot,
    create_psd_plot,
)
from .one_dim.distribution import create_hist_kde_plot, create_violin_plot, create_cdf_plot


def plot_1d(tensor, filename=None, dpi=100, show=True, remove_dc=True):
    """
    Generate a comprehensive visualization of a 1D tensor with a 3×3 grid of plots:

    Column 1 (Time Domain):
    - Line/scatter plot (auto-selects based on data characteristics)
    - Line/scatter plot with x-log scale
    - Line/scatter plot with y-log scale

    Column 2 (Frequency/Derived):
    - FFT Magnitude (auto log-scale magnitude if large dynamic range)
    - Autocorrelation
    - Power Spectral Density (PSD)

    Column 3 (Distribution):
    - Histogram + KDE (auto log-scale if sensible, discrete bar if few unique values)
    - Violin plot
    - Cumulative Distribution Function (CDF)

    Parameters:
        tensor (array-like): The input 1D tensor to plot
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        show (bool): Whether to display the plot interactively (True) or just return the figure (False).
                   Default is True except in test environments.
        remove_dc (bool): Whether to remove the DC component from the FFT plot. Default is True.

    Returns:
        matplotlib.figure.Figure: The figure containing the plots, or None if displayed
    """
    logger.info("Creating 1D plot with 3×3 grid layout")

    # Convert to numpy array using our robust conversion utility
    try:
        tensor_np = to_numpy(tensor).flatten()
        logger.debug(f"Converted tensor to numpy array of shape {tensor_np.shape}")
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise

    # Create a 3x3 grid of subplots with shared x-axes within columns
    # Further reduced figure size and spacing for more compact layout
    fig, axs = plt.subplots(3, 3, figsize=(6, 6), gridspec_kw={"hspace": 0.1, "wspace": 0.1})
    logger.debug("Created 3×3 subplot grid with compact layout")

    # Column 1: Time Domain plots
    # -------------------------------------

    # 1. Line/scatter plot (top-left)
    create_line_scatter_plot(tensor_np, ax=axs[0, 0], is_shared_x=True)

    # 2. Line/scatter plot with x-log scale (middle-left)
    create_line_scatter_logx_plot(tensor_np, ax=axs[1, 0], is_shared_x=True)

    # 3. Line/scatter plot with y-log scale (bottom-left)
    create_line_scatter_logy_plot(tensor_np, ax=axs[2, 0], is_shared_x=False)

    # Column 2: Frequency Domain plots
    # -------------------------------------

    # 4. FFT Magnitude (top-center)
    create_fft_magnitude_plot(tensor_np, ax=axs[0, 1], remove_dc=remove_dc, is_shared_x=True)

    # 5. Autocorrelation (middle-center)
    create_autocorrelation_plot(tensor_np, ax=axs[1, 1], is_shared_x=True)

    # 6. Power Spectral Density (bottom-center)
    create_psd_plot(tensor_np, ax=axs[2, 1], is_shared_x=False)

    # Column 3: Distribution plots
    # -------------------------------------

    # 7. Histogram + KDE (top-right)
    create_hist_kde_plot(tensor_np, ax=axs[0, 2], is_shared_x=True)

    # 8. Violin plot (middle-right)
    create_violin_plot(tensor_np, ax=axs[1, 2], is_shared_x=True)

    # 9. Cumulative Distribution Function (bottom-right)
    create_cdf_plot(tensor_np, ax=axs[2, 2], is_shared_x=False)

    # Add column headers (smaller font size and positioned closer to plots)
    axs[0, 0].text(
        0.5,
        1.25,
        "Time Domain",
        ha="center",
        va="center",
        transform=axs[0, 0].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axs[0, 1].text(
        0.5,
        1.25,
        "Frequency Domain",
        ha="center",
        va="center",
        transform=axs[0, 1].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axs[0, 2].text(
        0.5,
        1.25,
        "Distribution",
        ha="center",
        va="center",
        transform=axs[0, 2].transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # Make plot fonts smaller for all subplots
    for row in axs:
        for ax in row:
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_size(9)
            ax.title.set_size(10)

    # Use constrained_layout instead of tight_layout for better spacing management
    # Also, reduce the outer padding around the entire figure
    fig.set_constrained_layout(True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)

    # Remove extra whitespace around the plot area
    plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.07)

    logger.debug("Adjusted layout with minimal whitespace")

    # Save or display the plot
    if filename:
        logger.info(f"Saving plot to file: {filename}")
        try:
            plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
            logger.success(f"Plot saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save plot to {filename}: {e}")
            raise
        finally:
            plt.close(fig)
        return None
    elif show:
        logger.debug("Displaying plot interactively")
        plt.show()
        return None
    else:
        logger.debug("Returning figure without displaying")
        plt.close(fig)
        return fig
