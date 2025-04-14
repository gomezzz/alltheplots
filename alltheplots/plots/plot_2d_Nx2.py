import matplotlib.pyplot as plt
from ..utils.type_handling import to_numpy
from ..utils.logger import logger

# Import specialized plot functions for Nx2 data
from .two_dim.coordinate.scatter_trend import create_scatter_trend_plot
from .two_dim.coordinate.fft_spectrum import create_fft_spectrum_plot
from .two_dim.coordinate.spectral_analysis import create_spectral_analysis_plot
from .two_dim.coordinate.histogram_2d import create_histogram_2d_plot
from .two_dim.coordinate.hexbin_plot import create_hexbin_plot
from .two_dim.coordinate.kde_contour import create_kde_contour_plot
from .two_dim.coordinate.scatter_marginal import create_scatter_marginal_plot
from .two_dim.coordinate.delaunay_voronoi import create_delaunay_voronoi_plot
from .two_dim.coordinate.convex_hull import create_convex_hull_plot


def plot_2d_Nx2(tensor, filename=None, dpi=100, show=True):
    """
    Generate a specialized visualization for Nx2 tensors (coordinate data) with a 3×3 grid of plots:

    Column 1 (Time/Spatial & Fourier Analysis):
    - Raw Scatter with Trend Smoothing: Points with fitted smoothing line
    - FFT Magnitude Spectrum: FFT magnitude to highlight dominant frequencies
    - Power Spectral Density/Autocorrelation: Reveal periodicities

    Column 2 (Density and Distribution Visualization):
    - 2D Histogram Heatmap: Binned grid of point counts
    - Hexbin Plot: Hexagonal binning for density estimation
    - Bivariate KDE Contour Plot: Kernel density estimate contours

    Column 3 (Structural and Joint Analysis):
    - Scatter Plot with Marginal Distributions: Scatter with side histograms
    - Delaunay Triangulation or Voronoi Overlay: Local connectivity
    - Convex Hull Outline: Overall boundary of the distribution

    Parameters:
        tensor (array-like): The input Nx2 tensor to plot
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        show (bool): Whether to display the plot interactively (True) or just return the figure (False).

    Returns:
        matplotlib.figure.Figure: The figure containing the plots, or None if displayed
    """
    logger.info("Creating specialized Nx2 plot with 3×3 grid layout")

    # Convert to numpy array using our robust conversion utility
    try:
        tensor_np = to_numpy(tensor)
        if len(tensor_np.shape) != 2 or tensor_np.shape[1] != 2:
            raise ValueError(f"Expected Nx2 tensor, got shape {tensor_np.shape}")
        logger.debug(f"Converted tensor to numpy array of shape {tensor_np.shape}")
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise

    # Create a 3x3 grid of subplots with appropriate spacing
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.6)

    # Create all subplots
    axes = []
    for i in range(3):
        row = []
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        axes.append(row)

    try:
        # Column 1: Time/Spatial & Fourier Analysis
        create_scatter_trend_plot(tensor_np, ax=axes[0][0])
        create_fft_spectrum_plot(tensor_np, ax=axes[1][0])
        create_spectral_analysis_plot(tensor_np, ax=axes[2][0])

        # Column 2: Density and Distribution Visualization
        create_histogram_2d_plot(tensor_np, ax=axes[0][1])
        create_hexbin_plot(tensor_np, ax=axes[1][1])
        create_kde_contour_plot(tensor_np, ax=axes[2][1])

        # Column 3: Structural and Joint Analysis
        create_scatter_marginal_plot(tensor_np, ax=axes[0][2], fig=fig)
        create_delaunay_voronoi_plot(tensor_np, ax=axes[1][2])
        create_convex_hull_plot(tensor_np, ax=axes[2][2])

        # Add column headers
        axes[0][0].text(
            0.5,
            1.25,
            "Time/Spatial & Fourier Analysis",
            ha="center",
            va="center",
            transform=axes[0][0].transAxes,
            fontsize=11,
            fontweight="bold",
        )
        axes[0][1].text(
            0.5,
            1.25,
            "Density and Distribution",
            ha="center",
            va="center",
            transform=axes[0][1].transAxes,
            fontsize=11,
            fontweight="bold",
        )
        axes[0][2].text(
            0.5,
            1.25,
            "Structural and Joint Analysis",
            ha="center",
            va="center",
            transform=axes[0][2].transAxes,
            fontsize=11,
            fontweight="bold",
        )

    except Exception as e:
        logger.error(f"Failed to create one or more plots: {e}")
        # Clean up the figure in case of error
        plt.close(fig)
        raise

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
