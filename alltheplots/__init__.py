from .plots.plot import plot
from .utils.logger import logger, set_log_level
from .plots.three_dim import (
    create_xy_slice_plot,
    create_xz_slice_plot,
    create_yz_slice_plot,
    create_max_intensity_plot,
    create_mean_intensity_plot,
    create_std_intensity_plot,
    create_hist_kde_plot_3d,
    create_cdf_plot_3d,
)

__all__ = [
    "plot",
    "logger",
    "set_log_level",
    # 3D plotting functions
    "create_xy_slice_plot",
    "create_xz_slice_plot",
    "create_yz_slice_plot",
    "create_max_intensity_plot",
    "create_mean_intensity_plot",
    "create_std_intensity_plot",
    "create_hist_kde_plot_3d",
    "create_cdf_plot_3d",
]

__version__ = "0.1.0"
__author__ = "Pablo Gómez"
