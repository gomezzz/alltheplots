import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ...utils.logger import logger

def create_hist_kde_plot(tensor_np, ax=None, log_scale=False):
    """
    Create a histogram with KDE overlay plot for 1D data.
    
    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to create histogram for
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.
        log_scale (bool): Whether to use logarithmic scale for the y-axis. Default is False.
        
    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    scale_type = "log" if log_scale else "normal"
    logger.debug(f"Creating histogram with KDE ({scale_type} scale)")
    
    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    
    try:
        # Create histogram with KDE overlay
        sns.histplot(tensor_np, kde=True, ax=ax)
        
        # Set axis scale if log scale requested
        if log_scale:
            ax.set_yscale("log")
            ax.set_title("Histogram with KDE (Log Scale)")
        else:
            ax.set_title("Histogram with KDE")
            
        # Set plot labels
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        
    except Exception as e:
        error_msg = f"Failed to create {'log-scale ' if log_scale else ''}histogram: {e}"
        logger.error(error_msg)
        ax.text(
            0.5,
            0.5,
            f"Histogram Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"Histogram {('Log Scale ' if log_scale else '')}(Error)")
    
    return ax