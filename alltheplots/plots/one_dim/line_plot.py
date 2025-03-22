import numpy as np
import matplotlib.pyplot as plt
from ...utils.logger import logger

def create_line_plot(tensor_np, ax=None):
    """
    Create a time-domain line plot for 1D data. If the data appears to have high frequency
    components (many oscillations), a scatter plot will be used instead for better visibility.

    Parameters:
        tensor_np (numpy.ndarray): The 1D numpy array to plot
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new one is created.

    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    logger.debug("Creating time-domain plot")
    
    # Create ax if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    
    # Determine if we should use scatter instead of line plot
    # Calculate an approximation of frequency content using zero crossings
    zero_crossings = np.where(np.diff(np.signbit(tensor_np - np.mean(tensor_np))))[0]
    n_crossings = len(zero_crossings)
    n_points = len(tensor_np)
    
    # If there are many zero crossings relative to data length, use scatter plot
    use_scatter = n_crossings > n_points / 10  # threshold can be adjusted
    
    # Create the appropriate plot
    if use_scatter:
        logger.debug(f"Using scatter plot due to high frequency content (crossings: {n_crossings}, points: {n_points})")
        ax.scatter(np.arange(len(tensor_np)), tensor_np, s=3, alpha=0.7)
    else:
        logger.debug("Using line plot for time-domain visualization")
        ax.plot(tensor_np, linewidth=1)
    
    # Set plot labels
    ax.set_title("Time Domain")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    
    return ax