import seaborn as sns
import matplotlib.pyplot as plt


def plot(tensor, filename=None, dpi=100, style="darkgrid"):
    """
    Plot a 1D tensor using seaborn and matplotlib.

    Parameters:
        tensor (np.ndarray): The input tensor to plot.
        filename (str, optional): The name of the output file. If None, the plot will be shown instead.
        dpi (int): The resolution of the output file in dots per inch.
        style (str): The style of the plot. Default is "darkgrid".

    Returns:
        None
    """

    # Set the style for the plot
    sns.set(style=style)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the tensor
    ax.plot(tensor)

    # Set labels and title
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("1D Tensor Plot")

    # Save or show the plot
    if filename:
        plt.savefig(filename, dpi=dpi)
        print(f"Plot saved to {filename}")
    else:
        plt.show()
