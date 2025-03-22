import numpy as np
from ..utils.logger import logger


def to_numpy(tensor):
    """
    Convert tensor-like objects from various frameworks to NumPy arrays.

    This function handles conversion from:
    - numpy arrays (returned as-is)
    - PyTorch tensors (using .detach().cpu().numpy())
    - TensorFlow tensors (using .numpy())
    - JAX arrays (using np.asarray())
    - CuPy arrays (using .get())

    Parameters:
        tensor: A tensor-like object from numpy, torch, tensorflow, jax, or cupy

    Returns:
        numpy.ndarray: A NumPy array representation of the input tensor
    """
    # Get the module name to determine the tensor type
    tensor_type = type(tensor)
    module_name = tensor_type.__module__.split(".")[0]
    logger.debug(f"Converting tensor of type {tensor_type} from module {module_name} to numpy")

    # Handle different tensor types
    if module_name == "numpy" or module_name == "builtins":
        # NumPy arrays or Python native types
        logger.trace("Tensor is already numpy or native type")
        return np.asarray(tensor)

    elif module_name == "torch":
        # PyTorch tensors
        logger.debug("Converting PyTorch tensor to numpy")
        return tensor.detach().cpu().numpy()

    elif module_name == "tensorflow" or module_name == "tf":
        # TensorFlow tensors
        logger.debug("Converting TensorFlow tensor to numpy")
        return tensor.numpy()

    elif module_name == "jax" or module_name == "jaxlib":
        # JAX arrays
        logger.debug("Converting JAX array to numpy")
        return np.asarray(tensor)

    elif module_name == "cupy":
        # CuPy arrays - use get() method to avoid the implicit conversion error
        logger.debug("Converting CuPy array to numpy using .get()")
        try:
            # Force using get() method for CuPy arrays
            return tensor.get()
        except Exception as e:
            logger.error(f"Error converting CuPy array using .get(): {e}")
            raise

    else:
        # Try a generic conversion as fallback
        logger.warning(
            f"Unknown tensor type from module {module_name}, attempting generic conversion"
        )
        try:
            return np.asarray(tensor)
        except TypeError as e:
            logger.error(
                f"Unsupported tensor type: {tensor_type} from module {module_name}. Error: {e}"
            )
            raise TypeError(
                f"Unsupported tensor type: {tensor_type} from module {module_name}. Error: {e}"
            )
