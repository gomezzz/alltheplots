import warnings
import sys
from loguru import logger

def set_log_level(level):
    """
    Set the log level for alltheplots.

    Args:
        level (str): Log level (DEBUG, INFO, WARNING, ERROR, etc.)
    """
    # Remove default handler and add new one with specified level
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green>|<level>ATP-{level}</level>| <level>{message}</level>",
        colorize=True,
        level=level,
    )
    logger.info(f"Set log level to {level}")


# Suppress common warnings that clutter output
def suppress_common_warnings():
    """Suppress common warning messages that clutter output"""
    # Suppress sklearn deprecation warnings
    warnings.filterwarnings(
        "ignore",
        message="'force_all_finite' was renamed to 'ensure_all_finite'",
        category=FutureWarning,
    )

    # Suppress tqdm warnings in Jupyter
    warnings.filterwarnings(
        "ignore",
        message="IProgress not found. Please update jupyter and ipywidgets.",
        category=UserWarning,
    )

    # Suppress matplotlib warnings that are common and harmless
    warnings.filterwarnings(
        "ignore", message="Matplotlib is currently using agg", category=UserWarning
    )


# Call at import time to suppress warnings globally
suppress_common_warnings()
