import sys

from loguru import logger

from .plots.plot import plot
from .utils.logger import set_log_level


ATP_RELEASE_TOGGLE = True

if ATP_RELEASE_TOGGLE:
    # Disable alltheplots logger
    logger.disable("alltheplots")
else:
    # Enable alltheplots logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green>|<level>ATP-{level}</level>| <level>{message}</level>",
        colorize=True,
        level="INFO",
    )

__all__ = [
    "plot",
    "set_log_level",
]

__version__ = "0.1.0"
__author__ = "Pablo Gómez"
