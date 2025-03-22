import sys
from loguru import logger

# Configure default logger
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green>|ATP-<blue>{level}</blue>| <level>{message}</level>",
    filter="alltheplots",
)


def set_log_level(log_level: str):
    """Set the log level for the logger.

    Args:
        log_level (str): The log level to set. Options are 'TRACE','DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        level=log_level.upper(),
        format="<green>{time:HH:mm:ss}</green>|ATP-<blue>{level}</blue>| <level>{message}</level>",
        filter="alltheplots",
    )
    logger.debug(f"Setting LogLevel to {log_level.upper()}")
