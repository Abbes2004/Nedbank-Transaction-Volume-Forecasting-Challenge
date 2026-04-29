"""
src/utils/logger.py
-------------------
Centralised logging factory.
Every module calls get_logger(__name__) to obtain a consistently
formatted logger without duplicating handler setup.
"""

import logging
import sys
from pathlib import Path


_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger identified by *name* with a stdout StreamHandler.

    Repeated calls with the same name return the existing logger
    without adding duplicate handlers.

    Parameters
    ----------
    name  : Typically __name__ of the calling module.
    level : Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — return as-is to avoid duplicate output.
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def add_file_handler(
    logger: logging.Logger,
    log_path: Path,
    level: int = logging.DEBUG,
) -> None:
    """
    Attach a FileHandler to an existing logger.

    Parameters
    ----------
    logger   : Logger instance obtained from get_logger().
    log_path : Destination file (parent directory must exist).
    level    : Logging level for the file handler (default DEBUG).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    )

    logger.addHandler(file_handler)
