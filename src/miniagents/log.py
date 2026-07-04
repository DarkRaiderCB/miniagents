"""Logging setup helper for miniagents.

The library itself only attaches NullHandlers (standard library practice);
applications opt into output with configure_logging().
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    quiet_http: bool = True,
) -> logging.Logger:
    """
    Configure console (and optional file) logging for the 'miniagents' namespace.

    Args:
        level: Log level for miniagents loggers (name or numeric).
        log_file: If given, also append logs to this file.
        quiet_http: Suppress noisy INFO logs from the underlying HTTP stack.

    Returns:
        The configured 'miniagents' package logger.
    """
    pkg_logger = logging.getLogger("miniagents")
    pkg_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Avoid stacking duplicate handlers on repeated calls.
    pkg_logger.handlers = [
        h for h in pkg_logger.handlers if isinstance(h, logging.NullHandler)
    ]

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    pkg_logger.addHandler(console)

    if log_file is not None:
        target = Path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        pkg_logger.addHandler(file_handler)

    if quiet_http:
        for noisy in ("httpx", "httpcore", "groq"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    pkg_logger.propagate = False
    return pkg_logger


__all__ = ["configure_logging"]
