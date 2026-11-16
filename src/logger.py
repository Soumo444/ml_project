"""Logging setup shared by the project."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "project_beg.log"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "project_beg") -> logging.Logger:
    """Return a console logger and a rotating file handler when writable."""
    project_logger = logging.getLogger(name)
    project_logger.setLevel(logging.INFO)
    project_logger.propagate = False

    if project_logger.handlers:
        return project_logger

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    project_logger.addHandler(console_handler)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
    except OSError:
        # Elastic Beanstalk captures process output; keep the app usable if its
        # deployed source directory does not allow application log files.
        return project_logger

    file_handler.setFormatter(formatter)
    project_logger.addHandler(file_handler)
    return project_logger


logger = get_logger()
