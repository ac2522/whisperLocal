"""Logging configuration for whisper2text.

Provides a rotating file handler (DEBUG level) and a console handler
(WARNING level) so that detailed diagnostics go to disk while only
important messages appear on stderr.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_FILENAME = "app.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 2
_DEFAULT_SETTINGS_DIR = os.path.join(os.path.expanduser("~"), ".whisper2text")


def setup_logging(settings_dir=None, level=logging.INFO):
    """Configure the root logger with file and console handlers.

    Parameters
    ----------
    settings_dir : str or Path, optional
        Directory where the log file will be stored.
        Defaults to ``~/.whisper2text``.
    level : int, optional
        Overall logging level for the root logger.  Defaults to
        ``logging.INFO``.

    Returns
    -------
    str
        Absolute path to the log file.
    """
    if settings_dir is None:
        settings_dir = _DEFAULT_SETTINGS_DIR

    settings_dir = str(settings_dir)
    os.makedirs(settings_dir, exist_ok=True)

    log_file = os.path.join(settings_dir, _LOG_FILENAME)

    formatter = logging.Formatter(_LOG_FORMAT)

    # --- File handler: captures everything at DEBUG level ---
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # --- Console handler: only WARNING and above go to stderr ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    # --- Root logger ---
    root = logging.getLogger()
    # Clear any previously attached handlers to avoid duplicates.
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return log_file
