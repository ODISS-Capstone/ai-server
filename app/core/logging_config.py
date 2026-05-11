"""Application logging configuration."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import settings


def configure_logging() -> None:
    """Configure console and optional rotating file logging once."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)

    if not settings.log_to_file:
        return

    log_path = Path(settings.log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path = str(log_path.resolve())

    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler) and handler.baseFilename == resolved_log_path:
            return

    file_handler = RotatingFileHandler(
        resolved_log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
