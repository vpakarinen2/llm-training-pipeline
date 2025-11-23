"""Logging utilities."""

from __future__ import annotations

import logging
from typing import Optional


_LOGGER_INITIALIZED = False


def _ensure_basic_config() -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _LOGGER_INITIALIZED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return module-level logger with consistent format."""
    _ensure_basic_config()
    return logging.getLogger(name)
