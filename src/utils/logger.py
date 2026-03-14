"""Simple logger."""

import logging
from config import load_yaml_config

_log_config = load_yaml_config("logging_config.yaml")


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Create and return a logger."""
    if level is None:
        level = getattr(logging, _log_config.get("level", "INFO"))
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            _log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            datefmt=_log_config.get("datefmt", "%H:%M:%S"),
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
