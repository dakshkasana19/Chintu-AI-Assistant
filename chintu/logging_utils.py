from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import AssistantConfig, ROOT, load_config


def configure_logging(config: AssistantConfig | None = None) -> logging.Logger:
    if config is None:
        config = load_config()
    log_dir = ROOT / config.log_dir
    log_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger("chintu")
    logger.setLevel(getattr(logging, config.logging_level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            log_dir / "chintu.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
