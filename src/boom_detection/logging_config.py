"""
Logging configuration for boom detection.

Uses loguru for structured, colored logging with timing information.

Usage:
    from boom_detection.logging_config import logger

    logger.info("Starting training...")
    logger.debug("Processing simulation {}", sim_id)
    logger.warning("Low quality prediction")

Configure log level via environment variable:
    BOOM_LOG_LEVEL=DEBUG uv run python ...
"""
from __future__ import annotations

import os
import sys
from loguru import logger

# Remove default handler
logger.remove()

# Get log level from environment (default: INFO)
LOG_LEVEL = os.environ.get("BOOM_LOG_LEVEL", "INFO").upper()

# Add handler with nice formatting
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
    colorize=True,
)

# Export logger for use in other modules
__all__ = ["logger"]
