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

# Memory monitoring utility
def log_memory_usage(label: str = "") -> float:
    """
    Log current memory usage and return it in GB.

    Useful for debugging memory issues. Call at key points:
        log_memory_usage("after loading dataset")
        log_memory_usage("after feature extraction")
        log_memory_usage("after releasing simulation data")

    Args:
        label: Optional label to identify the measurement point

    Returns:
        Current memory usage in GB
    """
    import psutil
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    label_str = f" ({label})" if label else ""
    logger.info("Memory usage{}: {:.2f} GB", label_str, mem_gb)
    return mem_gb


# Export logger for use in other modules
__all__ = ["logger", "log_memory_usage"]
