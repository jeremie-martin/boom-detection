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

For experiment scripts with archival:
    from boom_detection.logging_config import logger, setup_run_logging

    run_dir = setup_run_logging("sweep_2model")
    logger.info("Results will be saved to {}", run_dir)
    # All logger output now goes to both terminal AND run_dir/run.log
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

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


def setup_run_logging(run_name: str, base_dir: Path | str = Path("runs")) -> Path:
    """
    Setup logging for an experiment run with timestamped directory.

    Creates: {base_dir}/{run_name}_{YYYYMMDD_HHMMSS}/
    Returns: Path to run directory

    Configures loguru to output to both terminal AND log file.
    The log file captures the same information as terminal output,
    providing automatic archival of experiment runs.

    Args:
        run_name: Name of the experiment (e.g., "sweep_2model")
        base_dir: Base directory for runs (default: "runs")

    Returns:
        Path to the created run directory

    Example:
        run_dir = setup_run_logging("sweep_quality")
        logger.info("Starting sweep...")
        # ... run experiment ...
        # Results saved to runs/sweep_quality_20251229_143022/
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler to loguru (in addition to existing stderr handler)
    log_file = run_dir / "run.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
    )

    logger.info("=" * 70)
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 70)

    return run_dir


# Export logger for use in other modules
__all__ = ["logger", "log_memory_usage", "setup_run_logging"]
