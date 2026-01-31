"""
src/utils/logging_config.py

Logging configuration utilities for Jupyter notebook environments.

Usage:
    from src.utils.logging_config import setup_notebook_logging
    setup_notebook_logging('INFO')  # Show training progress in notebooks
"""

import logging
import sys
from typing import Optional


class NotebookFormatter(logging.Formatter):
    """
    Clean formatter for Jupyter notebooks with optional color support.

    Formats log messages as: [LEVEL] module: message
    Example: [INFO] core: Iter 100: loss_LR=-282.88 Temp=-0.43
    """

    # ANSI color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'RESET': '\033[0m'      # Reset
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize formatter.

        Args:
            use_colors: Whether to use ANSI color codes in output
        """
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record for notebook display.

        Args:
            record: Log record to format

        Returns:
            Formatted log message string
        """
        # Extract short module name (e.g., "core" from "src.trainers.core")
        module_parts = record.name.split('.')
        module_short = module_parts[-1] if module_parts else record.name

        if self.use_colors:
            level_color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            return f"{level_color}[{record.levelname}]{reset} {module_short}: {record.getMessage()}"
        else:
            return f"[{record.levelname}] {module_short}: {record.getMessage()}"


def setup_notebook_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    use_colors: bool = True
) -> None:
    """
    Configure logging for Jupyter notebook environment.

    This function sets up logging handlers that output to the notebook cell
    and optionally to a log file. Call this once at the start of your notebook
    to see training progress and other log messages.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
               - 'DEBUG': Verbose output including convergence checks
               - 'INFO': Standard training progress (recommended)
               - 'WARNING': Only warnings and errors
        log_file: Optional path to save logs to file. If provided, all logs
                  are saved with full timestamps (always at DEBUG level)
        use_colors: Whether to use colored output in notebooks (default: True)

    Example:
        >>> from src.utils.logging_config import setup_notebook_logging
        >>>
        >>> # Basic setup (most common)
        >>> setup_notebook_logging('INFO')
        >>>
        >>> # Verbose mode with file logging
        >>> setup_notebook_logging('DEBUG', log_file='experiments/training_001.log')
        >>>
        >>> # Quiet mode (only warnings/errors)
        >>> setup_notebook_logging('WARNING')

    Notes:
        - This configures the root logger, affecting all modules
        - Removes existing handlers to avoid duplicate output
        - File logs always use DEBUG level regardless of console level
        - Color codes work in most modern Jupyter environments
    """
    # Validate level
    level_upper = level.upper()
    if not hasattr(logging, level_upper):
        raise ValueError(f"Invalid logging level: {level}. Use DEBUG, INFO, WARNING, ERROR, or CRITICAL")

    # Get root logger (affects all modules including src.trainers.*)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level_upper))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler for notebook output (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level_upper))

    # Use custom formatter for clean notebook output
    console_formatter = NotebookFormatter(use_colors=use_colors)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler for experiment tracking
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Always capture full detail to file

        # Use detailed formatter for file output
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        print(f"Logging configured: level={level}, file={log_file}")
    else:
        print(f"Logging configured: level={level}")


def disable_logging() -> None:
    """
    Disable all logging output.

    Useful for quick experiments where you don't want any log messages.
    Sets logging level to CRITICAL, effectively silencing all normal logs.

    Example:
        >>> from src.utils.logging_config import disable_logging
        >>> disable_logging()
        >>> # Now training runs silently (only returns results)
    """
    logging.getLogger().setLevel(logging.CRITICAL)
    print("Logging disabled (level=CRITICAL)")


def reset_logging() -> None:
    """
    Reset logging to Python's default configuration.

    Removes all handlers and sets level to WARNING.
    Useful for cleaning up after experiments.

    Example:
        >>> from src.utils.logging_config import reset_logging
        >>> reset_logging()
        >>> # Back to default Python logging behavior
    """
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.handlers.clear()
    print("Logging reset to defaults (level=WARNING, no handlers)")


def get_current_log_level() -> str:
    """
    Get the current logging level as a string.

    Returns:
        String representation of current log level (e.g., 'INFO', 'DEBUG')

    Example:
        >>> from src.utils.logging_config import get_current_log_level
        >>> level = get_current_log_level()
        >>> print(f"Current level: {level}")
    """
    level_num = logging.getLogger().getEffectiveLevel()
    return logging.getLevelName(level_num)
