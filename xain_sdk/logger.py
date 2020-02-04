"""Logging configuration."""

import logging

import structlog
from structlog._config import BoundLoggerLazyProxy

StructLogger = BoundLoggerLazyProxy


def get_logger(name: str, level: int = logging.INFO) -> BoundLoggerLazyProxy:
    """Wrap the python logger with the default configuration of structlog.

    Args:
        name (str): Identification name. For module name pass name=__name__.
        level (int): Threshold for this logger. Defaults to logging.INFO.

    Returns:
        ~structlog._config.BoundLoggerLazyProxy: The wrapped python logger with the default
            configuration of structlog.
    """

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter: structlog.stdlib.ProcessorFormatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(indent=2, sort_keys=True)
    )

    handler: logging.StreamHandler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger: logging.Logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)

    return structlog.wrap_logger(logger)
