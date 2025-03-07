import logging
import sys
from collections import OrderedDict

import structlog

loggers = {}


def configure_structlog():
    """
    Configure the structlog logger with a set of processors and options.
    :return: None
    """

    def reorder_keys(_, __, event_dict):
        """
        Reorder keys to ensure 'logger', 'level', and 'event' come first.
        """
        key_order = ["logger", "level", "event"]
        # Create an ordered dictionary with specified key order
        reordered = OrderedDict(
            (key, event_dict.pop(key)) for key in key_order if key in event_dict
        )
        # Append the remaining keys
        reordered.update(event_dict)
        return reordered

    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        reorder_keys,  # Custom processor to reorder keys
        structlog.processors.JSONRenderer(),  # Outputs events as JSON strings.
    ]

    # noinspection PyTypeChecker
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name, level=logging.INFO, log_file="logs.txt"):
    """
    Create and configure a logger with specified name.

    :param level:
    :param name: The name of the logger.
    :return: The configured logger instance.
    """
    logger = logging.getLogger(name)
    # set level
    logger.setLevel(level)
    # Console handler (standard output)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    # File handler (write logs to a file)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)

    # Only add handlers if they are not already present
    if not logger.hasHandlers():
        logger.addHandler(ch)
        logger.addHandler(fh)

    configure_structlog()
    logger = structlog.wrap_logger(logger)
    loggers[name] = logger
    return logger


def override_level(level):
    for logger_name in loggers:
        logger = loggers[logger_name]
        logger.setLevel(level)
        loggers[logger_name] = structlog.wrap_logger(logger)
