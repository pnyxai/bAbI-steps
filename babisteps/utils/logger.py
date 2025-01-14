import logging
import sys

import structlog
from collections import OrderedDict

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


def get_logger(name, level=logging.INFO):
    """
    Create and configure a logger with specified name.

    :param level:
    :param name: The name of the logger.
    :return: The configured logger instance.
    """
    logger = logging.getLogger(name)
    # set level
    logger.setLevel(level)
    # Add console handler for the logger
    ch = logging.StreamHandler(sys.stdout)
    # only add the handler if it doesn't already exist
    if not logger.hasHandlers():
        logger.addHandler(ch)
    configure_structlog()
    logger = structlog.wrap_logger(logger)
    loggers[name] = logger
    return logger


def override_level(level):
    for logger_name in loggers:
        logger = loggers[logger_name]
        logger.setLevel(level)
        loggers[logger_name] = structlog.wrap_logger(logger)
