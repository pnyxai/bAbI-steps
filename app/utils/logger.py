import logging
import structlog
import sys

loggers = {}


def configure_structlog():
    """
    Configure the structlog logger with a set of processors and options.
    :return: None
    """
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
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
    for logger_name in loggers.keys():
        logger = loggers[logger_name]
        logger.setLevel(level)
        loggers[logger_name] = structlog.wrap_logger(logger)
