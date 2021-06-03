"""
Package containing utility functions used in different packages.
Contains a statistics analyzer and a visualizer.
"""
import logging
import os


def setup_logger(name: str,
                 cd: str = None,
                 level=logging.DEBUG):
    """
    Setup an class or module specific logger instance
    to ensure readable output for users.

    :param str name:
        The name of the logger instance
    :param str cd:
        The path where to store the logfile.
        If None is given, logs are not stored.
    :param str level:
        The logging level, default is DEBUG

    .. versionadded:: 0.1.7
    """
    logger = logging.getLogger(name=name)
    # Set log-level
    logger.setLevel(level=level)
    # Check if logger was already instantiated. If so, return already.
    if logger.handlers:
        return logger
    # Add handlers
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
                                  datefmt='%d.%m.%Y-%H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=console)
    if cd is not None:
        os.makedirs(cd, exist_ok=True)
        file_handler = logging.FileHandler(filename=os.path.join(cd, f"{name}.log"))
        file_handler.setFormatter(fmt=formatter)
        logger.addHandler(hdlr=file_handler)
    return logger
