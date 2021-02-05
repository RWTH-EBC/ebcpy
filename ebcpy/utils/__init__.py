"""
Package containing utility functions used in different packages.
Contains a statistics analyzer and a visualizer.
"""
import logging
import os


def setup_logger(cd, name, level=logging.DEBUG):
    """
    Setup an class or module specific logger instance
    to ensure readable output for users.
    """
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    file_handler = logging.FileHandler(filename=os.path.join(cd, f"{name}.log"))
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
                                  datefmt='%d.%m.%Y-%H:%M:%S')
    console = logging.StreamHandler()
    file_handler.setFormatter(fmt=formatter)
    console.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=console)
    return logger
