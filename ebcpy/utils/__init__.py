"""
Package containing utility functions used in different packages.
Contains a statistics analyzer and a visualizer.
"""
import logging
import os
import re
from pathlib import Path
from typing import Union, List
import warnings


def setup_logger(name: str,
                 working_directory: Union[Path, str] = None,
                 level=logging.DEBUG):
    """
    Setup an class or module specific logger instance
    to ensure readable output for users.

    :param str name:
        The name of the logger instance
    :param str,Path working_directory:
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
    # Add handlers if not set already by logging.basicConfig and if path is specified
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
                                  datefmt='%d.%m.%Y-%H:%M:%S')
    if not logging.getLogger().hasHandlers():
        console = logging.StreamHandler()
        console.setFormatter(fmt=formatter)
        logger.addHandler(hdlr=console)
    if working_directory is not None:
        os.makedirs(working_directory, exist_ok=True)
        file_handler = logging.FileHandler(filename=working_directory.joinpath(f"{name}.log"))
        file_handler.setFormatter(fmt=formatter)
        logger.addHandler(hdlr=file_handler)
    return logger


def get_names(
        all_names: list,
        patterns: Union[str, List[str]],
        exclude: Union[str, List[str]] = None
) -> List[str]:
    """
    Filter a list of candidate names by literal values or glob-style patterns,
    optionally excluding names that match exclusion patterns.

    This function returns all names from ``all_names`` that match the provided
    ``patterns`` and do not match any of the ``exclude`` patterns.
    Patterns may be a single string or a list of strings, and may
    contain the wildcard ``*`` to match any sequence of characters. Literal names
    without ``*`` must match exactly.

    The returned list preserves the order of ``all_names``.

    :param list all_names:
        List of available names to filter.
    :param str,list[str] patterns:
        A pattern or list of patterns (with optional ``*`` wildcards)
        to match against ``all_names``.
    :param str,list[str] exclude:
        A pattern or list of patterns to exclude from the results.
        Names matching any exclusion pattern are removed after the
        inclusion step. Default is None (no exclusion).
    :return: A list of names from ``all_names`` that match any of the given
        patterns and none of the exclusion patterns, in original order.
    :rtype: list[str]
    :raises warning: If any inclusion pattern does not match at least one name.

    Example:

    >>> names = ["wall.layer[1].T", "wall.layer[2].T", "wall.layer[1].Q_flow"]
    >>> get_names(names, "wall.layer[*].T")
    ['wall.layer[1].T', 'wall.layer[2].T']
    >>> get_names(names, "wall.layer[*].*", exclude="*Q_flow")
    ['wall.layer[1].T', 'wall.layer[2].T']
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    matched = set()
    unmatched = []
    for pat in patterns:
        if '*' in pat:
            regex = '^' + re.escape(pat).replace(r'\*', '.*') + '$'
            hits = [k for k in all_names if re.match(regex, k)]
            if hits:
                matched.update(hits)
            else:
                unmatched.append(pat)
        else:
            if pat in all_names:
                matched.add(pat)
            else:
                unmatched.append(pat)

    if unmatched:
        warnings.warn(
            "The following variable names/patterns are not in the given .mat file: "
            + ", ".join(unmatched)
        )

    # Apply exclusion patterns
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        excluded = set()
        for pat in exclude:
            if '*' in pat:
                regex = '^' + re.escape(pat).replace(r'\*', '.*') + '$'
                excluded.update(k for k in matched if re.match(regex, k))
            else:
                if pat in matched:
                    excluded.add(pat)
        matched -= excluded

    # Preserve original order
    names = [var for var in all_names if var in matched]
    return names
