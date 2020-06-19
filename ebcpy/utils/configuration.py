"""
Module with functions to read and write config
files for objects in this and other repositories.
"""
import os
import yaml
import collections
# TODO: Add unit tests


def write_config(filepath, config):
    """
    Write the given config to the filepath.
    If the file already exists, the data is recursively
    updated.

    :param str,os.path.normpath filepath:
        Filepath with the config.
    :param: dict config:
        Config to be saved
    """
    if os.path.exists(filepath):
        existing_config = read_config(filepath)
        if existing_config:
            config = _update(existing_config, config)

    with open(filepath, "a+") as file:
        file.seek(0)
        file.truncate()
        yaml.dump(config, file)


def read_config(filepath):
    """
    Read the given file and return the yaml-config

    :param str,os.path.normpath filepath:
        Filepath with the config.
    :return: dict config:
        Loaded config
    """
    with open(filepath, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def _update(d, u):
    """Recursively update a given dictionary with a new one"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
