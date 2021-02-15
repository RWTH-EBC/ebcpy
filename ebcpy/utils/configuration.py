"""
Module with functions to read and write config
files for objects in this and other repositories.
"""
import os
import collections
import yaml
from ebcpy.simulationapi.dymola_api import DymolaAPI

# TODO: Add unit tests
# Specify solver-specific keyword-arguments depending on the solver and method you will use
kwargs_scipy_dif_evo = {"maxiter": 30,
                        "popsize": 5,
                        "mutation": (0.5, 1),
                        "recombination": 0.7,
                        "seed": None,
                        "polish": True,
                        "init": 'latinhypercube',
                        "atol": 0}

kwargs_dlib_min = {"num_function_calls": int(1e9),
                   "solver_epsilon": 0}

kwargs_scipy_min = {"tol": None,
                    "options": {"maxfun": 1},
                    "constraints": None,
                    "jac": None,
                    "hess": None,
                    "hessp": None}

default_sim_config = {"packages": None,
                      "model_name": None,
                      "type": "DymolaAPI",
                      "dymola_path": None,
                      "dymola_interface_path": None,
                      "equidistant_output": True,
                      "show_window": False,
                      "get_structural_parameters": True
                      }

tsd_config = {"filepath": "TODO: Specify the path to the target values measured",
              "key": None,
              "sheet_name": None,
              "sep": ","}


default_optimization_config = {"framework": "TODO: Choose the framework for calibration",
                               "method": "TODO: Choose the method of the framework",
                               "settings": {
                                   "scipy_differential_evolution": kwargs_scipy_dif_evo,
                                   "dlib_minimize": kwargs_dlib_min,
                                   "scipy_minimize": kwargs_scipy_min}
                               }

default_config = {
    "Working Directory": "TODO: Add the path where you want to work here",
    "SimulationAPI": default_sim_config,
    "Optimization": default_optimization_config
    }


def get_simulation_api_from_config(config):
    """
    Read the data for a SimulationAPI object.

    :param dict config:
        Config holding the following keys for
        - type: Type of the simulation API (e.g. DymolaAPI)
        - Further parameters as defined by the selected simulation api
    :return: SimulationAPI sim_api
        Loaded SimulationAPI
    """
    sim_type = config["type"]
    config.pop("type")
    if sim_type.lower() == "dymolaapi":
        return DymolaAPI(**config)

    raise KeyError(f"Given simulation type {sim_type} not supported.")


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


def _update(dic, new_dic):
    """Recursively update a given dictionary with a new one"""
    for key, val in new_dic.items():
        if isinstance(val, collections.abc.Mapping):
            dic[key] = _update(dic.get(key, {}), val)
        else:
            dic[key] = val
    return dic
