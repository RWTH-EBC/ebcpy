"""
Simulation APIs help you to perform automated
simulations for energy and building climate related models.
Parameters can easily be updated, and the initialization-process is
much more user-friendly than the provided APIs by Dymola or fmpy.
"""

import os
import warnings
from abc import abstractmethod
from ebcpy.utils import setup_logger


class SimulationAPI:
    """Base-class for simulation apis. Every simulation-api class
    must inherit from this class. It defines the structure of each class.

    :param str,os.path.normpath cd:
        Working directory path
    :param str model_name:
        Name of the model being simulated."""

    _default_sim_setup = {"initialValues": [],
                          "startTime": 0,
                          "stopTime": 1,
                          "outputInterval": 1,
                          }

    def __init__(self, cd, model_name):
        self._sim_setup = self._default_sim_setup.copy()
        self.cd = cd
        self.model_name = model_name
        # Setup the logger
        self.logger = setup_logger(cd=cd, name=self.__class__.__name__)
        self.logger.info(f'{"-" * 25}Initializing class {self.__class__.__name__}{"-" * 25}')
        self.inputs = []      # Inputs of model
        self.outputs = []     # Outputs of model
        self.parameters = []  # Parameter of model
        self.states = []      # States of model

    @abstractmethod
    def close(self):
        """Base function for closing the simulation-program."""
        raise NotImplementedError(f'{self.__class__.__name__}.close function is not defined')

    @abstractmethod
    def simulate(self, **kwargs):
        """Base function for simulating the simulation-model."""
        raise NotImplementedError(f'{self.__class__.__name__}.simulate function is not defined')

    @property
    def sim_setup(self) -> dict:
        """Return current sim_setup"""
        return self._sim_setup

    @sim_setup.setter
    def sim_setup(self, sim_setup: dict):
        """
        Overwrites multiple entries in the simulation
        setup dictionary for simulations with the used program.
        The object _number_values can be overwritten in child classes.

        :param dict sim_setup:
            Dictionary object with the same keys as this class's sim_setup dictionary
        """
        warnings.warn("Function will be removed in future versions. "
                      "Use the set function instead of the property setter directly e.g. "
                      "sim_api.set_sim_setup(sim_setup)", DeprecationWarning)
        self.set_sim_setup(sim_setup)

    @sim_setup.deleter
    def sim_setup(self):
        """In case user deletes the object, reset it to the default one."""
        self._sim_setup = self._default_sim_setup.copy()

    def set_sim_setup(self, sim_setup):
        """
        Replaced in v0.1.7 by property function
        """

        _diff = set(sim_setup.keys()).difference(self._default_sim_setup.keys())
        if _diff:
            raise KeyError(f"The given sim_setup contains the following keys "
                           f"({' ,'.join(list(_diff))}) which are not part of "
                           f"the sim_setup of class {self.__class__.__name__}")

        for key, value in sim_setup.items():
            _ref = type(self._default_sim_setup[key])
            if _ref in (float, int):
                _ref = (float, int)
            if isinstance(value, _ref):
                self._sim_setup[key] = value
            else:
                raise TypeError(f"{key} is of type {type(value).__name__} "
                                f"but should be type {_ref}")

    def set_initial_values(self, initial_values: list):
        """
        Overwrite inital values

        :param list initial_values:
            List containing initial values for the dymola interface
        """
        # Convert in case of np.array or similar
        self.sim_setup = {"initialValues": list(initial_values)}

    def set_cd(self, cd):
        """Base function for changing the current working directory."""
        self.cd = cd

    @property
    def cd(self) -> str:
        """Get the current working directory"""
        return self._cd

    @cd.setter
    def cd(self, cd: str):
        """Set the current working directory"""
        os.makedirs(cd, exist_ok=True)
        self._cd = cd

    @property
    def start_time(self) -> float:
        """Start time of the simulation"""
        return self.sim_setup["startTime"]

    @start_time.setter
    def start_time(self, start_time: float):
        """Set the start time of the simulation"""
        self.sim_setup["startTime"] = start_time

    @property
    def stop_time(self) -> float:
        """Stop time of the simulation"""
        return self.sim_setup["stopTime"]

    @stop_time.setter
    def stop_time(self, stop_time: float):
        """Set the stop time of the simulation"""
        self.sim_setup["stopTime"] = stop_time

    @property
    def output_interval(self) -> float:
        """
        Output interval of the simulation.
        Other meaning: The step size / sampling time of the simulation.
        """
        return self.sim_setup["outputInterval"]

    @output_interval.setter
    def output_interval(self, output_interval: float):
        """Set the stop time of the simulation"""
        self.sim_setup["outputInterval"] = output_interval
