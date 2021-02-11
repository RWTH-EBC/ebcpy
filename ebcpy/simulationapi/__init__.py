"""Module with the base class for all simulation APIs.
Different simulation modules like dymola_api or py_fmi
may inherit classes of this module."""

import os
from abc import abstractmethod
from ebcpy.utils import setup_logger


class SimulationAPI:
    """Base-class for simulation apis. Every simulation-api class
    must inherit from this class. It defines the structure of each class.

    :param str,os.path.normpath cd:
        Working directory path
    :param str model_name:
        Name of the model being simulated."""

    sim_setup = {}
    _number_values = []

    def __init__(self, cd, model_name):
        self.cd = cd
        self.model_name = model_name
        # Setup the logger
        self.logger = setup_logger(cd=cd, name=self.__class__.__name__)
        self.logger.info(f'{"-" * 25}Initializing class {self.__class__.__name__}{"-" * 25}')

    @abstractmethod
    def close(self):
        """Base function for closing the simulation-program."""
        raise NotImplementedError(f'{self.__class__.__name__}.close function is not defined')

    @abstractmethod
    def simulate(self):
        """Base function for simulating the simulation-model."""
        raise NotImplementedError(f'{self.__class__.__name__}.simulate function is not defined')

    def set_sim_setup(self, sim_setup):
        """
         Overwrites multiple entries in the simulation
         setup dictionary for simulations with the used program.
         The object _number_values can be overwritten in child classes.

         :param dict sim_setup:
             Dictionary object with the same keys as this class's sim_setup dictionary
         """
        _diff = set(sim_setup.keys()).difference(self.sim_setup.keys())
        if _diff:
            raise KeyError(f"The given sim_setup contains the following keys "
                           f"({' ,'.join(list(_diff))}) which are not part of "
                           f"the sim_setup of class {self.__class__.__name__}")

        for key, value in sim_setup.items():
            if key in self._number_values:
                _ref = (float, int)
            else:
                _ref = type(self.sim_setup[key])
            if isinstance(value, _ref):
                self.sim_setup[key] = value
            else:
                raise TypeError(f"{key} is of type {type(value).__name__} but should be type {_ref}")

    def set_cd(self, cd):
        """Base function for changing the current working directory."""
        self.cd = cd

    @property
    def cd(self) -> str:
        return self._cd

    @cd.setter
    def cd(self, cd: str):
        os.makedirs(cd, exist_ok=True)
        self._cd = cd
