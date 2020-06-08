"""Module with the base class for all simulation APIs.
Different simulation modules like dymola_api or py_fmi
may inherit classes of this module."""

from abc import abstractmethod
from ebcpy.utils import visualizer


class SimulationAPI:
    """Base-class for simulation apis. Every simulation-api class
    must inherit from this class. It defines the structure of each class.

    :param str,os.path.normpath cd:
        Working directory path
    :param str model_name:
        Name of the model being simulated."""

    def __init__(self, cd, model_name):
        self.cd = cd
        self.model_name = model_name
        # Setup the logger
        self.logger = visualizer.Logger(cd, "simulation_api")

    @abstractmethod
    def close(self):
        """Base function for closing the simulation-program."""
        raise NotImplementedError('{}.close function is not '
                                  'defined'.format(self.__class__.__name__))

    @abstractmethod
    def simulate(self):
        """Base function for simulating the simulation-model."""
        raise NotImplementedError('{}.simulate function is not '
                                  'defined'.format(self.__class__.__name__))

    @abstractmethod
    def set_sim_setup(self, sim_setup):
        """Base function for altering the simulation-setup."""
        raise NotImplementedError('{}.set_sim_setup function is not '
                                  'defined'.format(self.__class__.__name__))

    @abstractmethod
    def set_cd(self, cd):
        """Base function for changing the current working directory."""
        raise NotImplementedError('{}.set_cd function is not '
                                  'defined'.format(self.__class__.__name__))
