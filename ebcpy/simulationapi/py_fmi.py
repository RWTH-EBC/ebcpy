"""Module for classes using a fmu to
simulate models."""

from ebcpy import simulationapi
#import fmpy
#from fmpy.fmi2 import FMU2Slave
#import shutil


class PyFMI(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.
    """

    def __init__(self, cd, model_name):
        """Instantiate class parameters"""
        super().__init__(cd, model_name)
        if not model_name.lower().endswith(".fmu"):
            raise ValueError("{} is not a valid fmu file!".format(model_name))
        raise NotImplementedError

    def close(self):
        """
        Closes the fmu.
        :return:
            True on success
        """
        raise NotImplementedError

    def set_cd(self, cd):
        """
        Set current working directory for storing files etc.
        :param str,os.path.normpath cd:
            New working directory
        :return:
        """
        raise NotImplementedError

    def simulate(self, savepath_files):
        """
        Simulate current simulation-setup.

        :param str,os.path.normpath savepath_files:
            Savepath were to store result files of the simulation.
        :return:
            Filepath of the mat-file.
        """
        raise NotImplementedError

    def set_sim_setup(self, sim_setup):
        """
        Alter the simulation setup by changing the setup-dict.

        :param sim_setup:
        """
        raise NotImplementedError
