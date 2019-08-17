"""Module for classes using a fmu to
simulate models."""

from ebcpy import simulationapi
#import fmpy
#from fmpy.fmi2 import FMU2Slave
#import shutil
#import time as runtime


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
