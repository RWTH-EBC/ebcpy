"""Module for classes using a fmu to
simulate models."""

from ebcpy import simulationapi
import time
import fmpy
import sys
import os
import warnings
import atexit
import psutil
from ebcpy import simulationapi
from ebcpy import data_types
import pandas as pd
import shutil
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import logging as logger

class FmuApi(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.
    """
    sim_setup = {'startTime': 0.0,
                 'stopTime': 1.0}
    sim_time = 0

    def __init__(self, cd, model_name, speed):
        """Instantiate class parameters"""
        super().__init__(cd, model_name)
        if not model_name.lower().endswith(".fmu"):
            raise ValueError(f"{model_name} is no valid fmu file!")
        self.speed = speed
        self.fmu_path = cd + "\\" + model_name
        self.start_time = time.time()
        self.load()

    def load(self):
        # read the model description
        self.model_description = read_model_description(self.fmu_path)

        # collect the value references
        self.vrs = {}
        for variable in self.model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference

        self.unzipdir = extract(self.fmu_path)

        self.model = FMU2Slave(guid=self.model_description.guid,
                                   unzipDirectory=self.unzipdir,
                                   modelIdentifier=
                                   self.model_description.coSimulation.modelIdentifier,
                                   instanceName='instance1')

        self.model.instantiate()
        self.model.setupExperiment(startTime=0.0)
        self.model.enterInitializationMode()
        self.model.exitInitializationMode()


    def close(self):
        """
        Closes the fmu.
        :return:
            True on success
        """
        try:
            self.model.terminate()
            self.model.freeInstance()
            shutil.rmtree(self.unzipdir)
            logger.info('File successfully closed!')
        except Exception as e:
            logger.error('Failed to close file: ' + str(e))


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
        self.sim_setup=sim_setup

    def read(self, var):
        name = self.vrs[var]
        value = self.model.getReal([name])
        #self.proceed()
        return value

    def write(self, var, value):
        name = self.vrs[var]
        self.model.setReal([name], [value])


    def proceed(self):
        cur_time = time.time() - self.start_time
        incr = cur_time - self.prev_time

        if self.speed > 1:
            incr *= self.speed

        self.model.doStep(currentCommunicationPoint=self.start,
                              communicationStepSize=incr)

        self.prev_time = cur_time

        if self.speed > 1:
            self.start += incr
        else:
            self.start = self.prev_time
