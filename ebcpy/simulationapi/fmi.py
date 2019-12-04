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
from fmpy
import logging as logger


class FMI_API(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.
    """
    sim_setup = {'startTime': 0.0,
                 'stopTime': 1.0,
                 'solver': "CVode",  # Or "Euler"
                 'step_size': None,
                 'relative_tolerance': None,
                 'output_interval': None}
    sim_time = 0

    validate = True  # Whether to validate the given fmu-model description or not

    def __init__(self, cd, fmu_file, **kwargs):
        """Instantiate class parameters"""

        if not fmu_file.lower().endswith(".fmu"):
            raise TypeError("Given file is not a fmu-file")
        if not os.path.exists(fmu_file):
            raise FileNotFoundError("Given file does not exist on your machine")

        # Update Kwargs:
        self.__dict__.update(kwargs)

        # Model description
        # read the model description
        self.model_description = fmpy.read_model_description(self.fmu_file, validate=self.validate)
        super().__init__(cd, self.model_description.modelName)

        # collect the value references
        self.vrs = {}
        for variable in self.model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference

        self.speed = speed
        self.fmu_file = fmu_file
        self.start_time = time.time()
        self._setup_fmu()

    def _setup_fmu(self):
        self.unzipdir = fmpy.extract(self.fmu_file)

        self.fmu = fmpy.fmi2.FMU2Slave(guid=self.model_description.guid,
                                       unzipDirectory=self.unzipdir,
                                       modelIdentifier=
                                       self.model_description.coSimulation.modelIdentifier,
                                       instanceName='instance1')

        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    def close(self):
        """
        Closes the fmu.
        :return:
            True on success
        """
        try:
            self.fmu.terminate()
            self.fmu.freeInstance()
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
        fmpy.simulate_fmu(filename=filename,
                          validate=self.validate,
                          start_time=self.sim_setup["start_time"],
                          stop_time=self.sim_setup["stop_time"],
                          solver=self.sim_setup["solver"],
                          step_size=self.sim_setup["step_size"],
                          relative_tolerance=self.sim_setup["relative_tolerance"],
                          output_interval=self.sim_setup["output_interval"],
                          record_events=True,   # TODO
                          fmi_type=None,    # TODO
                          use_source_code=False,
                          start_values={},
                          apply_default_start_values=False,
                          input=None,  # TODO
                          output=None,  # TODO
                          timeout=None,  # TODO
                          debug_logging=False,   # TODO
                          logger=None,   # TODO
                          fmi_call_logger=None,   # TODO
                          step_finished=None,   # TODO
                          model_description=None)

    def set_sim_setup(self, sim_setup):
        """
        Alter the simulation setup by changing the setup-dict.

        :param sim_setup:
        """
        self.sim_setup = sim_setup

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
