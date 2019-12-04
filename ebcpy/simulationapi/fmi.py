"""Module for classes using a fmu to
simulate models."""

import time
import fmpy
import os
from ebcpy import simulationapi
import shutil
from fmpy


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

    _number_values = ["startTime", "stopTime", "step_size",
                      "relative_tolerance", "output_interval"]

    sim_time = 0

    validate = True  # Whether to validate the given fmu-model description or not
    equidistant_output = True
    instance_name = None

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
        if self.instance_name is None:
            self.instance_name = self.model_name
        self._setup_fmu()

    def _setup_fmu(self):
        self.unzipdir = fmpy.extract(self.fmu_file)

        self.fmu = fmpy.fmi2.FMU2Slave(guid=self.model_description.guid,
                                       unzipDirectory=self.unzipdir,
                                       modelIdentifier=
                                       self.model_description.coSimulation.modelIdentifier,
                                       instanceName=self.instance_name)

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
            self.logger.log('File successfully closed!')
        except Exception as e:
            self.logger.log('Failed to close file: ' + str(e))

    def set_cd(self, cd):
        """
        Set current working directory for storing files etc.
        :param str,os.path.normpath cd:
            New working directory
        :return:
        """
        # Check if path is valid
        if not os.path.isdir(cd):
            raise ValueError("Given working directory is not a valid path.")
        # Create path if it does not exist
        if not os.path.exists(cd):
            os.mkdir(cd)
        # Set the new working directory
        self.cd = cd

    def simulate(self, savepath_files):
        """
        Simulate current simulation-setup.

        :param str,os.path.normpath savepath_files:
            Savepath were to store result files of the simulation.
        :return:
            Filepath of the mat-file.
        """

        result = fmpy.simulate_fmu(filename=self.fmu_file,
                                   validate=self.validate,
                                   start_time=self.sim_setup["start_time"],
                                   stop_time=self.sim_setup["stop_time"],
                                   solver=self.sim_setup["solver"],
                                   step_size=self.sim_setup["step_size"],
                                   relative_tolerance=self.sim_setup["relative_tolerance"],
                                   output_interval=self.sim_setup["output_interval"],
                                   record_events=not self.equidistant_output,
                                   start_values={},
                                   input=None,
                                   output=None)

        return result

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
