"""Module for classes using a model to
simulate models."""

import shutil
import os
import fmpy
from ebcpy import simulationapi
import pandas as pd


class FMI_Stepwise_API(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.
    """
    sim_setup = {'startTime': 0.0,
                 'stopTime': 1.0,
                 'solver': "CVode",  # Or "Euler"
                 'step_size': None,
                 'relative_tolerance': None,
                 'numberOfIntervals': 0,
                 'outputInterval': 1,
                 'initialNames': [],
                 'initialValues': []
                 }

    _number_values = ["startTime", "stopTime", "step_size", "numberOfIntervals",
                      "relative_tolerance", "outputInterval"]

    sim_time = 0

    validate = True  # Whether to validate the given model-model description or not
    equidistant_output = True
    instance_name = None

    def __init__(self, cd, fmu_file, **kwargs):
        """Instantiate class parameters"""

        if not fmu_file.lower().endswith(".fmu"):
            raise TypeError("Given file is not a fmu-file")
        if not os.path.exists(fmu_file):
            raise FileNotFoundError("Given file does not exist on your machine")
        self.fmu_file = fmu_file

        # Update Kwargs:
        self.__dict__.update(kwargs)

        # Model description
        # read the model description
        self.model_description = fmpy.read_model_description(self.fmu_file, validate=self.validate)
        super().__init__(cd, self.model_description.modelName)

        # Collect all variables
        self.variables = {}
        for variable in self.model_description.modelVariables:
            self.variables[variable.name] = variable

        if self.instance_name is None:
            self.instance_name = self.model_name
        self._setup_fmu()

    def _setup_fmu(self):

        self.unzipdir = fmpy.extract(self.fmu_file)

        self.model = fmpy.fmi2.FMU2Slave(guid=self.model_description.guid,
                                         unzipDirectory=self.unzipdir,
                                         modelIdentifier=
                                         self.model_description.coSimulation.modelIdentifier,
                                         instanceName=self.instance_name)

        self.model.instantiate()
        self.model.setupExperiment(startTime=self.sim_setup["startTime"],
                                   stopTime=self.sim_setup["stopTime"],
                                   tolerance=self.sim_setup["relative_tolerance"])

        self.model.enterInitializationMode()
        self.model.exitInitializationMode()

    def simulate(self):
        """
        Simulate current simulation-setup.

        :param str,os.path.normpath savepath_files:
            Savepath were to store result files of the simulation.
        :return:
            Filepath of the mat-file.
        """

        _current_time = self.sim_setup["startTime"]
        _step_size = self.sim_setup["outputInterval"]
        result = {}
        for var_name in self.variables.keys():
            result[var_name] = []
        while _current_time < self.sim_setup["stopTime"]:
            # Make one simulation step
            self.model.doStep(currentCommunicationPoint=_current_time,
                              communicationStepSize=_step_size)
            for var_name in self.variables:
                result[var_name].append(self.get_value(var_name))
            _current_time += _step_size
        return pd.DataFrame(result)

    def close(self):
        """
        Closes the model.
        :return:
            True on success
        """
        try:
            self.model.terminate()
            self.model.freeInstance()
            shutil.rmtree(self.unzipdir)
            self.logger.log('FMU-File successfully closed!')
            return True
        except Exception as e:
            self.logger.log('Failed to close file: ' + str(e))
            return False

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

    def get_value(self, var_name):
        """
        Get a single variable.
        """

        variable = self.variables[var_name]
        vr = [variable.valueReference]

        if variable.type == 'Real':
            return self.model.getReal(vr)[0]
        elif variable.type in ['Integer', 'Enumeration']:
            return self.model.getInteger(vr)[0]
        elif variable.type == 'Boolean':
            value = self.model.getBoolean(vr)[0]
            return value != 0
        else:
            raise Exception("Unsupported type: %s" % variable.type)

    def set_value(self, var_name, value):
        """
        Set a single variable.
        var_name: str
        """

        variable = self.variables[var_name]
        vr = [variable.valueReference]

        if variable.type == 'Real':
            self.model.setReal(vr, [float(value)])
        elif variable.type in ['Integer', 'Enumeration']:
            self.model.setInteger(vr, [int(value)])
        elif variable.type == 'Boolean':

            self.model.setBoolean(vr, [value != 0.0 or value != False or value != "False"])
        else:
            raise Exception("Unsupported type: %s" % variable.type)

if __name__=="__main__":
    fmu_file = r"D:\00_testzone\AixCalTest_TestModel.fmu"
    cd = r"D:\00_testzone"

    fmu = FMI_Stepwise_API(cd, fmu_file, equidistant_output=False)
    SIM_SETUP = {"startTime": 0.0,
                 "stopTime": 3600,
                 "relative_tolerance": 0.01,
                 "outputInterval": 10}

    fmu.set_sim_setup(SIM_SETUP)
    r = fmu.simulate()

    import matplotlib.pyplot as plt
    plt.plot(r["heatExchanger.port_a.T"])
    plt.show()
