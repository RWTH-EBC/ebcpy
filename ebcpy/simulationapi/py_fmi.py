"""Module for classes using a fmu to
simulate models."""

from ebcpy import simulationapi
import fmpy
import pandas as pd
import numpy as np
import shutil
import os

# Klasse vergleichbar mit Hannah's "ModelicaFMU" Klasse
class FMU_API(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.
    """

    # Default attributes
    sim_setup = {'startTime': 0.0,
                 'stopTime': 1.0,
                 'numberOfIntervals': 0,
                 'outputInterval': 1,
                 'solver': 'CVode',
                 'initialNames': [],
                 'initialValues': [],
                 'initialBoundaries': [],
                 "inputNames": [],
                 'resultNames': [],}

    # Dynamic setup of simulation setup         # Notwendig?
    number_values = [key for key, value in sim_setup.items() if
                     (isinstance(value, (int, float)) and not isinstance(value, bool))]

    def __init__(self, cd, model_name):
        """Instantiate class parameters"""
        super().__init__(cd, model_name)
        if not model_name.suffix == ".fmu":
            raise ValueError("{} is not a valid fmu file!".format(model_name))

        # Read model description
        self.fmu_description = fmpy.read_model_description(model_name)

        # Collect all variables
        self.variables = {}
        for variable in self.fmu_description.modelVariables:
            self.variables[variable.name] = variable

        # extract the FMU
        self.unzipdir = fmpy.extract(self.model_name)

        # create fmu obj
        self.fmu = fmpy.fmi2.FMU2Slave(guid=self.fmu_description.guid,
                                       unzipDirectory=self.unzipdir,
                                       modelIdentifier=self.fmu_description.coSimulation.modelIdentifier,
                                       #instanceName=self.instanceName
                                       )

        # instantiate fmu
        self.fmu.instantiate()

        # initialize
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        # Extract inputs, outputs & tuner (lists from parent classes will be appended)
        for v in self.fmu_description.modelVariables:
            if v.causality == 'input':
                self.model_inp.append(v.name)
            if v.causality == 'output':
                self.model_out.append(v.name)
            if 'TunerParameter.' in v.name:
                self.model_tuner_names.append(v.name)
                self.model_tuner_initialvalues.append(float(v.start))
                if not type(v.min) == None or type(v.max) == None:
                    bounds_tuple = (float(v.min), float(v.max))
                    self.model_tuner_bounds.append(bounds_tuple)
                else:
                    raise Exception("No boundaries defined for parameter {} in the fmu file."
                                    " Please edit the model file".format(v))

        # Set inputs, outputs & tuner to simulation API
        self.set_sim_setup({
            "inputNames": self.model_inp,
            "resultNames": self.model_out,
            "initialNames": self.model_tuner_names,
            "initialValues": self.model_tuner_initialvalues,
            "initialBoundaries": self.model_tuner_bounds
        })

    def set_cd(self, cd):
        """
        Set current working directory for storing files etc.
        :param str,os.path.normpath cd:
            New working directory
        :return:
        """
        os.makedirs(cd, exist_ok=True)
        self.cd = cd

    def simulate(self, meas_input_data, **kwargs):              # %%%% TO-DO: Automatisieren. Anpassen auf InfluxDB.
        """
        Simulate current simulation-setup.

        :param dataframe meas_input_data:
            Pandas.Dataframe of the measured input data for simulating the FMU with fmpy
        :return dataframe sim_target_data:
            Pandas.Dataframe of simulated target values
        """
        # Dictionary with all tuner parameter names & -values
        start_values = {self.sim_setup["initialNames"][i]: value
                        for i, value in enumerate(self.sim_setup["initialValues"])}

        # Shift all columns, because "simulate_fmu" gets an input at timestep x and calculates the related output for timestep x+1
        shift_period = int(self.sim_setup["outputInterval"]/(meas_input_data.index[0]-meas_input_data.index[1]))
        meas_input_data = meas_input_data.shift(periods=shift_period)
        # Shift time column back
        meas_input_data.time = meas_input_data.time.shift(1, fill_value=0)
        # drop NANs
        meas_input_data = meas_input_data.dropna()

        # Convert df to structured numpy array for fmpy: simulate_fmu
        meas_input_tuples = [tuple(columns) for columns in meas_input_data.to_numpy()]
        dtype = [(i, np.double) for i in meas_input_data.columns]       # %%% TO-DO: implement more than "np.double" as type-possibilities
        meas_input_fmpy = np.array(meas_input_tuples, dtype=dtype)

        try:
            res = fmpy.simulate_fmu(
                     self.model_name,
                     validate=True,
                     start_time=self.sim_setup["startTime"],
                     stop_time=self.sim_setup["stopTime"],
                     solver=self.sim_setup["solver"],
                     step_size=self.sim_setup["numberOfIntervals"],      # !!Nur Einfluss, wenn Euler als Solver verwendet
                     relative_tolerance=None,
                     output_interval=self.sim_setup["outputInterval"],      # Hat sehr großen Einfluss auf die Ergebnisse
                     record_events=False,
                     fmi_type=None,
                     start_values=start_values,
                     apply_default_start_values=False,
                     input=meas_input_fmpy,                             # Ob Zeitintervall der inputs mit "outputInterval" übereinstimmt ist irrelevant
                     output=self.sim_setup["resultNames"],
                     timeout=None,
                     debug_logging=False,
                     visible=False,
                     logger=None,
                     fmi_call_logger=None,
                     step_finished=None,
                     model_description=None,
                     fmu_instance=None)
        except Exception as error:
            print(f"[SIMULATION ERROR] Error occured while running FMU: \n {error}")
            if kwargs.get("fail_on_error", False):
                raise error
            return None

        # Reshape result:
        _cols = ["Time"] + self.sim_setup["resultNames"]
        df = pd.DataFrame(res.tolist(), columns=_cols).set_index("Time")
        df.index = df.index.astype("float64")
        return df

    def do_step(self):
        # ...to add...

        # check if stop time is reached
        if self.current_time < self.stop_time:
            # do simulation step
            status = self.fmu.doStep(
                currentCommunicationPoint=self.current_time,
                communicationStepSize=self.step_size)
            # augment current time step
            self.current_time += self.step_size
            finished = False
        else:
            print('Simulation finished')
            finished = True

        return finished

    def overwrite_model(self):
        """
        Overwrites the simulation model after calibration.
        First the all optimized parameters will be overwritten in the model.
        Afterwards there will be an adjustment of the boundaries.

        :param type param_name:
            To add
        :return type returnname:
            To add
        """

        # To add if model can be permanently overwritten

        pass

    def setup(self):        # vielleicht noch hilfreich, bislang nicht genutzt
        # The current simulation time
        self.current_time = self.sim_setup["startTime"]

        # initialize model
        self.fmu.reset()
        self.fmu.setupExperiment(
            startTime=self.start_time, stopTime=self.stop_time, tolerance=self.sim_tolerance)


    def find_vars(self, start_str: str):
        """
        Retruns all variables starting with start_str
        """
        key = list(self.variables.keys())
        key_list = []
        for i in range(len(key)):
            if key[i].startswith(start_str):
                key_list.append(key[i])
        return key_list

    def get_value(self, var_name: str):
        """
        Get a single variable.
        """

        variable = self.variables[var_name]
        vr = [variable.valueReference]

        if variable.type == 'Real':
            return self.fmu.getReal(vr)[0]
        elif variable.type in ['Integer', 'Enumeration']:
            return self.fmu.getInteger(vr)[0]
        elif variable.type == 'Boolean':
            value = self.fmu.getBoolean(vr)[0]
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
            self.fmu.setReal(vr, [float(value)])
        elif variable.type in ['Integer', 'Enumeration']:
            self.fmu.setInteger(vr, [int(value)])
        elif variable.type == 'Boolean':
            self.fmu.setBoolean(vr, [value == 1.0 or value == True or value == "True"])
        else:
            raise Exception("Unsupported type: %s" % variable.type)

    def read_variables(self, vrs_list: list):
        """
        Reads multiple variable values of FMU.
        vrs_list as list of strings
        Method retruns a dict with FMU variable names as key
        """
        res = {}
        # read current variable values ans store in dict
        for var in vrs_list:
            res[var] = self.get_value(var)

        # add current time to results
        #res['SimTime'] = self.current_time

        return res

    def set_variables(self, var_dict: dict):
        '''
        Sets multiple variables.
        var_dict is a dict with variable names in keys.
        '''

        for key in var_dict:
            self.set_value(key, var_dict[key])
        return "Variable set!!"

    def close(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.unzipdir)
        print('FMU released')

