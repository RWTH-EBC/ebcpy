"""Module for classes using a fmu to
simulate models."""

import os
import logging
import shutil
import fmpy
from fmpy.model_description import read_model_description
import pandas as pd
import numpy as np
from ebcpy import simulationapi
# pylint: disable=broad-except


class FMU_API(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.

    .. versionadded:: 0.1.7
    """

    _default_sim_setup = {
        'startTime': 0.0,
        'stopTime': 1.0,
        'numberOfIntervals': 0,
        'outputInterval': 1,
        'solver': 'CVode',
        'initialNames': [],
        'initialValues': [],
        'resultNames': [],
        'timeout': np.inf}

    def __init__(self, cd, model_name):
        """Instantiate class parameters"""
        super().__init__(cd, model_name)
        if not model_name.lower().endswith(".fmu"):
            raise ValueError(f"{model_name} is not a valid fmu file!")
        # Init instance attributes
        self._unzip_dir = None
        self._fmu_instance = None
        self._model_description = None
        self._fmi_type = None
        self.log_fmu = True

        # Setup the fmu instance
        self.setup_fmu_instance()

    def close(self):
        """
        Closes the fmu.
        :return:
            True on success
        """
        try:
            self._fmu_instance.terminate()
        except Exception as error:  # This is due to fmpy which does not yield a narrow error
            self.logger.error(f"Could not terminate fmu instance: {error}")
        try:
            self._fmu_instance.freeInstance()
        except OSError as error:
            self.logger.error(f"Could not free fmu instance: {error}")
        # Remove the extracted files
        shutil.rmtree(self._unzip_dir)
        self._unzip_dir = None

    def simulate(self, **kwargs):
        """
        Simulate current simulation-setup.

        :param dataframe inputs:
            Pandas.Dataframe of the input data for simulating the FMU with fmpy
        :keyword Boolean fail_on_error:
            If True, an error in fmpy will trigger an error in this script.
            Default is false
        :return:
            Filepath of the mat-file.
        """
        # Dictionary with all tuner parameter names & -values
        start_values = {self.sim_setup["initialNames"][i]: value
                        for i, value in enumerate(self.sim_setup["initialValues"])}

        inputs = kwargs.get("inputs", None)
        if inputs is not None:
            inputs = inputs.copy() # Create save copy
            # Shift all columns, because "simulate_fmu" gets an input at
            # timestep x and calculates the related output for timestep x+1
            shift_period = int(self.sim_setup["outputInterval"] / (inputs.index[0] - inputs.index[1]))
            inputs = inputs.shift(periods=shift_period)
            # Shift time column back
            inputs.time = inputs.time.shift(-shift_period, fill_value=0)
            # drop NANs
            inputs = inputs.dropna()

            # Convert df to structured numpy array for fmpy: simulate_fmu
            inputs_tuple = [tuple(columns) for columns in inputs.to_numpy()]
            dtype = [(i, np.double) for i in
                     inputs.columns]  # %%% TO-DO: implement more than "np.double" as type-possibilities
            inputs = np.array(inputs_tuple, dtype=dtype)

        try:
            res = fmpy.simulate_fmu(
                self._unzip_dir,
                start_time=self.sim_setup["startTime"],
                stop_time=self.sim_setup["stopTime"],
                solver=self.sim_setup["solver"],
                step_size=self.sim_setup["numberOfIntervals"],
                relative_tolerance=None,
                output_interval=self.sim_setup["outputInterval"],
                record_events=False,  # Used for an equidistant output
                start_values=start_values,
                apply_default_start_values=False,  # As we pass start_values already
                input=inputs,   # TODO: Add custom input
                output=self.sim_setup["resultNames"],
                timeout=self.sim_setup["timeout"],
                step_finished=None,
                model_description=self._model_description,
                fmu_instance=self._fmu_instance,
                fmi_type=self._fmi_type,
            )
            self._fmu_instance.reset()

        except Exception as error:
            self.logger.error(f"[SIMULATION ERROR] Error occurred while running FMU: \n {error}")
            if kwargs.get("fail_on_error", False):
                raise error
            return None

        # Reshape result:
        df = pd.DataFrame(res).set_index("time")
        df.index = df.index.astype("float64")

        return df

    def setup_fmu_instance(self):
        """
        Manually set up and extract the data to
        avoid this step in the simulate function
        :return:
        """
        _unzipdir = os.path.join(self.cd,
                                 os.path.basename(self.model_name)[:-4] + "_extracted")
        os.makedirs(_unzipdir, exist_ok=True)
        self._unzip_dir = fmpy.extract(self.model_name,
                                       unzipdir=_unzipdir)
        self._model_description = read_model_description(self._unzip_dir,
                                                         validate=True)

        if self._model_description.coSimulation is None:
            self._fmi_type = 'ModelExchange'
        else:
            self._fmi_type = 'CoSimulation'

        # Extract inputs, outputs & tuner (lists from parent classes will be appended)
        for v in self._model_description.modelVariables:
            if v.causality == 'input':
                self.inputs.append(v)
            if v.causality == 'output':
                self.outputs.append(v)
            if v.causality == 'parameter' or v.causality == 'calculatedParameter':
                self.parameters.append(v)

        self._fmu_instance = fmpy.instantiate_fmu(
            unzipdir=self._unzip_dir,
            model_description=self._model_description,
            fmi_type=self._fmi_type,
            visible=False,
            debug_logging=False,
            logger=self._custom_logger,
            fmi_call_logger=None,
            use_remoting=False
        )

    def _custom_logger(self, component, instanceName, status, category, message):
        """ Print the FMU's log messages to the command line (works for both FMI 1.0 and 2.0) """
        # pylint: disable=unused-argument, invalid-name
        label = ['OK', 'WARNING', 'DISCARD', 'ERROR', 'FATAL', 'PENDING'][status]
        _level_map = {'OK': logging.INFO,
                      'WARNING': logging.WARNING,
                      'DISCARD': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'FATAL': logging.FATAL,
                      'PENDING': logging.FATAL}
        if self.log_fmu:
            self.logger.log(level=_level_map[label], msg=message.decode("utf-8"))
