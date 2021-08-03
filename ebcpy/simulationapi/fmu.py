"""Module for classes using a fmu to
simulate models."""

import os
import logging
import pathlib
import atexit
import shutil
import fmpy
from fmpy.model_description import read_model_description
from pydantic import Field
import pandas as pd
import numpy as np
from ebcpy import simulationapi, TimeSeriesData
from ebcpy.simulationapi import SimulationSetup, SimulationSetupClass, Variable
# pylint: disable=broad-except


class FMU_Setup(SimulationSetup):

    timeout: float = Field(
        title="timeout",
        default=np.inf,
        description="Timeout after which the simulation stops."
    )

    _default_solver = "CVode"
    _allowed_solvers = ["CVode", "Euler"]

class FMU_API(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.

    Example:

    >>> import matplotlib.pyplot as plt
    >>> from ebcpy import FMU_API
    >>> # Select any valid fmu. Replace the line below if
    >>> # you don't have this file on your device.
    >>> model_name = "Path to your fmu"
    >>> fmu_api = FMU_API(model_name)
    >>> fmu_api.sim_setup = {"stop_time": 3600}
    >>> result_df = fmu_api.simulate()
    >>> fmu_api.close()
    >>> # Select an exemplary column
    >>> col = result_df.columns[0]
    >>> plt.plot(result_df[col], label=col)
    >>> _ = plt.legend()
    >>> _ = plt.show()

    .. versionadded:: 0.1.7
    """

    _sim_setup_class: SimulationSetupClass = FMU_Setup

    def __init__(self, model_name, cd=None):
        """Instantiate class parameters"""
        if isinstance(model_name, pathlib.Path):
            model_name = str(model_name)
        if not model_name.lower().endswith(".fmu"):
            raise ValueError(f"{model_name} is not a valid fmu file!")
        if cd is None:
            cd = os.path.dirname(model_name)
        super().__init__(cd, model_name)

        # Init instance attributes
        self._unzip_dir = None
        self._fmu_instance = None
        self._model_description = None
        self._fmi_type = None
        self.log_fmu = True

        # Setup the fmu instance
        self.setup_fmu_instance()
        # Set all outputs to result_names:
        self.result_names = self.outputs.keys()
        # Register exit option
        atexit.register(self.close)

    def close(self):
        """
        Closes the fmu.

        :return: bool
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
        if self._unzip_dir is not None:
            shutil.rmtree(self._unzip_dir, ignore_errors=True)
        self._unzip_dir = None

    def simulate(self,
                 parameters: dict = None,
                 return_option: str = "time_series",
                 **kwargs):
        """
        Simulate current simulation-setup.


        Additional settings:
        :param dataframe inputs:
            Pandas.Dataframe of the input data for simulating the FMU with fmpy
        :keyword Boolean fail_on_error:
            If True, an error in fmpy will trigger an error in this script.
            Default is false
        :return:
            Filepath of the mat-file.
        """
        inputs = kwargs.get("inputs", None)
        if inputs is not None:
            inputs = inputs.copy()  # Create save copy
            # Shift all columns, because "simulate_fmu" gets an input at
            # timestep x and calculates the related output for timestep x+1
            shift_period = int(self.sim_setup.output_interval /
                               (inputs.index[0] - inputs.index[1]))
            inputs = inputs.shift(periods=shift_period)
            # Shift time column back
            inputs.time = inputs.time.shift(-shift_period, fill_value=0)
            # drop NANs
            inputs = inputs.dropna()

            # Convert df to structured numpy array for fmpy: simulate_fmu
            inputs_tuple = [tuple(columns) for columns in inputs.to_numpy()]
            # TODO: implement more than "np.double" as type-possibilities
            dtype = [(i, np.double) for i in
                     inputs.columns]
            inputs = np.array(inputs_tuple, dtype=dtype)
        if parameters is None:
            parameters = {}
        else:
            self.check_unsupported_variables(variables=list(parameters.keys()),
                                             type_of_var="parameters")
        try:
            res = fmpy.simulate_fmu(
                self._unzip_dir,
                start_time=self.sim_setup.start_time,
                stop_time=self.sim_setup.stop_time,
                solver=self.sim_setup.solver,
                step_size=self.sim_setup.fixedstepsize,
                relative_tolerance=None,
                output_interval=self.sim_setup.output_interval,
                record_events=False,  # Used for an equidistant output
                start_values=parameters,
                apply_default_start_values=False,  # As we pass start_values already
                input=inputs,
                output=self.result_names,
                timeout=self.sim_setup.timeout,
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
        df.index = np.round(df.index.astype("float64"),
                            str(self.sim_setup.output_interval)[::-1].find('.'))

        if return_option == "savepath":
            result_file_name = kwargs.get("result_file_name", 'resultFile')
            savepath = kwargs.get("savepath", None)

            if savepath is None:
                savepath = self.cd
            filepath = os.path.join(savepath, f"{result_file_name}.hdf")
            df.to_hdf(filepath,
                      key="simulation")
            return filepath
        if return_option == "last_point":
            return df.iloc[-1].to_dict()
        # Else return time series data
        tsd = TimeSeriesData(df, default_tag="sim")
        return tsd

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
                                                         validate=False)

        if self._model_description.coSimulation is None:
            self._fmi_type = 'ModelExchange'
        else:
            self._fmi_type = 'CoSimulation'

        def _to_bound(value):
            if value is None or \
                    not isinstance(value, (float, int, bool)):
                return np.inf
            return value

        # Extract inputs, outputs & tuner (lists from parent classes will be appended)
        for var in self._model_description.modelVariables:

            _var_ebcpy = Variable(
                min=-_to_bound(var.min),
                max=_to_bound(var.max),
                value=var.start
            )
            if var.causality == 'input':
                self.inputs[var.name] = _var_ebcpy
            elif var.causality == 'output':
                self.outputs[var.name] = _var_ebcpy
            elif var.causality == 'parameter' or var.causality == 'calculatedParameter':
                self.parameters[var.name] = _var_ebcpy
            elif var.causality == 'local':
                self.states[var.name] = _var_ebcpy
            else:
                self.logger.error(f"Could not map causality {var.causality}"
                                  f" to any variable type.")

        self._fmu_instance = fmpy.instantiate_fmu(
            unzipdir=self._unzip_dir,
            model_description=self._model_description,
            fmi_type=self._fmi_type,
            visible=False,
            debug_logging=False,
            logger=self._custom_logger,
            fmi_call_logger=None
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
