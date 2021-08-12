"""Module for classes using a fmu to
simulate models."""

import os
import logging
import pathlib
import atexit
import shutil
import time
from itertools import repeat
from typing import List, Union
import fmpy
from fmpy.model_description import read_model_description
from pydantic import Field
import pandas as pd
import numpy as np
from ebcpy import simulationapi, TimeSeriesData
from ebcpy.simulationapi import SimulationSetup, SimulationSetupClass, Variable
# pylint: disable=broad-except

#import multiprocessing as mp
#m = mp.Manager()
#m.dict()


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

    :keyword bool log_fmu:
        Whether to print fmu messages or not.

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
    _fmu_instances: dict = {}
    _unzip_dirs: dict = {}
    _type_map = {
        float: np.double,
        bool: np.bool_,
        int: np.int_
    }

    def __init__(self, cd, model_name, **kwargs):
        """Instantiate class parameters"""
        # Init instance attributes
        self._model_description = None
        self._fmi_type = None
        self.log_fmu = kwargs.get("log_fmu", True)
        self._single_unzip_dir: str = None

        if isinstance(model_name, pathlib.Path):
            model_name = str(model_name)
        if not model_name.lower().endswith(".fmu"):
            raise ValueError(f"{model_name} is not a valid fmu file!")
        if cd is None:
            cd = os.path.dirname(model_name)
        super().__init__(cd, model_name, **kwargs)
        # Register exit option
        atexit.register(self.close)

    def _update_model(self):
        # Setup the fmu instance
        self.setup_fmu_instance()

    def close(self):
        """
        Closes the fmu.

        :return: bool
            True on success
        """
        if self.use_mp:
            try:
                self.pool.map(self._close_multiprocessing,
                              [_ for _ in range(self.n_cpu)])
                self.pool.close()
                self.pool.join()
            except ValueError:
                pass  # Already closed prior to atexit
        else:
            if not self._fmu_instances:
                return  # Already closed
            self._single_close(fmu_instance=self._fmu_instances[0],
                               unzip_dir=self._unzip_dirs[0])
            self._unzip_dirs = {}
            self._fmu_instances = {}

    def _single_close(self, fmu_instance, unzip_dir):
        try:
            fmu_instance.terminate()
        except Exception as error:  # This is due to fmpy which does not yield a narrow error
            self.logger.error(f"Could not terminate fmu instance: {error}")
        try:
            fmu_instance.freeInstance()
        except OSError as error:
            self.logger.error(f"Could not free fmu instance: {error}")
        # Remove the extracted files
        if unzip_dir is not None:
            try:
                shutil.rmtree(unzip_dir)
            except FileNotFoundError:
                pass  # Nothing to delete
            except PermissionError:
                self.logger.error("Could not delete unzipped fmu "
                                  "in location %s. Delete it yourself.", unzip_dir)

    def _close_multiprocessing(self, _):
        """Small helper function"""
        idx_worker = self.worker_idx
        if idx_worker not in self._fmu_instances:
            return  # Already closed
        self.logger.error(f"Closing fmu for worker {idx_worker}")
        self._single_close(fmu_instance=self._fmu_instances[idx_worker],
                           unzip_dir=self._unzip_dirs[idx_worker])
        self._unzip_dirs = {}
        self._fmu_instances = {}

    def simulate(self,
                 parameters: Union[dict, List[dict]] = None,
                 return_option: str = "time_series",
                 **kwargs):
        """
        Perform the single simulation for the given
        unzip directory and fmu_instance.
        See the docstring of simulate() for information on kwargs.
        """
        return super().simulate(parameters=parameters, return_option=return_option, **kwargs)

    def _single_simulation(self, kwargs):
        """
        Perform the single simulation for the given
        unzip directory and fmu_instance.
        See the docstring of simulate() for information on kwargs.

        The single argument kwarg is to make this
        function accessible by multiprocessing pool.map.
        """
        # Unpack kwargs:
        parameters = kwargs.pop("parameters", None)
        return_option = kwargs.pop("return_option", "time_series")
        inputs = kwargs.get("inputs", None)
        fail_on_error = kwargs.get("fail_on_error", True)

        if self.use_mp:
            idx_worker = self.worker_idx
            if idx_worker not in self._fmu_instances:
                self._setup_single_fmu_instance(use_mp=True)
        else:
            idx_worker = 0

        fmu_instance = self._fmu_instances[idx_worker]
        unzip_dir = self._unzip_dirs[idx_worker]

        if inputs is not None:
            if not isinstance(inputs, (TimeSeriesData, pd.DataFrame)):
                raise TypeError("DataFrame or TimeSeriesData object expected for inputs.")
            inputs = inputs.copy()  # Create save copy
            if isinstance(inputs, TimeSeriesData):
                inputs = inputs.to_df(force_single_index=True)
            if "time" in inputs.columns:
                raise IndexError(
                    "Given inputs contain a column named 'time'. "
                    "The index is assumed to contain the time-information."
                )
            # Convert df to structured numpy array for fmpy: simulate_fmu
            inputs.insert(0, column="time", value=inputs.index)
            inputs_tuple = [tuple(columns) for columns in inputs.to_numpy()]
            # Try to match the type, default is np.double. 'time' is not in inputs and thus handled separately.
            dtype = [(inputs.columns[0], np.double)] + \
                    [(col,
                      self._type_map.get(self.inputs[col].type, np.double)
                      ) for col in inputs.columns[1:]]
            inputs = np.array(inputs_tuple, dtype=dtype)
        if parameters is None:
            parameters = {}
        else:
            self.check_unsupported_variables(variables=list(parameters.keys()),
                                             type_of_var="parameters")
        try:
            res = fmpy.simulate_fmu(
                filename=unzip_dir,
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
                fmu_instance=fmu_instance,
                fmi_type=self._fmi_type,
            )
            fmu_instance.reset()

        except Exception as error:
            self.logger.error(f"[SIMULATION ERROR] Error occurred while running FMU: \n {error}")
            if fail_on_error:
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
        avoid this step in the simulate function.
        """
        self.logger.info("Extracting fmu and reading fmu model description")
        # First load model description and extract variables
        self._single_unzip_dir = os.path.join(self.cd,
                                              os.path.basename(self.model_name)[:-4] + "_extracted")
        os.makedirs(self._single_unzip_dir, exist_ok=True)
        self._single_unzip_dir = fmpy.extract(self.model_name,
                                         unzipdir=self._single_unzip_dir)
        self._model_description = read_model_description(self._single_unzip_dir,
                                                         validate=True)

        if self._model_description.coSimulation is None:
            self._fmi_type = 'ModelExchange'
        else:
            self._fmi_type = 'CoSimulation'

        def _to_bound(value):
            if value is None or \
                    not isinstance(value, (float, int, bool)):
                return np.inf
            return value
        self.logger.info("Reading model variables")

        _types = {"Enumeration": int,
                  "Integer": int,
                  "Real": float,
                  "Boolean": bool, }
        # Extract inputs, outputs & tuner (lists from parent classes will be appended)
        for var in self._model_description.modelVariables:
            if var.start is not None:
                var.start = _types[var.type](var.start)

            _var_ebcpy = Variable(
                min=-_to_bound(var.min),
                max=_to_bound(var.max),
                value=var.start,
                type=_types[var.type]
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

        if self.use_mp:
            self.logger.info("Extracting fmu %s times for "
                             "multiprocessing on %s processes",
                             self.n_cpu, self.n_cpu)
            self.pool.map(
                self._setup_single_fmu_instance,
                [True for _ in range(self.n_cpu)]
            )
            self.logger.info("Instantiated fmu's on all processes.")
        else:
            self._setup_single_fmu_instance(use_mp=False)

    def _setup_single_fmu_instance(self, use_mp):
        if not use_mp:
            wrk_idx = 0
        else:
            wrk_idx = self.worker_idx
            if wrk_idx in self._fmu_instances:
                return True
        if use_mp:
            unzip_dir = self._single_unzip_dir + f"_worker_{wrk_idx}"
            unzip_dir = fmpy.extract(self.model_name,
                                     unzipdir=unzip_dir)
        else:
            unzip_dir = self._single_unzip_dir
        self.logger.info("Instantiating fmu for worker %s", wrk_idx)
        self._fmu_instances.update({wrk_idx: fmpy.instantiate_fmu(
            unzipdir=unzip_dir,
            model_description=self._model_description,
            fmi_type=self._fmi_type,
            visible=False,
            debug_logging=False,
            logger=self._custom_logger,
            fmi_call_logger=None)})
        self._unzip_dirs.update({
            wrk_idx: unzip_dir
        })
        return True

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
    import matplotlib.pyplot as plt
    # Setup the fmu-api:
    model_name = r"E:\04_git\ebcpy\tests\data\PumpAndValve_windows.fmu"
    cwd = r"D:\00_temp\test_mp_fmu"
    fmu_api = FMU_API(cd=cwd, model_name=model_name, n_cpu=10)
    fmu_api.result_names = ["heatCapacitor.T"]
    fmu_api.set_sim_setup({"stop_time": 10,
                           "output_interval": 0.001})
    PARAMETERS = [{"speedRamp.duration": 0.1 + 0.1 * i} for i in range(100)]
    res = fmu_api.simulate(parameters=PARAMETERS)
    for idx, _res in enumerate(res):
        plt.plot(_res["heatCapacitor.T"], label=idx)
    # Close the api to remove the created files:
    fmu_api.close()
    plt.legend()
    plt.show()
