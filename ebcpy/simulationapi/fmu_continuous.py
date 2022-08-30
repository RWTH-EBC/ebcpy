import os
import atexit
from typing import List, Union
import fmpy
import pandas as pd
import numpy as np
from ebcpy import TimeSeriesData
from ebcpy.simulationapi.fmu import FMU
from ebcpy.simulationapi import ContinuousSimulation
from ebcpy.simulationapi.config import *


class FMU_API(FMU, ContinuousSimulation):

    _sim_setup_class: SimulationSetupClass = SimulationSetupFMU_Continuous
    # _items_to_drop = ["pool"]
    _items_to_drop = ["pool", "_fmu_instance", "_unzip_dir"]
    _type_map = {
        float: np.double,
        bool: np.bool_,
        int: np.int_
    }

    def __init__(self, config, n_cpu, log_fmu: bool = True):  # todo: consider n_core and log_fmu in config -> requires more specific config classes
        self.config = self._exp_config_class.parse_obj(config)
        FMU.__init__(self, log_fmu=log_fmu)
        ContinuousSimulation.__init__(self, model_name=self.config.file_path, n_cpu=n_cpu)
        # Register exit option
        atexit.register(self.close)

    def simulate(self,
                 parameters: Union[dict, List[dict]] = None,
                 return_option: str = "time_series",
                 **kwargs):
        """
        Perform the single simulation for the given
        unzip directory and fmu_instance.
        See the docstring of simulate() for information on kwargs.

        Additional kwargs:
        :keyword str result_file_suffix:
            Suffix of the result file. Supported options can be extracted
            from the TimeSeriesData.save() function.
            Default is 'csv'.
        """
        return super().simulate(parameters=parameters, return_option=return_option, **kwargs)

    def _single_simulation(self, kwargs):  # todo: warum nicht ** kwargs und warum pop und get??
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
            if self._fmu_instance is None:
                self._setup_single_fmu_instance(use_mp=True)

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
            # Try to match the type, default is np.double.
            # 'time' is not in inputs and thus handled separately.
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
            # reset the FMU instance instead of creating a new one
            self._fmu_instance.reset()
            # Simulate
            res = fmpy.simulate_fmu(
                filename=self._unzip_dir,
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
            result_file_name = kwargs.get("result_file_name", "resultFile")
            result_file_suffix = kwargs.get("result_file_suffix", "csv")
            savepath = kwargs.get("savepath", None)

            if savepath is None:
                savepath = self.cd

            os.makedirs(savepath, exist_ok=True)
            filepath = os.path.join(savepath, f"{result_file_name}.{result_file_suffix}")
            TimeSeriesData(df).droplevel(1, axis=1).save(
                filepath=filepath,
                key="simulation"
            )

            return filepath
        if return_option == "last_point":
            return df.iloc[-1].to_dict()
        # Else return time series data
        tsd = TimeSeriesData(df, default_tag="sim")
        return tsd

    def close(self):
        """
        Closes the fmu.

        :return: bool
            True on success
        """
        print('FMU "{}" closed'.format(self._model_description.modelName))  # fixme: adjust for mp
        # Close MP of super class
        ContinuousSimulation.close(self)
        # Close if single process
        if not self.use_mp:
            if not self._fmu_instance:
                return  # Already closed
            self._single_close(fmu_instance=self._fmu_instance,
                               unzip_dir=self._unzip_dir)
            self._unzip_dir = None
            self._fmu_instance = None

    def _close_multiprocessing(self, _):
        """Small helper function"""
        idx_worker = self.worker_idx
        if self._fmu_instance is None:
            return  # Already closed
        self.logger.error(f"Closing fmu for worker {idx_worker}")
        self._single_close(fmu_instance=self._fmu_instance,
                           unzip_dir=self._unzip_dir)
        self._unzip_dir = None
        self._fmu_instance = None
        FMU_API._unzip_dir = None
        FMU_API._fmu_instance = None