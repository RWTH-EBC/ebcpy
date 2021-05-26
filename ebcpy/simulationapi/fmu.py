"""Module for classes using a fmu to
simulate models."""

import os
import logging
import shutil
import multiprocessing as mp
from typing import Any, List, Union
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

    :keyword bool log_fmu:
        Whether to print fmu messages or not.
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

    _fmu_instances: dict = {}
    _unzip_dirs: dict = {}

    def __init__(self, cd, model_name, **kwargs):
        """Instantiate class parameters"""
        super().__init__(cd, model_name, **kwargs)
        if not model_name.lower().endswith(".fmu"):
            raise ValueError(f"{model_name} is not a valid fmu file!")
        # Init instance attributes
        self._model_description = None
        self._fmi_type = None
        self.log_fmu = kwargs.get("log_fmu", True)

        # Setup the fmu instance
        self.setup_fmu_instance()

    def close(self):
        """
        Closes the fmu.
        :return:
            True on success
        """
        if self.use_mp:
            self.pool.map(self._close_multiprocessing, [_ for _ in range(self.n_cpu)])
        else:
            self._close_single(fmu_instance=self._fmu_instances[0],
                               unzip_dir=self._unzip_dirs[0])
        self._unzip_dirs = {}
        self._fmu_instances = {}

    def _close_single(self, fmu_instance, unzip_dir):
        try:
            fmu_instance.terminate()
        except Exception as error:  # This is due to fmpy which does not yield a narrow error
            self.logger.error(f"Could not terminate fmu instance: {error}")
        try:
            fmu_instance.freeInstance()
        except OSError as error:
            self.logger.error(f"Could not free fmu instance: {error}")
        # Remove the extracted files
        shutil.rmtree(unzip_dir)

    def _close_multiprocessing(self, _):
        idx_worker = self.get_worker_idx()
        self._close_single(fmu_instance=self._fmu_instances[idx_worker],
                           unzip_dir=self._unzip_dirs[idx_worker])

    def _single_simulation(self, kwargs):
        """
        Perform the single simulation for the given
        unzip directory and fmu_instance.
        See the docstring of simulate() for information on kwargs.

        The single argument kwarg is to make this
        function accessible by multiprocessing pool.map.
        """
        if self.use_mp:
            idx_worker = mp.current_process()._identity[0]
        else:
            idx_worker = 0
        print(self._fmu_instances)
        fmu_instance = self._fmu_instances[idx_worker]
        unzip_dir = self._unzip_dirs[idx_worker]

        # First update the simulation setup
        self.set_sim_setup(kwargs.get("sim_setup", {}))

        # Dictionary with all tuner parameter names & -values
        start_values = {self.sim_setup["initialNames"][i]: value
                        for i, value in enumerate(self.sim_setup["initialValues"])}

        inputs = kwargs.get("inputs", None)
        if inputs is not None:
            inputs = inputs.copy()  # Create save copy
            # Shift all columns, because "simulate_fmu" gets an input at
            # timestep x and calculates the related output for timestep x+1
            shift_period = int(self.sim_setup["outputInterval"] /
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

        try:
            res = fmpy.simulate_fmu(
                filename=unzip_dir,
                start_time=self.sim_setup["startTime"],
                stop_time=self.sim_setup["stopTime"],
                solver=self.sim_setup["solver"],
                step_size=self.sim_setup["numberOfIntervals"],
                relative_tolerance=None,
                output_interval=self.sim_setup["outputInterval"],
                record_events=False,  # Used for an equidistant output
                start_values=start_values,
                apply_default_start_values=False,  # As we pass start_values already
                input=inputs,  # TODO: Add custom input
                output=self.sim_setup["resultNames"],
                timeout=self.sim_setup["timeout"],
                step_finished=None,
                model_description=self._model_description,
                fmu_instance=fmu_instance,
                fmi_type=self._fmi_type,
            )
            fmu_instance.reset()

        except Exception as error:
            self.logger.error(f"[SIMULATION ERROR] Error occurred while running FMU: \n {error}")
            if kwargs.get("fail_on_error", False):
                raise error
            return None

        # Reshape result:
        df = pd.DataFrame(res).set_index("time")
        df.index = df.index.astype("float64")

        return df

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
        # Decide between mp and single core
        if self.use_mp:
            sim_setups = kwargs.get("sim_setup", {})
            if isinstance(sim_setups, dict):
                sim_setups = [sim_setups]
            inputs = kwargs.get("sim_setup", None)
            if isinstance(inputs, list):
                if len(inputs) != len(sim_setups):
                    raise ValueError(f"Mismatch in multiprocessing of "
                                     f"given sim_setups ({len(sim_setups)}) "
                                     f"and given inputs ({len(inputs)})")
            else:
                inputs = [inputs] * len(sim_setups)
            fail_on_error = kwargs.get("fail_on_error", False)
            if isinstance(fail_on_error, list):
                if len(fail_on_error) != len(sim_setups):
                    raise ValueError(f"Mismatch in multiprocessing of "
                                     f"given sim_setups ({len(sim_setups)}) "
                                     f"and given inputs ({len(fail_on_error)})")
            else:
                fail_on_error = [fail_on_error] * len(sim_setups)
            kwargs = []
            for _sim_setup, _inputs, _fail_on_error in zip(sim_setups,
                                                           inputs,
                                                           fail_on_error):
                kwargs.append(
                    {"sim_setup": _sim_setup,
                     "inputs": _inputs,
                     "fail_on_error": _fail_on_error,
                     }
                )
            return self.pool.map(self._single_simulation, kwargs)
        else:
            return self._single_simulation(kwargs)

    def setup_fmu_instance(self):
        """
        Manually set up and extract the data to
        avoid this step in the simulate function.
        """
        # First load model description and extract variables
        _unzip_dir_single = os.path.join(self.cd,
                                         os.path.basename(self.model_name)[:-4] + "_extracted")
        os.makedirs(_unzip_dir_single, exist_ok=True)
        _unzip_dir_single = fmpy.extract(self.model_name,
                                         unzipdir=_unzip_dir_single)
        self._model_description = read_model_description(_unzip_dir_single,
                                                         validate=True)

        if self._model_description.coSimulation is None:
            self._fmi_type = 'ModelExchange'
        else:
            self._fmi_type = 'CoSimulation'

        # Extract inputs, outputs & tuner (lists from parent classes will be appended)
        for var in self._model_description.modelVariables:
            if var.causality == 'input':
                self.inputs.append(var)
            if var.causality == 'output':
                self.outputs.append(var)
            if var.causality == 'parameter' or var.causality == 'calculatedParameter':
                self.parameters.append(var)

        if self.use_mp:
            _unzip_dirs = []
            for cpu_idx in range(self.n_cpu):
                _unzip_dir_cpu_idx = _unzip_dir_single + f"_worker_{cpu_idx}"
                _unzip_dir_cpu_idx = fmpy.extract(self.model_name,
                                                  unzipdir=_unzip_dir_cpu_idx)
                _unzip_dirs.append(_unzip_dir_cpu_idx)
            self.pool.map(self._setup_single_fmu_instance, _unzip_dirs)
        else:
            self._unzip_dirs = {0: _unzip_dir_single}
            self._fmu_instances = {0: fmpy.instantiate_fmu(
                unzipdir=_unzip_dir_single,
                model_description=self._model_description,
                fmi_type=self._fmi_type,
                visible=False,
                debug_logging=False,
                logger=self._custom_logger,
                fmi_call_logger=None,
                use_remoting=False)
            }

    def _setup_single_fmu_instance(self, unzip_dir):
        idx_worker = self.get_worker_idx()
        self._fmu_instances.update({idx_worker: fmpy.instantiate_fmu(
            unzipdir=unzip_dir,
            model_description=self._model_description,
            fmi_type=self._fmi_type,
            visible=False,
            debug_logging=False,
            logger=self._custom_logger,
            fmi_call_logger=None,
            use_remoting=False)})
        self._unzip_dirs.update({
            idx_worker: unzip_dir
        })
        return None

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
