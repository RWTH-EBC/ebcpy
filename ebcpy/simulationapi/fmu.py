"""Module for classes using a fmu to
simulate models."""

import os
import logging
import pathlib
import atexit
import shutil
from typing import List, Union
import fmpy
from fmpy.model_description import read_model_description
from pydantic import Field
import pandas as pd
import numpy as np
from ebcpy import simulationapi, TimeSeriesData
from ebcpy.simulationapi import SimulationSetup, SimulationSetupClass, Variable
from typing import Optional
import warnings
# pylint: disable=broad-except


class FMU_Setup(SimulationSetup):
    """
    Add's custom setup parameters for simulating FMU's
    to the basic `SimulationSetup`
    """

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
        self.current_time = None  # Used in simulation/initialisation using do_step()
        self.vrs = None  # Used in simulation/initialisation using do_step()

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
        # Close MP of super class
        super().close()
        # Close if single process
        if not self.use_mp:
            if not self._fmu_instances:
                return  # Already closed
            self._single_close(fmu_instance=self._fmu_instances[0],
                               unzip_dir=self._unzip_dirs[0])
            self._unzip_dirs = {}
            self._fmu_instances = {}

    def _single_close(self, **kwargs):
        fmu_instance = kwargs["fmu_instance"]
        unzip_dir = kwargs["unzip_dir"]
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

        Additional kwargs:
        :keyword str result_file_suffix:
            Suffix of the result file. Supported options can be extracted
            from the TimeSeriesData.save() function.
            Default is 'csv'.
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
            fmu_instance.reset()
            # Simulate
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

        _types = {
            "Enumeration": int,
            "Integer": int,
            "Real": float,
            "Boolean": bool,
            "String": str
        }
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

    """
    New function: do_step + additional functions related to it: 
    The do_step() function allows to perform a single simulation step of an FMU. 
    Using the function in a loop, a whole simulation can be conducted. 
    Compared to the simulate() function this offers the possibility for co-simulation with other FMUs 
    or the use of inputs that are dependent from the system behaviour during the simulation (e.g. applying control).
    """

    # TODO:
    # - Pandas append depreciated
    # -

    def set_variables(self, var_dict: dict, idx_worker: int = 0):  # todo: how to deal with idx worker
        """
        Sets multiple variables.
        var_dict is a dict with variable names in keys.
        """
        for key, value in var_dict.items():
            var = self.vrs[key]
            vr = [var.valueReference]

            if var.type == 'Real':
                self._fmu_instances[idx_worker].setReal(vr, [float(value)])
            elif var.type in ['Integer', 'Enumeration']:
                self._fmu_instances[idx_worker].setInteger(vr, [int(value)])
            elif var.type == 'Boolean':
                self._fmu_instances[idx_worker].setBoolean(vr, [value == 1.0 or value or value == "True"])
            else:
                raise Exception("Unsupported type: %s" % var.type)

    def read_variables(self, vrs_list: list, idx_worker: int = 0):  # todo: how to deal with idx worker
        """
        Reads multiple variable values of FMU.
        vrs_list as list of strings
        Method returns a dict with FMU variable names as key
        """

        # initialize dict for results of simulation step
        res = {}

        for name in vrs_list:
            var = self.vrs[name]
            vr = [var.valueReference]

            if var.type == 'Real':
                res[name] = self._fmu_instances[idx_worker].getReal(vr)[0]
            elif var.type in ['Integer', 'Enumeration']:
                res[name] = self._fmu_instances[idx_worker].getInteger(vr)[0]
            elif var.type == 'Boolean':
                value = self._fmu_instances[idx_worker].getBoolean(vr)[0]
                res[name] = value != 0
            else:
                raise Exception("Unsupported type: %s" % var.type)

            res['SimTime'] = self.current_time

        return res

    def do_step(self, idx_worker: int = 0):  # todo: how to deal with idx worker
        """
        perform simulation step; return True if stop time reached
        """
        # check if stop time is reached
        if self.current_time < self.sim_setup.stop_time:
            # do simulation step
            status = self._fmu_instances[idx_worker].doStep(
                currentCommunicationPoint=self.current_time,
                communicationStepSize=self.sim_setup.output_interval)
            # update current time and determine status
            self.current_time += self.sim_setup.output_interval
            finished = False
        else:
            print('Simulation finished')
            finished = True
        return finished


    def initialize_fmu_for_do_step(self,
                                   parameters: Optional[dict] = None,
                                   init_values: Optional[dict] = None,
                                   tolerance: Optional[float] = None):  # todo: tol is not a user input in simulate()
        """
        Initialisation of FMU. To be called before using the do_step() function.
        Parameters and initial values can be set. An empty tsd object is created to store the results.
        """

        # FOLLOWING STEPS OF INITIALISATION ALREADY COVERED BY INSTANTIATING FMU API:
        # - Read model description
        # - extract .fmu file
        # - Create FMU2 Slave
        # - instantiate fmu instance (instead of fmu_instance.instantiate(), instantiate_fmu() is used)

        # Create dict of variable names with variable references from model description
        self.vrs = {}
        for variable in self._model_description.modelVariables:
            self.vrs[variable.name] = variable

        # Check for mp setting
        if self.use_mp:  # todo: check if this is enough to turn of falsely activated mp
            warnings.warn('Multi processing not available for step-by-step simulation using do_step. '
                          'Multi processing turned off')
            self.use_mp = False
            self.n_cpu = 1
        idx_worker = 0

        # Reset FMU instance
        self._fmu_instances[idx_worker].reset()

        # Set up experiment
        self._fmu_instances[idx_worker].setupExperiment(startTime=self.sim_setup.start_time,
                                                        stopTime=self.sim_setup.stop_time,
                                                        tolerance=tolerance)

        # initialize current time for use in do_step function
        self.current_time = self.sim_setup.start_time

        # Set parameters and initial values
        if init_values is None:
            init_values = {}
        if parameters is None:
            parameters = {}
        # merge initial values and parameters in one dict as they are treated similarly
        start_values = init_values.copy()
        start_values.update(parameters)

        # write parameters and initial values to fmu
        self.set_variables(var_dict=start_values, idx_worker=idx_worker)

        # Finalise initialisation
        self._fmu_instances[idx_worker].enterInitializationMode()
        self._fmu_instances[idx_worker].exitInitializationMode()

        # Initialize TimeSeriesObject to store results
        df = pd.DataFrame(columns=self.result_names)
        res = TimeSeriesData(df, default_tag="sim")
        # res.rename_axis('time').rename_axis(['Variables', 'Tags'], axis='columns')

        return res

    def do_step_read_write(self, input_step: Optional[dict] = None, input_table: Optional[TimeSeriesData] = None):
        """
        Function do perform a single simulation step (useful for co-simulation or control).
        1. read variables from FMU  # todo: discuss order!!
        2. write values to FMU
        3. perform simulation step
        Different types of inputs can be specified. Either as a tsd table (input_table) that contains values for the whole simulation
        or as inputs that are used for the specific step only (input_step).
        If both are given, input_step overwrites input_table.
        Returns dict of results for single step (res_tsd) and the boolean finished that indicates the simulation status
        """
        def bisection(arr,
                      val):  # todo: give alternative route in case the steps match or set simulation step based on iput step
            """
            Returns an index j such that val is between arr[j]
            and arr[j+1]. arr must be monotonic increasing.
            If val exceeds the last arr entry, the index of the last array entry is returned.
            Used in do_step to find correct value in input data table.
            """

            n = len(arr)
            if val < arr[0]:
                raise Exception('Fist entry in time table is above current time')
            elif val > arr[n - 1]:
                warnings.warn('Current time exceeds last val of input time table. Last val is hold')
            jl = 0  # Initialize lower
            ju = n - 1  # and upper limits.
            while ju - jl > 1:  # If we are not yet done,
                jm = (ju + jl) >> 1  # compute a midpoint with a bitshift  # todo: how does this binary operation work?
                if val >= arr[jm]:
                    jl = jm  # and replace either the lower limit
                else:
                    ju = jm  # or the upper limit, as appropriate.
                # Repeat until the test condition is satisfied.
            if val == arr[0]:  # edge cases at bottom
                return 0
            elif val == arr[n - 1]:  # and top
                return n - 1
            else:
                return jl

        # Check for mp setting  # todo: duplicate to initialize_fmu_for_do_step()
        if self.use_mp:  # todo: check if this is enough to turn off falsely activated mp
            warnings.warn('Multi processing not available for step-by-step simulation using do_step. '
                          'Multi processing turned off')
            self.use_mp = False
            self.n_cpu = 1
        idx_worker = 0

        # read variables from fmu # todo: extra function like in fmu_handler by aku?  # todo: discuss order/overwriting
        res = self.read_variables(vrs_list=self.result_names, idx_worker=idx_worker)

        # reshape res dictionary for tsd format
        res_tsd = {}
        for key, value in res.items():
            res_tsd[(key, 'sim')] = value

        # get input from input table (overwrite with specific input for single step)
        single_input = {}
        if input_table is not None:
            # extract value from input time table
            input_time_index = input_table.index
            idx_low = bisection(input_time_index, self.current_time)
            if isinstance(input_table, TimeSeriesData):
                input_table = input_table.to_df(force_single_index=True)
            single_input = input_table.iloc[idx_low].to_dict()#(orient='records')

        if input_step is not None:
            # overwrite with input for step
            single_input.update(input_step)

        # write inputs to fmu  # todo: extra function like in fmu_handler by aku?  ä also used in initialisation
        if single_input:
            self.set_variables(var_dict=single_input, idx_worker=idx_worker)

        finished = self.do_step(idx_worker=idx_worker)
        # if self.current_time < self.sim_setup.stop_time:
        #     # do simulation step
        #     status = self._fmu_instances[idx_worker].doStep(
        #         currentCommunicationPoint=self.current_time,
        #         communicationStepSize=self.sim_setup.output_interval)
        #     # update current time and determine status
        #     self.current_time += self.sim_setup.output_interval
        #     finished = False
        # else:
        #     print('Simulation finished')
        #     finished = True

        return res_tsd, finished


