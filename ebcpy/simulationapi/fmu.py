"""Module for classes using a fmu to
simulate models. It contains FMU base functionalities,
and an api for continuous and for discrete fmu simulation."""

import os
import logging
import pathlib
import shutil
import atexit
import warnings
from typing import Dict, List, Union, Optional
import fmpy
from fmpy.model_description import read_model_description
from pydantic import FilePath
import numpy as np
import pandas as pd
from ebcpy import TimeSeriesData
from ebcpy.simulationapi import ContinuousSimulation, DiscreteSimulation
from ebcpy.simulationapi import Variable
from ebcpy.simulationapi.config import ExperimentConfigFMU_Continuous, SimulationSetupFMU_Continuous
from ebcpy.simulationapi.config import ExperimentConfigFMU_Discrete, SimulationSetupFMU_Discrete
from ebcpy.simulationapi.config import ExperimentConfigurationClass, SimulationSetupClass
from ebcpy.utils.interpolation import interp_df
from ebcpy.utils.reproduction import CopyFile


class FMU:
    """
    Base class for simulation using the fmpy library and
    a functional mockup interface as a model input.
    This class has to be inherited besides the Model base class.

    :param str file_path:
        File path to the fmu file.
    :param str cd:
        Working directory in which the fmu files are extracted.
    :param log_fmu:
         Whether to print fmu messages or not.
    """

    _fmu_instance = None
    _unzip_dir: Optional[str] = None

    def __init__(self, file_path: str, cd: str, log_fmu: bool = True):
        self._unzip_dir = None
        self._fmu_instance = None
        path = file_path
        if isinstance(file_path, pathlib.Path):
            path = str(file_path)
        if not path.lower().endswith(".fmu"):
            raise ValueError(f"{file_path} is not a valid fmu file!")
        self.path = path
        if cd is not None:
            self.cd = cd
        else:
            self.cd = os.path.dirname(path)
        self.log_fmu = log_fmu
        self._var_refs: Optional[dict] = None  # Dict of variables and their references
        self._model_description = None
        self._fmi_type = None
        self._single_unzip_dir: Optional[str] = None
        # initialize logger
        self.logger = None
        # initialize model variables
        self.inputs: Dict[str, Variable] = {}  # Inputs of model
        self.outputs: Dict[str, Variable] = {}  # Outputs of model
        self.parameters: Dict[str, Variable] = {}  # Parameter of model
        self.states: Dict[str, Variable] = {}  # States of model
        # Placeholders for variables that are required by subclass
        # todo: self.n_cpu, self.use_mp and self.pool are not connected with FMU class
        self.n_cpu = None
        self.use_mp = None
        self.pool = None

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

    def find_vars(self, start_str: str):
        """
        Returns all variables starting with start_str
        .
        :param str start_str:
            All variables starting with start_str are considered
        :return:
            List: List of variables that fulfill the search criteria
        """

        key = list(self._var_refs.keys())
        key_list = []
        for _, k in enumerate(key):
            if k.startswith(start_str):
                key_list.append(k)
        return key_list

    def set_variables(self, var_dict: dict):
        """
        Sets multiple variables.
        :param dict var_dict:
            Dictionary with variable name in key and variable value in value
        """

        for key, value in var_dict.items():
            var = self._var_refs[key]
            var_ref = [var.valueReference]

            if var.type == 'Real':
                self._fmu_instance.setReal(var_ref, [float(value)])
            elif var.type in ['Integer', 'Enumeration']:
                self._fmu_instance.setInteger(var_ref, [int(value)])
            elif var.type == 'Boolean':
                self._fmu_instance.setBoolean(var_ref, [value == 1.0 or value or value == "True"])
            else:
                raise Exception(f"Unsupported type: {var.type}")

    def read_variables(self, vrs_list: list):
        """
        Reads multiple variable values
        :param list vrs_list:
            List of variables to be read from FMU
        :return:
            Dict: Dictionary with requested variables and their values
        """

        # initialize dict for results of simulation step
        res = {}

        for name in vrs_list:
            var = self._var_refs[name]
            var_ref = [var.valueReference]

            if var.type == 'Real':
                res[name] = self._fmu_instance.getReal(var_ref)[0]
            elif var.type in ['Integer', 'Enumeration']:
                res[name] = self._fmu_instance.getInteger(var_ref)[0]
            elif var.type == 'Boolean':
                value = self._fmu_instance.getBoolean(var_ref)[0]
                res[name] = value != 0
            else:
                raise Exception(f"Unsupported type: {var.type}")

        return res

    def _update_model(self):
        # Setup the fmu instance
        self.setup_fmu_instance()

    def setup_fmu_instance(self):
        """
        Manually set up and extract the data to
        avoid this step in the simulate function.
        """
        self.logger.info("Extracting fmu and reading fmu model description")
        self._single_unzip_dir = os.path.join(self.cd,
                                              os.path.basename(self.path)[:-4] + "_extracted")
        os.makedirs(self._single_unzip_dir, exist_ok=True)
        self._single_unzip_dir = fmpy.extract(self.path,
                                              unzipdir=self._single_unzip_dir)
        self._model_description = read_model_description(self._single_unzip_dir,
                                                         validate=True)

        if self._model_description.coSimulation is None:
            self._fmi_type = 'ModelExchange'
        else:
            self._fmi_type = 'CoSimulation'

        # Create dict of variable names with variable references from model description
        self._var_refs = {}
        for variable in self._model_description.modelVariables:
            self._var_refs[variable.name] = variable

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
                min=var.min,
                max=var.max,
                value=var.start,
                type=_types[var.type]
            )
            if var.causality == 'input':
                self.inputs[var.name] = _var_ebcpy
            elif var.causality == 'output':
                self.outputs[var.name] = _var_ebcpy
            elif var.causality in ('parameter', 'calculatedParameter'):
                self.parameters[var.name] = _var_ebcpy
            elif var.causality == 'local':
                self.states[var.name] = _var_ebcpy
            else:
                self.logger.error(f"Could not map causality {var.causality}"
                                  f" to any variable type.")
                print()

        if self.use_mp:
            self.logger.info(f"Extracting fmu {self.n_cpu} times for "
                             f"multiprocessing on {self.n_cpu} processes")
            self.pool.map(
                self._setup_single_fmu_instance,
                [True for _ in range(self.n_cpu)]
            )
            self.logger.info("Instantiated fmu's on all processes.")
        else:
            self._setup_single_fmu_instance(use_mp=False)

    def _setup_single_fmu_instance(self, use_mp):
        if use_mp:
            wrk_idx = self.worker_idx
            if self._fmu_instance is not None:
                return True
            unzip_dir = self._single_unzip_dir + f"_worker_{wrk_idx}"
            fmpy.extract(self.path,
                         unzipdir=unzip_dir)
        else:
            wrk_idx = 0
            unzip_dir = self._single_unzip_dir

        self.logger.info("Instantiating fmu for worker %s", wrk_idx)
        fmu_instance = fmpy.instantiate_fmu(
            unzipdir=unzip_dir,
            model_description=self._model_description,
            fmi_type=self._fmi_type,
            visible=False,
            debug_logging=False,
            logger=self._custom_logger,
            fmi_call_logger=None)
        if use_mp:
            FMU._fmu_instance = fmu_instance
            FMU._unzip_dir = unzip_dir
        else:
            self._fmu_instance = fmu_instance
            self._unzip_dir = unzip_dir
        return True

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


class FMU_API(FMU, ContinuousSimulation):
    """
    Class for continuous simulation using the fmpy library and
    a functional mockup interface as a model input.

    :param dict config:
         Dictionary with experiment configuration
    :param int n_cpu:
        Number of cores to be used by simulation.
        If None is given, single core will be used.
        Maximum number equals the cpu count of the device.
    :param bool log_fmu:
        Whether to print fmu messages or not.

    Example:
    >>> import matplotlib.pyplot as plt
    >>> from ebcpy import FMU_API
    >>> # Select any valid fmu. Replace the line below if
    >>> # you don't have this file on your device.
    >>> path = "Path to your fmu"
    >>> fmu_api = FMU_API({'file_path': path})
    >>> fmu_api.set_sim_setup({"stop_time": 3600})
    >>> result_df = fmu_api.simulate()
    >>> fmu_api.close()
    >>> # Select an exemplary column
    >>> col = result_df.columns[0]
    >>> plt.plot(result_df[col], label=col)
    >>> _ = plt.legend()
    >>> _ = plt.show()
    """

    _sim_setup_class: SimulationSetupClass = SimulationSetupFMU_Continuous
    _exp_config_class: ExperimentConfigurationClass = ExperimentConfigFMU_Continuous
    _items_to_drop = ["pool", "_fmu_instance", "_unzip_dir"]
    _type_map = {
        float: np.double,
        bool: np.bool_,
        int: np.int_
    }

    def __init__(self,
                 config: Optional[dict] = None,
                 n_cpu: int = 1,
                 log_fmu: bool = True,
                 **kwargs):
        config = self._check_config(config, **kwargs)  # generate config out of outdated arguments
        self.config = self._exp_config_class.parse_obj(config)

        FMU.__init__(self, file_path=self.config.file_path, cd=self.config.cd, log_fmu=log_fmu)
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
        # Close MP of super class
        ContinuousSimulation.close(self)
        # Close if single process
        if not self.use_mp:
            if not self._fmu_instance:
                return  # Already closed
            self.logger.info(f"Closing fmu {self._model_description.modelName} ")
            self._single_close(fmu_instance=self._fmu_instance,
                               unzip_dir=self._unzip_dir)
            self._unzip_dir = None
            self._fmu_instance = None

    def _close_multiprocessing(self, _):
        """Small helper function"""
        idx_worker = self.worker_idx
        if self._fmu_instance is None:
            return  # Already closed
        self.logger.info(f"Closing fmu {self._model_description.modelName} for worker {idx_worker}")
        self._single_close(fmu_instance=self._fmu_instance,
                           unzip_dir=self._unzip_dir)
        self._unzip_dir = None
        self._fmu_instance = None
        FMU_API._unzip_dir = None
        FMU_API._fmu_instance = None

    def save_for_reproduction(self,
                              title: str,
                              path: pathlib.Path = None,
                              files: list = None,
                              **kwargs):
        """
        Additionally to the basic reproduction, add info
        for FMU files.
        """
        if files is None:
            files = []
        files.append(CopyFile(
            filename="FMU/" + pathlib.Path(self.model_name).name,
            sourcepath=pathlib.Path(self.model_name),
            remove=False
        ))
        return super().save_for_reproduction(
            title=title,
            path=path,
            files=files,
            **kwargs
        )

    def _check_config(self, cfg, **kwargs):
        """
        Checks if instead of a config dict, the user is using the
        outdated arguments 'cd' and 'model_name' for initialization of the fmu api.
        To provide backwards-compatibility the required config
        is constructed out of these arguments (at least if arguments are provided with key).
        """
        if not cfg:
            cd_depr = kwargs.pop('cd', None)
            model_name_depr = kwargs.pop('model_name', None)
            if model_name_depr is not None:
                warnings.warn(f"Arguments 'model_name' and 'cd' will be depreciated "
                              f"in future versions. "
                              f"Please use a configuration instead "
                              f"and consider the available fields: "
                              f"{self.get_experiment_config_fields()}", FutureWarning)
                if cd_depr is not None:
                    return {'file_path': model_name_depr,
                            'cd': cd_depr
                            }
                return {'file_path': model_name_depr
                            }
            raise TypeError(f"No configuration given for instantiation. "
                            f"Please use the 'config' argument and "
                            f"consider the available fields: "
                            f"{self.get_experiment_config_fields()}")
        return cfg


class FMUDiscrete(FMU, DiscreteSimulation):
    """
    Class for discrete/stepwise simulation using the fmpy library and
    a functional mockup interface as a model input.

    :param dict config:
       Dictionary with experiment configuration
    :param bool log_fmu:
      Whether to print fmu messages or not.

    Example:

    >>> import matplotlib.pyplot as plt
    >>> from ebcpy import FMUDiscrete
    >>> # Select any valid fmu. Replace the line below if
    >>> # you don't have this file on your device.
    >>> path = "Path to your fmu"
    >>> fmu_api = FMUDiscrete({'file_path': path})
    >>> fmu_api.set_sim_setup({"stop_time": 3600})
    >>> # initialize FMU for discrete simulation
    >>> fmu_api.initialize_discrete_sim()
    >>> # simulation loop (typically there is interaction to other FMUs or python code)
    >>> # for straight simulation use api for continuous fmu simulation
    >>> while not fmu_api.finished:
    >>>     fmu_api.do_step()
    >>> result_df = fmu_api.get_results()
    >>> fmu_api.close()
    >>> # Select an exemplary column
    >>> col = result_df.columns[0]
    >>> plt.plot(result_df[col], label=col)
    >>> _ = plt.legend()
    >>> _ = plt.show()
"""

    _sim_setup_class: SimulationSetupClass = SimulationSetupFMU_Discrete
    _exp_config_class: ExperimentConfigurationClass = ExperimentConfigFMU_Discrete
    objs = []  # to use the close_all method

    def __init__(self, config: dict, log_fmu: bool = True):
        FMUDiscrete.objs.append(self)
        self.config = self._exp_config_class.parse_obj(config)

        FMU.__init__(self, file_path=self.config.file_path, cd=self.config.cd, log_fmu=log_fmu)
        self.use_mp = False  # no mp for stepwise FMU simulation
        # in case of fmu: file path, in case of dym: model_name are passed
        DiscreteSimulation.__init__(self, model_name=self.config.file_path)

        # define input data (can be adjusted during simulation using the setter)
        # calling the setter to distinguish depending on type and filtering
        self._input_data_on_grid = False  # if false the input data does not
        # cover the required grid. Need to hold last value or interpolate
        self.input_table = self.config.input_data
        # if false, last value of input table is hold, otherwise interpolated
        self.interp_input_table = False

    def get_results(self, tsd_format: bool = False):
        """
            returns the simulation results either as pd.DataFrame or as TimeSeriesData
        :param bool tsd_format:
            Whether to return as TimeSeriesData or pd.Dataframe
        :return:
            Results as TimeSeriesData or pd.Dataframe
        """

        if not tsd_format:
            results = self.sim_res_df
        else:
            results = TimeSeriesData(self.sim_res_df, default_tag="sim")
            results.rename_axis(['Variables', 'Tags'], axis='columns')
            results.index.names = ['Time']
        return results

    @classmethod
    def close_all(cls):
        """close multiple FMUs at once. Useful for co-simulation."""
        for obj in cls.objs:
            obj.close()

    @property
    def input_table(self):
        """input data that holds for longer parts of the simulation"""
        return self._input_table

    @input_table.setter
    def input_table(self, inp: Union[FilePath, pd.DataFrame, TimeSeriesData]):
        """setter allows the input data to change during discrete simulation"""
        # update config to trigger pydantic checks
        self._update_config({'input_data': inp})
        if inp is not None:
            input_table_raw = None
            if isinstance(inp, (str, pathlib.Path)):
                if not str(inp).endswith('csv'):
                    raise TypeError(
                        f"input data {inp} is not a .csv file. "
                        f"Instead of passing a file "
                        f"consider passing a pd.Dataframe or TimeSeriesData object")
                input_table_raw = pd.read_csv(inp, index_col='time')
            else:  # pd frame or tsd object; wrong type already caught by pydantic
                if isinstance(inp, TimeSeriesData):
                    input_table_raw = inp.to_df(force_single_index=True)
                elif isinstance(inp, pd.DataFrame):
                    input_table_raw = inp

            # check unsupported vars:
            self.check_unsupported_variables(input_table_raw.columns.to_list(), "inputs")
            # only consider columns in input table that refer to inputs of the FMU
            input_matches = list(set(self.inputs.keys()).intersection(set(input_table_raw.columns)))
            self._input_table = input_table_raw[input_matches]

            # check if the input data satisfies the whole time grid.
            self._check_input_data_grid()
        else:
            print('No long-term input data set! '
                  'Setter method can still be used to set input data to "input_table" attribute')
            self._input_table = None

    def read_variables(self, vrs_list: list):
        """
        Extends the read_variables() function of the FMU class by
        adding the current time to the results read from the fmu.

        Reads multiple variable values

        :param list vrs_list:
            List of variables to be read from FMU
        :return:
            Dict: Dictionary with requested variables and their values + the current time
        """

        res = super().read_variables(vrs_list)

        # add current time
        res['SimTime'] = self.current_time

        return res

    def initialize_discrete_sim(self,
                                parameters: dict = None,
                                init_values: dict = None
                                ):
        """
        Initialisation of FMU. To be called before stepwise simulation.
        Parameters and initial values can be set.

        :param dict parameters:
            Name (key) and value (value) of parameter to be set
        :param init_values:
            Name (key) and value (value) of initial value to be set

        """

        # THE FOLLOWING STEPS OF FMU INITIALISATION ALREADY COVERED BY INSTANTIATING FMU API:
        # - Read model description
        # - extract .fmu file
        # - Create FMU2 Slave
        # - instantiate fmu

        # check if input valid
        if parameters is not None:
            self.check_unsupported_variables(list(parameters.keys()), "parameters")
        if init_values is not None:
            self.check_unsupported_variables(list(init_values.keys()), "variables")

        # Reset FMU instance
        self._fmu_instance.reset()

        # Set up experiment
        self._fmu_instance.setupExperiment(startTime=self.sim_setup.start_time,
                                           stopTime=self.sim_setup.stop_time,
                                           tolerance=self.sim_setup.tolerance)

        # initialize current time and communication step size for stepwise FMU simulation
        self.current_time = self.sim_setup.start_time

        # Set parameters and initial values
        if init_values is None:
            init_values = {}
        if parameters is None:
            parameters = {}
        # merge initial values and parameters in one dict as they are treated similarly
        start_values = init_values.copy()
        start_values.update(parameters)

        # write parameters and initial values to FMU
        self.set_variables(var_dict=start_values)

        # Finalise initialisation
        self._fmu_instance.enterInitializationMode()
        self._fmu_instance.exitInitializationMode()

        # Initialize dataframe to store results
        res = self.read_variables(vrs_list=self.result_names)

        self.sim_res_df = pd.DataFrame(res,
                                       index=[res['SimTime']],
                                       columns=self.result_names
                                       )

        self.logger.info(f"FMU '{self._model_description.modelName}' "
                         f"initialized for discrete simulation")

        # initialize status indicator
        self.finished = False

        # reset step count
        self.step_count = 0

    def step_only(self):
        """
        Perform simulation step, return True if stop time reached.
        """
        # check if stop time is reached
        if self.current_time < self.sim_setup.stop_time:
            if self.step_count == 0:
                self.logger.info(f"Starting simulation of FMU "
                                 f"'{self._model_description.modelName}'")
            # do simulation step
            self._fmu_instance.doStep(
                currentCommunicationPoint=self.current_time,
                communicationStepSize=self.sim_setup.comm_step_size)
            # step count
            self.step_count += 1
            # update current time and determine status
            self.current_time += self.sim_setup.comm_step_size
            self.finished = False
        else:
            self.finished = True
            self.logger.info(f"Simulation of FMU '{self._model_description.modelName}' finished")
        return self.finished

    def do_step(self, input_step: dict = None, close_when_finished: bool = False):
        """
        Wrapper function for step_only.
        Write values from input_step and attribute input_table to fmu,
        performs simulation step,
        read from fmu right after simulation step (variables in result_names attribute considered).

        :param dict input_step:
            Input variable name (key) and value (value) to be set to fmu before step
        :param bool close_when_finished:
            Whether to close fmu when stop time is reached
        :return:
            Dict: Results after the simulation step
        """
        # check for unsupported input
        if input_step is not None:
            self.check_unsupported_variables(list(input_step.keys()), 'inputs')
        # collect inputs
        # get input from input table (overwrite with specific input for single step)
        single_input = {}
        if self.input_table is not None:
            # extract value from input time table
            if self._input_data_on_grid:
                # In the case that all indices within the required grid (req_grid) are present
                # values can be directly accessed.
                # There is no need to find the last available index or interpolation.
                single_input = self.input_table.loc[self.current_time].to_dict()
            else:
                single_input = interp_df(t_act=self.current_time,
                                         df=self.input_table,
                                         interpolate=self.interp_input_table)

        if input_step is not None:
            # overwrite with input for step
            single_input.update(input_step)

        # write inputs to fmu
        if single_input:
            self.set_variables(var_dict=single_input)

        # perform simulation step
        self.step_only()

        # read results
        res = self.read_variables(vrs_list=self.result_names)
        if not self.finished:
            # append
            if self.current_time % self.sim_setup.output_interval == 0:
                self.sim_res_df = pd.concat(   # because frame.append will be depreciated
                    [self.sim_res_df,
                     pd.DataFrame.from_records([res],
                                               index=[res['SimTime']],
                                               columns=self.sim_res_df.columns)])

        else:
            if close_when_finished:
                self.close()
        return res

    def _set_result_names(self):
        """
        Adds input names to list result_names in addition to outputs.
        In discrete simulation the inputs are typically relevant.
        """
        self.result_names = list(self.outputs.keys()) + list(self.inputs.keys())

    def close(self):
        """ Closes the fmu."""
        # No MP for discrete simulation
        if not self._fmu_instance:
            return  # Already closed
        self.logger.info(f"Closing fmu {self._model_description.modelName} ")
        self._single_close(fmu_instance=self._fmu_instance,
                           unzip_dir=self._unzip_dir)
        self._unzip_dir = None
        self._fmu_instance = None

    def _check_input_data_grid(self):
        """
        Checks whether the input data in the input_table attribute
        covers the time grid specified by the sim_setup attribute.
        """
        if hasattr(self, "_input_table") :
            if self.input_table is not None:
                # time grid defined by sim_setup
                sim_setup_idx = list(np.arange(self.sim_setup.start_time,
                                          self.sim_setup.stop_time + self.sim_setup.comm_step_size,
                                          self.sim_setup.comm_step_size))
                if set(sim_setup_idx).issubset(set(self.input_table.index.tolist())):
                    self._input_data_on_grid = True
                else:
                    self._input_data_on_grid = False

    def set_sim_setup(self, sim_setup):
        """
        Extends the set_sim_setup method of the Model class by triggering the check,
        whether the input data satisfies the time grid.

        Updates only those entries that are given as arguments.
        """

        super().set_sim_setup(sim_setup)
        self._check_input_data_grid()

    # todo: make class method out of it to consider multiple discrete fmu apis;
    # todo: consider attribute interp_input data and input_data_on_grid
    def save_for_reproduction(self,
                              title: str,
                              path: pathlib.Path = None,
                              files: list = None,
                              **kwargs):
        """
        Additionally to the basic reproduction, add info
        for FMU files.
        """
        if files is None:
            files = []
        files.append(CopyFile(
            filename="FMU/" + pathlib.Path(self.model_name).name,
            sourcepath=pathlib.Path(self.model_name),
            remove=False
        ))
        return super().save_for_reproduction(
            title=title,
            path=path,
            files=files,
            **kwargs
        )
