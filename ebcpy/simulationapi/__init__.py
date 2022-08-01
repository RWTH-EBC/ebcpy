"""
Simulation APIs help you to perform automated
simulations for energy and building climate related models.
Parameters can easily be updated, and the initialization-process is
much more user-friendly than the provided APIs by Dymola or fmpy.
"""
import logging
import os
import itertools
from typing import Dict, Union, TypeVar, Any, List
from abc import abstractmethod
import multiprocessing as mp
from pydantic import BaseModel, Field, validator
import numpy as np
from ebcpy.utils import setup_logger
import fmpy
from fmpy.model_description import read_model_description


class Variable(BaseModel):
    """
    Data-Class to store relevant information for a
    simulation variable (input, parameter, output or local/state).
    """
    value: Any = Field(
        description="Default variable value"
    )
    max: Union[float, int] = Field(
        default=np.inf,
        title='max',
        description='Maximal value (upper bound) of the variables value'
    )
    min: Union[float, int] = Field(
        default=-np.inf,
        title='min',
        description='Minimal value (lower bound) of the variables value'
    )
    type: Any = Field(
        default=None,
        title='type',
        description='Type of the variable'
    )


class SimulationSetup(BaseModel):
    """
    pydantic BaseModel child to define relevant
    parameters to setup the simulation.
    """
    start_time: float = Field(
        default=0,
        description="The start time of the simulation",
        title="start_time"
    )
    stop_time: float = Field(
        default=1,
        description="The stop / end time of the simulation",
        title="stop_time"
    )
    output_interval: float = Field(
        default=1,
        description="The step size of the simulation and "
                    "thus also output interval of results.",
        title="output_interval"
    )
    fixedstepsize: float = Field(
        title="fixedstepsize",
        default=0.0,
        description="Fixed step size for Euler"
    )
    solver: str = Field(
        title="solver",
        default="",  # Is added in the validator
        description="The solver to be used for numerical integration."
    )
    _default_solver: str = None
    _allowed_solvers: list = []

    @validator("solver", always=True, allow_reuse=True)
    def check_valid_solver(cls, solver):
        """
        Check if the solver is in the list of valid solvers
        """
        if not solver:
            return cls.__private_attributes__['_default_solver'].default
        allowed_solvers = cls.__private_attributes__['_allowed_solvers'].default
        if solver not in allowed_solvers:
            raise ValueError(f"Given solver '{solver}' is not supported! "
                             f"Supported are '{allowed_solvers}'")
        return solver

    class Config:
        """Overwrite default pydantic Config"""
        extra = 'forbid'
        underscore_attrs_are_private = True


SimulationSetupClass = TypeVar("SimulationSetupClass", bound=SimulationSetup)


class SimulationAPI:
    """Base-class for simulation apis. Every simulation-api class
    must inherit from this class. It defines the structure of each class.

    :param str,os.path.normpath cd:
        Working directory path
    :param str model_name:
        Name of the model being simulated.
    :keyword int n_cpu:
        Number of cores to be used by simulation.
        If None is given, single core will be used.
        Maximum number equals the cpu count of the device.
        **Warning**: Logging is not yet fully working on multiple processes.
        Output will be written to the stream handler, but not to the created .log files.

    """
    _sim_setup_class: SimulationSetupClass = SimulationSetup
    _items_to_drop = [
        'pool',
    ]

    def __init__(self, cd, model_name, **kwargs):
        # Private helper attrs for multiprocessing
        self._n_sim_counter = 0
        self._n_sim_total = 0
        self._progress_int = 0
        # Setup the logger
        self.logger = setup_logger(cd=cd, name=self.__class__.__name__)
        self.logger.info(f'{"-" * 25}Initializing class {self.__class__.__name__}{"-" * 25}')
        # Check multiprocessing
        self.n_cpu = kwargs.get("n_cpu", 1)
        if self.n_cpu > mp.cpu_count():
            raise ValueError(f"Given n_cpu '{self.n_cpu}' is greater "
                             "than the available number of "
                             f"cpus on your machine '{mp.cpu_count()}'")
        if self.n_cpu > 1:
            # pylint: disable=consider-using-with
            self.pool = mp.Pool(processes=self.n_cpu)
            self.use_mp = True
        else:
            self.pool = None
            self.use_mp = False
        # Setup the model
        self._sim_setup = self._sim_setup_class()
        self.cd = cd
        self.inputs: Dict[str, Variable] = {}       # Inputs of model
        self.outputs: Dict[str, Variable] = {}      # Outputs of model
        self.parameters: Dict[str, Variable] = {}   # Parameter of model
        self.states: Dict[str, Variable] = {}       # States of model
        self.result_names = []
        self.model_name = model_name

    # MP-Functions
    @property
    def worker_idx(self):
        """Index of the current worker"""
        _id = mp.current_process()._identity
        if _id:
            return _id[0]
        return None

    def __getstate__(self):
        """Overwrite magic method to allow pickling the api object"""
        self_dict = self.__dict__.copy()
        for item in self._items_to_drop:
            del self_dict[item]
        #return deepcopy(self_dict)
        return self_dict

    def __setstate__(self, state):
        """Overwrite magic method to allow pickling the api object"""
        self.__dict__.update(state)

    def close(self):
        """Base function for closing the simulation-program."""
        if self.use_mp:
            try:
                self.pool.map(self._close_multiprocessing,
                              list(range(self.n_cpu)))
                self.pool.close()
                self.pool.join()
            except ValueError:
                pass  # Already closed prior to atexit

    @abstractmethod
    def _close_multiprocessing(self, _):
        raise NotImplementedError(f'{self.__class__.__name__}.close '
                                  f'function is not defined')

    @abstractmethod
    def _single_close(self, **kwargs):
        """Base function for closing the simulation-program of a single core"""
        raise NotImplementedError(f'{self.__class__.__name__}._single_close '
                                  f'function is not defined')

    @abstractmethod
    def simulate(self,
                 parameters: Union[dict, List[dict]] = None,
                 return_option: str = "time_series",
                 **kwargs):
        """
        Base function for simulating the simulation-model.

        :param dict parameters:
            Parameters to simulate.
            Names of parameters are key, values are value of the dict.
            Default is an empty dict.
        :param str return_option:
            How to handle the simulation results. Options are:
            - 'time_series': Returns a DataFrame with the results and does not store anything.
            Only variables specified in result_names will be returned.
            - 'last_point': Returns only the last point of the simulation.
            Relevant for integral metrics like energy consumption.
            Only variables specified in result_names will be returned.
            - 'savepath': Returns the savepath where the results are stored.
            Depending on the API, different kwargs may be used to specify file type etc.
        :keyword str,os.path.normpath savepath:
            If path is provided, the relevant simulation results will be saved
            in the given directory.
            Only relevant if return_option equals 'savepath' .
        :keyword str result_file_name:
            Name of the result file. Default is 'resultFile'.
            Only relevant if return_option equals 'savepath'.
        :keyword (TimeSeriesData, pd.DataFrame) inputs:
            Pandas.Dataframe of the input data for simulating the FMU with fmpy
        :keyword Boolean fail_on_error:
            If True, an error in fmpy will trigger an error in this script.
            Default is True

        :return: str,os.path.normpath filepath:
            Only if return_option equals 'savepath'.
            Filepath of the result file.
        :return: dict:
            Only if return_option equals 'last_point'.
        :return: Union[List[pd.DataFrame],pd.DataFrame]:
            If parameters are scalar and squeeze=True,
            a DataFrame with the columns being equal to
            self.result_names.
            If multiple set's of initial values are given, one
            dataframe for each set is returned in a list
        """
        # Convert inputs to equally sized objects of lists:
        if parameters is None:
            parameters = [{}]
        if isinstance(parameters, dict):
            parameters = [parameters]
        new_kwargs = {}
        kwargs["return_option"] = return_option  # Update with arg
        # Handle special case for saving files:
        if return_option == "savepath" and len(parameters) > 1:
            savepath = kwargs.get("savepath", [])
            result_file_name = kwargs.get("result_file_name", [])
            if (len(set(savepath)) != len(parameters) and
                    len(set(result_file_name)) != len(parameters)):
                raise TypeError(
                    "Simulating multiple parameter set's on "
                    "the same savepath will overwrite old "
                    "results or even cause errors. "
                    "Specify a result_file_name or savepath for each "
                    "parameter combination"
                )
        for key, value in kwargs.items():
            if isinstance(value, list):
                if len(value) != len(parameters):
                    raise ValueError(f"Mismatch in multiprocessing of "
                                     f"given parameters ({len(parameters)}) "
                                     f"and given {key} ({len(value)})")
                new_kwargs[key] = value
            else:
                new_kwargs[key] = [value] * len(parameters)
        kwargs = []
        for _idx, _parameters in enumerate(parameters):
            kwargs.append(
                {"parameters": _parameters,
                 **{key: value[_idx] for key, value in new_kwargs.items()}
                 }
            )
        # Decide between mp and single core
        if self.use_mp:
            self._n_sim_counter = 0
            self._n_sim_total = len(kwargs)
            self._progress_int = 0
            self.logger.info("Starting %s simulations on %s cores",
                             self._n_sim_total, self.n_cpu)
            _async_jobs = []
            for _kwarg in kwargs:
                _async_jobs.append(
                    self.pool.apply_async(
                        func=self._single_simulation,
                        args=(_kwarg,),
                        callback=self._log_simulation_process)
                )
            results = []
            for _async_job in _async_jobs:
                _async_job.wait()
                results.append(_async_job.get())
        else:
            results = [self._single_simulation(kwargs={
                "parameters": _single_kwargs["parameters"],
                "return_option": _single_kwargs["return_option"],
                **_single_kwargs
            }) for _single_kwargs in kwargs]
        if len(results) == 1:
            return results[0]
        return results

    def _log_simulation_process(self, _):
        """Log the simulation progress"""
        self._n_sim_counter += 1
        progress = int(self._n_sim_counter / self._n_sim_total * 100)
        if progress == self._progress_int + 10:
            if self.logger.isEnabledFor(level=logging.INFO):
                self.logger.info(f"Finished {progress} % of all {self._n_sim_total} simulations")
            self._progress_int = progress

    @abstractmethod
    def _single_simulation(self, kwargs):
        """
        Same arguments and function as simulate().
        Used to differ between single- and multi-processing simulation"""
        raise NotImplementedError(f'{self.__class__.__name__}._single_simulation '
                                  f'function is not defined')

    @property
    def sim_setup(self) -> SimulationSetupClass:
        """Return current sim_setup"""
        return self._sim_setup

    @sim_setup.deleter
    def sim_setup(self):
        """In case user deletes the object, reset it to the default one."""
        self._sim_setup = self._sim_setup_class()

    def set_sim_setup(self, sim_setup):
        """
        Replaced in v0.1.7 by property function
        """
        new_setup = self._sim_setup.dict()
        new_setup.update(sim_setup)
        self._sim_setup = self._sim_setup_class(**new_setup)

    @property
    def model_name(self) -> str:
        """Name of the model being simulated"""
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        """
        Set new model_name and trigger further functions
        to load parameters etc.
        """
        self._model_name = model_name
        # Empty all variables again.
        if self.worker_idx:
            return
        self.outputs = {}
        self.parameters = {}
        self.states = {}
        self.inputs = {}
        self._update_model()
        # Set all outputs to result_names:
        self.result_names = list(self.outputs.keys())

    @abstractmethod
    def _update_model(self):
        """
        Reimplement this to change variables etc.
        based on the new model.
        """
        raise NotImplementedError(f'{self.__class__.__name__}._update_model '
                                  f'function is not defined')

    def set_cd(self, cd):
        """Base function for changing the current working directory."""
        self.cd = cd

    @property
    def cd(self) -> str:
        """Get the current working directory"""
        return self._cd

    @cd.setter
    def cd(self, cd: str):
        """Set the current working directory"""
        os.makedirs(cd, exist_ok=True)
        self._cd = cd

    @property
    def result_names(self) -> List[str]:
        """
        The variables names which to store in results.

        Returns:
            list: List of string where the string is the
            name of the variable to store in the result.
        """
        return self._result_names

    @result_names.setter
    def result_names(self, result_names):
        """
        Set the result names. If the name is not supported,
        an error is logged.
        """
        self.check_unsupported_variables(variables=result_names,
                                         type_of_var="variables")
        self._result_names = result_names

    @property
    def variables(self):
        """
        All variables of the simulation model
        """
        return list(itertools.chain(self.parameters.keys(),
                                    self.outputs.keys(),
                                    self.inputs.keys(),
                                    self.states.keys()))

    def check_unsupported_variables(self, variables: List[str], type_of_var: str):
        """Log warnings if variables are not supported."""
        if type_of_var == "parameters":
            ref = self.parameters.keys()
        elif type_of_var == "outputs":
            ref = self.outputs.keys()
        elif type_of_var == "inputs":
            ref = self.inputs.keys()
        elif type_of_var == "inputs":
            ref = self.states.keys()
        else:
            ref = self.variables

        diff = set(variables).difference(ref)
        if diff:
            self.logger.warning(
                "Variables '%s' not found in model '%s'. "
                "Will most probably trigger an error when simulating.",
                ', '.join(diff), self.model_name
            )
            return True
        return False

    @classmethod
    def get_simulation_setup_fields(cls):
        """Return all fields in the chosen SimulationSetup class."""
        return list(cls._sim_setup_class.__fields__.keys())

class FMU:
    """
    Base class for simulations with FMUs.
    """
    def __init__(self, **kwargs):
        self._fmu_instances: dict = {}  # fixme: kbe: as class attribute its not possible to instantiate two fmu's in parralel for co simulation
        self._unzip_dirs: dict = {}
        self._single_unzip_dir: str = None
        self._model_description = None
        self._fmi_type = None
        self.log_fmu = kwargs.get("log_fmu", True)
        self.var_refs = None

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


    def _set_variables(self, var_dict: dict, idx_worker: int = 0):  # todo: idx_worker not nice
        """
        Sets multiple variables.
        var_dict is a dict with variable names in keys.
        """

        for key, value in var_dict.items():
            var = self.var_refs[key]
            vr = [var.valueReference]

            if var.type == 'Real':
                self._fmu_instances[idx_worker].setReal(vr, [float(value)])
            elif var.type in ['Integer', 'Enumeration']:
                self._fmu_instances[idx_worker].setInteger(vr, [int(value)])
            elif var.type == 'Boolean':
                self._fmu_instances[idx_worker].setBoolean(vr, [value == 1.0 or value or value == "True"])
            else:
                raise Exception("Unsupported type: %s" % var.type)

    def _read_variables(self, vrs_list: list, idx_worker: int = 0):  # todo: idx_worker not nice
        """
        Reads multiple variable values of FMU.
        vrs_list as list of strings
        Method returns a dict with FMU variable names as key
        """

        # initialize dict for results of simulation step
        res = {}

        for name in vrs_list:
            var = self.var_refs[name]
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

    def _find_vars(self, start_str: str):
        """
        Returns all variables starting with start_str
        """

        key = list(self.var_refs.keys())
        key_list = []
        for i in range(len(key)):
            if key[i].startswith(start_str):
                key_list.append(key[i])
        return key_list

