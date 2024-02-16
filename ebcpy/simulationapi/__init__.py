"""
Simulation APIs help you to perform automated
simulations for energy and building climate related models.
Parameters can easily be updated, and the initialization-process is
much more user-friendly than the provided APIs by Dymola or fmpy.
"""
import pathlib
import warnings
import os
import sys
import itertools
import time
from pathlib import Path
from datetime import timedelta
from typing import Dict, Union, TypeVar, Any, List
from abc import abstractmethod
import multiprocessing as mp

import pydantic
from pydantic import BaseModel, Field, field_validator
import numpy as np
from ebcpy.utils import setup_logger
from ebcpy.utils.reproduction import save_reproduction_archive
from shutil import disk_usage


class Variable(BaseModel):
    """
    Data-Class to store relevant information for a
    simulation variable (input, parameter, output or local/state).
    """
    type: Any = Field(
        default=None,
        title='type',
        description='Type of the variable'
    )
    value: Any = Field(
        description="Default variable value"
    )
    max: Any = Field(
        default=None,
        title='max',
        description='Maximal value (upper bound) of the variables value. '
                    'Only for ints and floats variables.'
    )
    min: Any = Field(
        default=None,
        title='min',
        description='Minimal value (lower bound) of the variables value. '
                    'Only for ints and floats variables.'
    )

    @field_validator("value")
    @classmethod
    def check_value_type(cls, value, info: pydantic.FieldValidationInfo):
        """Check if the given value has correct type"""
        _type = info.data["type"]
        if _type is None:
            return value   # No type -> no conversion
        if value is None:
            return value  # Setting None is allowed.
        if not isinstance(value, _type):
            return _type(value)
        return value

    @field_validator('max', 'min')
    @classmethod
    def check_value(cls, value, info: pydantic.FieldValidationInfo):
        """Check if the given bounds are correct."""
        # Check if the variable type even allows for min/max bounds
        _type = info.data["type"]
        if _type is None:
            return value   # No type -> no conversion
        if _type not in (float, int, bool):
            if value is not None:
                warnings.warn(
                    "Setting a min/max for variables "
                    f"of type {_type} is not supported."
                )
            return None
        if value is not None:
            return _type(value)
        if info.field_name == "min":
            return -np.inf if _type != bool else False
        # else it is max
        return np.inf if _type != bool else True


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
        default="",  # Is added in the field_validator
        description="The solver to be used for numerical integration."
    )
    _default_solver: str = None
    _allowed_solvers: list = []

    @field_validator("solver")
    @classmethod
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


SimulationSetupClass = TypeVar("SimulationSetupClass", bound=SimulationSetup)


class SimulationAPI:
    """Base-class for simulation apis. Every simulation-api class
    must inherit from this class. It defines the structure of each class.

    :param str,Path working_directory:
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

    def __init__(self, working_directory: Union[Path, str], model_name: str, **kwargs):
        # Private helper attrs for multiprocessing
        self._n_sim_counter = 0
        self._n_sim_total = 0
        self._progress_int = 0
        # Handle deprecation warning
        self.working_directory = working_directory
        self.logger = setup_logger(
            working_directory=self.working_directory,
            name=self.__class__.__name__
        )
        # Setup the logger
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
            It is also possible to specify a list of multiple parameter
            dicts for different parameter variations to be simulated.
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
        :keyword str,Path savepath:
            If path is provided, the relevant simulation results will be saved
            in the given directory. For multiple parameter variations also a list
            of savepaths for each parameterset can be specified.
            The savepaths for each parameter set must be unique.
            Only relevant if return_option equals 'savepath' .
        :keyword str result_file_name:
            Name of the result file. Default is 'resultFile'.
            For multiple parameter variations a list of names
            for each result must be specified. 
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

        if return_option not in ["time_series", "savepath", "last_point"]:
            raise ValueError(f"Given return option '{return_option}' is not supported.")

        new_kwargs = {}
        kwargs["return_option"] = return_option  # Update with arg
        n_simulations = len(parameters)
        # Handle special case for saving files:
        if return_option == "savepath" and n_simulations > 1:
            savepath = kwargs.get("savepath", [])
            if isinstance(savepath, (str, os.PathLike, Path)):
                savepath = [savepath] * n_simulations
            result_file_name = kwargs.get("result_file_name", [])
            if isinstance(result_file_name, str):
                result_file_name = [result_file_name] * n_simulations
            if len(savepath) != len(result_file_name):
                raise ValueError("Given savepath and result_file_name "
                                 "have not the same length.")
            joined_save_paths = []
            for _single_save_path, _single_result_name in zip(savepath, result_file_name):
                joined_save_paths.append(os.path.join(_single_save_path, _single_result_name))
            if len(set(joined_save_paths)) != n_simulations:
                raise ValueError(
                    "Simulating multiple parameter set's on "
                    "the same combination of savepath and result_file_name "
                    "will override results or even cause errors. "
                    "Specify a unique result_file_name-savepath combination "
                    "for each parameter combination"
                )
        for key, value in kwargs.items():
            if isinstance(value, list):
                if len(value) != n_simulations:
                    raise ValueError(f"Mismatch in multiprocessing of "
                                     f"given parameters ({n_simulations}) "
                                     f"and given {key} ({len(value)})")
                new_kwargs[key] = value
            else:
                new_kwargs[key] = [value] * n_simulations
        kwargs = []
        for _idx, _parameters in enumerate(parameters):
            kwargs.append(
                {"parameters": _parameters,
                 **{key: value[_idx] for key, value in new_kwargs.items()}
                 }
            )
        # Decide between mp and single core
        t_sim_start = time.time()
        if self.use_mp:
            self._n_sim_counter = 0
            self._n_sim_total = len(kwargs)
            self._progress_int = 0
            self.logger.info("Starting %s simulations on %s cores",
                             self._n_sim_total, self.n_cpu)
            results = []
            for result in self.pool.imap(self._single_simulation, kwargs):
                results.append(result)
                self._n_sim_counter += 1
                # Assuming that all worker start and finish their first simulation
                # at the same time, so that the time estimation begins after
                # n_cpu simulations. Otherwise, the translation and start process
                # could falsify the time estimation.
                if self._n_sim_counter == self.n_cpu:
                    t1 = time.time()
                if self._n_sim_counter > self.n_cpu:
                    self._remaining_time(t1)
                if self._n_sim_counter == 1 and return_option == 'savepath':
                    self._check_disk_space(result)
            sys.stderr.write("\r")
        else:
            results = [self._single_simulation(kwargs={
                "parameters": _single_kwargs["parameters"],
                "return_option": _single_kwargs["return_option"],
                **_single_kwargs
            }) for _single_kwargs in kwargs]
        self.logger.info(f"Finished {n_simulations} simulations on {self.n_cpu} processes in "
                         f"{timedelta(seconds=int(time.time() - t_sim_start))}")
        if len(results) == 1:
            return results[0]
        return results

    def _remaining_time(self, t1):
        """
        Helper function to calculate the remaining simulation time and log the finished simulations.
        The function can first be used when a simulation has finished on each used cpu, so that the
        translation of the model is not considered in the time estimation.

        :param float t1:
            Start time after n_cpu simulations.
        """
        t_remaining = (time.time() - t1) / (self._n_sim_counter - self.n_cpu) * (
                    self._n_sim_total - self._n_sim_counter)
        p_finished = self._n_sim_counter / self._n_sim_total * 100
        sys.stderr.write(f"\rFinished {np.round(p_finished, 1)} %. "
                         f"Approximately remaining time: {timedelta(seconds=int(t_remaining))} ")

    def _check_disk_space(self, filepath):
        """
        Checks how much disk space all simulations will need on a hard drive
        and throws a warning when less than 5 % would be free on the hard drive
        after all simulations.
        Works only for multiprocessing.
        """

        def convert_bytes(size):
            suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
            suffix_idx = 0
            while size >= 1024 and suffix_idx < len(suffixes):
                suffix_idx += 1
                size = size / 1024.0
            return f'{str(np.round(size, 2))} {suffixes[suffix_idx]}'

        sim_file_size = os.stat(filepath).st_size
        sim_files_size = sim_file_size * self._n_sim_total
        self.logger.info(f"Simulations files need approximately {convert_bytes(sim_files_size)} of disk space")
        total, used, free = disk_usage(filepath)
        if sim_files_size > free - 0.05 * total:
            warnings.warn(f"{convert_bytes(free)} of free disk space on {filepath[:2]} "
                          f"is not enough for all simulation files.")

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
        # Only update if the model_name actually changes
        if hasattr(self, "_model_name") and self._model_name == model_name:
            return
        self._model_name = model_name
        # Only update model if it's the first setup. On multiprocessing,
        # all objects are duplicated and thus this setter is triggered again.
        # This if statement catches this case.
        if self.worker_idx and self.use_mp:
            return
        # Empty all variables again.
        self._update_model_variables()

    def _update_model_variables(self):
        """
        Function to empty all variables and update them again
        """
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

    def set_working_directory(self, working_directory: Union[Path, str]):
        """Base function for changing the current working directory."""
        self.working_directory = working_directory

    @property
    def working_directory(self) -> Path:
        """Get the current working directory"""
        return self._working_directory

    @working_directory.setter
    def working_directory(self, working_directory: Union[Path, str]):
        """Set the current working directory"""
        if isinstance(working_directory, str):
            working_directory = Path(working_directory)
        os.makedirs(working_directory, exist_ok=True)
        self._working_directory = working_directory

    def set_cd(self, cd: Union[Path, str]):
        warnings.warn("cd was renamed to working_directory in all classes. "
                      "Use working_directory instead instead.", category=DeprecationWarning)
        self.working_directory = cd

    @property
    def cd(self) -> Path:
        warnings.warn("cd was renamed to working_directory in all classes. "
                      "Use working_directory instead instead.", category=DeprecationWarning)
        return self.working_directory

    @cd.setter
    def cd(self, cd: Union[Path, str]):
        warnings.warn("cd was renamed to working_directory in all classes. "
                      "Use working_directory instead instead.", category=DeprecationWarning)
        self.working_directory = cd

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

    def save_for_reproduction(self,
                              title: str,
                              path: pathlib.Path = None,
                              files: list = None,
                              **kwargs):
        """
        Save the settings of the SimulationAPI in order to
        reproduce the settings of the used simulation.

        Should be extended by child-classes to allow custom
        saving.

        :param str title:
            Title of the study
        :param pathlib.Path path:
            Where to store the .zip file. If not given, self.cd is used.
        :param list files:
            List of files to save along the standard ones.
            Examples would be plots, tables etc.
        :param dict kwargs:
            All keyword arguments except title, files, and path of the function
            `save_reproduction_archive`. Most importantly, `log_message` may be
            specified to avoid input during execution.
        """
        if path is None:
            path = self.cd
        return save_reproduction_archive(
            title=title,
            path=path,
            files=files,
            **kwargs
        )
