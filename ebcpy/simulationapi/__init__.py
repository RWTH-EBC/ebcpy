"""
Simulation APIs help you to perform automated
simulations for energy and building climate related models.
Parameters can easily be updated, and the initialization-process is
much more user-friendly than the provided APIs by Dymola or fmpy.
"""

import logging
import os
import itertools
from typing import Union
from typing import Dict, Any, List
from abc import abstractmethod
import multiprocessing as mp
from pydantic import BaseModel, Field
import numpy as np
from ebcpy.utils import setup_logger
from ebcpy.simulationapi.config import *


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


class Model:
    """
    Base-class for simulation apis. Every simulation-api class
    must inherit from this class. It defines the basic model structure.

    :param model_name:
        Name of the model being simulated.
    """

    _sim_setup_class: SimulationSetupClass = SimulationSetup
    _exp_config_class: ExperimentConfigurationClass = ExperimentConfiguration

    def __init__(self, model_name):
        # initialize sim setup with class default
        self._sim_setup = self._sim_setup_class()
        # update sim setup if given in config; if not update config
        if self.config.sim_setup is not None:
            self.set_sim_setup(self.config.sim_setup)
        else:
            self._update_config({'sim_setup': self._sim_setup})
        # current directory
        if not hasattr(self, 'cd'):  # in case of FMU, cd is set already by now
            if self.config.cd is not None:
                self.cd = self.config.cd
            else:
                self.cd = pathlib.Path(__file__).parent.joinpath("results")
        # Setup the logger
        self.logger = setup_logger(cd=self.cd, name=self.__class__.__name__)
        self.logger.info(f'{"-" * 25}Initializing class {self.__class__.__name__}{"-" * 25}')
        # initialize model variables
        self.inputs: Dict[str, Variable] = {}  # Inputs of model
        self.outputs: Dict[str, Variable] = {}  # Outputs of model
        self.parameters: Dict[str, Variable] = {}  # Parameter of model
        self.states: Dict[str, Variable] = {}  # States of model
        # results
        self.result_names = []  # initialize list of tracked variables
        self.model_name = model_name

    def _update_config(self, config_update: dict):
        """
        Updates config attribute.
        To be called in methods that modify an element within the config.
        This assures that config is up-to-date and triggers pydantic check.
        Not to be called by user as updating the config after initialization is not intended
        (because updates are not forwarded)

        :param config_update:
            Dictionary containing updates to the experiment configuration
        """
        new_config = self.config.dict()
        new_config.update(config_update)
        self.config = self._exp_config_class(**new_config)

    def set_cd(self, cd):
        """Base function for changing the current working directory"""
        self.cd = cd

    @property
    def cd(self) -> str:
        """Get the current working directory"""
        return self._cd

    @cd.setter
    def cd(self, cd: str):
        """Set the current working directory and update the configuration accordingly"""
        # update config and thereby trigger pydantic validator
        self._update_config({'cd': cd})
        # create dir and set attribute
        os.makedirs(cd, exist_ok=True)
        self._cd = cd

    @classmethod
    def get_simulation_setup_fields(cls):
        """Return all fields in the chosen SimulationSetup class."""
        return list(cls._sim_setup_class.__fields__.keys())

    @classmethod
    def get_experiment_config_fields(cls):
        """Return all fields in the chosen ExperimentConfig class."""
        return list(cls._exp_config_class.__fields__.keys())

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
        Updates only those entries that are given as arguments
        """
        new_setup = self._sim_setup.dict()
        new_setup.update(sim_setup)
        self._sim_setup = self._sim_setup_class(**new_setup)

        # update config (redundant in case the sim_setup dict comes from config,
        # but relevant if set afterwards)
        self._update_config({'sim_setup': new_setup})

    @property
    def model_name(self) -> str:
        """Name of the model being simulated"""
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        """Set new model_name and trigger further functions to load parameters etc."""
        self._model_name = model_name
        # Empty all variables again.
        # TODO: Review: review this condition!
        #  It would be better to get rid off worker_idx at level of Model-class
        if self.use_mp:
            if self.worker_idx:
                return
        self._update_model_variables()

    def _update_model_variables(self):
        """ Function to empty all variables and update them again"""
        self.outputs = {}
        self.parameters = {}
        self.states = {}
        self.inputs = {}
        self._update_model()
        # Set all outputs to result_names:
        self._set_result_names()

    def _set_result_names(self):
        """
        By default, keys of the output variables are passed to result_names list.
        Method may be overwritten by child.
        """
        self.result_names = list(self.outputs.keys())

    @abstractmethod
    def _update_model(self):
        """ Reimplement this to change variables etc. based on the new model. """
        raise NotImplementedError(f'{self.__class__.__name__}._update_model '
                                  f'function is not defined')

    @property
    def result_names(self) -> List[str]:
        """
        The variable names which to store in results.

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
        """All variables of the simulation model"""
        return list(itertools.chain(self.parameters.keys(),
                                    self.outputs.keys(),
                                    self.inputs.keys(),
                                    self.states.keys()))

    def check_unsupported_variables(self, variables: List[str], type_of_var: str):
        """
        Checks if variables are in the model as a specified type.

        :param list variables:
            List of variables to check
        :param str type_of_var:
            Variable type to search for
        :return:
            bool: Returns True if unsupported variables occur
        """
        # Log warnings if variables are not supported
        if type_of_var == "parameters":
            ref = self.parameters.keys()
        elif type_of_var == "outputs":
            ref = self.outputs.keys()
        elif type_of_var == "inputs":
            ref = self.inputs.keys()
        elif type_of_var == "states":
            ref = self.states.keys()
        else:
            ref = self.variables

        diff = set(variables).difference(ref)
        if diff:
            if type_of_var not in ["parameters", "outputs", "inputs", "states"]:
                type_of_var = "variables"  # to specify warning
            self.logger.warning(
                "Variables '%s' are no '%s' in model '%s'. "
                "Will most probably trigger an error when simulating "
                "or being ignored.",  # in case of input table
                ', '.join(diff), type_of_var, self.model_name
            )
            return True
        return False

    @ abstractmethod
    def close(self):
        """ close model carrier (i.e. fmu, dymola) """
        raise NotImplementedError(f'{self.__class__.__name__}.close '
                                  f'function is not defined')


class ContinuousSimulation(Model):
    """
    Simulation apis for continuous simulations must inherit from ContinuousSimulation class.
    It includes methods for multi-processing

    :param str model_name:
        Name of the model being simulated
    :param int n_cpu:
        Number of cores to be used by simulation.
        If None is given, single core will be used.
        Maximum number equals the cpu count of the device.
        **Warning**: Logging is not yet fully working on multiple processes.
        Output will be written to the stream handler, but not to the created .log files.
    """

    _items_to_drop = [
        'pool',
    ]

    def __init__(self, model_name, n_cpu: int = 1):
        # Private helper attrs for multiprocessing
        self._n_sim_counter = 0
        self._n_sim_total = 0
        self._progress_int = 0
        # Check multiprocessing
        self.n_cpu = n_cpu
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

        super().__init__(model_name=model_name)

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
        # return deepcopy(self_dict)
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
        Used to differ between single- and multi-processing simulation
        """
        raise NotImplementedError(f'{self.__class__.__name__}._single_simulation '
                                  f'function is not defined')

    @abstractmethod
    def _update_model(self):
        """ Reimplement this to change variables etc. based on the new model. """
        raise NotImplementedError(f'{self.__class__.__name__}._update_model '
                                  f'function is not defined')


class DiscreteSimulation(Model):
    """
    Simulation apis for discrete simulations must inherit from DiscreteSimulation class.
    Defines abstract methods that must be implemented in sub-classes

    :param model_name:
    """
    def __init__(self, model_name):
        # attributes for discrete simulation
        self.current_time = None
        self.finished = None
        self.step_count = None  # counting simulation steps
        self.sim_res_df = None  # attribute that stores simulation result
        # pass model name to super class
        super().__init__(model_name=model_name)

    @abstractmethod
    def _update_model(self):
        """ Reimplement this to change variables etc. based on the new model. """
        raise NotImplementedError(f'{self.__class__.__name__}._update_model '
                                  f'function is not defined')

    @abstractmethod
    def close(self):
        raise NotImplementedError(f'{self.__class__.__name__}.close '
                                  f'function is not defined')

    @abstractmethod
    def step_only(self):
        """
        Reimplement this, to perform a single simulation step.
        In the method call the attributes step_count, current_time, and finished should be updated.
        """
        raise NotImplementedError(f'{self.__class__.__name__}.step_only '
                                  f'function is not defined')

    @abstractmethod
    def do_step(self):
        """
        Reimplement this, as a wrapper for step only.
        It extends the step functionality by considering inputs
        and writing the results to the sim_res_df attribute.
        """
        raise NotImplementedError(f'{self.__class__.__name__}.do_step '
                                  f'function is not defined')

    @abstractmethod
    def get_results(self):
        """
        Reimplement this, to return the sim_res_df attribute.
        """
        raise NotImplementedError(f'{self.__class__.__name__}.get_results '
                                  f'function is not defined')
