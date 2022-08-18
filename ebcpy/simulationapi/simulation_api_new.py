from pydantic import BaseModel, Field#, validator
from pydantic import FilePath, DirectoryPath
from typing import Union, Optional
from typing import TypeVar, Dict, Any, List
import numpy as np
from abc import abstractmethod
import pathlib
import pandas as pd
from ebcpy import TimeSeriesData
import os
import fmpy
from fmpy.model_description import read_model_description
import matplotlib.pyplot as plt
import shutil
from ebcpy.utils import setup_logger
import multiprocessing as mp
import logging
import itertools
import atexit

# todo:
# - easy: add simple side functions cd setter etc.
# - discuss update_model and model_name triggering other functions
# - Frage: wie genau funktioniert setter und _ notation
# - discuss output step, comm step
# - is log_fmu woirking? or is it only for contoinuous simulation
# - logger: instance name/index additionally to class name for co simulation? Alternatively FMU name
# - decompose disctete fmu sim iniialize func to inbtegfrate given mehtods that atre already use for continuous sim


class PID:  # todo: used for testing; remove once done and move to example
    '''
    PID implementation from aku and pst, simplified for the needs in this example by kbe
    '''
    def __init__(self, Kp=1.0, Ti=100.0, Td=0.0, lim_low=0.0, lim_high=100.0,
                 reverse_act=False, fixed_dt=1.0):

        self.x_act = 0  # measurement
        self.x_set = 0  # set point
        self.e = 0  # control difference
        self.e_last = 0  # control difference of previous time step
        self.y = 0  # controller output
        self.i = 0  # integrator value

        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.lim_low = lim_low  # low control limit
        self.lim_high = lim_high  # high control limit
        self.reverse_act = reverse_act  # control action
        self.dt = fixed_dt

    # -------- PID algorithm -----------------
    def run(self, x_act, x_set):
        self.x_act = x_act
        self.x_set = x_set

        # control difference depending on control direction
        if self.reverse_act:
            self.e = -(self.x_set - self.x_act)
        else:
            self.e = (self.x_set - self.x_act)

        # Integral
        if self.Ti > 0:
            self.i = 1 / self.Ti * self.e * self.dt + self.i
        else:
            self.i = 0

        # differential
        if self.dt > 0 and self.Td:
            de = self.Td * (self.e - self.e_last) / self.dt
        else:
            de = 0

        # PID output
        self.y = self.Kp * (self.e + self.i + de)

        # Limiter
        if self.y < self.lim_low:
            self.y = self.lim_low
            self.i = self.y / self.Kp - self.e
        elif self.y > self.lim_high:
            self.y = self.lim_high
            self.i = self.y / self.Kp - self.e

        self.e_last = self.e
        return self.y


def interp_df(t: int, df: pd.DataFrame,
              interpolate: bool = False):  # todo: move to utilities script
    """
    The function returns the values of the dataframe (row) at a given index.
    If the index is not present in the dataframe, either the next lower index
    is chosen or values are interpolated. If the last or first index value is exceeded the
    value is hold. In both cases a warning is printed.
    """
    # todo: consider check if step of input time step matches communication step size
    #  (or is given at a higher but aligned frequency).
    #  This might be the case very often and potentially inefficient df interpolation can be omitted in these cases.

    # initialize dict that represents row in dataframe with interpolated or hold values
    row = {}

    # catch values that are out of bound
    if t < df.index[0]:
        row.update(df.iloc[0].to_dict())
        warnings.warn(
            'Time {} s is below the first entry of the dataframe {} s, which is hold. Please check input data!'.format(
                t, df.index[0]))
    elif t >= df.index[-1]:
        row.update(df.iloc[-1].to_dict())
        # a time mathing the last index value causes problems with interpolation but should not raise a warning
        if t > df.index[-1]:
            warnings.warn(
                'Time {} s is above the last entry of the dataframe {} s, which is hold. Please check input data!'.format(
                    t, df.index[-1]))
    # either hold value of last index or interpolate
    else:
        # look for next lower index
        idx_l = df.index.get_indexer([t], method='pad')[0]  # get_loc() depreciated

        # return values at lower index
        if not interpolate:
            row = df.iloc[idx_l].to_dict()

        # return interpolated values
        else:
            idx_r = idx_l + 1
            for column in df.columns:
                row.update({column: np.interp(t, [df.index[idx_l], df.index[idx_r]],
                                              df[column].iloc[idx_l:idx_r + 1])})
    return row


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


""" Simulation Setup """
# Base - Dymola/FMU_continuous/FMU_discrete


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
    comm_step_size: float = Field(
        title="communication step size",
        default=1,
        description="step size in which the do_step() function is called"
    )
    solver: str = Field(
        title="solver",
        default="",  # Is added in the validator
        description="The solver to be used for numerical integration."
    )
    _default_solver: str = None
    _allowed_solvers: list = []


class SimulationSetupDymola(SimulationSetup):
    """
    Add's custom setup parameters for simulating in Dymola
    to the basic `SimulationSetup`
    """
    tolerance: float = Field(
        title="tolerance",
        default=0.0001,
        description="Tolerance of integration"
    )

    fixedstepsize: float = Field(
        title="fixedstepsize",
        default=0.0,
        description="Fixed step size for Euler"
    )


    _default_solver = "Dassl"
    _allowed_solvers = ["Dassl", "Euler", "Cerk23", "Cerk34", "Cerk45",
                        "Esdirk23a", "Esdirk34a", "Esdirk45a", "Cvode",
                        "Rkfix2", "Rkfix3", "Rkfix4", "Lsodar",
                        "Radau", "Dopri45", "Dopri853", "Sdirk34hw"]


class SimulationSetupFMU_Continuous(SimulationSetup):
    """
    Add's custom setup parameters for simulating FMU's continuously
    to the basic `SimulationSetup`
    """
    tolerance: float = Field(
        title="tolerance",
        default=0.0001,
        description="Total tolerance of integration"
    )

    fixedstepsize: float = Field(
        title="fixedstepsize",
        default=0.0,
        description="Fixed step size for Euler"
    )

    timeout: float = Field(
        title="timeout",
        default=np.inf,
        description="Timeout after which the simulation stops."
    )

    _default_solver = "CVode"
    _allowed_solvers = ["CVode", "Euler"]


class SimulationSetupFMU_Discrete(SimulationSetup):
    """
    Add's custom setup parameters for simulating FMUs stepwise
    to the basic `SimulationSetup`
    """
    comm_step_size: float = Field(
        title="communication step size",
        default=1,
        description="step size in which the do_step() function is called"
    )

    tolerance: float = Field(
        title="tolerance",
        default=0.0001,
        description="Absolute tolerance of integration"
    )
    _default_solver = "CVode"
    _allowed_solvers = ["CVode", "Euler"]


"""" Configuration """
# Base - Dymola/FMU


class ExperimentConfiguration(BaseModel):
    """
    pydantic BaseModel child to define a full simulation configuration
    """
    cd: Optional[DirectoryPath]
    sim_setup: Optional[SimulationSetup]


class ExperimentConfigurationFMU(ExperimentConfiguration):
    """
    in case of FMU simulation the fmu file path defines the model
    """
    file_path: Optional[FilePath]
    input_file: Optional[FilePath]


class ExperimentConfigurationDymola(ExperimentConfiguration):
    """
    in case of a Dymola simulation the package and model name define the model
    """
    package: Optional[FilePath]
    model_name: Optional[str]


SimulationSetupClass = TypeVar("SimulationSetupClass", bound=SimulationSetup)
ExperimentConfigurationClass = TypeVar("ExperimentConfigurationClass", bound=ExperimentConfiguration)


class Model:

    _sim_setup_class: SimulationSetupClass = SimulationSetup
    _exp_config_class: ExperimentConfigurationClass = ExperimentConfiguration

    def __init__(self, model_name):
        # initialize sim setup with specific class defaults.
        self._sim_setup = self._sim_setup_class()  # todo: why _ notation here?
        # update sim setup with config entries if given
        if hasattr(self.config, 'sim_setup'):
            self.set_sim_setup(self.config.sim_setup)
        # current directory
        if not hasattr(self, 'cd'):  # in case of FMU, cd is set already by now  # todo: not nice
            if hasattr(self.config, 'cd'):
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
        # self.worker_idx = False  # todo: evaluate if needed here
        # results
        self.result_names = []  # initialize list of tracked variables
        self.model_name = model_name  # todo: discuss setting model name triggers further functions
        self.sim_res = None  # todo: implement functionality for dym and fmu continuous

    @classmethod
    def get_simulation_setup_fields(cls):
        """Return all fields in the chosen SimulationSetup class."""
        return list(cls._sim_setup_class.__fields__.keys())

    def get_results(self, tsd_format: bool = False):
        """
        returns the simulation results either as pd.DataFrame or as TimeSeriesData
        """
        if not tsd_format:
            results = self.sim_res
        else:
            results = TimeSeriesData(self.sim_res, default_tag="sim")
            results.rename_axis(['Variables', 'Tags'], axis='columns')
            results.index.names = ['Time']  # todo: in ebcpy tsd example only sometimes
        return results

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
        Updates only those entries that are given as arguments  # todo: consider resetting to default first
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
        self._set_result_names()

    def _set_result_names(self):
        """
        by default, keys of the output variables are passed to result_names list.
        Method may be overwritten by child.
        """
        self.result_names = list(self.outputs.keys())

    @abstractmethod
    def _update_model(self):
        """
        Reimplement this to change variables etc.
        based on the new model.
        """
        raise NotImplementedError(f'{self.__class__.__name__}._update_model '
                                  f'function is not defined')

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

    def check_unsupported_variables(self, variables: List[str], type_of_var: str):  # todo: use this functionality in discrete simulation!!
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

    @ abstractmethod
    def close(self):
        raise NotImplementedError(f'{self.__class__.__name__}.close '
                                  f'function is not defined')


class FMU:

    _exp_config_class: ExperimentConfigurationClass = ExperimentConfigurationFMU
    _fmu_instances: dict = {}  # Dict of FMU instances  # fixme: mp in continuous requires class attribute..
    _unzip_dirs: dict = {}  # Dict of directories for fmu extraction  # fixme: mp in continuous requires class attribute..

    def __init__(self, log_fmu: bool = True):
        path = self.config.file_path
        if isinstance(self.config.file_path, pathlib.Path):
            path = str(self.config.file_path)
        if not path.lower().endswith(".fmu"):
            raise ValueError(f"{self.config.file_path} is not a valid fmu file!")
        self.path = path
        if hasattr(self.config, 'cd'):
            self.cd = self.config.cd
        else:
            self.cd = os.path.dirname(fmu_path)
        self.log_fmu = log_fmu  # todo consider moving to config
        # self._fmu_instances: dict = {}  # Dict of FMU instances  # fixme: mp in continuous requires class attribute..
        # self._unzip_dirs: dict = {}  # Dict of directories for fmu extraction  # fixme: mp in continuous requires class attribute..
        self._var_refs: dict = None  # Dict of variables and their references
        self._model_description = None
        self._fmi_type = None
        self._single_unzip_dir: str = None
        self.log_fmu = None

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
        """

        key = list(self.var_refs.keys())
        key_list = []
        for i in range(len(key)):
            if key[i].startswith(start_str):
                key_list.append(key[i])
        return key_list

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

        # Create dict of variable names with variable references from model description
        self.var_refs = {}
        for variable in self._model_description.modelVariables:
            self.var_refs[variable.name] = variable

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
                print()

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

    # def close(self):
    #     """
    #     Closes the fmu.
    #
    #     :return: bool
    #         True on success
    #     """
    #     # Close MP of super class
    #     # super().close()  # fixme:..back in? does super work? -> fmu is parent class; otherwise consider only single close in fmu class here
    #     # Close if single process
    #     if not self.use_mp:
    #         if not self._fmu_instances:
    #             return  # Already closed
    #         self._single_close(fmu_instance=self._fmu_instances[0],
    #                            unzip_dir=self._unzip_dirs[0])
    #         self._unzip_dirs = {}
    #         self._fmu_instances = {}

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
        self.logger.info('FMU "{}" closed'.format(self._model_description.modelName))
        if unzip_dir is not None:
            try:
                shutil.rmtree(unzip_dir)
            except FileNotFoundError:
                pass  # Nothing to delete
            except PermissionError:
                self.logger.error("Could not delete unzipped fmu "
                                  "in location %s. Delete it yourself.", unzip_dir)


class FMU_Discrete(FMU, Model):

    _sim_setup_class: SimulationSetupClass = SimulationSetupFMU_Discrete

    def __init__(self, config, log_fmu: bool = True):
        self.use_mp = False  # no mp for stepwise FMU simulation
        self.config = self._exp_config_class.parse_obj(config)
        self.worker_idx = None  # todo: evaluate where to place
        FMU.__init__(self, log_fmu)
        Model.__init__(self, model_name=self.config.file_path)  # todo: in case of fmu: file path, in case of dym: model_name, find better way to deal with; consider getting rid of model_name. For now it is to make the old methods work
        # used for stepwise simulation
        self.current_time = None
        self.finished = None
        # define input data (can be adjusted during simulation using the setter)
        if hasattr(self.config, 'input_file'):
            self._input_table = pd.read_csv(self.config.input_file, index_col='timestamp')
        else:
            print('No long-term input data set. '
                  'Setter method can still be used to set input data to "input_table" attribute')
            self.input_table = None
        self.interp_input_table = False  # if false, last value of input table is hold, otherwise interpolated
        self.step_count = None  # counting simulation steps

    @property
    def input_table(self):
        """
        input data that holds for longer parts of the simulation
        """
        return self._input_table

    @input_table.setter
    def input_table(self, input_data):
        """
        setter allows the input data to change during discrete simulation
        """
        if isinstance(input_data, TimeSeriesData):
            input_data = input_table.to_df(force_single_index=True)
        self._input_table = input_data

    def initialize_discrete_sim(self,
                                parameters: dict = None,
                                init_values: dict = None  # todo: consider as attributes
                                ): 
        """
        Initialisation of FMU. To be called before using stepwise simulation  # todo: consider calling automaically after fmu setup or before first step
        Parameters and initial values can be set.
        """

        # THE FOLLOWING STEPS OF INITIALISATION ALREADY COVERED BY INSTANTIATING FMU API:
        # - Read model description
        # - extract .fmu file
        # - Create FMU2 Slave
        # - instantiate fmu instance

        idx_worker = 0  # no mp for discrete simulation

        # Reset FMU instance
        self._fmu_instances[idx_worker].reset()

        # Set up experiment
        self._fmu_instances[idx_worker].setupExperiment(startTime=self.sim_setup.start_time,
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
        self._set_variables(var_dict=start_values, idx_worker=idx_worker)

        # Finalise initialisation
        self._fmu_instances[idx_worker].enterInitializationMode()
        self._fmu_instances[idx_worker].exitInitializationMode()

        # Initialize dataframe to store results
        # empty
        # self.sim_res = pd.DataFrame(columns=self.result_names)
        # initialized
        res = self._read_variables(vrs_list=self.result_names)
        self.sim_res = pd.DataFrame(res,
                                    index=[res['SimTime']],
                                    columns=self.result_names
                                    )

        self.logger.info('FMU "{}" initialized for discrete simulation'.format(self._model_description.modelName))

        # initialize status indicator
        self.finished = False

        # reset step count
        self.step_count = 0

    def _do_step(self, ret_res: bool = False, idx_worker: int = 0):
        """
        perform simulation step; return True if stop time reached.
        The results are appended to the sim_res results frame, just after the step -> ground truth
        If ret_res, additionally the results of the step are returned
        """

        # check if stop time is reached
        if self.current_time < self.sim_setup.stop_time:
            if self.step_count == 0:
                self.logger.info('Starting simulation of FMU "{}"'.format(self._model_description.modelName))
            # do simulation step
            status = self._fmu_instances[idx_worker].doStep(
                currentCommunicationPoint=self.current_time,
                communicationStepSize=self.sim_setup.comm_step_size)
            # step count
            self.step_count+=1
            # update current time and determine status
            self.current_time += self.sim_setup.comm_step_size
            self.finished = False
        else:
            self.finished = True
            self.logger.info('Simulation of FMU "{}" finished'.format(self._model_description.modelName))
        # read results
        res = self._read_variables(
            vrs_list=self.result_names)
        if not self.finished:
            # append
            if self.current_time % self.sim_setup.output_interval == 0:  # todo: output_step > comm_step -> the last n results of results attribute can no be used for mpc!!! consider downsampling in get_results or second results attribute that keeps the last n values? On the other hand, if user needs mpc with css step, he can set output_step =css
                self.sim_res = pd.concat(
                    [self.sim_res, pd.DataFrame.from_records([res_step],  # because frame.append will be depreciated
                                                             index=[res_step['SimTime']],
                                                             columns=self.sim_res.columns)])
        if ret_res:
            return self.finished, res
        else:
            return self.finished

    def do_step_wrapper(self, input_step: dict = None):  # todo: consider automatic close in here again. after results are read there is no need for the fmu to stay
        # collect inputs
        # get input from input table (overwrite with specific input for single step)
        single_input = {}
        if self.input_table is not None:
            # extract value from input time table
            # only consider columns in input table that refer to inputs of the FMU
            input_matches = list(set(self.inputs.keys()).intersection(set(self.input_table.columns)))
            input_table_filt = self.input_table[input_matches]  # todo: consider moving to setter for efficiency, if so, inputs must be identified before
            single_input = interp_df(t=self.current_time, df=input_table_filt, interpolate=self.interp_input_table)

        if input_step is not None:
            # overwrite with input for step
            single_input.update(input_step)

        # write inputs to fmu
        if single_input:
            self._set_variables(var_dict=single_input)

        # perform simulation step
        res_step = self._do_step(ret_res=True)[1]

        return res_step

    def _set_result_names(self):
        """
        In discrete simulation the inputs are typically relevant too.
        """
        self.result_names = list(self.outputs.keys()) + list(self.inputs.keys())


class ContinuousSimulation(Model):

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

    def close(self):  # todo: check if this is overwritten anyway?? Counts only for MP?? whats happending at else??
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

    @abstractmethod  # todo: why abstract method?
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


class FMU_Continuous(FMU, ContinuousSimulation):

    _sim_setup_class: SimulationSetupClass = SimulationSetupFMU_Continuous

    _type_map = {
        float: np.double,
        bool: np.bool_,
        int: np.int_
    }

    def __init__(self, config, n_cpu, log_fmu: bool = True):  # todo: consider use mp and n_core in config -> requires more specific config classes
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
            if not self._fmu_instances:
                return  # Already closed
            self._single_close(fmu_instance=self._fmu_instances[0],
                               unzip_dir=self._unzip_dirs[0])
            self._unzip_dirs = {}
            self._fmu_instances = {}

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


if __name__ == '__main__':

    """ FMU discrete """
    # # ---- Settings ---------
    # output_step = 60 * 10  # step size of simulation results in seconds (resolution of results data)
    # comm_step = 60 / 3  # step size of FMU communication in seconds (in this interval, values are set to or read from the fmu)
    # start = 0  # start time
    # stop = 86400 * 3  # stop time
    # t_start = 293.15 - 5
    # t_start_amb = 293.15 - 15
    #
    # input_data = pd.read_csv('D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/data/ThermalZone_input.csv', index_col='timestamp')
    #
    # # store simulation setup as dict
    # simulation_setup = {"start_time": start,
    #                     "stop_time": stop,
    #                     "output_interval": output_step,
    #                     "comm_step_size": comm_step
    #                     }
    #
    # config_obj = {
    #               'file_path': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/data/ThermalZone_bus.fmu',
    #               'cd': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/results',  # fixme: if not exists -> pydantic returns error instead of creating it
    #               'sim_setup': simulation_setup,
    #               'input_file': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/data/ThermalZone_input.csv'
    #               }
    #
    # sys = FMU_Discrete(config_obj)
    # ctr = PID(Kp=0.01, Ti=300, lim_high=1, reverse_act=False, fixed_dt=comm_step)
    #
    # sys.initialize_discrete_sim(parameters={'T_start': t_start}, init_values={'bus.disturbance[1]': t_start_amb})
    #
    # res_step = sys.sim_res.iloc[-1]
    # while not sys.finished:
    #     # Call controller (for advanced control strategies that require previous results, use the attribute sim_res)
    #     ctr_action = ctr.run(res_step['bus.processVar'], input_data.loc[sys.current_time][
    #         'bus.setPoint'])
    #     # Apply control action to system and perform simulation step
    #     res_step = sys.do_step_wrapper(input_step={
    #         'bus.controlOutput': ctr_action})  # fixme consider returning the last n values for mpc if n==1 return dict, otherwise list of dicts
    #
    # sys.close()
    #
    # # ---- Results ---------
    # # return simulation results as pd data frame
    # results_study_A = sys.get_results(tsd_format=False)
    #
    # # format settings
    # import matplotlib
    #
    # # plot settings
    # matplotlib.rcParams['mathtext.fontset'] = 'custom'
    # matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    # matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    # matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    #
    # matplotlib.rcParams['mathtext.fontset'] = 'stix'
    # matplotlib.rcParams['font.family'] = 'STIXGeneral'
    # matplotlib.rcParams['font.size'] = 9
    # matplotlib.rcParams['lines.linewidth'] = 0.75
    #
    # cases = [results_study_A]
    # time_index_out = np.arange(0, stop + comm_step, output_step)  # time index with output interval step
    # fig, axes_mat = plt.subplots(nrows=3, ncols=1)
    # for i in range(len(cases)):
    #     axes = axes_mat
    #     axes[0].plot(time_index_out, cases[i]['bus.processVar'] - 273.15, label='mea', color='b')
    #     axes[0].plot(input_data.index, input_data['bus.setPoint'] - 273.15, label='set', color='r')  # fixme: setpoint not available in results
    #     axes[1].plot(time_index_out, cases[i]['bus.controlOutput'], label='control output', color='b')
    #     axes[2].plot(time_index_out, cases[i]['bus.disturbance[1]'] - 273.15, label='dist', color='b')
    #
    #     # x label
    #     axes[2].set_xlabel('Time / s')
    #     # title and y label
    #     if i == 0:
    #         axes[0].set_title('System FMU - Python controller')
    #         axes[0].set_ylabel('Zone temperature / °C')
    #         axes[1].set_ylabel('Rel. heating power / -')
    #         axes[2].set_ylabel('Ambient temperature / °C')
    #     if i == 1:
    #         axes[0].set_title('System FMU - Controller FMU')
    #         axes[0].legend(loc='upper right')
    #     # grid
    #     for ax in axes:
    #         ax.grid(True, 'both')
    #         if i > 0:
    #             # ignore y labels for all but the first
    #             ax.set_yticklabels([])
    #
    # plt.tight_layout()
    # plt.show()

    """ FMU continuous """
    n_sim = 10
    simulation_setup = {"start_time": 0,
                        "stop_time": 3600,
                        "output_interval": 100}
    config_obj = {
                  'file_path': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/data/HeatPumpSystemWithInput.fmu',
                  'cd': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/results',  # fixme: if not exists -> pydantic returns error instead of creating it
                  'sim_setup': simulation_setup,
                  'input_file': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/data/ThermalZone_input.csv'
                  }

    sys = FMU_Continuous(config_obj, n_cpu=2)

    time_index = np.arange(
        sys.sim_setup.start_time,
        sys.sim_setup.stop_time,
        sys.sim_setup.output_interval
    )
    # Apply some sinus function for the outdoor air temperature
    t_dry_bulb = np.sin(time_index / 3600 * np.pi) * 10 + 263.15
    df_inputs = TimeSeriesData({"TDryBul": t_dry_bulb}, index=time_index)

    hea_cap_c = sys.parameters['heaCap.C'].value
    # Let's alter it from 10% to 1000 % in n_sim simulations:
    sizings = np.linspace(0.1, 10, n_sim)
    parameters = []
    for sizing in sizings:
        parameters.append({"heaCap.C": hea_cap_c * sizing})

    sys.result_names = ["heaCap.T", "TDryBul"]

    results = sys.simulate(parameters=parameters,
                           inputs=df_inputs)

    # Plot the result
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].set_ylabel("TDryBul in K")
    ax[1].set_ylabel("T_Cap in K")
    ax[1].set_xlabel("Time in s")
    ax[0].plot(df_inputs, label="Inputs", linestyle="--")
    for res, sizing in zip(results, sizings):
        ax[0].plot(res['TDryBul'])
        ax[1].plot(res['heaCap.T'], label=sizing)
    for _ax in ax:
        _ax.legend(bbox_to_anchor=(1, 1.05), loc="upper left")

    plt.show()