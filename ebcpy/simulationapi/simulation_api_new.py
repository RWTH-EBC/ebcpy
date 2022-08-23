from pydantic import BaseModel, Field, validator
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
import sys
from ebcpy.modelica import manipulate_ds
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt
import time # for timing code only

# TODO:
# - add function: print supported exp setup options (like sim setup)
# - add check for variable names whenever var_names are passed (using "check_unsupported_vars")
# todo:
# bug: single unzip dir not deleted in continuous simulation
# - easy: add simple side functions cd setter etc.
# - discuss update_model and model_name triggering other functions
# - Frage: wie genau funktioniert setter und _ notation
# - discuss output step, comm step
# - is log_fmu woirking? or is it only for contoinuous simulation
# - logger: instance name/index additionally to class name for co simulation? Alternatively FMU name
# - decompose disctete fmu sim iniialize func to inbtegfrate given mehtods that atre already use for continuous sim
# fixme:
# mit python console läuft fmu conti nicht und dymola läuft nicht bis zum ende durch
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

# pd.DataFrame und TimeSeriesData as type to be validated by pydantic
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
TimeSeriesDataObject = TypeVar('TimeSeriesData')


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
        default=None,  # to select fmpy's default
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
    input_data: Optional[Union[FilePath, PandasDataFrame, TimeSeriesDataObject]]  # fixme: not needed in continuous fmu sim


class ExperimentConfigurationDymola(ExperimentConfiguration):
    """
    in case of a Dymola simulation the package and model name define the model
    """
    packages: Optional[List[FilePath]]
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
        if self.config.sim_setup is not None:
            self.set_sim_setup(self.config.sim_setup)
        # current directory
        if not hasattr(self, 'cd'):  # in case of FMU, cd is set already by now  # todo: not nice
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
        self.model_name = model_name  # todo: discuss setting model name triggers further functions
        self.sim_res = None  # todo: implement functionality for dym and fmu continuous

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
        if self.use_mp:  # todo: review this condition? It would be better to get rid off worker_idx at level of Model-class
            if self.worker_idx:  # todo: what is this actually for???
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
    _fmu_instance = None
    _unzip_dir: str = None

    def __init__(self, log_fmu: bool = True):
        self._unzip_dir = None
        self._fmu_instance = None
        path = self.config.file_path
        if isinstance(self.config.file_path, pathlib.Path):
            path = str(self.config.file_path)
        if not path.lower().endswith(".fmu"):
            raise ValueError(f"{self.config.file_path} is not a valid fmu file!")
        self.path = path
        if hasattr(self.config, 'cd') and self.config.cd is not None:
            self.cd = self.config.cd
        else:
            self.cd = os.path.dirname(path)
        self.log_fmu = log_fmu  # todo consider moving to config
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

    def _set_variables(self, var_dict: dict):
        """
        Sets multiple variables.
        var_dict is a dict with variable names in keys.
        """

        for key, value in var_dict.items():
            var = self.var_refs[key]
            vr = [var.valueReference]

            if var.type == 'Real':
                self._fmu_instance.setReal(vr, [float(value)])
            elif var.type in ['Integer', 'Enumeration']:
                self._fmu_instance.setInteger(vr, [int(value)])
            elif var.type == 'Boolean':
                self._fmu_instance.setBoolean(vr, [value == 1.0 or value or value == "True"])
            else:
                raise Exception("Unsupported type: %s" % var.type)

    def _read_variables(self, vrs_list: list):  #
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
                res[name] = self._fmu_instance.getReal(vr)[0]
            elif var.type in ['Integer', 'Enumeration']:
                res[name] = self._fmu_instance.getInteger(vr)[0]
            elif var.type == 'Boolean':
                value = self._fmu_instance.getBoolean(vr)[0]
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
        if use_mp:
            wrk_idx = self.worker_idx
            if self._fmu_instance is not None:
                return True
            unzip_dir = self._single_unzip_dir + f"_worker_{wrk_idx}"
            fmpy.extract(self.model_name,
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
    objs = []  # to use the close_all method

    def __init__(self, config, log_fmu: bool = True):
        FMU_Discrete.objs.append(self)
        self.use_mp = False  # no mp for stepwise FMU simulation
        self.config = self._exp_config_class.parse_obj(config)
        FMU.__init__(self, log_fmu)
        Model.__init__(self, model_name=self.config.file_path)  # todo: in case of fmu: file path, in case of dym: model_name, find better way to deal with; consider getting rid of model_name. For now it is to make the old methods work
        # used for stepwise simulation
        self.current_time = None
        self.finished = None
        # define input data (can be adjusted during simulation using the setter)
        self.input_table = self.config.input_data  # calling the setter to distinguish depending on type
        self.interp_input_table = False  # if false, last value of input table is hold, otherwise interpolated
        self.step_count = None  # counting simulation steps

    @classmethod
    def close_all(cls):
        """
        close multiple FMUs at once. Useful for co-simulation
        """
        for obj in cls.objs:
            obj.close()

    @property
    def input_table(self):
        """
        input data that holds for longer parts of the simulation
        """
        return self._input_table

    @input_table.setter
    def input_table(self, inp: Union[FilePath, PandasDataFrame, TimeSeriesDataObject]):
        """
        setter allows the input data to change during discrete simulation
        """
        if inp is not None:
            if isinstance(inp, (str, pathlib.Path)):  # fixme: why does pydantcs FilePath does not work jhere
                if not str(inp).endswith('csv'):
                    raise TypeError(
                        'input data {} is not passed as .csv file.'
                        'Instead of passing a file consider passing a pd.Dataframe or TimeSeriesData object'.format(inp)
                    )
                self._input_table = pd.read_csv(inp, index_col='time')
            else:  # pd frame or tsd object; wrong type already caught by pydantic
                if isinstance(inp, TimeSeriesData):
                    self._input_table = inp.to_df(force_single_index=True)
                elif isinstance(inp, pd.DataFrame):
                    self._input_table = inp
        else:
            print('No long-term input data set!'
                  'Setter method can still be used to set input data to "input_table" attribute')
            self._input_table = None

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
        self._set_variables(var_dict=start_values)

        # Finalise initialisation
        self._fmu_instance.enterInitializationMode()
        self._fmu_instance.exitInitializationMode()

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

    def _do_step(self):
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
            status = self._fmu_instance.doStep(
                currentCommunicationPoint=self.current_time,
                communicationStepSize=self.sim_setup.comm_step_size)
            # step count
            self.step_count += 1
            # update current time and determine status
            self.current_time += self.sim_setup.comm_step_size
            self.finished = False
        else:
            self.finished = True
            self.logger.info('Simulation of FMU "{}" finished'.format(self._model_description.modelName))

        return self.finished

    def inp_step_read(self, input_step: dict = None):  # todo: consider automatic close in here again. after results are read there is no need for the fmu to stay
        # collect inputs
        # get input from input table (overwrite with specific input for single step)
        single_input = {}
        if self.input_table is not None:
            # extract value from input time table
            # only consider columns in input table that refer to inputs of the FMU
            input_matches = list(set(self.inputs.keys()).intersection(set(self.input_table.columns)))
            input_table_filt = self.input_table[input_matches]  # todo: consider moving filter to setter for efficiency, if so, inputs must be identified before
            single_input = interp_df(t=self.current_time, df=input_table_filt, interpolate=self.interp_input_table)

        if input_step is not None:
            # overwrite with input for step
            single_input.update(input_step)

        # write inputs to fmu
        if single_input:
            self._set_variables(var_dict=single_input)

        # perform simulation step
        self._do_step()

        # read results
        res = self._read_variables(
            vrs_list=self.result_names)
        if not self.finished:
            # append
            if self.current_time % self.sim_setup.output_interval == 0:  # todo: output_step > comm_step -> the last n results of results attribute can no be used for mpc!!! consider downsampling in get_results or second results attribute that keeps the last n values? On the other hand, if user needs mpc with css step, he can set output_step =css
                self.sim_res = pd.concat(
                    [self.sim_res, pd.DataFrame.from_records([res],  # because frame.append will be depreciated
                                                             index=[res['SimTime']],
                                                             columns=self.sim_res.columns)])
        return res

    def _set_result_names(self):
        """
        Adds input names to list result_names in addition to outputs.
        In discrete simulation the inputs are typically relevant.
        """
        self.result_names = list(self.outputs.keys()) + list(self.inputs.keys())

    def close(self):
        # No MP for discrete simulation
        if not self._fmu_instance:
            return  # Already closed
        self._single_close(fmu_instance=self._fmu_instance,
                           unzip_dir=self._unzip_dir)
        self._unzip_dir = None
        self._fmu_instance = None


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


class FMU_API(FMU, ContinuousSimulation):

    _sim_setup_class: SimulationSetupClass = SimulationSetupFMU_Continuous
    # _items_to_drop = ["pool"]
    _items_to_drop = ["pool", "_fmu_instance", "_unzip_dir"]
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


class DymolaAPI(ContinuousSimulation):
    """
    API to a Dymola instance.

    :param str,os.path.normpath cd:
        Dirpath for the current working directory of dymola
    :param str model_name:
        Name of the model to be simulated
    :param list packages:
        List with path's to the packages needed to simulate the model
    :keyword Boolean show_window:
        True to show the Dymola window. Default is False
    :keyword Boolean modify_structural_parameters:
        True to automatically set the structural parameters of the
        simulation model via Modelica modifiers. Default is True.
        See also the keyword ``structural_parameters``
        of the ``simulate`` function.
    :keyword Boolean equidistant_output:
        If True (Default), Dymola stores variables in an
        equisdistant output and does not store variables at events.
    :keyword str dymola_path:
         Path to the dymola installation on the device. Necessary
         e.g. on linux, if we can't find the path automatically.
         Example: ``dymola_path="C://Program Files//Dymola 2020x"``
    :keyword int n_restart:
        Number of iterations after which Dymola should restart.
        This is done to free memory. Default value -1. For values
        below 1 Dymola does not restart.
    :keyword bool extract_variables:
        If True (the default), all variables of the model will be extracted
        on init of this class.
        This required translating the model.
    :keyword bool debug:
        If True (not the default), the dymola instance is not closed
        on exit of the python script. This allows further debugging in
        dymola itself if API-functions cause a python error.
    :keyword str mos_script_pre:
        Path to a valid mos-script for Modelica/Dymola.
        If given, the script is executed **prior** to laoding any
        package specified in this API.
        May be relevant for handling version conflicts.
    :keyword str mos_script_post:
        Path to a valid mos-script for Modelica/Dymola.
        If given, the script is executed before closing Dymola.
    :keyword str dymola_version:
        Version of Dymola to use.
        If not given, newest version will be used.
        If given, the Version needs to be equal to the folder name
        of your installation.

        **Example:** If you have two version installed at

        - ``C://Program Files//Dymola 2021`` and
        - ``C://Program Files//Dymola 2020x``

        and you want to use Dymola 2020x, specify
        ``dymola_version='Dymola 2020x'``.

        This parameter is overwritten if ``dymola_path`` is specified.

    Example:

    >>> import os
    >>> from ebcpy import DymolaAPI
    >>> # Specify the model name
    >>> model_name = "Modelica.Thermal.FluidHeatFlow.Examples.PumpAndValve"
    >>> dym_api = DymolaAPI(cd=os.getcwd(),
    >>>                     model_name=model_name,
    >>>                     packages=[],
    >>>                     show_window=True)
    >>> dym_api.sim_setup = {"start_time": 100,
    >>>                      "stop_time": 200}
    >>> dym_api.simulate()
    >>> dym_api.close()

    """

    _exp_config_class: ExperimentConfigurationClass = ExperimentConfigurationDymola
    _sim_setup_class: SimulationSetupClass = SimulationSetupDymola
    _items_to_drop = ["pool", "dymola", "_dummy_dymola_instance"]
    dymola = None
    # Default simulation setup
    _supported_kwargs = [
        "show_window",
        "modify_structural_parameters",
        "dymola_path",
        "equidistant_output",
        "n_restart",
        "debug",
        "mos_script_pre",
        "mos_script_post",
        "dymola_version"
    ]

    def __init__(self, config, n_cpu, **kwargs):
        """Instantiate class objects."""
        self.config = self._exp_config_class.parse_obj(config)
        packages = self.config.packages

        self.dymola = None  # Avoid key-error in get-state. Instance attribute needs to be there.

        # Update kwargs with regard to what kwargs are supported.
        self.extract_variables = kwargs.pop("extract_variables", True)
        self.fully_initialized = False
        self.debug = kwargs.pop("debug", False)
        self.show_window = kwargs.pop("show_window", False)
        self.modify_structural_parameters = kwargs.pop("modify_structural_parameters", True)
        self.equidistant_output = kwargs.pop("equidistant_output", True)
        self.mos_script_pre = kwargs.pop("mos_script_pre", None)
        self.mos_script_post = kwargs.pop("mos_script_post", None)
        self.dymola_version = kwargs.pop("dymola_version", None)
        for mos_script in [self.mos_script_pre, self.mos_script_post]:
            if mos_script is not None:
                if not os.path.isfile(mos_script):
                    raise FileNotFoundError(
                        f"Given mos_script '{mos_script}' does "
                        f"not exist."
                    )
                if not str(mos_script).endswith(".mos"):
                    raise TypeError(
                        f"Given mos_script '{mos_script}' "
                        f"is not a valid .mos file."
                    )

        # Convert to modelica path
        if self.mos_script_pre is not None:
            self.mos_script_pre = self._make_modelica_normpath(self.mos_script_pre)
        if self.mos_script_post is not None:
            self.mos_script_post = self._make_modelica_normpath(self.mos_script_post)

        super().__init__(model_name=self.config.model_name,
                         n_cpu=n_cpu)

        # First import the dymola-interface
        dymola_path = kwargs.pop("dymola_path", None)
        if dymola_path is not None:
            if not os.path.exists(dymola_path):
                raise FileNotFoundError(f"Given path '{dymola_path}' can not be found on "
                                        "your machine.")
            _dym_install = dymola_path
        else:
            # Get the dymola-install-path:
            _dym_installations = self.get_dymola_install_paths()
            if _dym_installations:
                if self.dymola_version:
                    _found_version = False
                    for _dym_install in _dym_installations:
                        if _dym_install.endswith(self.dymola_version):
                            _found_version = True
                            break
                    if not _found_version:
                        raise ValueError(
                            f"Given dymola_version '{self.dymola_version}' not found in "
                            f"the list of dymola installations {_dym_installations}"
                        )
                else:
                    _dym_install = _dym_installations[0]  # 0 is the newest
                self.logger.info("Using dymola installation at %s", _dym_install)
            else:
                raise FileNotFoundError("Could not find a dymola-interface on your machine.")
        dymola_exe_path = self.get_dymola_path(_dym_install)
        self.logger.info("Using dymola.exe: %s", dymola_exe_path)
        dymola_interface_path = self.get_dymola_interface_path(_dym_install)
        self.logger.info("Using dymola interface: %s", dymola_interface_path)

        # Set the path variables:
        self.dymola_interface_path = dymola_interface_path
        self.dymola_exe_path = dymola_exe_path

        self.packages = []
        if packages is not None:
            for package in packages:
                if isinstance(package, pathlib.Path):
                    self.packages.append(str(package))
                elif isinstance(package, str):
                    self.packages.append(package)
                else:
                    raise TypeError(f"Given package is of type {type(package)}"
                                    f" but should be any valid path.")

        # Import n_restart
        self.sim_counter = 0
        self.n_restart = kwargs.pop("n_restart", -1)
        if not isinstance(self.n_restart, int):
            raise TypeError(f"n_restart has to be type int but "
                            f"is of type {type(self.n_restart)}")

        self._dummy_dymola_instance = None  # Ensure self._close_dummy gets the attribute.
        if self.n_restart > 0:
            self.logger.info("Open blank placeholder Dymola instance to ensure"
                             " a licence during Dymola restarts")
            self._dummy_dymola_instance = self._open_dymola_interface()
            atexit.register(self._close_dummy)

        # List storing structural parameters for later modifying the simulation-name.
        # Parameter for raising a warning if to many dymola-instances are running
        self._critical_number_instances = 10 + self.n_cpu
        # Register the function now in case of an error.
        if not self.debug:
            atexit.register(self.close)
        if self.use_mp:
            self.pool.map(self._setup_dymola_interface, [True for _ in range(self.n_cpu)])
        # For translation etc. always setup a default dymola instance
        self.dymola = self._setup_dymola_interface(use_mp=False)

        self.fully_initialized = True
        # Trigger on init.
        self._update_model()
        # Set result_names to output variables.
        self.result_names = list(self.outputs.keys())

        # Check if some kwargs are still present. If so, inform the user about
        # false usage of kwargs:
        if kwargs:
            self.logger.error(
                "You passed the following kwargs which "
                "are not part of the supported kwargs and "
                "have thus no effect: %s.", " ,".join(list(kwargs.keys())))

    def _update_model(self):
        # Translate the model and extract all variables,
        # if the user wants to:
        if self.extract_variables and self.fully_initialized:
            self.extract_model_variables()

    def simulate(self,
                 parameters: Union[dict, List[dict]] = None,
                 return_option: str = "time_series",
                 **kwargs):
        """
        Simulate the given parameters.

        Additional settings:

        :keyword List[str] model_names:
            List of Dymola model-names to simulate. Should be either the size
            of parameters or parameters needs to be sized 1.
            Keep in mind that different models may use different parameters!
        :keyword Boolean show_eventlog:
            Default False. True to show evenlog of simulation (advanced)
        :keyword Boolean squeeze:
            Default True. If only one set of initialValues is provided,
            a DataFrame is returned directly instead of a list.
        :keyword str table_name:
            If inputs are given, you have to specify the name of the table
            in the instance of CombiTimeTable. In order for the inputs to
            work the value should be equal to the value of 'tableName' in Modelica.
        :keyword str file_name:
            If inputs are given, you have to specify the file_name of the table
            in the instance of CombiTimeTable. In order for the inputs to
            work the value should be equal to the value of 'fileName' in Modelica.
        :keyword List[str] structural_parameters:
            A list containing all parameter names which are structural in Modelica.
            This means a modifier has to be created in order to change
            the value of this parameter. Internally, the given list
            is added to the known states of the model. Hence, you only have to
            specify this keyword argument if your structural parameter
            does not appear in the dsin.txt file created during translation.

            Example:
            Changing a record in a model:

            >>> sim_api.simulate(
            >>>     parameters={"parameterPipe": "AixLib.DataBase.Pipes.PE_X.DIN_16893_SDR11_d160()"},
            >>>     structural_parameters=["parameterPipe"])

        """
        # Handle special case for structural_parameters
        if "structural_parameters" in kwargs:
            _struc_params = kwargs["structural_parameters"]
            # Check if input is 2-dimensional for multiprocessing.
            # If not, make it 2-dimensional to avoid list flattening in
            # the super method.
            if not isinstance(_struc_params[0], list):
                kwargs["structural_parameters"] = [_struc_params]
        if "model_names" in kwargs:
            model_names = kwargs["model_names"]
            if not isinstance(model_names, list):
                raise TypeError("model_names needs to be a list.")
            if isinstance(parameters, dict):
                # Make an array of parameters to enable correct use of super function.
                parameters = [parameters] * len(model_names)
            if parameters is None:
                parameters = [{}] * len(model_names)
        return super().simulate(parameters=parameters, return_option=return_option, **kwargs)

    def _single_simulation(self, kwargs):
        # Unpack kwargs
        show_eventlog = kwargs.get("show_eventlog", False)
        squeeze = kwargs.get("squeeze", True)
        result_file_name = kwargs.get("result_file_name", 'resultFile')
        parameters = kwargs.get("parameters")
        return_option = kwargs.get("return_option")
        model_names = kwargs.get("model_names")
        inputs = kwargs.get("inputs", None)
        fail_on_error = kwargs.get("fail_on_error", True)
        structural_parameters = kwargs.get("structural_parameters", [])

        # Handle multiprocessing
        if self.use_mp:
            idx_worker = self.worker_idx
            if self.dymola is None:
                self._setup_dymola_interface(use_mp=True)


        # Handle eventlog
        if show_eventlog:
            self.dymola.experimentSetupOutput(events=True)
            self.dymola.ExecuteCommand("Advanced.Debug.LogEvents = true")
            self.dymola.ExecuteCommand("Advanced.Debug.LogEventsInitialization = true")

        # Restart Dymola after n_restart iterations
        self._check_restart()

        # Handle custom model_names
        if model_names is not None:
            # Custom model_name setting
            _res_names = self.result_names.copy()
            self._model_name = model_names
            self._update_model_variables()
            if _res_names != self.result_names:
                self.logger.info(
                    "Result names changed due to setting the new model. "
                    "If you do not expect custom result names, ignore this warning."
                    "If you do expect them, please raise an issue to add the "
                    "option when using the model_names keyword.")
                self.logger.info(
                    "Difference: %s",
                    " ,".join(list(set(_res_names).difference(self.result_names)))
                )

        # Handle parameters:
        if parameters is None:
            parameters = {}
            unsupported_parameters = False
        else:
            unsupported_parameters = self.check_unsupported_variables(
                variables=list(parameters.keys()),
                type_of_var="parameters"
            )

        # Handle structural parameters

        if (unsupported_parameters and
                (self.modify_structural_parameters or
                 structural_parameters)):
            # Alter the model_name for the next simulation
            model_name, parameters_new = self._alter_model_name(
                parameters=parameters,
                model_name=self.model_name,
                structural_params=list(self.states.keys()) + structural_parameters
            )
            # Trigger translation only if something changed
            if model_name != self.model_name:
                _res_names = self.result_names.copy()
                self.model_name = model_name
                self.result_names = _res_names  # Restore previous result names
                self.logger.warning(
                    "Warning: Currently, the model is re-translating "
                    "for each simulation. You should add to your Modelica "
                    "parameters \"annotation(Evaluate=false)\".\n "
                    "Check for these parameters: %s",
                    ', '.join(set(parameters.keys()).difference(parameters_new.keys()))
                )
            parameters = parameters_new
            # Check again
            unsupported_parameters = self.check_unsupported_variables(
                variables=list(parameters.keys()),
                type_of_var="parameters"
            )

        initial_names = list(parameters.keys())
        initial_values = list(parameters.values())
        # Convert to float for Boolean and integer types:
        try:
            initial_values = [float(v) for v in initial_values]
        except (ValueError, TypeError) as err:
            raise TypeError("Dymola only accepts float values. "
                            "Could bot automatically convert the given "
                            "parameter values to float.") from err

        # Handle inputs
        if inputs is not None:
            # Unpack additional kwargs
            try:
                table_name = kwargs["table_name"]
                file_name = kwargs["file_name"]
            except KeyError as err:
                raise KeyError("For inputs to be used by DymolaAPI.simulate, you "
                               "have to specify the 'table_name' and the 'file_name' "
                               "as keyword arguments of the function. These must match"
                               "the values 'tableName' and 'fileName' in the CombiTimeTable"
                               " model in your modelica code.") from err
            # Generate the input in the correct format
            offset = self.sim_setup.start_time - inputs.index[0]
            filepath = convert_tsd_to_modelica_txt(
                tsd=inputs,
                table_name=table_name,
                save_path_file=file_name,
                offset=offset
            )
            self.logger.info("Successfully created Dymola input file at %s", filepath)

        if return_option == "savepath":
            if unsupported_parameters:
                raise KeyError("Dymola does not accept invalid parameter "
                               "names for option return_type='savepath'. "
                               "To use this option, delete unsupported "
                               "parameters from your setup.")
            res = self.dymola.simulateExtendedModel(
                self.model_name,
                startTime=self.sim_setup.start_time,
                stopTime=self.sim_setup.stop_time,
                numberOfIntervals=0,
                outputInterval=self.sim_setup.output_interval,
                method=self.sim_setup.solver,
                tolerance=self.sim_setup.tolerance,
                fixedstepsize=self.sim_setup.fixedstepsize,
                resultFile=result_file_name,
                initialNames=initial_names,
                initialValues=initial_values)
        else:
            if not parameters and not self.parameters:
                raise ValueError(
                    "Sadly, simulating a model in Dymola "
                    "with no parameters returns no result. "
                    "Call this function using return_option='savepath' to get the results."
                )
            if not parameters:
                random_name = list(self.parameters.keys())[0]
                initial_values = [self.parameters[random_name].value]
                initial_names = [random_name]

            # Handle 1 and 2 D initial names:
            # Convert a 1D list to 2D list
            if initial_values and isinstance(initial_values[0], (float, int)):
                initial_values = [initial_values]

            # Handle the time of the simulation:
            res_names = self.result_names.copy()
            if "Time" not in res_names:
                res_names.append("Time")

            # Internally convert output Interval to number of intervals
            # (Required by function simulateMultiResultsModel
            number_of_intervals = (self.sim_setup.stop_time - self.sim_setup.start_time) / \
                                  self.sim_setup.output_interval
            if int(number_of_intervals) != number_of_intervals:
                raise ValueError(
                    "Given output_interval and time interval did not yield "
                    "an integer numberOfIntervals. To use this functions "
                    "without savepaths, you have to provide either a "
                    "numberOfIntervals or a value for output_interval "
                    "which can be converted to numberOfIntervals.")

            res = self.dymola.simulateMultiResultsModel(
                self.model_name,
                startTime=self.sim_setup.start_time,
                stopTime=self.sim_setup.stop_time,
                numberOfIntervals=int(number_of_intervals),
                method=self.sim_setup.solver,
                tolerance=self.sim_setup.tolerance,
                fixedstepsize=self.sim_setup.fixedstepsize,
                resultFile=None,
                initialNames=initial_names,
                initialValues=initial_values,
                resultNames=res_names)

        if not res[0]:
            self.logger.error("Simulation failed!")
            self.logger.error("The last error log from Dymola:")
            log = self.dymola.getLastErrorLog()
            # Only print first part as output is sometimes to verbose.
            self.logger.error(log[:10000])
            dslog_path = os.path.join(self.cd, 'dslog.txt')
            try:
                with open(dslog_path, "r") as dslog_file:
                    dslog_content = dslog_file.read()
                    self.logger.error(dslog_content)
            except Exception:
                dslog_content = "Not retreivable. Open it yourself."
            msg = f"Simulation failed: Reason according " \
                  f"to dslog, located at '{dslog_path}': {dslog_content}"
            if fail_on_error:
                raise Exception(msg)
            # Don't raise and return None
            self.logger.error(msg)
            return None

        if return_option == "savepath":
            _save_name_dsres = f"{result_file_name}.mat"
            savepath = kwargs.pop("savepath", None)
            # Get the cd of the current dymola instance
            self.dymola.cd()
            # Get the value and convert it to a 100 % fitting str-path
            dymola_cd = str(pathlib.Path(self.dymola.getLastErrorLog().replace("\n", "")))
            if savepath is None or str(savepath) == dymola_cd:
                return os.path.join(dymola_cd, _save_name_dsres)
            os.makedirs(savepath, exist_ok=True)
            for filename in [_save_name_dsres, "dslog.txt", "dsfinal.txt"]:
                # Delete existing files
                try:
                    os.remove(os.path.join(savepath, filename))
                except OSError:
                    pass
                # Move files
                shutil.copy(os.path.join(dymola_cd, filename),
                            os.path.join(savepath, filename))
                os.remove(os.path.join(dymola_cd, filename))
            return os.path.join(savepath, _save_name_dsres)

        data = res[1]  # Get data
        if return_option == "last_point":
            results = []
            for ini_val_set in data:
                results.append({result_name: ini_val_set[idx][-1] for idx, result_name
                                in enumerate(res_names)})
            if len(results) == 1 and squeeze:
                return results[0]
            return results
        # Else return as dataframe.
        dfs = []
        for ini_val_set in data:
            df = pd.DataFrame({result_name: ini_val_set[idx] for idx, result_name
                               in enumerate(res_names)})
            # Set time index
            df = df.set_index("Time")
            # Convert it to float
            df.index = df.index.astype("float64")
            dfs.append(df)
        # Most of the cases, only one set is provided. In that case, avoid
        if len(dfs) == 1 and squeeze:
            return TimeSeriesData(dfs[0], default_tag="sim")
        return [TimeSeriesData(df, default_tag="sim") for df in dfs]

    def translate(self):
        """
        Translates the current model using dymola.translateModel()
        and checks if erros occur.
        """
        res = self.dymola.translateModel(self.model_name)
        if not res:
            self.logger.error("Translation failed!")
            self.logger.error("The last error log from Dymola:")
            self.logger.error(self.dymola.getLastErrorLog())
            raise Exception("Translation failed - Aborting")

    def set_compiler(self, name, path, dll=False, dde=False, opc=False):
        """
        Set up the compiler and compiler options on Windows.
        Optional: Specify if you want to enable dll, dde or opc.

        :param str name:
            Name of the compiler, avaiable options:
            - 'vs': Visual Studio
            - 'gcc': GCC
        :param str,os.path.normpath path:
            Path to the compiler files.
            Example for name='vs': path='C:/Program Files (x86)/Microsoft Visual Studio 10.0/Vc'
            Example for name='gcc': path='C:/MinGW/bin/gcc'
        :param Boolean dll:
            Set option for dll support. Check Dymolas Manual on what this exactly does.
        :param Boolean dde:
            Set option for dde support. Check Dymolas Manual on what this exactly does.
        :param Boolean opc:
            Set option for opc support. Check Dymolas Manual on what this exactly does.
        :return: True, on success.
        """
        # Lookup dict for internal name of CCompiler-Variable
        _name_int = {"vs": "MSVC",
                     "gcc": "GCC"}

        if "win" not in sys.platform:
            raise OSError(f"set_compiler function only implemented "
                          f"for windows systems, you are using {sys.platform}")
        # Manually check correct input as Dymola's error are not a help
        name = name.lower()
        if name not in ["vs", "gcc"]:
            raise ValueError(f"Given compiler name {name} not supported.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Given compiler path {path} does not exist on your machine.")
        # Convert path for correct input
        path = self._make_modelica_normpath(path)
        if self.use_mp:
            raise ValueError("Given function is not yet supported for multiprocessing")

        res = self.dymola.SetDymolaCompiler(name.lower(),
                                            [f"CCompiler={_name_int[name]}",
                                             f"{_name_int[name]}DIR={path}",
                                             f"DLL={int(dll)}",
                                             f"DDE={int(dde)}",
                                             f"OPC={int(opc)}"])

        return res

    def import_initial(self, filepath):
        """
        Load given dsfinal.txt into dymola

        :param str,os.path.normpath filepath:
            Path to the dsfinal.txt to be loaded
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Given filepath {filepath} does not exist")
        if not os.path.splitext(filepath)[1] == ".txt":
            raise TypeError('File is not of type .txt')
        if self.use_mp:
            raise ValueError("Given function is not yet supported for multiprocessing")
        res = self.dymola.importInitial(dsName=filepath)
        if res:
            self.logger.info("Successfully loaded dsfinal.txt")
        else:
            raise Exception("Could not load dsfinal into Dymola.")

    @Model.cd.setter
    def cd(self, cd):
        """Set the working directory to the given path"""
        self._cd = cd
        if self.dymola is None:  # Not yet started
            return
        # Also set the cd in the dymola api
        self.set_dymola_cd(dymola=self.dymola,
                           cd=cd)
        if self.use_mp:
            self.logger.warning("Won't set the cd for all workers, "
                                "not yet implemented.")

    def set_dymola_cd(self, dymola, cd):
        """
        Set the cd of the Dymola Instance.
        Before calling the Function, create the path and
        convert to a modelica-normpath.
        """
        os.makedirs(cd, exist_ok=True)
        cd_modelica = self._make_modelica_normpath(path=cd)
        res = dymola.cd(cd_modelica)
        if not res:
            raise OSError(f"Could not change working directory to {cd}")

    def close(self):
        """Closes dymola."""
        # Close MP of super class
        super().close()
        # Always close main instance
        self._single_close(dymola=self.dymola)

    def _close_multiprocessing(self, _):
        self._single_close()
        DymolaAPI.dymola = None

    def _single_close(self, **kwargs):
        """Closes a single dymola instance"""
        if self.dymola is None:
            return  # Already closed prior
        # Execute the mos-script if given:
        if self.mos_script_post is not None:
            self.logger.info("Executing given mos_script_post "
                             "prior to closing.")
            self.dymola.RunScript(self.mos_script_post)
            self.logger.info("Output of mos_script_post: %s", self.dymola.getLastErrorLog())
        self.logger.info('Closing Dymola')
        self.dymola.close()
        self.logger.info('Successfully closed Dymola')
        self.dymola = None

    def _close_dummy(self):
        """
        Closes dummy instance at the end of the execution
        """
        if self._dummy_dymola_instance is not None:
            self.logger.info('Closing dummy Dymola instance')
            self._dummy_dymola_instance.close()
            self.logger.info('Successfully closed dummy Dymola instance')

    def extract_model_variables(self):
        """
        Extract all variables of the model by
        translating it and then processing the dsin
        using the manipulate_ds module.
        """
        # Translate model
        self.logger.info("Translating model '%s' to extract model variables ",
                         self.model_name)
        self.translate()
        # Get path to dsin:
        dsin_path = os.path.join(self.cd, "dsin.txt")
        df = manipulate_ds.convert_ds_file_to_dataframe(dsin_path)
        # Convert and return all parameters of dsin to initial values and names
        for idx, row in df.iterrows():
            _max = float(row["4"])
            _min = float(row["3"])
            if _min >= _max:
                _var_ebcpy = Variable(value=float(row["2"]))
            else:
                _var_ebcpy = Variable(
                    min=_min,
                    max=_max,
                    value=float(row["2"])
                )
            if row["5"] == "1":
                self.parameters[idx] = _var_ebcpy
            elif row["5"] == "5":
                self.inputs[idx] = _var_ebcpy
            elif row["5"] == "4":
                self.outputs[idx] = _var_ebcpy
            else:
                self.states[idx] = _var_ebcpy

    def _setup_dymola_interface(self, use_mp):
        """Load all packages and change the current working directory"""
        dymola = self._open_dymola_interface()
        self._check_dymola_instances()
        if use_mp:
            cd = os.path.join(self.cd, f"worker_{self.worker_idx}")
        else:
            cd = self.cd
        # Execute the mos-script if given:
        if self.mos_script_pre is not None:
            self.logger.info("Executing given mos_script_pre "
                             "prior to loading packages.")
            dymola.RunScript(self.mos_script_pre)
            self.logger.info("Output of mos_script_pre: %s", dymola.getLastErrorLog())

        # Set the cd in the dymola api
        self.set_dymola_cd(dymola=dymola, cd=cd)

        for package in self.packages:
            self.logger.info("Loading Model %s", os.path.dirname(package).split("\\")[-1])
            res = dymola.openModel(package, changeDirectory=False)
            if not res:
                raise ImportError(dymola.getLastErrorLog())
        self.logger.info("Loaded modules")
        if self.equidistant_output:
            # Change the Simulation Output, to ensure all
            # simulation results have the same array shape.
            # Events can also cause errors in the shape.
            dymola.experimentSetupOutput(equidistant=True,
                                         events=False)
        if not dymola.RequestOption("Standard"):
            warnings.warn("You have no licence to use Dymola. "
                          "Hence you can only simulate models with 8 or less equations.")
        if use_mp:
            DymolaAPI.dymola = dymola
            return None
        return dymola

    def _open_dymola_interface(self):
        """Open an instance of dymola and return the API-Object"""
        if self.dymola_interface_path not in sys.path:
            sys.path.insert(0, self.dymola_interface_path)
        try:
            from dymola.dymola_interface import DymolaInterface
            from dymola.dymola_exception import DymolaConnectionException
            return DymolaInterface(showwindow=self.show_window,
                                   dymolapath=self.dymola_exe_path)
        except ImportError as error:
            raise ImportError("Given dymola-interface could not be "
                              "loaded:\n %s" % self.dymola_interface_path) from error
        except DymolaConnectionException as error:
            raise ConnectionError(error) from error

    def to_dict(self):
        """
        Store the most relevant information of this class
        into a dictionary. This may be used for future configuration.

        :return: dict config:
            Dictionary with keys to re-init this class.
        """
        config = {"cd": self.cd,
                  "packages": self.packages,
                  "model_name": self.model_name,
                  "type": "DymolaAPI",
                  }
        # Update kwargs
        config.update({kwarg: self.__dict__.get(kwarg, None)
                       for kwarg in self._supported_kwargs})

        return config

    @staticmethod
    def _make_modelica_normpath(path):
        """
        Convert given path to a path readable in dymola.
        If the base path does not exist, create it.

        :param str,os.path.normpath path:
            Either a file or a folder path. The base to this
            path is created in non existent.
        :return: str
            Path readable in dymola
        """
        if isinstance(path, pathlib.Path):
            path = str(path)

        path = path.replace("\\", "/")
        # Search for e.g. "D:testzone" and replace it with D:/testzone
        loc = path.find(":")
        if path[loc + 1] != "/" and loc != -1:
            path = path.replace(":", ":/")
        return path

    @staticmethod
    def get_dymola_interface_path(dymola_install_dir):
        """
        Function to get the path of the newest dymola interface
        installment on the used machine

        :param str dymola_install_dir:
            The dymola installation folder. Example:
            "C://Program Files//Dymola 2020"
        :return: str
            Path to the dymola.egg-file
        """
        path_to_egg_file = os.path.normpath("Modelica/Library/python_interface/dymola.egg")
        egg_file = os.path.join(dymola_install_dir, path_to_egg_file)
        if not os.path.isfile(egg_file):
            raise FileNotFoundError(f"The given dymola installation directory "
                                    f"'{dymola_install_dir}' has no "
                                    f"dymola-interface egg-file.")
        return egg_file

    @staticmethod
    def get_dymola_path(dymola_install_dir, dymola_name=None):
        """
        Function to get the path of the dymola exe-file
        on the current used machine.

        :param str dymola_install_dir:
            The dymola installation folder. Example:
            "C://Program Files//Dymola 2020"
        :param str dymola_name:
            Name of the executable. On Windows it is always Dymola.exe, on
            linux just dymola.
        :return: str
            Path to the dymola-exe-file.
        """
        if dymola_name is None:
            if "linux" in sys.platform:
                dymola_name = "dymola"
            elif "win" in sys.platform:
                dymola_name = "Dymola.exe"
            else:
                raise OSError(f"Your operating system {sys.platform} has no default dymola-name."
                              f"Please provide one.")

        bin_64 = os.path.join(dymola_install_dir, "bin64", dymola_name)
        bin_32 = os.path.join(dymola_install_dir, "bin", dymola_name)
        if os.path.isfile(bin_64):  # First check for 64bit installation
            dym_file = bin_64
        elif os.path.isfile(bin_32):  # Else use the 32bit version
            dym_file = bin_32
        else:
            raise FileNotFoundError(
                f"The given dymola installation has not executable at '{bin_32}'. "
                f"If your dymola_path exists, please raise an issue."
            )

        return dym_file

    @staticmethod
    def get_dymola_install_paths(basedir=None):
        """
        Function to get all paths of dymola installations
        on the used machine. Supported platforms are:
        * Windows
        * Linux
        * Mac OS X
        If multiple installation of Dymola are found, the newest version will be returned.
        This assumes the names are sortable, e.g. Dymola 2020, Dymola 2019 etc.

        :param str basedir:
            The base-directory to search for the dymola-installation.
            The default value depends on the platform one is using.
            On Windows it is "C://Program Files" or "C://Program Files (x86)" (for 64 bit)
            On Linux it is "/opt" (based on our ci-Docker configuration
            On Mac OS X "/Application" (based on the default)
        :return: str
            Path to the dymola-installation
        """

        if basedir is None:
            if "linux" in sys.platform:
                basedir = os.path.normpath("/opt")
            elif "win" in sys.platform:
                basedir = os.path.normpath("C:/Program Files")
            elif "darwin" in sys.platform:
                basedir = os.path.normpath("/Applications")
            else:
                raise OSError(f"Your operating system ({sys.platform})does not support "
                              f"a default basedir. Please provide one.")

        syspaths = [basedir]
        # Check if 64bit is installed (Windows only)
        systempath_64 = os.path.normpath("C://Program Files (x86)")
        if os.path.exists(systempath_64):
            syspaths.append(systempath_64)
        # Get all folders in both path's
        temp_list = []
        for systempath in syspaths:
            temp_list += os.listdir(systempath)
        # Filter programs that are not Dymola
        dym_versions = []
        for folder_name in temp_list:
            # Catch both Dymola and dymola folder-names
            if "dymola" in folder_name.lower():
                dym_versions.append(folder_name)
        del temp_list
        # Find the newest version and return the egg-file
        # This sorting only works with a good Folder structure, eg. Dymola 2020, Dymola 2019 etc.
        dym_versions.sort()
        valid_paths = []
        for dym_version in reversed(dym_versions):
            for system_path in syspaths:
                full_path = os.path.join(system_path, dym_version)
                if os.path.isdir(full_path):
                    valid_paths.append(full_path)
        return valid_paths

    def _check_dymola_instances(self):
        """
        Check how many dymola instances are running on the machine.
        Raise a warning if the number exceeds a certain amount.
        """
        # The option may be useful. However the explicit requirement leads to
        # Problems on linux, therefore the feature is not worth the trouble.
        # pylint: disable=import-outside-toplevel
        try:
            import psutil
        except ImportError:
            return
        counter = 0
        for proc in psutil.process_iter():
            try:
                if "Dymola" in proc.name():
                    counter += 1
            except psutil.AccessDenied:
                continue
        if counter >= self._critical_number_instances:
            warnings.warn("There are currently %s Dymola-Instances "
                          "running on your machine!" % counter)

    @staticmethod
    def _alter_model_name(parameters, model_name, structural_params):
        """
        Creates a modifier for all structural parameters,
        based on the modelname and the initalNames and values.

        :param dict parameters:
            Parameters of the simulation
        :param str model_name:
            Name of the model to be modified
        :param list structural_params:
            List of strings with structural parameters
        :return: str altered_modelName:
            modified model name
        """
        # the structural parameter needs to be removed from paramters dict
        new_parameters = parameters.copy()
        model_name = model_name.split("(")[0]  # Trim old modifier
        if parameters == {}:
            return model_name
        all_modifiers = []
        for var_name, value in parameters.items():
            # Check if the variable is in the
            # given list of structural parameters
            if var_name in structural_params:
                all_modifiers.append(f"{var_name}={value}")
                # removal of the structural parameter
                new_parameters.pop(var_name)
        altered_model_name = f"{model_name}({','.join(all_modifiers)})"
        return altered_model_name, new_parameters

    def _check_restart(self):
            """Restart Dymola every n_restart iterations in order to free memory"""

            if self.sim_counter == self.n_restart:
                self.logger.info("Closing and restarting Dymola to free memory")
                self.close()
                self._dummy_dymola_instance = self._setup_dymola_interface(use_mp=False)
                self.sim_counter = 1
            else:
                self.sim_counter += 1



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
    #     res_step = sys.inp_step_read(input_step={
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
    # t0 = time.time()
    # n_sim = 200
    # simulation_setup = {"start_time": 0,
    #                     "stop_time": 3600,
    #                     "output_interval": 100}
    # config_obj = {
    #               'file_path': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/data/HeatPumpSystemWithInput.fmu',
    #               'cd': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/results',  # fixme: if not exists -> pydantic returns error instead of creating it
    #               'sim_setup': simulation_setup,
    #               'input_file': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/data/ThermalZone_input.csv'
    #               }
    #
    # sys = FMU_API(config_obj, n_cpu=4)
    #
    # time_index = np.arange(
    #     sys.sim_setup.start_time,
    #     sys.sim_setup.stop_time,
    #     sys.sim_setup.output_interval
    # )
    #
    # # Apply some sinus function for the outdoor air temperature
    # t_dry_bulb = np.sin(time_index / 3600 * np.pi) * 10 + 263.15
    # df_inputs = TimeSeriesData({"TDryBul": t_dry_bulb}, index=time_index)
    #
    # hea_cap_c = sys.parameters['heaCap.C'].value
    # # Let's alter it from 10% to 1000 % in n_sim simulations:
    # sizings = np.linspace(0.1, 10, n_sim)
    # parameters = []
    # for sizing in sizings:
    #     parameters.append({"heaCap.C": hea_cap_c * sizing})
    #
    # sys.result_names = ["heaCap.T", "TDryBul"]
    #
    # results = sys.simulate(parameters=parameters,
    #                        inputs=df_inputs)
    #
    # print('time', time.time()-t0)
    #
    # # Plot the result
    # fig, ax = plt.subplots(2, sharex=True)
    # ax[0].set_ylabel("TDryBul in K")
    # ax[1].set_ylabel("T_Cap in K")
    # ax[1].set_xlabel("Time in s")
    # ax[0].plot(df_inputs, label="Inputs", linestyle="--")
    # for res, sizing in zip(results, sizings):
    #     ax[0].plot(res['TDryBul'])
    #     ax[1].plot(res['heaCap.T'], label=sizing)
    # for _ax in ax:
    #     _ax.legend(bbox_to_anchor=(1, 1.05), loc="upper left")
    #
    # plt.show()



    """ Dymola """
    aixlib_mo = 'D:/02_workshop/AixLib/AixLib/package.mo'
    simulation_setup = {"start_time": 0,
                        "stop_time": 3600,
                        "output_interval": 100}
    config_obj = {
                  'model_name': 'AixLib.Systems.HeatPumpSystems.Examples.HeatPumpSystem',
                  'cd': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/results',
                  'sim_setup': simulation_setup,
                  'packages': [aixlib_mo]
                  }

    # ######################### Simulation API Instantiation ##########################
    # %% Setup the Dymola-API:
    sys = DymolaAPI(
        config_obj,
        n_cpu=1,
        show_window=True,
        n_restart=-1,
        equidistant_output=False,
        get_structural_parameters=True
        # Only necessary if you need a specific dymola version
        #dymola_path=None,
        #dymola_version=None
    )

    p_el_name = "heatPumpSystem.heatPump.sigBus.PelMea"
    sys.result_names = [p_el_name, 'timTab.y[1]']
    table_name = "myCustomInput"
    file_name = pathlib.Path(aixlib_mo).parent.joinpath("Resources", "my_custom_input.txt")
    time_index = np.arange(
        sys.sim_setup.start_time,
        sys.sim_setup.stop_time,
        sys.sim_setup.output_interval
    )
    # Apply some sinus function for the outdoor air temperature
    internal_gains = np.sin(time_index/3600*np.pi) * 1000
    tsd_input = TimeSeriesData({"InternalGains": internal_gains}, index=time_index)
    # To generate the input in the correct format, use the convert_tsd_to_modelica_txt function:
    filepath = convert_tsd_to_modelica_txt(
        tsd=tsd_input,
        table_name=table_name,
        save_path_file=file_name
    )

    result_time_series = sys.simulate(
        return_option="time_series",
        # Info: You would not need these following keyword-arguments,
        # as we've already created our file above.
        # However, you can also pass the arguments
        # from above directly into the function call:
        inputs=tsd_input,
        table_name=table_name,
        file_name=file_name
    )
    print(type(result_time_series))
    print(result_time_series)
    result_last_point = sys.simulate(
        return_option="last_point"
    )
    print(type(result_last_point))
    print(result_last_point)
