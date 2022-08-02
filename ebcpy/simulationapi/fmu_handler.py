from pydantic import BaseModel, Field, validator
import numpy as np
import fmpy
from fmpy.model_description import read_model_description
import pathlib
from abc import abstractmethod
import pydantic
from typing import Union, Optional
from pydantic import FilePath, DirectoryPath
from typing import Dict, Union, TypeVar, Any, List


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
    communication_step_size: float = Field(
        title="communication step size",
        default=1,
        description="step size in which the do_step() function is called"
    )

class ExperimentConfiguration(BaseModel):
    # file_path: Optional[FilePath]
    wd: Optional[DirectoryPath]
    # package: Optional[FilePath]
    # model_name: Optional[str]
    sim_setup: Optional[SimulationSetup]

class ExperimentConfigurationFMU(ExperimentConfiguration):
    file_path: Optional[FilePath]

class FMU_SetupContinuous(SimulationSetup):
    """
    Add's custom setup parameters for simulating FMU_Handler's continuously
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


SimulationSetupClass = TypeVar("SimulationSetupClass", bound=SimulationSetup)
ExperimentConfigurationClass = TypeVar("ExperimentConfigurationClass", bound=ExperimentConfiguration)


class Model:
    def __init__(self, config):
        # read config
        self.config = self._exp_config_class.parse_obj(config)
        self.path = self.config.file_path  # todo: does it make sense to extract the entries of the config again?, how to automate??
        self.wd = self.config.wd
        self.sim_setup = self.config.sim_setup
        # initialize model variables
        self.inputs: Dict[str, Variable] = {}  # Inputs of model
        self.outputs: Dict[str, Variable] = {}  # Outputs of model
        self.parameters: Dict[str, Variable] = {}  # Parameter of model
        self.states: Dict[str, Variable] = {}  # States of model
        # results
        self.result_names = []

        @abstractmethod
        def _read_model_vars(self):
            raise NotImplementedError(f'{self.__class__.__name__}._read_model_vars '
                                      f'function is not defined')


class FMU:

    _exp_config_class : ExperimentConfigurationClass = ExperimentConfigurationFMU

    def __init__(self, n_instances):
        self.n_instances = n_instances  # number of instances of the same fmu for multiprocessing
        self._fmu_instances = None  # Dict of FMU instances
        self._unzip_dirs = None  # List of directories for fmu extraction
        self._var_refs = None  # Dict of variables and their references
        self._model_description = None
        self._fmi_type = None

    def _set_variables(self, var_dict: dict, instance: int = 0):
        """
        Sets multiple variables.
        var_dict is a dict with variable names in keys.
        """

        for key, value in var_dict.items():
            var = self._var_refs[key]
            vr = [var.valueReference]

            if var.type == 'Real':
                self._fmu_instances[instance].setReal(vr, [float(value)])
            elif var.type in ['Integer', 'Enumeration']:
                self._fmu_instances[instance].setInteger(vr, [int(value)])
            elif var.type == 'Boolean':
                self._fmu_instances[instance].setBoolean(vr, [value == 1.0 or value or value == "True"])
            else:
                raise Exception("Unsupported type: %s" % var.type)

    def _read_variables(self, vrs_list: list, instance: int = 0):  # todo: idx_worker not nice
        """
        Reads multiple variable values of FMU_Handler.
        vrs_list as list of strings
        Method returns a dict with FMU_Handler variable names as key
        """

        # initialize dict for results of simulation step
        res = {}

        for name in vrs_list:
            var = self._var_refs[name]
            vr = [var.valueReference]

            if var.type == 'Real':
                res[name] = self._fmu_instances[instance].getReal(vr)[0]
            elif var.type in ['Integer', 'Enumeration']:
                res[name] = self._fmu_instances[instance].getInteger(vr)[0]
            elif var.type == 'Boolean':
                value = self._fmu_instances[instance].getBoolean(vr)[0]
                res[name] = value != 0
            else:
                raise Exception("Unsupported type: %s" % var.type)

        res['SimTime'] = self.current_time

        return res

    def _find_vars(self, start_str: str):
        """
        Returns all variables starting with start_str
        """

        key = list(self._var_refs.keys())
        key_list = []
        for i in range(len(key)):
            if key[i].startswith(start_str):
                key_list.append(key[i])
        return key_list

    def setup_fmu_instance(self):
        """
        Manually set up and extract the data to
        avoid this step in the simulate function.
        """
        self.logger.info("Extracting fmu and reading fmu model description")
        # First load model description and extract variables
        self._single_unzip_dir = os.path.join(self.cd,
                                              os.path.basename(self.fmu_path)[:-4] + "_extracted")
        os.makedirs(self._single_unzip_dir, exist_ok=True)
        self._single_unzip_dir = fmpy.extract(self.fmu_path,
                                         unzipdir=self._single_unzip_dir)
        self._model_description = read_model_description(self._single_unzip_dir,
                                                         validate=True)
        self._read_model_vars()

        # Create dict of variable names with variable references from model description
        self._var_refs = {}
        for variable in self._model_description.modelVariables:
            self._var_refs[variable.name] = variable

        if self._model_description.coSimulation is None:
            self._fmi_type = 'ModelExchange'
        else:
            self._fmi_type = 'CoSimulation'



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

    def _read_model_vars(self):
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


class FMU_stepwise(Model, FMU):

    _sim_setup_class: SimulationSetupClass = FMU_SetupContinuous

    def __init__(self, config):
        Model.__init__(self, config)
        FMU.__init__(self, n_instances=1)  # no mp for stepwise FMU simulation


if __name__ == '__main__':

    config_obj = {
                  'file_path': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/data/ThermalZone_bus.fmu',
                  'wd': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/results',
                  'sim_setup': {"start_time": 0,
                              "stop_time": 100,
                              "output_interval": 10,
                              "communication_step_size": 1}
                  }

    fmu_model = FMU_stepwise(config_obj)


