from pydantic import BaseModel, Field#, validator
from pydantic import FilePath, DirectoryPath
from typing import Union, Optional
from typing import TypeVar, Dict #, Any, List
import numpy as np
from abc import abstractmethod
import pathlib
# import fmpy
# from fmpy.model_description import read_model_description


# import pydantic


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
    communication_step_size: float = Field(
        title="communication step size",
        default=1,
        description="step size in which the do_step() function is called"
    )


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
    communication_step_size: float = Field(
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

    def __init__(self, config, model_name):

        # read config
        self.config = self._exp_config_class.parse_obj(config)
        # initialize sim setup with specific class defaults.
        self._sim_setup = self._sim_setup_class()
        # update sim setup with config entries if given
        if hasattr(self.config, 'sim_setup'):
            self._sim_setup = self.config.sim_setup
        # initialize model variables
        self.inputs: Dict[str, Variable] = {}  # Inputs of model
        self.outputs: Dict[str, Variable] = {}  # Outputs of model
        self.parameters: Dict[str, Variable] = {}  # Parameter of model
        self.states: Dict[str, Variable] = {}  # States of model
        self.model_name = model_name
        # results
        self.result_names = []
        self.sim_res = None  # todo: implement functionality for dym and fmu continuous


class FMU:

    _exp_config_class: ExperimentConfigurationClass = ExperimentConfigurationFMU

    def __init__(self, n_instances):
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
        self.n_instances = n_instances  # number of instances of the same fmu for multiprocessing
        self._fmu_instances: dict = {}  # Dict of FMU instances
        self._unzip_dirs: dict = {}  # Dict of directories for fmu extraction
        self._var_refs: dict = None  # Dict of variables and their references
        self._model_description = None
        self._fmi_type = None
        self._single_unzip_dir: str = None


class FMU_Discrete(FMU, Model):

    _sim_setup_class: SimulationSetupClass = SimulationSetupFMU_Discrete

    def __init__(self, config):
        Model.__init__(self, config=config,
                       model_name=config['file_path'])  # todo: in case of fmu: file path, in case of dym: model_name, find better way to deal with; consider getting rid of model_name. For now it is to make the old methods work
        FMU.__init__(self, n_instances=1)  # no mp for stepwise FMU simulation

        # used for stepwise simulation
        self.current_time = None
        self.finished = None
        # define input data (can be adjusted during simulation using the setter)
        self.input_table = None
        self.interp_input_table = None


if __name__ == '__main__':

    config_obj = {
                  'file_path': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/data/ThermalZone_bus.fmu',
                  'cd': 'D:/pst-kbe/tasks/08_ebcpy_restructure/ebcpy/examples/results',  # fixme: if not exists -> pydantic returns error instead of creating it
                  'sim_setup': {"start_time": 0,
                              "stop_time": 100,
                              "output_interval": 10,
                              "communication_step_size": 2}
                  }

    fmu_model = FMU_Discrete(config_obj)