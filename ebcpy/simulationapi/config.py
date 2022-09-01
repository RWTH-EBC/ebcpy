from pydantic import BaseModel, Field, validator
from pydantic import FilePath, DirectoryPath
from typing import Union, Optional
from typing import TypeVar, List
import numpy as np
import pandas as pd

# pd.DataFrame und TimeSeriesData as type to be validated by pydantic
PandasDataFrameType = TypeVar('pd.DataFrame')  # todo: does this make sense? does it need boudn? does it need typeVar at all?
TimeSeriesDataObjectType = TypeVar('TimeSeriesData')

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

    tolerance: Union[float, None] = Field(
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

    class Config:
        """Overwrite default pydantic Config"""
        extra = 'forbid'



class ExperimentConfigurationFMU(ExperimentConfiguration):
    """
    in case of FMU simulation the fmu file path defines the model
    """
    file_path: Optional[FilePath]


class ExperimentConfigurationFMU_Discrete(ExperimentConfigurationFMU):
    """
    in case of discrete FMU simulation long-term input data can be passed
    """
    input_data: Optional[
        Union[FilePath, PandasDataFrameType, TimeSeriesDataObjectType]]


class ExperimentConfigurationDymola(ExperimentConfiguration):
    """
    in case of a Dymola simulation the package and model name define the model
    """
    packages: Optional[List[FilePath]]
    model_name: Optional[str]


SimulationSetupClass = TypeVar("SimulationSetupClass", bound=SimulationSetup)
ExperimentConfigurationClass = TypeVar("ExperimentConfigurationClass", bound=ExperimentConfiguration)