"""
Module contains pydantic-based models to define experiment configuration and simulation setup
in both dymola and fmu api.
"""

from typing import Union, Optional
from typing import TypeVar, List
from pydantic import BaseModel, Field, validator
from pydantic import FilePath, DirectoryPath
import numpy as np
import pandas as pd
from ebcpy import TimeSeriesData


# ############## Simulation Setup ###########################
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
    Add's custom setup parameters for continuous FMU simulation
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
    Add's custom setup parameters for stepwise/discrete FMU simulation
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


# ############## Experiment Configuration ###########################
class ExperimentConfiguration(BaseModel):
    """
    pydantic BaseModel child to define a full simulation configuration
    """
    cd: Optional[DirectoryPath]  # Dirpath of the fmu or the current working directory of dymola
    sim_setup: Optional[SimulationSetup]

    class Config:
        """Overwrite default pydantic Config"""
        extra = 'forbid'
        arbitrary_types_allowed = True  # to validate pandas dataframe and tsd


class ExperimentConfigFMU_Continuous(ExperimentConfiguration):
    """
    Add's custom parameters for continuous FMU simulation
    to the basic `ExperimentConfiguration`
    """
    sim_setup: Optional[SimulationSetupFMU_Continuous]
    file_path: FilePath


class ExperimentConfigFMU_Discrete(ExperimentConfiguration):
    """
    Add's custom parameters for discrete FMU simulation
    to the basic `ExperimentConfiguration`
    """
    file_path: FilePath
    sim_setup: Optional[SimulationSetupFMU_Discrete]
    input_data: Optional[
        Union[FilePath, pd.DataFrame, TimeSeriesData]]


class ExperimentConfigDymola(ExperimentConfiguration):
    """
    Add's custom parameters for simulating Dymola models
    to the basic `ExperimentConfiguration`
    """
    packages: Optional[List[FilePath]]  # List with path's to the packages needed to simulate the mode
    model_name: Optional[str]  # Name of the model to be simulated
    sim_setup: Optional[SimulationSetupDymola]


SimulationSetupClass = TypeVar("SimulationSetupClass",
                               bound=SimulationSetup)
ExperimentConfigurationClass = TypeVar("ExperimentConfigurationClass",
                                       bound=ExperimentConfiguration)
