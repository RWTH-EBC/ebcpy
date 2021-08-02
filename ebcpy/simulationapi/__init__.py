"""
Simulation APIs help you to perform automated
simulations for energy and building climate related models.
Parameters can easily be updated, and the initialization-process is
much more user-friendly than the provided APIs by Dymola or fmpy.
"""

import os
import warnings
from typing import Dict, Union, TypeVar
from pydantic import BaseModel, Field, validator
from abc import abstractmethod
from ebcpy.utils import setup_logger


class SimulationSetup(BaseModel):
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
        description="The solver to be used for numerical integration."
    )
    _default_solver = None
    _allowed_solvers = []

    @validator("solver", always=True)
    def check_valid_solver(cls, solver):
        if solver is None:
            return cls._default_solver
        if solver not in cls._allowed_solvers:
            raise ValueError("Given solver is not supported!")
        return solver

    class Config:
        extra = 'forbid'
        underscore_attrs_are_private = True


SimulationSetupClass = TypeVar("SimulationSetupClass", bound=SimulationSetup)


class SimulationAPI:
    """Base-class for simulation apis. Every simulation-api class
    must inherit from this class. It defines the structure of each class.

    :param str,os.path.normpath cd:
        Working directory path
    :param str model_name:
        Name of the model being simulated."""

    _sim_setup_class: SimulationSetupClass = SimulationSetup

    def __init__(self, cd, model_name):
        self._sim_setup = self._sim_setup_class()
        self.cd = cd
        self.model_name = model_name
        # Setup the logger
        self.logger = setup_logger(cd=cd, name=self.__class__.__name__)
        self.logger.info(f'{"-" * 25}Initializing class {self.__class__.__name__}{"-" * 25}')
        self.inputs = []      # Inputs of model
        self.outputs = []     # Outputs of model
        self.parameters = []  # Parameter of model
        self.states = []      # States of model

    @abstractmethod
    def close(self):
        """Base function for closing the simulation-program."""
        raise NotImplementedError(f'{self.__class__.__name__}.close function is not defined')

    @abstractmethod
    def simulate(self, **kwargs):
        """Base function for simulating the simulation-model."""
        raise NotImplementedError(f'{self.__class__.__name__}.simulate function is not defined')

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

    def set_cd(self, cd):
        """Base function for changing the current working directory."""
        warnings.warn("Function will be removed in future versions. "
                      "Use the property setter function instead of this setter: "
                      "sim_api.cd = cd", DeprecationWarning)
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
