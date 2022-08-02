"""Module for classes using a fmu to
simulate models."""

import os
import logging
import pathlib
import atexit
import shutil
from typing import List, Union
import fmpy
from fmpy.model_description import read_model_description
from pydantic import Field
import pandas as pd
import numpy as np
from ebcpy import simulationapi, TimeSeriesData
from ebcpy.simulationapi import SimulationSetup, SimulationSetupClass, Variable
from ebcpy.utils.interpolation import interp_df
from typing import Optional
import warnings
# pylint: disable=broad-except


class FMU_Setup_Stepwise(SimulationSetup):
    """
    Add's custom setup parameters for simulating FMU_Handler's stepwise
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


class FMU_API_stepwise(simulationapi.SimulationAPI, simulationapi.FMU_Handler):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.

    :keyword bool log_fmu:
        Whether to print fmu messages or not.

    Example:

    >>> import matplotlib.pyplot as plt
    >>> from ebcpy import FMU_API_continuous
    >>> # Select any valid fmu. Replace the line below if
    >>> # you don't have this file on your device.
    >>> model_name = "Path to your fmu"
    >>> sys_fmu_A = FMU_API_continuous(model_name)
    >>> sys_fmu_A.sim_setup = {"stop_time": 3600}
    >>> result_df = sys_fmu_A.simulate()
    >>> sys_fmu_A.close()
    >>> # Select an exemplary column
    >>> col = result_df.columns[0]
    >>> plt.plot(result_df[col], label=col)
    >>> _ = plt.legend()
    >>> _ = plt.show()

    .. versionadded:: 0.1.7
    """

    _sim_setup_class: SimulationSetupClass = FMU_Setup_Stepwise

    _type_map = {
        float: np.double,
        bool: np.bool_,
        int: np.int_
    }

    def __init__(self, cd, fmu_path, **kwargs):
        """Instantiate class parameters"""
        # Init instance attributes
        self._fmu_instances: dict = {}
        self._unzip_dirs: dict = {}
        # used for stepwise simulation
        self.current_time = None
        self.sim_res = None
        self.finished = None

        if cd is None:
            cd = os.path.dirname(fmu_path)

        simulationapi.FMU_Handler.__init__(self, fmu_path=fmu_path)
        simulationapi.SimulationAPI.__init__(self,
                                             cd=cd,
                                             model_name=self.fmu_path,
                                             ncp=1,  # no mp for stepwise fmu simulation
                                             **kwargs
                                             )

        # Register exit option
        atexit.register(self.close)

    def _update_model(self):
        # Setup the fmu instance
        self.setup_fmu_instance()

    def close(self):
        """
        Closes the fmu.

        :return: bool
            True on success
        """
        # Close MP of super class
        super().close()
        # Close if single process
        if not self.use_mp:
            if not self._fmu_instances:
                return  # Already closed
            self._single_close(fmu_instance=self._fmu_instances[0],
                               unzip_dir=self._unzip_dirs[0])
            self._unzip_dirs = {}
            self._fmu_instances = {}

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
        if unzip_dir is not None:
            try:
                shutil.rmtree(unzip_dir)
            except FileNotFoundError:
                pass  # Nothing to delete
            except PermissionError:
                self.logger.error("Could not delete unzipped fmu "
                                  "in location %s. Delete it yourself.", unzip_dir)

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

    """
    New function: do_step + additional functions related to it: 
    The do_step() function allows to perform a single simulation step of an FMU_Handler. 
    Using the function in a loop, a whole simulation can be conducted. 
    Compared to the simulate() function this offers the possibility for co-simulation with other FMUs 
    or the use of inputs that are dependent from the system behaviour during the simulation (e.g. applying control).
    """

    def do_step(self, automatic_close: bool = False, idx_worker: int = 0):  # todo: idx worker not nice
        """
        perform simulation step; return True if stop time reached
        """

        # check if stop time is reached
        if self.current_time < self.sim_setup.stop_time:
            # do simulation step
            status = self._fmu_instances[idx_worker].doStep(
                currentCommunicationPoint=self.current_time,
                communicationStepSize=self.sim_setup.communication_step_size
                )
            # update current time and determine status
            self.current_time += self.sim_setup.communication_step_size
            self.finished = False
        else:
            self.finished = True
            print('Simulation of FMU_Handler "{}" finished'.format(self._model_description.modelName))
            if automatic_close:
                # close FMU_Handler
                self.close()
                print('FMU_Handler "{}" closed'.format(self._model_description.modelName))
        return self.finished

    def _add_inputs_to_result_names(self):
        """
        Inputs and output variables are added to the result_names (names of variables that are read from the fmu)
        """
        self.result_names.extend(list(self.inputs.keys()))
        print("Added FMU_Handler inputs to the list of variables to read from the fmu")

    def initialize_fmu_for_do_step(self,
                                   parameters: dict = None,
                                   init_values: dict = None,
                                   store_input: bool = True):
        """
        Initialisation of FMU_Handler. To be called before using stepwise simulation
        Parameters and initial values can be set.
        """

        # THE FOLLOWING STEPS OF INITIALISATION ALREADY COVERED BY INSTANTIATING FMU_Handler API:
        # - Read model description
        # - extract .fmu file
        # - Create FMU2 Slave
        # - instantiate fmu instance  # todo: (instead of fmu_instance.instantiate(), instantiate_fmu() is used)??

        # Create dict of variable names with variable references from model description
        self.var_refs = {}
        for variable in self._model_description.modelVariables:
            self.var_refs[variable.name] = variable

        # Check for mp setting
        if self.use_mp:
            raise Exception('Multi processing not available for stepwise FMU_Handler simulation')

        idx_worker = 0

        # Reset FMU_Handler instance
        self._fmu_instances[idx_worker].reset()

        # Set up experiment
        self._fmu_instances[idx_worker].setupExperiment(startTime=self.sim_setup.start_time,
                                                        stopTime=self.sim_setup.stop_time,
                                                        tolerance=self.sim_setup.tolerance)

        # initialize current time for stepwise FMU_Handler simulation
        self.current_time = self.sim_setup.start_time

        # Set parameters and initial values
        if init_values is None:
            init_values = {}
        if parameters is None:
            parameters = {}
        # merge initial values and parameters in one dict as they are treated similarly
        start_values = init_values.copy()
        start_values.update(parameters)

        # write parameters and initial values to FMU_Handler
        self._set_variables(var_dict=start_values)

        # Finalise initialisation
        self._fmu_instances[idx_worker].enterInitializationMode()
        self._fmu_instances[idx_worker].exitInitializationMode()

        # add inputs to result_names
        if store_input:
            self._add_inputs_to_result_names()

        # Initialize dataframe to store results
        self.sim_res = pd.DataFrame(columns=self.result_names)

        # initialize status indicator
        self.finished = False

    def get_results(self, tsd_format: bool = False):
        """
        returns the simulation results either as pd.DataFrame or as TimeSeriesData
        """
        # delete duplicates
        res_clean = self.sim_res
        res_clean['SimTime'] = res_clean.index.to_list()
        res_clean.drop_duplicates(inplace=True)
        res_clean.drop(columns=['SimTime'], inplace=True)
        # check if there is still entries with the same index/time
        index_as_list = res_clean.index.to_list()
        if len(index_as_list) > len(set(index_as_list)):
            raise Exception('The simulation results contain ambigious entries. '
                            'Check the use and order of read_variables() and set_variables()')


        if not tsd_format:
            results = res_clean
        else:
            results = TimeSeriesData(res_clean, default_tag="sim")
            results.rename_axis(['Variables', 'Tags'], axis='columns')
            results.index.names = ['Time']  # todo: in ebcpy tsd example only sometimes
        return results



    def read_variables_wr(self, save_results: bool = True):

        # read results for current time from FMU_Handler
        res_step = self._read_variables(vrs_list=self.result_names)

        # store results in df
        if save_results:
            if self.current_time % self.sim_setup.output_interval == 0:
                self.sim_res = pd.concat([self.sim_res, pd.DataFrame.from_records([res_step],  # because frame.append will be depreciated
                                                                                  index=[res_step['SimTime']],
                                                                                  columns=self.sim_res.columns)])
        return res_step

    def set_variables_wr(self,
                         input_step: dict = None,
                         input_table: pd.DataFrame = None,
                         interp_table: bool = False,
                         do_step: bool = True,
                         automatic_close: bool = False):

        # get input from input table (overwrite with specific input for single step)
        single_input = {}
        if input_table is not None:
            # extract value from input time table
            if isinstance(input_table, TimeSeriesData):
                input_table = input_table.to_df(force_single_index=True)
            # only consider columns in input table that refer to inputs of the FMU_Handler
            input_matches = list(set(self.inputs.keys()).intersection(set(input_table.columns)))
            input_table_filt = input_table[input_matches]
            single_input = interp_df(t=self.current_time, df=input_table_filt, interpolate=interp_table)

        if input_step is not None:
            # overwrite with input for step
            single_input.update(input_step)

        # write inputs to fmu
        if single_input:
            self._set_variables(var_dict=single_input)

        # optional: perform simulation step
        if do_step:
            self.do_step(automatic_close=automatic_close)





