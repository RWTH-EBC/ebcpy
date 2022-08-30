import pathlib
from typing import Union
import pandas as pd
from ebcpy import TimeSeriesData
from ebcpy.simulationapi.fmu import FMU
from ebcpy.simulationapi import Model
from ebcpy.simulationapi.config import *
from ebcpy.utils.interpolation import interp_df


class FMU_Discrete(FMU, Model):

    _sim_setup_class: SimulationSetupClass = SimulationSetupFMU_Discrete
    _exp_config_class: ExperimentConfigurationClass = ExperimentConfigurationFMU_Discrete
    objs = []  # to use the close_all method

    def __init__(self, config, log_fmu: bool = True):
        FMU_Discrete.objs.append(self)
        self.use_mp = False  # no mp for stepwise FMU simulation
        self.config = self._exp_config_class.parse_obj(config)
        FMU.__init__(self, log_fmu)
        Model.__init__(self, model_name=self.config.file_path)  # in case of fmu: file path, in case of dym: model_name
        # used for stepwise simulation
        self.current_time = None
        self.finished = None
        # define input data (can be adjusted during simulation using the setter)
        self.input_table = self.config.input_data  # calling the setter to distinguish depending on type
        self.interp_input_table = False  # if false, last value of input table is hold, otherwise interpolated
        self.step_count = None  # counting simulation steps
        self.sim_res = None  # attribute that stores simulation result

    def get_results(self, tsd_format: bool = False):
        """
        returns the simulation results either as pd.DataFrame or as TimeSeriesData
        """
        if not tsd_format:
            results = self.sim_res
        else:
            results = TimeSeriesData(self.sim_res, default_tag="sim")
            results.rename_axis(['Variables', 'Tags'], axis='columns')
            results.index.names = ['Time']
        return results

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
    def input_table(self, inp: Union[FilePath, PandasDataFrameType, TimeSeriesDataObjectType]):
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
            # check unsupported vars:
            self.check_unsupported_variables(self._input_table.columns.to_list(), "inputs")
        else:
            print('No long-term input data set! '
                  'Setter method can still be used to set input data to "input_table" attribute')
            self._input_table = None

    def initialize_discrete_sim(self,
                                parameters: dict = None,
                                init_values: dict = None
                                ):
        """
        Initialisation of FMU. To be called before using stepwise simulation
        Parameters and initial values can be set.
        """

        # THE FOLLOWING STEPS OF FMU INITIALISATION ALREADY COVERED BY INSTANTIATING FMU API:
        # - Read model description
        # - extract .fmu file
        # - Create FMU2 Slave
        # - instantiate fmu instance

        # check if input valid
        if parameters is not None:
            self.check_unsupported_variables(parameters.keys(), "parameters")
        if init_values is not None:
            self.check_unsupported_variables(init_values.keys(), "variables")


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
        start_values.update(parameters)  # todo: is it necessary to distuinguis?

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

    def inp_step_read(self, input_step: dict = None, close_when_finished: bool = False):
        # check for unsupported input
        if input_step is not None:
            self.check_unsupported_variables(input_step.keys(), 'inputs')
        # collect inputs
        # get input from input table (overwrite with specific input for single step)
        single_input = {}
        if self.input_table is not None:
            # extract value from input time table
            # only consider columns in input table that refer to inputs of the FMU
            input_matches = list(set(self.inputs.keys()).intersection(set(self.input_table.columns)))
            input_table_filt = self.input_table[input_matches]  # todo: consider moving this filter to setter for efficiency, if so, inputs must be identified before
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
            if self.current_time % self.sim_setup.output_interval == 0:
                self.sim_res = pd.concat(
                    [self.sim_res, pd.DataFrame.from_records([res],  # because frame.append will be depreciated
                                                             index=[res['SimTime']],
                                                             columns=self.sim_res.columns)])
        else:
            if close_when_finished:
                self.close()
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