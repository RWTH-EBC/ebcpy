"""Module for classes using a fmu to
simulate models."""

import os
import fmpy
import shutil
import pandas as pd
import numpy as np
from ebcpy import simulationapi
from fmpy.model_description import read_model_description


class FMU_API(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.
    """

    sim_setup = {'startTime': 0.0,
                 'stopTime': 1.0,
                 'numberOfIntervals': 0,
                 'outputInterval': 1,
                 'solver': 'CVode',
                 'initialNames': [],
                 'initialValues': [],
                 'resultNames': [],
                 'timeout': np.inf}

    # Dynamic setup of simulation setup
    _number_values = [key for key, value in sim_setup.items() if
                      (isinstance(value, (int, float)) and not isinstance(value, bool))]

    def __init__(self, cd, model_name):
        """Instantiate class parameters"""
        super().__init__(cd, model_name)
        if not model_name.lower().endswith(".fmu"):
            raise ValueError("{} is not a valid fmu file!".format(model_name))
        # Init instance attributes
        self._unzip_dir = None
        self._fmu_instance = None
        self._model_description = None
        self._fmi_type = None
        self.log_fmu = True

        # Setup the fmu instance
        self.setup_fmu_instance()

    def close(self):
        """
        Closes the fmu.
        :return:
            True on success
        """
        try:
            self._fmu_instance.terminate()
        except Exception:  # This is due to fmpy which does not yield a narrow error
            pass
        try:
            self._fmu_instance.freeInstance()
        except OSError:
            pass
        # Remove the extracted files
        shutil.rmtree(self._unzip_dir)
        self._unzip_dir = None

    def set_cd(self, cd):
        """
        Set current working directory for storing files etc.
        :param str,os.path.normpath cd:
            New working directory
        :return:
        """
        os.makedirs(cd, exist_ok=True)
        self.cd = cd

    def simulate(self, **kwargs):
        """
        Simulate current simulation-setup.

        :param str,os.path.normpath savepath_files:
            Savepath were to store result files of the simulation.
        :keyword Boolean fail_on_error:
            If True, an error in fmpy will trigger an error in this script.
            Default is false
        :return:
            Filepath of the mat-file.
        """
        start_values = {self.sim_setup["initialNames"][i]: value
                        for i, value in enumerate(self.sim_setup["initialValues"])}
        try:
            res = fmpy.simulate_fmu(
                     self._unzip_dir,
                     start_time=self.sim_setup["startTime"],
                     stop_time=self.sim_setup["stopTime"],
                     solver=self.sim_setup["solver"],
                     step_size=self.sim_setup["numberOfIntervals"],
                     relative_tolerance=None,
                     output_interval=self.sim_setup["outputInterval"],
                     record_events=False,  # Used for an equidistant output
                     start_values=start_values,
                     apply_default_start_values=False,  # As we pass start_values already
                     input=None,   # TODO: Add custom input
                     output=self.sim_setup["resultNames"],
                     timeout=self.sim_setup["timeout"],
                     step_finished=None,
                     model_description=self._model_description,
                     fmu_instance=self._fmu_instance,
                     fmi_type=self._fmi_type,
            )
            self._fmu_instance.reset()

        except Exception as error:
            print(f"[SIMULATION ERROR] Error occurred while running FMU: \n {error}")
            if kwargs.get("fail_on_error", False):
                raise error
            return None

        # Reshape result:
        df = pd.DataFrame(res).set_index("time")
        df.index = df.index.astype("float64")

        return df

    def setup_fmu_instance(self):
        """
        Manually set up and extract the data to
        avoid this step in the simulate function
        :return:
        """
        _unzipdir = os.path.join(self.cd,
                                 os.path.basename(self.model_name)[:-4] + "_extracted")
        os.makedirs(_unzipdir, exist_ok=True)
        self._unzip_dir = fmpy.extract(self.model_name,
                                       unzipdir=_unzipdir)
        self._model_description = read_model_description(self._unzip_dir,
                                                         validate=True)

        self._fmi_type = 'CoSimulation' if self._model_description.coSimulation is not None else 'ModelExchange'

        self._fmu_instance = fmpy.instantiate_fmu(
            unzipdir=self._unzip_dir,
            model_description=self._model_description,
            fmi_type=self._fmi_type,
            visible=False,
            debug_logging=False,
            logger=self._custom_logger,
            fmi_call_logger=None,
            use_remoting=False
        )

    def set_initial_values(self, initial_values):
        """
        Overwrite inital values

        :param list initial_values:
            List containing initial values for the dymola interface
        """
        self.sim_setup["initialValues"] = list(initial_values)

    def _custom_logger(self, component, instanceName, status, category, message):
        """ Print the FMU's log messages to the command line (works for both FMI 1.0 and 2.0) """
        label = ['OK', 'WARNING', 'DISCARD', 'ERROR', 'FATAL', 'PENDING'][status]
        if self.log_fmu:
            self.logger.log("[%s] %s" % (label, message.decode("utf-8")))
