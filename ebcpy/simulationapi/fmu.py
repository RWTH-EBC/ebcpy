import os
import logging
import pathlib
import shutil
import fmpy
from fmpy.model_description import read_model_description
import numpy as np
from ebcpy.simulationapi import Variable
from ebcpy.simulationapi.config import *


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
        self.log_fmu = log_fmu
        self._var_refs: dict = None  # Dict of variables and their references
        self._model_description = None
        self._fmi_type = None
        self._single_unzip_dir: str = None

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

    def set_variables(self, var_dict: dict):
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

    def read_variables(self, vrs_list: list):  #
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