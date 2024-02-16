"""Module containing the DymolaAPI used for simulation
of Modelica-Models."""

import sys
import os
import shutil
import pathlib
import warnings
import atexit
import json
import time
import socket
from pathlib import Path
from contextlib import closing
from typing import Union, List

from pydantic import Field
import pandas as pd

from ebcpy import TimeSeriesData
from ebcpy.modelica import manipulate_ds
from ebcpy.simulationapi import SimulationSetup, SimulationAPI, \
    SimulationSetupClass, Variable
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt


class DymolaSimulationSetup(SimulationSetup):
    """
    Adds ``tolerance`` to the list of possible
    setup fields.
    """
    tolerance: float = Field(
        title="tolerance",
        default=0.0001,
        description="Tolerance of integration"
    )

    _default_solver = "Dassl"
    _allowed_solvers = ["Dassl", "Euler", "Cerk23", "Cerk34", "Cerk45",
                        "Esdirk23a", "Esdirk34a", "Esdirk45a", "Cvode",
                        "Rkfix2", "Rkfix3", "Rkfix4", "Lsodar",
                        "Radau", "Dopri45", "Dopri853", "Sdirk34hw"]


class DymolaAPI(SimulationAPI):
    """
    API to a Dymola instance.

    :param str,Path working_directory:
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

        **Example:** If you have two versions installed at

        - ``C://Program Files//Dymola 2021`` and
        - ``C://Program Files//Dymola 2020x``

        and you want to use Dymola 2020x, specify
        ``dymola_version='Dymola 2020x'``.

        This parameter is overwritten if ``dymola_path`` is specified.
    :keyword str dymola_path:
         Path to the dymola installation on the device. Necessary
         e.g. on linux, if we can't find the path automatically.
         Example: ``dymola_path="C://Program Files//Dymola 2020x"``
    :keyword str dymola_interface_path:
        Direct path to the .egg-file of the dymola interface.
        Only relevant when the dymola_path
        differs from the interface path.
    :keyword str dymola_exe_path:
        Direct path to the dymola executable.
        Only relevant if the dymola installation do not follow
        the official guideline.
    :keyword float time_delay_between_starts:
        If starting multiple Dymola instances on multiple
        cores, a time delay between each start avoids weird
        behaviour, such as requiring to set the C-Compiler again
        as Dymola overrides the default .dymx setup file.
        If you start e.g. 20 instances and specify `time_delay_between_starts=5`,
        each 5 seconds one instance will start, taking in total
        100 seconds. Default is no delay.

    Example:

    >>> import os
    >>> from ebcpy import DymolaAPI
    >>> # Specify the model name
    >>> model_name = "Modelica.Thermal.FluidHeatFlow.Examples.PumpAndValve"
    >>> dym_api = DymolaAPI(working_directory=os.getcwd(),
    >>>                     model_name=model_name,
    >>>                     packages=[],
    >>>                     show_window=True)
    >>> dym_api.sim_setup = {"start_time": 100,
    >>>                      "stop_time": 200}
    >>> dym_api.simulate()
    >>> dym_api.close()

    """
    _sim_setup_class: SimulationSetupClass = DymolaSimulationSetup
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
        "dymola_version",
        "dymola_interface_path",
        "dymola_exe_path",
        "time_delay_between_starts"
    ]

    def __init__(
            self,
            working_directory: Union[Path, str],
            model_name: str,
            packages: List[Union[Path, str]] = None,
            **kwargs
    ):
        """Instantiate class objects."""
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
        self.dymola_interface_path = kwargs.pop("dymola_interface_path", None)
        self.dymola_exe_path = kwargs.pop("dymola_exe_path", None)
        _time_delay_between_starts = kwargs.pop("time_delay_between_starts", 0)
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

        super().__init__(working_directory=working_directory,
                         model_name=model_name,
                         n_cpu=kwargs.pop("n_cpu", 1))

        # First import the dymola-interface
        dymola_path = kwargs.pop("dymola_path", None)
        if dymola_path is not None:
            if not os.path.exists(dymola_path):
                raise FileNotFoundError(f"Given path '{dymola_path}' can not be found on "
                                        "your machine.")
        else:
            # Get the dymola-install-path:
            _dym_installations = self.get_dymola_install_paths()
            if _dym_installations:
                if self.dymola_version:
                    dymola_path = _get_dymola_path_of_version(
                        dymola_installations=_dym_installations,
                        dymola_version=self.dymola_version
                    )
                else:
                    dymola_path = _dym_installations[0]  # 0 is the newest
                self.logger.info("Using dymola installation at %s", dymola_path)
            else:
                if self.dymola_exe_path is None or self.dymola_interface_path is None:
                    raise FileNotFoundError(
                        "Could not find dymola on your machine. "
                        "Thus, not able to find the `dymola_exe_path` and `dymola_interface_path`. "
                        "Either specify both or pass an existing `dymola_path`."
                    )
        if self.dymola_exe_path is None:
            self.dymola_exe_path = self.get_dymola_path(dymola_path)
        self.logger.info("Using dymola.exe: %s", self.dymola_exe_path)
        if self.dymola_interface_path is None:
            self.dymola_interface_path = self.get_dymola_interface_path(dymola_path)
        self.logger.info("Using dymola interface: %s", self.dymola_interface_path)

        self.packages = []
        if packages is not None:
            for package in packages:
                if isinstance(package, Path):
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
            # Use standard port allocation, should always work
            self._dummy_dymola_instance = self._open_dymola_interface(port=-1)
            atexit.register(self._close_dummy)

        # List storing structural parameters for later modifying the simulation-name.
        # Parameter for raising a warning if to many dymola-instances are running
        self._critical_number_instances = 10 + self.n_cpu
        # Register the function now in case of an error.
        if not self.debug:
            atexit.register(self.close)
        if self.use_mp:
            ports = _get_n_available_ports(n_ports=self.n_cpu)
            self.pool.map(
                self._setup_dymola_interface,
                [dict(use_mp=True, port=port, time_delay=i * _time_delay_between_starts)
                 for i, port in enumerate(ports)]
            )
        # For translation etc. always setup a default dymola instance
        self.dymola = self._setup_dymola_interface(dict(use_mp=False))

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
        show_eventlog = kwargs.pop("show_eventlog", False)
        squeeze = kwargs.pop("squeeze", True)
        result_file_name = kwargs.pop("result_file_name", 'resultFile')
        parameters = kwargs.pop("parameters")
        return_option = kwargs.pop("return_option")
        model_names = kwargs.pop("model_names", None)
        inputs = kwargs.pop("inputs", None)
        fail_on_error = kwargs.pop("fail_on_error", True)
        structural_parameters = kwargs.pop("structural_parameters", [])
        table_name = kwargs.pop("table_name", None)
        file_name = kwargs.pop("file_name", None)
        savepath = kwargs.pop("savepath", None)
        if kwargs:
            self.logger.error(
                "You passed the following kwargs which "
                "are not part of the supported kwargs and "
                "have thus no effect: %s.", " ,".join(list(kwargs.keys())))

        # Handle multiprocessing
        if self.use_mp:
            idx_worker = self.worker_idx
            if self.dymola is None:
                # This should not affect #119, as this rarely happens. Thus, the
                # method used in the DymolaInterface should work.
                self._setup_dymola_interface(dict(use_mp=True))

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
            if table_name is None or file_name is None:
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
            dslog_path = self.working_directory.joinpath('dslog.txt')
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
            # Get the working_directory of the current dymola instance
            self.dymola.cd()
            # Get the value and convert it to a 100 % fitting str-path
            dymola_working_directory = str(Path(self.dymola.getLastErrorLog().replace("\n", "")))
            if savepath is None or str(savepath) == dymola_working_directory:
                return os.path.join(dymola_working_directory, _save_name_dsres)
            os.makedirs(savepath, exist_ok=True)
            for filename in [_save_name_dsres]:
                # Copying dslogs and dsfinals can lead to errors,
                # as the names are not unique
                # for filename in [_save_name_dsres, "dslog.txt", "dsfinal.txt"]:
                # Delete existing files
                try:
                    os.remove(os.path.join(savepath, filename))
                except OSError:
                    pass
                # Move files
                shutil.copy(os.path.join(dymola_working_directory, filename),
                            os.path.join(savepath, filename))
                os.remove(os.path.join(dymola_working_directory, filename))
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

    @SimulationAPI.working_directory.setter
    def working_directory(self, working_directory: Union[Path, str]):
        """Set the working directory to the given path"""
        if isinstance(working_directory, str):
            working_directory = Path(working_directory)
        self._working_directory = working_directory
        if self.dymola is None:  # Not yet started
            return
        # Also set the working_directory in the dymola api
        self.set_dymola_cd(dymola=self.dymola,
                           cd=working_directory)
        if self.use_mp:
            self.logger.warning("Won't set the working_directory for all workers, "
                                "not yet implemented.")

    @SimulationAPI.cd.setter
    def cd(self, cd):
        warnings.warn("cd was renamed to working_directory in all classes. Use working_directory instead.", category=DeprecationWarning)
        self.working_directory = cd

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

    def _setup_dymola_interface(self, kwargs: dict):
        """Load all packages and change the current working directory"""
        use_mp = kwargs["use_mp"]
        port = kwargs.get("port", -1)
        time_delay = kwargs.get("time_delay", 0)
        time.sleep(time_delay)
        dymola = self._open_dymola_interface(port=port)
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

    def _open_dymola_interface(self, port):
        """Open an instance of dymola and return the API-Object"""
        if self.dymola_interface_path not in sys.path:
            sys.path.insert(0, self.dymola_interface_path)
        try:
            from dymola.dymola_interface import DymolaInterface
            from dymola.dymola_exception import DymolaConnectionException
            return DymolaInterface(showwindow=self.show_window,
                                   dymolapath=self.dymola_exe_path,
                                   port=port)
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
        # Convert Path to str to enable json-dumping
        config = {"cd": str(self.cd),
                  "packages": [str(pack) for pack in self.packages],
                  "model_name": self.model_name,
                  "type": "DymolaAPI",
                  }
        # Update kwargs
        config.update({kwarg: self.__dict__.get(kwarg, None)
                       for kwarg in self._supported_kwargs})

        return config

    def get_packages(self):
        """
        Get the currently loaded packages of Dymola
        """
        packages = self.dymola.ExecuteCommand(
            'ModelManagement.Structure.AST.Misc.ClassesInPackage("")'
        )
        if packages is None:
            self.logger.error("Could not load packages from Dymola, using self.packages")
            packages = []
            for pack in self.packages:
                pack = Path(pack)
                if pack.name == "package.mo":
                    packages.append(pack.parent.name)
        valid_packages = []
        for pack in packages:
            current_package = f"modelica://{pack}/package.order"
            pack_path = self.dymola.ExecuteCommand(
                f'Modelica.Utilities.Files.loadResource("{current_package}")'
            )
            if not isinstance(pack_path, str):
                self.logger.error("Could not load model resource for package %s", pack)
            if os.path.isfile(pack_path):
                valid_packages.append(Path(pack_path).parent)
        return valid_packages

    def save_for_reproduction(
            self,
            title: str,
            path: Path = None,
            files: list = None,
            save_total_model: bool = True,
            export_fmu: bool = True,
            **kwargs
    ):
        """
        Additionally to the basic reproduction, add info
        for Dymola packages.

        Content which is saved:
        - DymolaAPI configuration
        - Information on Dymola: Version, flags
        - All loaded packages
        - Total model, if save_total_model = True
        - FMU, if export_fmu = True

        :param bool save_total_model:
            True to save the total model
        :param bool export_fmu:
            True to export the FMU of the current model.
        """
        # Local import to require git-package only when called
        from ebcpy.utils.reproduction import ReproductionFile, CopyFile, get_git_information

        if files is None:
            files = []
        # DymolaAPI Info:
        files.append(ReproductionFile(
            filename="Dymola/DymolaAPI_config.json",
            content=json.dumps(self.to_dict(), indent=2)
        ))
        # Dymola info:
        self.dymola.ExecuteCommand("list();")
        _flags = self.dymola.getLastErrorLog()
        dymola_info = [
            self.dymola.ExecuteCommand("DymolaVersion()"),
            str(self.dymola.ExecuteCommand("DymolaVersionNumber()")),
            "\n\n"
        ]
        files.append(ReproductionFile(
            filename="Dymola/DymolaInfo.txt",
            content="\n".join(dymola_info) + _flags
        ))

        # Packages
        packages = self.get_packages()
        package_infos = []
        for pack_path in packages:

            for pack_dir_parent in [pack_path] + list(pack_path.parents):
                repo_info = get_git_information(
                    path=pack_dir_parent,
                    zip_folder_path="Dymola"
                )
                if not repo_info:
                    continue

                files.extend(repo_info.pop("difference_files"))
                pack_path = str(pack_path) + "; " + "; ".join([f"{key}: {value}" for key, value in repo_info.items()])
                break
            package_infos.append(str(pack_path))
        files.append(ReproductionFile(
            filename="Dymola/Modelica_packages.txt",
            content="\n".join(package_infos)
        ))
        # Total model
        if save_total_model:
            _total_model_name = f"Dymola/{self.model_name.replace('.', '_')}_total.mo"
            _total_model = Path(self.cd).joinpath(_total_model_name)
            os.makedirs(_total_model.parent, exist_ok=True)  # Create to ensure model can be saved.
            res = self.dymola.saveTotalModel(
                fileName=str(_total_model),
                modelName=self.model_name
            )
            if res:
                files.append(ReproductionFile(
                    filename=_total_model_name,
                    content=_total_model.read_text()
                ))
                os.remove(_total_model)
            else:
                self.logger.error("Could not save total model: %s",
                                  self.dymola.getLastErrorLog())
        # FMU
        if export_fmu:
            _fmu_path = self._save_to_fmu(fail_on_error=False)
            if _fmu_path is not None:
                files.append(CopyFile(
                    sourcepath=_fmu_path,
                    filename="Dymola/" + _fmu_path.name,
                    remove=True
                ))

        return super().save_for_reproduction(
            title=title,
            path=path,
            files=files,
            **kwargs
        )

    def _save_to_fmu(self, fail_on_error):
        """Save model as an FMU"""
        res = self.dymola.translateModelFMU(
            modelToOpen=self.model_name,
            storeResult=False,
            modelName='',
            fmiVersion='2',
            fmiType='all',
            includeSource=False,
            includeImage=0
        )
        if not res:
            msg = "Could not export fmu: %s" % self.dymola.getLastErrorLog()
            self.logger.error(msg)
            if fail_on_error:
                raise Exception(msg)
        else:
            path = Path(self.cd).joinpath(res + ".fmu")
            return path

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
        if isinstance(path, Path):
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
            self._dummy_dymola_instance = self._setup_dymola_interface(dict(use_mp=False))
            self.sim_counter = 1
        else:
            self.sim_counter += 1


def _get_dymola_path_of_version(dymola_installations: list, dymola_version: str):
    """
    Helper function to get the path associated to the dymola_version
    from the list of all installations
    """
    for dymola_path in dymola_installations:
        if dymola_path.endswith(dymola_version):
            return dymola_path
    # If still here, version was not found
    raise ValueError(
        f"Given dymola_version '{dymola_version}' not found in "
        f"the list of dymola installations {dymola_installations}"
    )


def _get_n_available_ports(n_ports: int, start_range: int = 44000, end_range: int = 44400):
    """
    Get a specified number of available network ports within a given range.

    This function uses socket connections to check the availability of ports within the specified range.
    If the required number of open ports is found, it returns a list of those ports. If not, it raises
    a ConnectionError with a descriptive message indicating the failure to find the necessary ports.

    Parameters:
    - n_ports (int): The number of open ports to find.
    - start_range (int, optional):
        The starting port of the range to check (inclusive).
        Default is 44000.
    - end_range (int, optional):
        The ending port of the range to check (exclusive).
        Default is 44400.

    Returns:
    - list of int:
        A list containing the available ports.
        The length of the list is equal to 'n_ports'.

    Raises:
    - ConnectionError:
        If the required number of open ports cannot
        be found within the specified range.

    Example:

    ```
    try:
        open_ports = _get_n_available_ports(3, start_range=50000, end_range=50500)
        print(f"Found open ports: {open_ports}")
    except ConnectionError as e:
        print(f"Error: {e}")
    ```
    """
    ports = []
    for port in range(start_range, end_range):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
            ports.append(port)
        except OSError:
            pass
        if len(ports) == n_ports:
            return ports
    raise ConnectionError(
        f"Could not find {n_ports} open ports in range {start_range}-{end_range}."
        f"Can't open {n_ports} Dymola instances"
    )
