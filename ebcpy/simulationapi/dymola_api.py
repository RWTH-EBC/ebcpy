"""Module containing the DymolaAPI used for simulation
of Modelica-Models."""

import sys
import os
import warnings
import atexit
import psutil
import pandas as pd
from ebcpy import simulationapi
from ebcpy import data_types
from ebcpy.modelica import manipulate_ds
DymolaInterface = None  # Create dummy to later be used for global-import
DymolaConnectionException = None  # Create dummy to later be used for global-import


class DymolaAPI(simulationapi.SimulationAPI):
    """
    Dymola interface class

    :param str,os.path.normpath cd:
        Dirpath for the current working directory of dymola
    :param str model_name:
        Name of the model to be simulated
    :param list packages:
        List with path's to the packages needed to simulate the model
    """

    show_window = False
    get_structural_parameters = True
    # Alter the output-format so all simulations will result in the same array-length
    equidistant_output = True
    _supported_kwargs = ["show_window",
                         "get_structural_parameters",
                         "dymola_path",
                         "dymola_interface_path",
                         "equidistant_output"]
    dymola_path = ""
    _bit_64 = True  # Whether to use 32 bit or not.

    dymola = None
    # Default simulation setup
    sim_setup = {'startTime': 0.0,
                 'stopTime': 1.0,
                 'numberOfIntervals': 0,
                 'outputInterval': 1,
                 'method': 'Dassl',
                 'tolerance': 0.0001,
                 'fixedstepsize': 0.0,
                 'resultFile': 'resultFile',
                 'autoLoad': False,
                 'initialNames': [],
                 'initialValues': []}

    _number_values = ["startTime", "stopTime", "numberOfIntervals",
                      "outputInterval", "tolerance", "fixedstepsize"]

    def __init__(self, cd, model_name, packages, **kwargs):
        """Instantiate class objects."""
        super().__init__(cd, model_name)
        # Get the dymola-install-path:
        _dym_install = self.get_dymola_install_path()
        if _dym_install:
            self.dymola_path = self.get_dymola_path(_dym_install)
            if "bin64" not in self.dymola_path:
                self._bit_64 = False

        # First import the dymola-interface
        if "dymola_interface_path" in kwargs:
            if not kwargs["dymola_interface_path"].endswith(".egg"):
                raise TypeError("Please provide an .egg-file for the dymola-interface.")
            if os.path.isfile(kwargs["dymola_interface_path"]):
                dymola_interface_path = kwargs["dymola_interface_path"]
            else:
                raise FileNotFoundError("Given dymola-interface could not be found.")
        else:
            if _dym_install is None:
                raise FileNotFoundError("Could not find a dymola-interface on your machine.")
            dymola_interface_path = self.get_dymola_interface_path(_dym_install)

        self._global_import_dymola(dymola_interface_path)
        self.packages = packages

        # Update kwargs with regard to what kwargs are supported.
        _not_supported = set(kwargs.keys()).difference(self._supported_kwargs)
        if _not_supported:
            raise KeyError("The following keyword-arguments are not "
                           "supported: \n{}".format(", ".join(list(_not_supported))))

        # By know only supported kwargs are in the dictionary.
        self.__dict__.update(kwargs)

        # List storing structural parameters for later modifying the simulation-name.
        self._structural_params = []
        # Parameter for raising a warning if to many dymola-instances are running
        self._critical_number_instances = 10
        self._setup_dymola_interface(self.show_window)
        # Register this class to the atexit module to always close dymola-instances

    def simulate(self, savepath_files=""):
        """
        Simulate the current setup.
        If simulation terminates without an error and the files should be saved,
        the files are moved to a folder based on the current datetime.
        Returns the filepath of the result-matfile.

        :param str,os.path.normpath savepath_files:
            If path is provided, the relevant simulation results will be saved
            in the given directory.
        :return: str,os.path.normpath filepath:
            Filepath of the result file.
        """
        if self._structural_params:
            warnings.warn("Warning: Currently, the model is re-translating for each simulation.\n"
                          "Check for these parameters: %s" % ", ".join(self._structural_params))
            # Alter the model_name for the next simulation
            self.model_name = self._alter_model_name(self.sim_setup,
                                                     self.model_name, self._structural_params)
        res = self.dymola.simulateExtendedModel(
                    self.model_name,
                    startTime=self.sim_setup['startTime'],
                    stopTime=self.sim_setup['stopTime'],
                    numberOfIntervals=self.sim_setup['numberOfIntervals'],
                    outputInterval=self.sim_setup['outputInterval'],
                    method=self.sim_setup['method'],
                    tolerance=self.sim_setup['tolerance'],
                    fixedstepsize=self.sim_setup['fixedstepsize'],
                    resultFile=self.sim_setup['resultFile'],
                    initialNames=self.sim_setup['initialNames'],
                    initialValues=self.sim_setup['initialValues'])
        if not res[0]:
            self.logger.log("Simulation failed!")
            self.logger.log("The last error log from Dymola:")
            self.logger.log(self.dymola.getLastErrorLog())
            raise Exception("Simulation failed: Look into dslog.txt at {} of the "
                            "simulation.".format(os.path.join(self.cd + "dslog.txt")))

        _save_name_dsres = "{}.mat".format(self.sim_setup["resultFile"])

        if savepath_files:
            if not os.path.isdir(savepath_files):
                os.mkdir(savepath_files)
            for filepath in [_save_name_dsres, "dslog.txt", "dsfinal.txt"]:
                # Delete existing files
                if os.path.isfile(os.path.join(savepath_files, filepath)):
                    os.remove(os.path.join(savepath_files, filepath))
                # Move files
                os.rename(os.path.join(self.cd, filepath),
                          os.path.join(savepath_files, filepath))
        else:
            savepath_files = self.cd

        if self.get_structural_parameters:
            # Get the structural parameters based on the error log
            self._structural_params = self._filter_error_log(self.dymola.getLastErrorLog())

        return os.path.join(savepath_files, _save_name_dsres)

    def set_initial_values(self, initial_values):
        """
        Overwrite inital values

        :param list initial_values:
            List containing initial values for the dymola interface
        """
        self.sim_setup["initialValues"] = list(initial_values)

    def import_initial(self, filepath):
        """
        Load given dsfinal.txt into dymola

        :param str,os.path.normpath filepath:
            Path to the dsfinal.txt to be loaded
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError("Given filepath {} does not exist".format(filepath))
        if not os.path.splitext(filepath)[1] == ".txt":
            raise TypeError('File is not of type .txt')
        res = self.dymola.importInitial(dsName=filepath)
        if res:
            self.logger.log("\nSuccessfully loaded dsfinal.txt")
        else:
            raise Exception("Could not load dsfinal into Dymola.")

    def set_cd(self, cd):
        """Set the working directory to the given path"""
        modelica_normpath = self._make_modelica_normpath(cd)
        # Check if path exists, if not create it.
        if not os.path.exists(modelica_normpath):
            os.mkdir(modelica_normpath)
        res = self.dymola.cd(modelica_normpath)
        if res:
            self.cd = cd
        else:
            raise OSError("Could not change working directory to {}".format(cd))

    def close(self):
        """Closes dymola."""

        # Change so the atexit function works without an error.
        if self.dymola is not None:
            self.dymola.close()
        # Set dymola object to None to avoid further access to it.
        self.dymola = None

    def get_all_tuner_parameters(self):
        """Get all tuner-parameters of the model by
        translating it and then processing the dsin
        using modelicares."""
        # Translate model
        res = self.dymola.translateModel(self.model_name)
        if not res:
            self.logger.log("Translation failed!")
            self.logger.log("The last error log from Dymola:")
            self.logger.log(self.dymola.getLastErrorLog())
            raise Exception("Translation failed!")
        # Get path to dsin:
        dsin_path = os.path.join(self.cd, "dsin.txt")
        df = manipulate_ds.convert_ds_file_to_dataframe(dsin_path)
        # Convert and return all parameters of dsin as a TunerParas-object.
        df = df[df["5"] == "1"]
        names = df.index
        initial_values = pd.to_numeric(df["2"].values)
        # Get min and max-values
        bounds = [(float(df["3"][idx]), float(df["4"][idx])) for idx in df.index]
        try:
            tuner_paras = data_types.TunerParas(list(names),
                                                initial_values,
                                                bounds=bounds)
        except ValueError:
            # Sometimes, not all parameters have bounds. In this case, no bounds
            # are specified.
            tuner_paras = data_types.TunerParas(list(names),
                                                initial_values,
                                                bounds=None)
        return tuner_paras

    def _setup_dymola_interface(self, show_window):
        """Load all packages and change the current working directory"""
        try:
            self.dymola = DymolaInterface(showwindow=show_window,
                                          dymolapath=self.dymola_path)
        except DymolaConnectionException as error:
            raise ConnectionError(error)
        # Register the function now in case of an error.
        atexit.register(self.close)
        self._check_dymola_instances()
        self.set_cd(self.cd)
        for package in self.packages:
            self.logger.log("Loading Model %s" % os.path.dirname(package).split("\\")[-1])
            res = self.dymola.openModel(package, changeDirectory=False)
            if not res:
                raise ImportError(self.dymola.getLastErrorLog())
        self.logger.log("Loaded modules")
        if self.equidistant_output:
            # Change the Simulation Output, to ensure all
            # simulation results have the same array shape.
            # Events can also cause errors in the shape.
            self.dymola.experimentSetupOutput(equidistant=True,
                                              events=False)

    def _check_dymola_instances(self):
        """
        Check how many dymola instances are running on the machine.
        Raise a warning if the number exceeds a certain amount.
        """
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
    def _alter_model_name(sim_setup, model_name, structural_params):
        """
        Creates a modifier for all structural parameters,
        based on the modelname and the initalNames and values.

        :param dict sim_setup:
            Simulation setup dictionary
        :param str model_name:
            Name of the model to be modified
        :param list structural_params:
            List of strings with structural parameters
        :return: str altered_modelName:
            modified model name
        """
        initial_values = sim_setup["initialValues"]
        initial_names = sim_setup["initialNames"]
        model_name = model_name.split("(")[0] # Trim old modifier
        if structural_params == [] or initial_names == []:
            return model_name
        all_modifiers = []
        for structural_para in structural_params:
            # Checks if the structural parameter is inside the initialNames to be altered
            if structural_para in initial_names:
                # Get the location of the parameter for
                # extraction of the corresponding initial value
                k = initial_names.index(structural_para)
                all_modifiers.append("%s = %s" % (structural_para, initial_values[k]))
        altered_model_name = "%s(%s)" % (model_name, ",".join(all_modifiers))
        return altered_model_name

    @staticmethod
    def _filter_error_log(error_log):
        """
        Filters the error log to detect recurring errors or structural parameters.
        Each structural parameter will raise this warning:
        'Warning: Setting n has no effect in model.\n
        After translation you can only set literal start-values\n
        and non-evaluated parameters.'
        Filtering of this string will extract 'n' in the given case.

        :param str error_log:
            Error log from the dymola_interface.getLastErrorLog() function
        :return: str filtered_log:
        """
        _trigger_string = "After translation you can only set " \
                          "literal start-values and non-evaluated parameters"
        structural_params = []
        split_error = error_log.split("\n")
        for i in range(1, len(split_error)):  # First line will never match the string
            if _trigger_string in split_error[i]:
                prev_line = split_error[i - 1]
                prev_line = prev_line.replace("Warning: Setting ", "")
                param = prev_line.replace(" has no effect in model.", "")
                structural_params.append(param)
        return structural_params

    @staticmethod
    def _global_import_dymola(dymola_interface_path):
        sys.path.insert(0, dymola_interface_path)
        global DymolaInterface
        global DymolaConnectionException
        try:
            from dymola.dymola_interface import DymolaInterface
            from dymola.dymola_exception import DymolaConnectionException
        except ImportError:
            raise ImportError("Given dymola-interface could "
                              "not be loaded:\n %s" % dymola_interface_path)

    @staticmethod
    def _make_modelica_normpath(path):
        """
        Convert given path to a path readable in dymola.
        If the path does not exist, create it.

        :param str,os.path.normpath path:
        :return: str
        Path readable in dymola
        """
        if not os.path.isdir(path):
            os.makedirs(path)
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
            "C:\Program Files\Dymola 2020"
        :return: str
            Path to the dymola.egg-file
        """
        path_to_egg_file = os.path.normpath("Modelica/Library/python_interface/dymola.egg")
        egg_file = os.path.join(dymola_install_dir, path_to_egg_file)
        if not os.path.isfile(egg_file):
            raise FileNotFoundError(f"The given dymola installation directory {dymola_install_dir}"
                                    " has no dymola-interface egg-file.")
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
        if os.path.isfile(bin_64): # First check for 64bit installation
            dym_file = bin_64
        elif os.path.isfile(bin_32): # Else use the 32bit version
            dym_file = bin_32
        else:
            raise FileNotFoundError(f"The given dymola file{bin_32} is not found. Either the "
                                    f"dymola_install_dir, or the dymola_name have false values.")

        return dym_file

    @staticmethod
    def get_dymola_install_path(basedir=None):
        """
        Function to get the path of the newest dymola installment
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
        systempath_64 = os.path.normpath("C:\Program Files (x86)")
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
        for dym_version in reversed(dym_versions):
            for system_path in syspaths:
                full_path = os.path.join(system_path, dym_version)
                if os.path.isdir(full_path):
                    return full_path
        # If still inside the function, no interface was found
        return None
