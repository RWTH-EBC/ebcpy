"""
This module provides useful classes for all ebcpy.
Every data_type class should include every parameter
other classes like optimization etc. may need. The checking
of correct input is especially relevant here as the correct
format of data-types will prevent errors during simulations,
optimization etc.
"""

import os
import warnings
from PyQt5 import QtWidgets
import modelicares.simres as sr
import numpy as np
import pandas as pd
from ebcpy.utils import statistics_analyzer
from ebcpy import preprocessing
import ebcpy.modelica.simres as ebc_sr
# pylint: disable=I1101


class TimeSeriesData:
    """
    Base class for time series data in the framework. This class
    provides functions for all it's children.

    :param str,os.path.normpath filepath:
        Filepath ending with either .hdf, .mat or .csv containing
        time-dependent data to be loaded as a pandas.DataFrame
    :keyword str key:
        Name of the table in a .hdf-file if the file
        contains multiple tables.
    :keyword str sep:
        separator for the use of a csv file.
    """

    key, sep = "", ","
    sheet_name = None

    def __init__(self, filepath, **kwargs):
        """Initialize class-objects and check correct input."""
        self.data_type = None
        self.df = pd.DataFrame()
        # Check whether the file exists
        if not os.path.isfile(filepath):
            raise FileNotFoundError("The given filepath {} could not be opened".format(filepath))
        self.filepath = filepath
        # Used for import of .hdf-files, as multiple tables can be stored inside on file.
        supported_kwargs = ["key", "sep", "sheet_name"]
        for keyword in supported_kwargs:
            setattr(self, keyword, kwargs.get(keyword))
        self._load_data()

    def _load_data(self):
        """
        Private function to load the data in the
        file in filepath and convert it to a dataframe.
        """
        # Open based on file suffix.
        # Currently, hdf, csv, and Modelica result files (mat) are supported.
        file_suffix = self.filepath.split(".")[-1].lower()
        if file_suffix == "hdf":
            self._load_hdf()
        elif file_suffix == "csv":
            self.df = pd.read_csv(self.filepath, sep=self.sep)
        elif file_suffix == "mat":
            sim = sr.SimRes(self.filepath)
            self.df = ebc_sr.to_pandas(sim, with_unit=False)
        elif file_suffix == "xlsx":
            self.df = pd.read_excel(io=self.filepath, sheet_name=self.sheet_name)
        else:
            raise TypeError("Only .hdf, .csv and .mat are supported!")

    def _load_hdf(self):
        """
        Load the current file as a hdf to a dataframe.
        As specifying the key can be a problem, the user will
        get all keys of the file if one is necessary but not provided.
        """
        try:
            self.df = pd.read_hdf(self.filepath, key=self.key)
        except (ValueError, KeyError):
            keys = ", ".join(get_keys_of_hdf_file(self.filepath))
            raise KeyError("key must be provided when HDF5 file contains multiple datasets. "
                           "Here are all keys in the given hdf-file: %s" % keys)

    def get_df(self):
        """Returns the dataframe constructed in this class

        :return pd.DataFrame
            DataFrame of this class"""
        return self.df

    def set_df(self, df):
        """Set's the dataframe of this class to the given df

        :param pd.DataFrame df:
            DataFrame to be used as TimeSeriesData
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Given df is of type {} but should be "
                            "of type pd.DataFrame".format(type(df).__name__))
        self.df = df


class MeasTargetData(TimeSeriesData):
    """
    Class for measurement target data. This class is used for all
    measured data (in an experiment), which will be used to evaluate
    the objective function of different classes, like calibration etc.
    """

    def __init__(self, filepath, **kwargs):
        """Set data_type object"""
        # Inherit all objects from TimeSeriesData
        super().__init__(filepath, **kwargs)
        self.data_type = "MeasTargetData"


class MeasInputData(TimeSeriesData):
    """
    Class for measurement input data. Such data is necessary for
    different uses-cases:

    1. Classification: Based on data like mass flow, pump signal etc.
        a classification is made
    2. Sensitivity Analysis and Calibration: The data will ensure that
        the model behaves exactly like the real device in the experiment.
        This is wy it's called input data, as it is an input to the
        simulation
    """
    def __init__(self, filepath, **kwargs):
        """Set data_type object"""
        # Inherit all objects from TimeSeriesData
        super().__init__(filepath, **kwargs)
        self.data_type = "MeasInputData"


class SimTargetData(TimeSeriesData):
    """
    Class for simulation target data. This class is used for all
    simulated data (based on a model), which will be used to evaluate
    the objective function of different classes, like calibration etc.
    The data given to this class should always be a simulation output,
    e.g. a mat-file from a modelica-simulation.
    """
    def __init__(self, filepath, **kwargs):
        """Set data_type object"""
        # Inherit all objects from TimeSeriesData
        super().__init__(filepath, **kwargs)
        self.data_type = "SimTargetData"


class TunerParas:
    """
    Class for tuner parameters.

    :param list names:
        List of names of the tuner parameters
    :param float,int initial_values:
        Initial values for optimization
    :param list,tuple bounds:
        Tuple or list of float or ints for lower and upper bound to the tuner parameter
    """
    def __init__(self, names, initial_values, bounds=None):
        """Initialize class-objects and check correct input."""
        # Check if the given input-parameters are of correct format. If not, raise an error.
        for name in names:
            if not isinstance(name, str):
                raise TypeError("Given name is of type {} and not of "
                                "type str.".format(type(name).__name__))
        try:
            # Calculate the sum, as this will fail if the elements are not float or int.
            sum(initial_values)
        except TypeError:
            raise TypeError("initial_values contains other instances than float or int.")
        if len(names) != len(initial_values):
            raise ValueError("shape mismatch: names has length {} and initial_values "
                             "{}.".format(len(names), len(initial_values)))
        self.bounds = bounds
        if bounds is None:
            _bound_min = -np.inf
            _bound_max = np.inf
        else:
            if len(bounds) != len(names):
                raise ValueError("shape mismatch: bounds has length {} "
                                 "and names {}.".format(len(bounds), len(names)))
            _bound_min, _bound_max = [], []
            for bound in bounds:
                _bound_min.append(bound[0])
                _bound_max.append(bound[1])

        self._df = pd.DataFrame({"names": names,
                                 "initial_value": initial_values,
                                 "min": _bound_min,
                                 "max": _bound_max})
        self._df = self._df.set_index("names")
        self._set_scale()

    def __str__(self):
        """Overwrite string method to present the TunerParas-Object more
        nicely."""
        return str(self._df)

    def scale(self, descaled):
        """
        Scales the given value to the bounds of the tuner parameter between 0 and 1

        :param np.array,list descaled:
            Value to be scaled
        :return: np.array scaled:
            Scaled value between 0 and 1
        """
        # If no bounds are given, scaling is not possible--> descaled = scaled
        if self.bounds is None:
            return descaled
        _scaled = (descaled - self._df["min"])/self._df["scale"]
        if not all((_scaled >= 0) & (_scaled <= 1)):
            warnings.warn("Given descaled values are outside of bounds."
                          "Automatically limiting the values with respect to the bounds.")
        return np.clip(_scaled, a_min=0, a_max=1)

    def descale(self, scaled):
        """
        Converts the given scaled value to an descaled one.

        :param np.array,list scaled:
            Scaled input value between 0 and 1
        :return: np.array descaled:
            descaled value based on bounds.
        """
        # If no bounds are given, scaling is not possible--> descaled = scaled
        if not self.bounds:
            return scaled
        _scaled = np.array(scaled)
        if not all((_scaled >= 0-1e4) & (_scaled <= 1+1e4)):
            warnings.warn("Given scaled values are outside of bounds. "
                          "Automatically limiting the values with respect to the bounds.")
        _scaled = np.clip(_scaled, a_min=0, a_max=1)
        return _scaled*self._df["scale"] + self._df["min"]

    def get_names(self):
        """Return the names of the tuner parameters"""
        return list(self._df.index)

    def get_initial_values(self):
        """Return the initial values of the tuner parameters"""
        return self._df["initial_value"].values

    def get_bounds(self):
        """Return the bound-values of the tuner parameters"""
        return self._df["min"].values, self._df["max"].values

    def get_value(self, name, col):
        """Function to get a value of a specific tuner parameter"""
        return self._df[col][name]

    def set_value(self, name, col, value):
        """Function to set a value of a specific tuner parameter"""
        if not isinstance(value, (float, int)):
            raise ValueError("Given value is of type {} but float or "
                             "int is required".format(type(value).__name__))
        if col not in ["max", "min", "initial_value"]:
            raise KeyError("Can only alter max, min and initial_value")
        self._df[col][name] = value
        self._set_scale()

    def remove_names(self, names):
        """
        Remove gives list of names from the Tuner-parameters

        :param list names:
            List with names inside of the TunerParas-dataframe
        """
        self._df = self._df.loc[~self._df.index.isin(names)]

    def show(self):
        """
        Shows the tuner parameters and stores the altered values to
        the object if wanted.
        """
        import sys
        from ebcpy._io import tuner_paras_gui
        try:
            app = QtWidgets.QApplication(sys.argv)
            main_window = QtWidgets.QMainWindow()
            gui = tuner_paras_gui.TunerParasUI(main_window)
            gui.set_data(self._df[["initial_value", "min", "max"]])
            main_window.show()
            sys.exit(app.exec_())
        except SystemExit:
            tuner_paras = gui.tuner_paras
            if tuner_paras is not None:
                self._df = tuner_paras._df.copy()

    def _set_scale(self):
        self._df["scale"] = self._df["max"] - self._df["min"]
        if not self._df[self._df["scale"] <= 0].empty:
            raise ValueError("The given lower bounds are greater equal than the upper bounds,"
                             "resulting in a negative scale: \n{}".format(str(self._df["scale"])))


class Goals:
    """
    Class for one or multiple goals. Used to evaluate the
    difference between current simulation and measured data

    :param list,str meas_columns:
        List of strings or one string with names to the columns
        inside measured_data.
    :param list,str sim_columns:
        List of strings or one string with names to the columns
        inside simulated_data.
    :param MeasTargetData meas_target_data:
        The dataset to be used as a reference for the simulation output.
    :param SimTargetData, None sim_target_data:
        Class holding the dataframe of the simulated data.
        Can be empty for instantiation, however has to be set before
        calling eval_difference().
    :param list weightings:
        Values between 0 and 1 to account for multiple Goals to be evaluated.
        If multiple goals are selected, and weightings is None, each
        weighting will be equal to 1/(Number of goals).
        The weigthing is scaled so that the sum will equal 1.
    """

    _meas_df = pd.DataFrame()
    _sim_df = pd.DataFrame()

    def __init__(self, meas_columns, sim_columns, meas_target_data,
                 sim_target_data=None, weightings=None):
        """Initialize class-objects and check correct input."""

        # Convert given str to list for identical processing:
        if isinstance(meas_columns, str):
            self._meas_columns = [meas_columns]
        elif isinstance(meas_columns, list):
            self._meas_columns = meas_columns
        else:
            raise TypeError("Given sim_columns is pf type {} but should be "
                            "float or string".format(type(meas_columns).__name__))

        if isinstance(sim_columns, str):
            self._sim_columns = [sim_columns]
        elif isinstance(sim_columns, list):
            self._sim_columns = sim_columns
        else:
            raise TypeError("Given sim_columns is pf type {} but should be "
                            "float or string".format(type(sim_columns).__name__))

        if len(self._meas_columns) != len(self._sim_columns):
            raise ValueError("The given amount of meas_columns ({}) does not equal"
                             "the amount of sim_columns ({}). Can't map goals if the number"
                             "is not equal.".format(len(self._meas_columns),
                                                    len(self._sim_columns)))

        # Open the meas target data:
        if not isinstance(meas_target_data, MeasTargetData):
            raise TypeError("Given meas_target_data is of type {} but MeasTargetData "
                            "is required.".format(type(meas_target_data).__name__))

        self._meas_target_data = meas_target_data
        _meas_keys = list(self._meas_target_data.df.keys())

        _diff_meas = self._get_difference(self._meas_columns, _meas_keys)
        if _diff_meas:
            raise KeyError("Given meas_columns not found in "
                           "meas_target_data:\n{}".format(", ".join(_diff_meas)))

        # Set the weightings, if not specified.
        self._num_goals = len(self._meas_columns)
        if weightings is None:
            self._weightings = np.array([1/self._num_goals for i in range(self._num_goals)])
        else:
            if not isinstance(weightings, (list, np.ndarray)):
                raise TypeError("weightings is of type {} but should be of type"
                                " list.".format(type(weightings).__name__))
            if len(weightings) != self._num_goals:
                raise IndexError("The given number of weightings ({}) does not match the number"
                                 " of goals ({})".format(len(weightings), self._num_goals))
            self._weightings = np.array(weightings) / sum(weightings)

        # Extract the dataframe from the meas_target_data and sim_target_data
        self._meas_df = self._meas_target_data.df

        # Create an array for the goals.
        self._goals = []
        # Eventually set the given sim_target_data
        if sim_target_data is not None:
            self.set_sim_target_data(sim_target_data)

    def _update_goals(self):
        """Function to create or update the goals-list. As
        set_sim_target_data will alter the sim-object of a goal,
        the goals-object has to be created again after calling
        set_sim_target_data."""
        # Create local class Goal
        class Goal:
            """
            Single Goals class.

            :param np.array meas:
                Array with measurement data.
            :param np.array sim:
                Array with simulated data
            :param float weighting:
                Weighting of the Goal.
            """

            def __init__(self, meas, sim, weighting):
                """Instantiate class parameters."""
                self.meas = meas
                self.sim = sim
                self.weighting = weighting

        self._goals.clear()
        for goal_num in range(self._num_goals):
            _meas_data = self._meas_df[self._meas_columns[goal_num]]
            _sim_data = self._sim_df[self._sim_columns[goal_num]]
            _meas_data = preprocessing.build_average_on_duplicate_rows(_meas_data)
            _sim_data = preprocessing.build_average_on_duplicate_rows(_sim_data)
            _weighting = self._weightings[goal_num]
            self._goals.append(Goal(_meas_data,
                                    _sim_data,
                                    _weighting))

    def eval_difference(self, statistical_measure):
        """
        Evaluate the difference of the measurement and simulated data based on the
        given statistical_measure.

        :param str statistical_measure:
            Method supported by statistics_analyzer.StatisticsAnalyzer, e.g. RMSE
        :return: float total_difference
            weighted ouput for all goals.
        """
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer(statistical_measure)
        total_difference = 0
        for goal in self._goals:
            total_difference += goal.weighting * stat_analyzer.calc(goal.meas, goal.sim)
        return total_difference

    def set_sim_target_data(self, sim_target_data):
        """Alter the object self._sim_target_data based on given
        sim_target_data.

        :param SimTargetData sim_target_data:
            Object with simulation target data.
        """
        if not isinstance(sim_target_data, SimTargetData):
            raise TypeError("Given sim_target_data is of type {} but SimTargetData "
                            "is required.".format(type(sim_target_data).__name__))

        _diff = self._get_difference(self._sim_columns, sim_target_data.df.keys())
        if _diff:
            raise KeyError("Given sim_target_data does not contain all required column-keys "
                           "for this Goals object.\nMissing: {}".format(", ".join(_diff)))
        self._sim_target_data = sim_target_data
        self._sim_df = self._sim_target_data.df
        self._update_goals()

    def set_relevant_time_interval(self, start_time, end_time):
        """
        For many calibration-uses cases, different time-intervals of the measured
        and simulated data are relevant. Set the interval to be used with this function.
        This will change both measured and simulated data. Therefore, the eval_difference
        function can be called at every moment.

        :param float start_time:
            Start-time of the relevant time interval
        :param float end_time:
            End-time of the relevant time interval
        """
        _sim_df_ref = self._sim_target_data.get_df().copy()
        _meas_df_ref = self._meas_target_data.get_df().copy()
        self._meas_df = _meas_df_ref.loc[(_meas_df_ref.index >= start_time) &
                                         (_meas_df_ref.index <= end_time)]
        self._sim_df = _sim_df_ref.loc[(_sim_df_ref.index >= start_time)
                                       &
                                       (_sim_df_ref.index <= end_time)]
        self._update_goals()

    def set_meas_target_data(self, meas_target_data):
        """Alter the object self._meas_target_data based on given
        sim_target_data.

        :param MeasTargetData meas_target_data:
            Object with simulation target data."""
        if not isinstance(meas_target_data, MeasTargetData):
            raise TypeError("Given sim_target_data is of type {} but SimTargetData "
                            "is required.".format(type(meas_target_data).__name__))
        _diff = self._get_difference(self._meas_columns, meas_target_data.df.keys())
        if _diff:
            raise KeyError("Given sim_target_data does not contain all required column-keys"
                           " for this Goals object.\nMissing: {}".format(", ".join(_diff)))
        self._meas_target_data = meas_target_data
        self._meas_df = self._meas_target_data.df
        self._update_goals()

    def get_goal_names(self, index_of_goal):
        """Return a dictionary with all relevant names of
        the given index of goal.

        :param int index_of_goal:
            Index of the goals-list
        :returns: dict
            Dict containing name of the simulation data and name
            of the measurement data.
        """
        return {"sim_name": self._sim_columns[index_of_goal],
                "meas_name": self._meas_columns[index_of_goal]}

    def get_goals_list(self):
        """Get the internal list containing all goals."""
        return self._goals

    @staticmethod
    def _get_difference(list_1, list_2):
        return list(set(list_1).difference(list_2))


def get_keys_of_hdf_file(filepath):
    """
    Find all keys in a given hdf-file.

    :param str,os.path.normpath filepath:
        Path to the .hdf-file
    :return: list
        List with all keys in the given file.
    """
    import h5py
    hdf_file = h5py.File(filepath, 'r')
    return list(hdf_file.keys())
