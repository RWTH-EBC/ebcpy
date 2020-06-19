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
from datetime import datetime
import modelicares.simres as sr
import numpy as np
import pandas as pd
import ebcpy.modelica.simres as ebc_sr
from ebcpy import preprocessing
# pylint: disable=I1101


class TimeSeriesData(pd.DataFrame):
    """
      Class for handling time series data using a pandas dataframe.
      This class works file-based and makes the import of different
      file-types into a pandas DataFrame more user-friendly.
      Furthermore, functions to support multi-indexing are provided to
      efficiently handle variable passed processing and provide easy
      visualization access.

      :param str,os.path.normpath filepath:
          Filepath ending with either .hdf, .mat or .csv containing
          time-dependent data to be loaded as a pandas.DataFrame
      :keyword str key:
          Name of the table in a .hdf-file if the file
          contains multiple tables.
      :keyword str sep:
          separator for the use of a csv file. If none is provided,
          a comma (",") is used as a default value.
      :keyword str sheet_name:
          Name of the sheet you want to load data from. Required keyword
          argument when loading a xlsx-file.

    """

    def __init__(self, filepath, **kwargs):
        """Initialize class-objects and check correct input."""
        # Two possibles inputs. first argument is actually data provided by pandas
        # and kwargs hold further information or is it an actual filepath.
        if not isinstance(filepath, str):
            super().__init__(data=filepath,
                             index=kwargs.get("index", None),
                             columns=kwargs.get("columns", None),
                             dtype=kwargs.get("dtype", None),
                             copy=kwargs.get("copy", False))
            # Already return as everything is set up
            return
        # Check whether the file exists
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                "The given filepath {} could not be opened".format(filepath))

        # Set the kwargs
        key = kwargs.get("key")
        if key == "":
            key = None  # Avoid cryptic error in pandas by converting empty string to None
        sep = kwargs.get("sep", ",")  # Set default to most common separator, the comma
        sheet_name = kwargs.get("sheet_name")

        # Open based on file suffix.
        # Currently, hdf, csv, and Modelica result files (mat) are supported.
        file_suffix = filepath.split(".")[-1].lower()
        if file_suffix == "hdf":
            # Load the current file as a hdf to a dataframe.
            # As specifying the key can be a problem, the user will
            # get all keys of the file if one is necessary but not provided.
            try:
                _df_loaded = pd.read_hdf(filepath, key=key)
            except (ValueError, KeyError):
                keys = ", ".join(get_keys_of_hdf_file(filepath))
                raise KeyError("key must be provided when HDF5 file contains multiple datasets. "
                               "Here are all keys in the given hdf-file: %s" % keys)
        elif file_suffix == "csv":
            _df_loaded = pd.read_csv(filepath, sep=sep)
        elif file_suffix == "mat":
            sim = sr.SimRes(filepath)
            _df_loaded = ebc_sr.to_pandas(sim, with_unit=False)
        elif file_suffix == "xlsx":
            if sheet_name is None:
                raise KeyError("sheet_name is a required keyword argument to load xlsx-files."
                               "Please pass a string to specify the name "
                               "of the sheet you want to load.")
            _df_loaded = pd.read_excel(io=filepath, sheet_name=sheet_name)
        else:
            raise TypeError("Only .hdf, .csv and .mat are supported!")

        _multi_col_names = ["Variables", "Tags"]
        _default_tag = ["raw"]

        if _df_loaded.columns.nlevels == 1:
            multi_col = pd.MultiIndex.from_product([[var_name for var_name in _df_loaded.columns],
                                                    _default_tag], names=_multi_col_names)
            _df_loaded.columns = multi_col
        elif _df_loaded.columns.nlevels == 2:
            if _df_loaded.columns.names != _multi_col_names:
                raise TypeError("Loaded dataframe has a different 2-Level header format than "
                                "it is supported by this class. The names have to match.")
        else:
            raise TypeError("Only DataFrames with Multi-Columns with 2 "
                            "Levels are supported by this class.")

        super().__init__(_df_loaded)

    @property
    def _constructor(self):
        """Overwrite constructor method according to:
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas"""
        return TimeSeriesData

    def save(self, filepath, **kwargs):
        """
        Save the current time-series-data into the given file-format.
        Currently supported are .hdf (easy and fast storage) and
        .csv (easy-readable).

        :param str,os.path.normpath filepath:
            Filepath were to store the data. Either .hdf or .csv
            has to be the file-ending.
        :keyword str key:
            Necessary keyword-argument for saving a .hdf-file.
            Specifies the key of the table in the .hdf-file.
        :keyword str sep:
            Separator used for saving as .csv. Default is ','.
        :return:
        """
        _df_to_store = pd.DataFrame(self)
        if filepath.lower().endswith(".hdf"):
            _df_to_store.to_hdf(filepath, key=kwargs.get("key"))
        elif filepath.lower().endswith(".csv"):
            _df_to_store.to_csv(filepath, sep=kwargs.get("sep", ","))
        else:
            raise TypeError("Given file-format is not supported."
                            "You can only store TimeSeriesData as .hdf or .csv")

    def get_columns_by_tag(self, tag, columns=None, return_type='pandas'):
        """
        Returning all columns with defined tag in the form of ndarray.
        :return: ndarray of input signals
        """
        #Extract columns
        if columns:
            _ret = self.loc[:, columns]
        else:
            _ret = self

        _ret = _ret.xs(tag, axis=1, level=1)

        # Return based on the given return_type
        if return_type.lower() == 'pandas':
            return _ret
        elif return_type.lower() in ['numpy', 'scipy', 'sp', 'np']:
            return _ret.to_numpy()
        elif return_type.lower() == 'control':
            return _ret.to_numpy().transpose()
        else:
            raise TypeError("Unknown return type")

    def set_data_by_tag(self, data, tag, variables=None):
        """
        Data can be an array for single variables, or a dataframe itself.
        :param data:
        :param str tag:
            New tag for the data
        :param variables:
        :return:
        """
        self.loc[:, (variables, tag)] = data

    def to_datetime_index(self, unit_of_index="s", origin=datetime.now()):
        """
        Convert the current index to a float based index using
        ebcpy.preprocessing.convert_index_to_datetime_index()

        :param str unit_of_index: default 's'
            The unit of the given index. Used to convert to
            total_seconds later on.
        :param datetime.datetime origin:
            The reference datetime object for the first index.
            Default is the current system time.
        """
        preprocessing.convert_index_to_datetime_index(df=self,
                                                      unit_of_index=unit_of_index,
                                                      origin=origin)

    def to_float_index(self, offset=0):
        """
        Convert the current index to a float based index using
        ebcpy.preprocessing.convert_datetime_index_to_float_index()

        :param float offset:
            Offset in seconds
        """
        if not isinstance(self.index, pd.DatetimeIndex):
            return
        preprocessing.convert_datetime_index_to_float_index(df=self,
                                                            offset=offset)

    def clean_and_space_equally(self, desired_freq):
        """
        Call to the preprocessing function
        ebcpy.preprocessing.clean_and_space_equally_time_series()
        See the docstring of this function to know what is happening.

        :param str desired_freq:
            Frequency to determine number of elements in processed dataframe.
            Options are for example:
            - s: second-based
            - 5s: Every 5 seconds
            - 6min: Every 6 minutes
            This also works for h, d, m, y, ms etc.
        """
        df = preprocessing.clean_and_space_equally_time_series(df=self,
                                                               desired_freq=desired_freq)
        super().__init__(df)


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

    def _set_scale(self):
        self._df["scale"] = self._df["max"] - self._df["min"]
        if not self._df[self._df["scale"] <= 0].empty:
            raise ValueError("The given lower bounds are greater equal than the upper bounds,"
                             "resulting in a negative scale: \n{}".format(str(self._df["scale"])))


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
