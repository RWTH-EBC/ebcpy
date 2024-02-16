"""
This module provides useful classes for all ebcpy.
Every data_type class should include every parameter
other classes like optimization etc. may need. The checking
of correct input is especially relevant here as the correct
format of data-types will prevent errors during simulations,
optimization etc.
"""

import os
from pathlib import Path
from typing import List, Union, Any
from datetime import datetime
from pandas.core.internals import BlockManager
import pandas as pd
import numpy as np
import ebcpy.modelica.simres as sr
from ebcpy import preprocessing

# pylint: disable=I1101
# pylint: disable=too-many-ancestors

__all__ = ['TimeSeries',
           'TimeSeriesData',
           'numeric_indexes',
           'datetime_indexes']

numeric_index_dtypes = [
    pd.Index([], dtype=dtype).dtype for dtype in
    ["int8", "int16", "int32", "int64",
     "uint8", "uint16", "uint32", "uint64",
     "float32", "float64"]
]

datetime_indexes = [
    pd.DatetimeIndex
]


def index_is_numeric(index: pd.Index):
    """Check if pandas Index is numeric"""
    return isinstance(index, pd.RangeIndex) or index.dtype in numeric_index_dtypes


class TimeSeriesData(pd.DataFrame):
    """
    Most data related to energy and building
    climate related problems is time-variant.

    Class for handling time series data using a pandas dataframe.
    This class works file-based and makes the import of different
    file-types into a pandas DataFrame more user-friendly.
    Furthermore, functions to support multi-indexing are provided to
    efficiently handle variable passed processing and provide easy
    visualization and preprocessing access.

    :param str,os.path.normpath,pd.DataFrame data:
        Filepath ending with either .hdf, .mat, .csv, .parquet,
        or .parquet.COMPRESSION_NAME containing
        time-dependent data to be loaded as a pandas.DataFrame.
        Alternative option is to pass a DataFrame directly.
    :keyword str key:
        Name of the table in a .hdf-file if the file
        contains multiple tables.
    :keyword str sep:
        separator for the use of a csv file. If none is provided,
        a comma (",") is used as a default value.
        See pandas.read_csv() docs for further information.
    :keyword int, list header:
        Header columns for .csv files.
        See pandas.read_csv() docs for further information.
        Default is first row (0).
    :keyword int,str index_col:
        Column to be used as index in .csv files.
        See pandas.read_csv() docs for further information.
        Default is first column (0).
    :keyword str sheet_name:
        Name of the sheet you want to load data from. Required keyword
        argument when loading a xlsx-file.
    :keyword str default_tag:
        Which value to use as tag. Default is 'raw'
    :keyword str engine:
        Chose the engine for reading .parquet files. Default is 'pyarrow'
        Other option is 'fastparquet' (python>=3.9).
    :keyword list variable_names:
        List of variable names to load from .mat file. If you
        know which variables you want to plot, this may speed up
        loading significantly, and reduce memory size drastically.

    Examples:

    First let's see the usage for a common dataframe.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from ebcpy import TimeSeriesData
    >>> df = pd.DataFrame({"my_variable": np.random.rand(5)})
    >>> tsd = TimeSeriesData(df)
    >>> tsd.to_datetime_index()
    >>> tsd.save("my_new_data.csv")

    Now, let's load the recently created file.
    As we just created the data, we specify the tag
    'sim' to indicate it is some sort of simulated value.

    >>> tsd = TimeSeriesData("my_new_data.csv", tag='sim')
    """

    # normal properties
    _metadata = [
        "_filepath",
        "_loader_kwargs",
        "_default_tag",
        "_multi_col_names"
    ]

    def __init__(self, data: Union[str, Any], **kwargs):
        """Initialize class-objects and check correct input."""
        # Initialize as default
        self._filepath = None
        self._loader_kwargs = {}
        self._multi_col_names = ["Variables", "Tags"]

        self._default_tag = kwargs.pop("default_tag", "raw")
        if not isinstance(self._default_tag, str):
            raise TypeError(f"Invalid type for default_tag! Expected 'str' but "
                            f"received {type(self._default_tag)}")

        # Two possibles inputs. first argument is actually data provided by pandas
        # and kwargs hold further information or is it an actual filepath.
        if isinstance(data, BlockManager):
            super().__init__(data=data)
            return

        if not isinstance(data, (str, Path)):
            _df_loaded = pd.DataFrame(data=data,
                                      index=kwargs.get("index", None),
                                      columns=kwargs.get("columns", None),
                                      dtype=kwargs.get("dtype", None),
                                      copy=kwargs.get("copy", False))
        else:
            file = Path(data)
            self._loader_kwargs = kwargs.copy()
            _df_loaded = self._load_df_from_file(file=file)
            self._filepath = file

        if _df_loaded.columns.nlevels == 1:
            # Check if first level is named Tags.
            # If so, don't create MultiIndex-DF as the method is called by the pd constructor
            if _df_loaded.columns.name != self._multi_col_names[1]:
                multi_col = pd.MultiIndex.from_product(
                    [_df_loaded.columns, [self._default_tag]],
                    names=self._multi_col_names
                )
                _df_loaded.columns = multi_col

        elif _df_loaded.columns.nlevels == 2:
            if _df_loaded.columns.names != self._multi_col_names:
                raise TypeError("Loaded dataframe has a different 2-Level "
                                "header format than it is supported by this "
                                "class. The names have to match.")
        else:
            raise TypeError("Only DataFrames with Multi-Columns with 2 "
                            "Levels are supported by this class.")

        super().__init__(_df_loaded)

    @property
    def _constructor(self):
        """Overwrite constructor method according to:
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas"""
        return TimeSeriesData

    @property
    def _constructor_sliced(self):
        """Overwrite constructor method according to:
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas"""
        return TimeSeries

    @property
    def filepath(self) -> str:
        """Get the filepath associated with the time series data"""
        return self._filepath

    @filepath.setter
    def filepath(self, filepath: str):
        """Set the filepath associated with the time series data"""
        self._filepath = Path(filepath)

    @property
    def default_tag(self) -> str:
        """Get the default of time series data object"""
        return self._default_tag

    @default_tag.setter
    def default_tag(self, tag: str) -> None:
        """Set the default_tag of the time series data object
        :param tag: new tag
        :type tag: String
        """
        if not isinstance(tag, str):
            raise TypeError(f"Invalid type for default_tag! Expected 'str' but "
                            f"received {type(tag)}")
        if tag not in self.get_tags():
            raise KeyError(f"Tag '{tag}' does not exist for current data set!"
                           f"\n Available tags: {self.get_tags()}")
        self._default_tag = tag

    def save(self, filepath: str = None, **kwargs) -> None:
        """
        Save the current time-series-data into the given file-format.
        Currently supported are .hdf, which is an easy and fast storage,
        and, .csv is supported as an easy-readable option.
        Also, .parquet, and with additional compression .parquet.COMPRESSION_NAME
        are supported. Compressions could be gzip, brotli or snappy. For all possible
        compressions see the documentation of the parquet engines.
        For a small comparison of these data formats see https://github.com/RWTH-EBC/ebcpy/issues/81

        :param str,os.path.normpath filepath:
            Filepath were to store the data. Either .hdf, .csv, .parquet
            or .parquet.COMPRESSION_NAME has to be the file-ending.
            Default is current filepath of class.
        :keyword str key:
            Necessary keyword-argument for saving a .hdf-file.
            Specifies the key of the table in the .hdf-file.
        :keyword str sep:
            Separator used for saving as .csv. Default is ','.
        :keyword str engine:
            Chose the engine for reading .parquet files. Default is 'pyarrow'
            Other option is 'fastparquet' (python>=3.9).
        :return:
        """
        # If new settings are needed, update existing ones
        self._loader_kwargs.update(kwargs)
        # Set filepath if not given
        if filepath is None:
            filepath = self.filepath
        else:
            filepath = Path(filepath)
        # Check if filepath is still None (if no filepath was used in init)
        if filepath is None:
            raise ValueError("Current TimeSeriesData instance "
                             "has no filepath, please specify one.")
        # Save based on file suffix
        if filepath.suffix == ".hdf":
            if "key" not in kwargs:
                raise KeyError("Argument 'key' must be "
                               "specified to save a .hdf file")
            pd.DataFrame(self).to_hdf(filepath, key=kwargs.get("key"))

        elif filepath.suffix == ".csv":
            pd.DataFrame(self).to_csv(filepath, sep=kwargs.get("sep", ","))
        elif ".parquet" in filepath.name:
            parquet_split = filepath.name.split(".parquet")
            pd.DataFrame(self).to_parquet(filepath, engine=kwargs.get('engine', 'pyarrow'),
                                          compression=parquet_split[-1][1:] if parquet_split[-1] else None,
                                          index=True)
        else:
            raise TypeError("Given file-format is not supported."
                            "You can only store TimeSeriesData as .hdf, .csv, .parquet, "
                            "and .parquet.COMPRESSION_NAME with additional compression options")

    def to_df(self, force_single_index=False):
        """
        Return the dataframe version of the current TimeSeriesData object.
        If all tags are equal, the tags are dropped.
        Else, the object is just converted.

        :param bool force_single_index:
            If True (not the default), the conversion to a standard
            DataFrame with a single index column (only variable names)
            is only done if no variable contains multiple tags.
        """
        if len(self.get_variables_with_multiple_tags()) == 0:
            return pd.DataFrame(self.droplevel(1, axis=1))
        if force_single_index:
            raise IndexError(
                "Can't automatically drop all tags "
                "as the following variables contain multiple tags: "
                f"{' ,'.join(self.get_variables_with_multiple_tags())}. "
            )
        return pd.DataFrame(self)

    def _load_df_from_file(self, file):
        """Function to load a given filepath into a dataframe"""
        # Check whether the file exists
        if not os.path.isfile(file):
            raise FileNotFoundError(
                f"The given filepath {file} could not be opened")

        # Open based on file suffix.
        # Currently, hdf, csv, and Modelica result files (mat) are supported.
        if file.suffix == ".hdf":
            # Load the current file as a hdf to a dataframe.
            # As specifying the key can be a problem, the user will
            # get all keys of the file if one is necessary but not provided.
            key = self._loader_kwargs.get("key")
            if key == "":
                key = None  # Avoid cryptic error in pandas by converting empty string to None
            try:
                df = pd.read_hdf(file, key=key)
            except (ValueError, KeyError) as error:
                keys = ", ".join(get_keys_of_hdf_file(file))
                raise KeyError(f"key must be provided when HDF5 file contains multiple datasets. "
                               f"Here are all keys in the given hdf-file: {keys}") from error
        elif file.suffix == ".csv":
            # Check if file was previously a TimeSeriesData object
            with open(file, "r") as _f:
                lines = [_f.readline() for _ in range(2)]
                if (lines[0].startswith(self._multi_col_names[0]) and
                        lines[1].startswith(self._multi_col_names[1])):
                    _hea_def = [0, 1]
                else:
                    _hea_def = 0

            df = pd.read_csv(
                file,
                sep=self._loader_kwargs.get("sep", ","),
                index_col=self._loader_kwargs.get("index_col", 0),
                header=self._loader_kwargs.get("header", _hea_def)
            )
        elif file.suffix == ".mat":
            df = sr.mat_to_pandas(
                fname=file,
                with_unit=False,
                names=self._loader_kwargs.get("variable_names")
            )
        elif file.suffix in ['.xlsx', '.xls', '.odf', '.ods', '.odt']:
            sheet_name = self._loader_kwargs.get("sheet_name")
            if sheet_name is None:
                raise KeyError("sheet_name is a required keyword argument to load xlsx-files."
                               "Please pass a string to specify the name "
                               "of the sheet you want to load.")
            df = pd.read_excel(io=file, sheet_name=sheet_name)
        elif ".parquet" in file.name:
            df = pd.read_parquet(path=file, engine=self._loader_kwargs.get('engine', 'pyarrow'))
        else:
            raise TypeError("Only .hdf, .csv, .xlsx and .mat are supported!")
        if not isinstance(df.index, tuple(datetime_indexes)) and not index_is_numeric(df.index):
            try:
                df.index = pd.DatetimeIndex(df.index)
            except Exception as err:
                raise IndexError(
                    f"Given data has index of type {type(df.index)}. "
                    f"Currently only numeric indexes and the following are supported:"
                    f"{' ,'.join([str(idx) for idx in [pd.RangeIndex] + datetime_indexes])} "
                    f"Automatic conversion to pd.DateTimeIndex failed"
                    f"see error above."
                ) from err
        return df

    def get_variable_names(self) -> List[str]:
        """
        Return an alphabetically sorted list of all variables

        :return: List[str]
        """
        return sorted(self.columns.get_level_values(0).unique())

    def get_variables_with_multiple_tags(self) -> List[str]:
        """
        Return an alphabetically sorted list of all variables
        that contain more than one tag.

        :return: List[str]
        """
        var_names = self.columns.get_level_values(0)
        return sorted(var_names[var_names.duplicated()])

    def get_tags(self, variable: str = None) -> List[str]:
        """
        Return an alphabetically sorted list of all tags

        :param str variable:
            If given, tags of this variable are returned

        :return: List[str]
        """
        if variable:
            tags = self.loc[:, variable].columns
            return sorted(tags)
        return sorted(self.columns.get_level_values(1).unique())

    def get_columns_by_tag(self,
                           tag: str,
                           variables: list = None,
                           return_type: str = 'pandas',
                           drop_level: bool = False):
        """
        Returning all columns with defined tag in the form of ndarray.

        :param str tag:
            Define the tag which return columns have to
            match.
        :param list variables:
            Besides the given tag, specify the
            variables names matching the return criteria as well.
        :param boolean drop_level:
            If tag should be included in the response.
            Default is True.
        :param str return_type:
            Return format. Options are:
            - pandas (pd.series)
            - numpy, scipy, sp, and np (np.array)
            - control (transposed np.array)
        :return: ndarray of input signals
        """
        # Extract columns
        if variables:
            _ret = self.loc[:, variables]
        else:
            _ret = self

        _ret = _ret.xs(tag, axis=1, level=1, drop_level=drop_level)

        # Return based on the given return_type
        if return_type.lower() == 'pandas':
            return _ret
        if return_type.lower() in ['numpy', 'scipy', 'sp', 'np']:
            return _ret.to_numpy()
        if return_type.lower() == 'control':
            return _ret.to_numpy().transpose()
        raise TypeError("Unknown return type")

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

    def low_pass_filter(self, crit_freq, filter_order, variable,
                        tag=None, new_tag="low_pass_filter"):
        """
        Call to the preprocessing function
        ebcpy.preprocessing.low_pass_filter()
        See the docstring of this function to know what is happening.

        :param float crit_freq:
            The critical frequency or frequencies.
        :param int filter_order:
            The order of the filter
        :param str variable:
            The variable name to apply the filter to
        :param str tag:
            If this variable has more than one tag, specify which one
        :param str new_tag:
            The new tag to pass to the variable.
            Default is 'low_pass_filter'
        """
        if tag is None:
            data = self.loc[:, variable].to_numpy()
        else:
            data = self.loc[:, (variable, tag)].to_numpy()

        result = preprocessing.low_pass_filter(
            data=data,
            filter_order=filter_order,
            crit_freq=crit_freq
        )
        self.loc[:, (variable, new_tag)] = result

    def moving_average(self, window, variable,
                       tag=None, new_tag="moving_average"):
        """
        Call to the preprocessing function
        ebcpy.preprocessing.moving_average()
        See the docstring of this function to know what is happening.

        :param int window:
            sample rate of input
        :param str variable:
            The variable name to apply the filter to
        :param str tag:
            If this variable has more than one tag, specify which one
        :param str new_tag:
            The new tag to pass to the variable.
            Default is 'low_pass_filter'
        """
        if tag is None:
            data = self.loc[:, variable].to_numpy()
        else:
            data = self.loc[:, (variable, tag)].to_numpy()

        result = preprocessing.moving_average(
            data=data,
            window=window,
        )
        self.loc[:, (variable, new_tag)] = result

    def number_lines_totally_na(self):
        """
        Returns the number of rows in the given dataframe
        that are filled with NaN-values.
        """
        return preprocessing.number_lines_totally_na(self)

    @property
    def frequency(self):
        """
        The frequency of the time series data.
        Returns's the mean and the standard deviation of
        the index.

        :returns:
            float: Mean value
            float: Standard deviation
        """
        return preprocessing.get_df_index_frequency_mean_and_std(
            df_index=self.index
        )


class TimeSeries(pd.Series):
    """Overwrites pd.Series to enable correct slicing
    and expansion in the TimeSeriesData class

    .. versionadded:: 0.1.7
    """

    @property
    def _constructor(self):
        """Overwrite constructor method according to:
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas"""
        return TimeSeries

    @property
    def _constructor_expanddim(self):
        """Overwrite constructor method according to:
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas"""
        return TimeSeriesData


def get_keys_of_hdf_file(filepath):
    """
    Find all keys in a given hdf-file.

    :param str,os.path.normpath filepath:
        Path to the .hdf-file
    :return: list
        List with all keys in the given file.
    """
    # pylint: disable=import-outside-toplevel
    try:
        import h5py
        with h5py.File(filepath, 'r') as hdf_file:
            return list(hdf_file.keys())
    except ImportError:
        return ["ERROR: Could not obtain keys as h5py is not installed"]
