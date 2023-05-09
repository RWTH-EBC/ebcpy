"""
Module with functions to convert
certain format into other formats.
"""
import pathlib
import scipy.io as spio
import numpy as np
import pandas as pd

from ebcpy.data_types import index_is_numeric, datetime_indexes


def convert_tsd_to_modelica_mat(tsd, save_path_file, **kwargs):
    """
    Function to convert a tsd to a mat-file readable within Dymola.

    :param TimeSeriesData tsd:
        TimeSeriesData object
    :param str,os.path.normpath save_path_file:
        File path and name where to store the output .mat file.
    :keyword list columns:
        A list with names of columns that should be saved to .mat file.
        If no list is provided, all columns are converted.
    :keyword float offset:
        Offset for time in seconds, default 0
    :returns mat_file:
        Returns the version 4 mat-file

    :return:
        str,os.path.normpath:
            Path where the data is saved.
            Equal to save_path_file

    Examples:

    >>> import os
    >>> from ebcpy import TimeSeriesData
    >>> project_dir = os.path.dirname(os.path.dirname(__file__))
    >>> example_file = os.path.normpath(project_dir + "//tests//data//example_data.csv")
    >>> save_path = os.path.normpath(project_dir + "//tests//data//example_data_converted.mat")
    >>> cols = ["sine.freqHz / Hz"]
    >>> tsd = TimeSeriesData(example_file, sep=";")
    >>> filepath = convert_tsd_to_modelica_mat(tsd,
    >>>                                        save_path, columns=cols)
    >>> os.remove(filepath)
    """
    if isinstance(save_path_file, pathlib.Path):
        save_path_file = str(save_path_file)

    if not save_path_file.endswith(".mat"):
        raise ValueError("Given savepath for txt-file is not a .mat file!")

    # Load the relevant part of the df
    df_sub, _ = _convert_to_subset(
        df=tsd,
        columns=kwargs.get("columns", None),
        offset=kwargs.get("offset", 0)
    )

    # Convert np.array into a list and create a dict with 'table' as matrix name
    new_mat = {'table': df_sub.values.tolist()}
    # Save matrix as a MATLAB *.mat file, which is readable by Modelica.
    spio.savemat(save_path_file, new_mat, format="4")
    # Provide user feedback whether the conversion was successful.
    return save_path_file


def convert_tsd_to_clustering_txt(tsd, save_path_file, columns=None):
    """
    Function to convert a TimeSeriesData object
    to a txt-file readable within the TICC-module.

    :param TimeSeriesData tsd:
        TimeSeriesData object
    :param str,os.path.normpath save_path_file:
        File path and name where to store the output .mat file.
    :param list columns:
        A list with names of columns that should be saved to .mat file.
        If no list is provided, all columns are converted.
    :returns True on Success, savepath of txt-file:
        Returns the version 4 mat-file

    :return:
        str,os.path.normpath:
            Path where the data is saved.
            Equal to save_path_file

    Examples:

    >>> import os
    >>> project_dir = os.path.dirname(os.path.dirname(__file__))
    >>> example_file = os.path.normpath(project_dir + "//tests//data//example_data.csv")
    >>> save_path = os.path.normpath(project_dir + "//tests//data//example_data_converted.txt")
    >>> cols = ["sine.freqHz / Hz"]
    >>> tsd = TimeSeriesData(example_file, sep=";")
    >>> filepath = convert_tsd_to_clustering_txt(tsd,
    >>>                                          save_path, columns=cols)
    >>> os.remove(filepath)
    """
    # Get the subset of the dataFrame
    df_sub, _ = _convert_to_subset(df=tsd, columns=columns, offset=0)

    # Convert np.array into a list and create a list as matrix name
    df_sub.values.tolist()
    # Save matrix as a *.txt file, which is readable by TICC.
    np.savetxt(save_path_file, df_sub, delimiter=',', fmt='%.4f')
    # Provide user feedback whether the conversion was successful.
    return save_path_file


def convert_tsd_to_modelica_txt(tsd, table_name, save_path_file, **kwargs):
    """
    Convert a TimeSeriesData object to modelica readable text. This is especially useful
    for generating input data for a modelica simulation.

    :param TimeSeriesData tsd:
        TimeSeriesData object
    :param str table_name:
        Name of the table for modelica.
        Needed in Modelica to correctly load the file.
    :param str,os.path.normpath save_path_file:
        File path and name where to store the output .txt file.
    :keyword list columns:
        A list with names of columns that should be saved to .mat file.
        If no list is provided, all columns are converted.
    :keyword float offset:
        Offset for time in seconds, default 0
    :keyword str sep:
        Separator used to separate values between columns
    :keyword Boolean with_tag:
        Use True each variable and tag is written to the file
        If False, only the variable name is written to the file.

    :return:
        str,os.path.normpath:
            Path where the data is saved.
            Equal to save_path_file

    Examples:

    >>> import os
    >>> from ebcpy import TimeSeriesData
    >>> project_dir = os.path.dirname(os.path.dirname(__file__))
    >>> example_file = os.path.normpath(project_dir + "//tests//data//example_data.csv")
    >>> save_path = os.path.normpath(project_dir + "//tests//data//example_data_converted.txt")
    >>> cols = ["sine.freqHz / Hz"]
    >>> tsd = TimeSeriesData(example_file, sep=";")
    >>> filepath = convert_tsd_to_modelica_txt(tsd, "dummy_input_data", columns=cols)
    >>> os.remove(filepath)
    """
    if isinstance(save_path_file, pathlib.Path):
        save_path_file = str(save_path_file)
    if not save_path_file.endswith(".txt"):
        raise ValueError("Given savepath for txt-file is not a .txt file!")

    # Load the relavant part of the df
    df_sub, headers = _convert_to_subset(
        df=tsd,
        columns=kwargs.get("columns", None),
        offset=kwargs.get("offset", 0)
    )

    # Unpack kwargs
    sep = kwargs.get("sep", "\t")

    n_cols = len(headers)
    n_rows = len(df_sub.index)
    # Comment header line
    _temp_str = ""
    if kwargs.get("with_tag", True):
        # Convert ("variable", "tag") to "variable_tag"
        _temp_str = sep.join(["_".join(variable_tag) for variable_tag in headers])
    else:
        for idx, var in enumerate(headers):
            if idx == 0:
                # Convert time with tag to one string as unit is important
                _temp_str += "_".join(var)
            else:
                # Convert ("variable", "tag") to "variable"
                _temp_str += sep + var[0]
    content_as_lines = [f"#{_temp_str}\n"]
    content_as_lines.insert(0, f"double {table_name}({n_rows}, {n_cols})\n")
    content_as_lines.insert(0, "#1\n")  # Print Modelica table no

    # Open file and write the header
    with open(file=save_path_file, mode="a+", encoding="utf-8") as file:
        file.seek(0)
        file.truncate()  # Delete possible old content
        file.writelines(content_as_lines)

    # Append the data directly using to_csv from pandas
    df_sub.to_csv(save_path_file, header=None, index=None, sep=sep, mode="a")

    return save_path_file


def _convert_to_subset(df, columns, offset):
    """
    Private function to ensure lean conversion to either mat or txt.
    """
    df = df.copy()
    if columns:
        headers = df[columns].columns.values.tolist()
    else:
        headers = df.columns.values.tolist()

    _time_header = ('time', 'in_s')
    headers.insert(0, _time_header)  # Ensure time will be at first place

    if isinstance(df.index, tuple(datetime_indexes)):
        df.index = df.index - df.iloc[0].name.to_datetime64()  # Make index zero based
        df[_time_header] = df.index.total_seconds() + offset
    elif index_is_numeric(df.index):
        df[_time_header] = df.index - df.iloc[0].name + offset
    else:
        # Should not happen as error is raised in data_types. But just to be sure:
        raise IndexError(f"Given index of type {type(df.index)} is not supported.")
    # Avoid 1e-8 errors in timedelta calculation.
    df[_time_header] = df[_time_header].round(4)

    # Check if nan values occur
    if df.loc[:, headers].isnull().values.sum() > 0:
        raise ValueError("Selected columns contain NaN values. This would lead to errors"
                         "in the simulation environment.")

    return df.loc[:, headers], headers
