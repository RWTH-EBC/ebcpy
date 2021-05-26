"""
Module with functions to convert
certain format into other formats.
"""
import os
import scipy.io as spio
import numpy as np
import pandas as pd
from ebcpy import data_types


def convert_hdf_to_modelica_mat(filepath, save_path_file=None, columns=None,
                                key=None, offset=0):
    """
    Function to convert a hdf file to a mat-file readable within Dymola.

    :param str,os.path.normpath filepath:
        String or even os.path.normpath.
        Must point to a valid hdf file.
    :param str,os.path.normpath save_path_file:
        File path and name where to store the output .mat file.
        If None is provided, the filepath will be used just with the
        .mat-ending
    :param list columns:
        A list with names of columns that should be saved to .mat file.
        If no list is provided, all columns are converted.
    :param str key:
        The name of the dataframe inside the given hdf-file.
        Only needed if multiple tables are stored within tht given file.
    :param float offset:
        Offset for time in seconds, default 0
    :returns mat_file:
        Returns the version 4 mat-file

    Examples:

    >>> import os
    >>> project_dir = os.path.dirname(os.path.dirname(__file__))
    >>> example_file = os.path.normpath(project_dir + "//examples//data//example_data.hdf")
    >>> save_path = os.path.normpath(project_dir + "//examples//data//example_data_converted.mat")
    >>> cols = ["sine.y / "]
    >>> key = "trajectories"
    >>> success, filepath = convert_hdf_to_modelica_mat(example_file,
    >>>                         save_path, columns=cols, key=key)
    >>> print(success)
    True
    >>> os.remove(filepath)
    """
    if save_path_file and not save_path_file.endswith(".mat"):
        raise ValueError("Given savepath for txt-file is not a .mat file!")

    if save_path_file is None:
        # Change file extension
        pre, _ = os.path.splitext(filepath)
        save_path_file = pre + ".mat"

    # Load the relavant part of the df
    df_sub, _ = _convert_hdf_to_df_subset(filepath, key, columns, offset)

    # Convert np.array into a list and create a dict with 'table' as matrix name
    new_mat = {'table': df_sub.values.tolist()}
    # Save matrix as a MATLAB *.mat file, which is readable by Modelica.
    spio.savemat(save_path_file, new_mat, format="4")
    # Provide user feedback whether the conversion was successful.
    return True, save_path_file


def convert_hdf_to_clustering_txt(filepath, save_path_file, columns=None, key=None):
    """
    Function to convert a hdf file to a txt-file readable within the TICC-module.

    :param str,os.path.normpath filepath:
        String or even os.path.normpath.
        Must point to a valid hdf file.
    :param str,os.path.normpath save_path_file:
        File path and name where to store the output .mat file.
    :param list columns:
        A list with names of columns that should be saved to .mat file.
        If no list is provided, all columns are converted.
    :param str key:
        The name of the dataframe inside the given hdf-file.
        Only needed if multiple tables are stored within tht given file.
    :returns True on Success, savepath of txt-file:
        Returns the version 4 mat-file

    Examples:

    >>> import os
    >>> project_dir = os.path.dirname(os.path.dirname(__file__))
    >>> example_file = os.path.normpath(project_dir + "//examples//data//example_data.hdf")
    >>> save_path = os.path.normpath(project_dir + "//examples//data//example_data_converted.txt")
    >>> cols = ["sine.y / "]
    >>> key = "trajectories"
    >>> success, filepath = convert_hdf_to_clustering_txt(example_file,
    >>>                         save_path, columns=cols, key=key)
    >>> print(success)
    True
    >>> os.remove(filepath)
    """
    # Get the subset of the dataFrame
    df_sub, _ = _convert_hdf_to_df_subset(filepath, key, columns, offset=0)

    # Convert np.array into a list and create a list as matrix name
    df_sub.values.tolist()
    # Save matrix as a *.txt file, which is readable by TICC.
    np.savetxt(save_path_file, df_sub, delimiter=',', fmt='%.4f')
    # Provide user feedback whether the conversion was successful.
    return True, save_path_file


def convert_hdf_to_modelica_txt(filepath, table_name, save_path_file=None,
                                columns=None, key=None, offset=0, sep="\t",
                                with_tag=True):
    """
    Convert a hdf file to modelica readable text. This is especially useful
    for generating input data for a modelica simulation.

    :param str,os.path.normpath filepath:
        String or even os.path.normpath.
        Must point to a valid hdf file.
    :param str table_name:
        Name of the table for modelica.
        Needed in Modelica to correctly load the file.
    :param str,os.path.normpath save_path_file:
        File path and name where to store the output .txt file.
        If None is provided, the filepath will be used just with the
        .txt-ending
    :param list columns:
        A list with names of columns that should be saved to .mat file.
        If no list is provided, all columns are converted.
    :param str key:
        The name of the dataframe inside the given hdf-file.
        Only needed if multiple tables are stored within the given file.
    :param float offset:
        Offset for time in seconds, default 0
    :param str sep:
        Separator used to separate values between columns
    :param Boolean with_tag:
        Use True each variable and tag is written to the file
        If False, only the variable name is written to the file.
    :return:

    Examples:

    >>> import os
    >>> project_dir = os.path.dirname(os.path.dirname(__file__))
    >>> example_file = os.path.normpath(project_dir + "//examples//data//example_data.hdf")
    >>> save_path = os.path.normpath(project_dir + "//examples//data//example_data_converted.txt")
    >>> cols = ["sine.y / "]
    >>> key = "trajectories"
    >>> success, filepath = convert_hdf_to_modelica_txt(example_file,
    >>>                              "dummy_input_data", columns=cols, key=key)
    >>> print(success)
    True
    >>> os.remove(filepath)
    """
    if save_path_file and not save_path_file.endswith(".txt"):
        raise ValueError("Given savepath for txt-file is not a .txt file!")

    if save_path_file is None:
        # Change file extension
        pre, _ = os.path.splitext(filepath)
        save_path_file = pre + ".txt"
    # Load the relavant part of the df
    df_sub, headers = _convert_hdf_to_df_subset(filepath, key, columns, offset)

    n_cols = len(headers)
    n_rows = len(df_sub.index)
    # Comment header line
    _temp_str = ""
    if with_tag:
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

    return True, save_path_file


def _convert_hdf_to_df_subset(filepath, key, columns, offset):
    """
    Private function to ensure lean conversion to either mat or txt.
    """
    df = data_types.TimeSeriesData(filepath, key=key)

    if columns:
        headers = df[columns].columns.values.tolist()
    else:
        headers = df.columns.values.tolist()

    _time_header = ('time', 'in_s')
    headers.insert(0, _time_header)  # Ensure time will be at first place

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index - df.iloc[0].name.to_datetime64()  # Make index zero based
        df[_time_header] = df.index.total_seconds() + offset
    elif isinstance(df.index, pd.Float64Index):
        df[_time_header] = df.index - df.iloc[0].name + offset
    # Avoid 1e-8 errors in timedelta calculation.
    df[_time_header] = df[_time_header].round(4)

    # Check if nan values occur
    if df.loc[:, headers].isnull().values.sum() > 0:
        raise ValueError("Selected columns contain NaN values. This would lead to errors"
                         "in the simulation environment.")

    return df.loc[:, headers], headers


if __name__ == '__main__':
    PROJECT_DIR = r"D:\02_git\ebcpy\ebcpy"
    EXAMPLE_FILE = os.path.normpath(PROJECT_DIR + "//examples//data//example_data.hdf")
    SAVE_PATH = os.path.normpath(PROJECT_DIR + "//examples//data//example_data_converted.txt")
    COLS = ["sine.y / "]
    KEY = "trajectories"
    SUCCESS, FILEPATH = convert_hdf_to_modelica_txt(EXAMPLE_FILE,
                                                    "dummy_input_data",
                                                    columns=COLS,
                                                    key=KEY,
                                                    offset=0)
