"""
Module with functions to convert
certain format into other formats.
"""
import os
import scipy.io as spio
from ebcpy import data_types
import numpy as np


def convert_hdf_to_mat(filepath, save_path_file, columns=None, key=None, set_time_to_zero=True, offset=0):
    """
    Function to convert a hdf file to a mat-file readable within Dymola.

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
    :param bool set_time_to_zero: (default True),
        If True, the index, which is the time, will start at a zero base.
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
    >>> success, filepath = convert_hdf_to_mat(example_file, save_path, columns=cols, key=key)
    >>> print(success)
    True
    >>> os.remove(filepath)
    """
    data = data_types.TimeSeriesData(filepath, **{"key": key})
    df = data.get_df().copy()
    if set_time_to_zero:
        df.index = df.index - df.iloc[0].name.to_datetime64()  # Make index zero based
    df['time_vector'] = df.index.total_seconds() + offset  # Copy values of index as seconds into new column
    if not columns:
        columns = df.columns
    # Add column name of time in front of desired columns,
    # since time must be first column in the *.mat file.
    columns = ['time_vector'] + list(columns)
    # Store desired columns in new variable, which is a np.array
    subset = df[columns]
    # Convert np.array into a list and create a dict with 'table' as matrix name
    new_mat = {'table': subset.values.tolist()}
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
    >>> success, filepath = convert_hdf_to_clustering_txt(example_file, save_path, columns=cols, key=key)
    >>> print(success)
    True
    >>> os.remove(filepath)
    """
    data = data_types.TimeSeriesData(filepath, **{"key": key})
    df = data.get_df().copy()
    # Store desired columns in new variable, which is a np.array
    if columns is not None:
        subset = df[columns]
    else:
        subset = df
    # Convert np.array into a list and create a list as matrix name
    subset.values.tolist()
    # Save matrix as a *.txt file, which is readable by TICC.
    np.savetxt(save_path_file, subset, delimiter=',', fmt='%.4f')
    # Provide user feedback whether the conversion was successful.
    return True, save_path_file


def convert_hdf_to_modelica_txt(filepath, table_name, save_path_file=None,
                                columns=None, key=None, offset=0, sep="\t"):
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
        File path and name where to store the output .mat file.
    :param list columns:
        A list with names of columns that should be saved to .mat file.
        If no list is provided, all columns are converted.
    :param str key:
        The name of the dataframe inside the given hdf-file.
        Only needed if multiple tables are stored within tht given file.
    :param float offset:
        Offset for time in seconds, default 0
    :param str sep:
        Separator used to separate values between columns
    :return:

    Examples:

    >>> import os
    >>> project_dir = os.path.dirname(os.path.dirname(__file__))
    >>> example_file = os.path.normpath(project_dir + "//examples//data//example_data.hdf")
    >>> save_path = os.path.normpath(project_dir + "//examples//data//example_data_converted.txt")
    >>> cols = ["sine.y / "]
    >>> key = "trajectories"
    >>> success, filepath = convert_hdf_to_modelica_txt(example_file, "dummy_input_data", columns=cols, key=key)
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

    data = data_types.TimeSeriesData(filepath, **{"key": key})
    df = data.get_df().copy()

    df.index = df.index - df.iloc[0].name.to_datetime64()  # Make index zero based
    df['time_in_s'] = df.index.total_seconds() + offset

    if columns:
        columns.insert(0, 'time_in_s')
        headers = df[columns].columns.values.tolist()
    else:
        headers = df.columns.values.tolist()

    n_cols = len(headers)
    n_rows = len(df.index)
    content_as_lines = ["#" + sep.join(headers)]  # Comment header line
    content_as_lines.insert(0, f"double {table_name}({n_rows}, {n_cols})\n")
    content_as_lines.insert(0, "#1\n")  # Print Modelica table no

    # Open file and write the header
    f = open(file=save_path_file, mode="a+", encoding="utf-8")
    f.seek(0)
    f.truncate()  # Delete possible old content
    f.writelines(content_as_lines)
    f.close()
    # Append the data directly using to_csv from pandas
    df_sub = df[headers]
    df_sub.to_csv(save_path_file, header=None, index=None, sep=sep, mode="a")

    return True, save_path_file


if __name__ == '__main__':
    import doctest
    doctest.testmod()


