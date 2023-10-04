"""Functions to manipulate (or extract information of) the
dsfinal.txt and dsin.txt files created by Modelica."""

from io import StringIO
import re

import pandas as pd


def convert_ds_file_to_dataframe(filename):
    """
    Function to convert a given dsfinal or dsfin file to a DataFrame.
    The index is the name of the variable. Further,
    the following columns are used analog to the dsfinal:
    column 1: Type of initial value:
    = -2: special case: for continuing simulation (column 2 = value)
    = -1: fixed value (column 2 = fixed value)
    =  0: free value, i.e., no restriction (column 2 = initial value)
    >  0: desired value (column 1 = weight for optimization,
    column 2 = desired value)
    use weight=1, since automatic scaling usually
    leads to equally weighted terms
    column 2: fixed, free or desired value according to column 1.
    column 3: Minimum value (ignored, if Minimum >= Maximum).
    column 4: Maximum value (ignored, if Minimum >= Maximum).
    Minimum and maximum restrict the search range in initial
    value calculation. They might also be used for scaling.
    column 5: Category of variable:
    = 1: parameter.
    = 2: state.
    = 3: state derivative.
    = 4: output.
    = 5: input.
    = 6: auxiliary variable.
    column 6: Data type of variable and flags according to dsBaseType

    :param str,os.path.normpath filename:
        Filepath to the dsfinal or dsinto be loaded.
    :return: pd.DataFrame
        Converted DataFrame
    """
    # Open file and get relevant content by splitting the lines.
    with open(filename, "r") as file:
        content = file.read().split("\n")

    # Gets the X out of 'char initialName(X,Y)'
    pattern_size_initial_name = r'^char initialName\((\d+),(\d+)\)$'
    for number_line_initial_name, line in enumerate(content):
        match_size_initial_name = re.match(pattern_size_initial_name, line)
        if match_size_initial_name:
            size_initial_names = int(match_size_initial_name.string.split("(")[-1].split(",")[0])
            break
    else:
        raise ValueError("Could not find initial names in file")

    # Number of line below line "double initialValue(X,Y)"
    number_line_initial_value = number_line_initial_name + size_initial_names + 3
    if "double initialValue" in content[number_line_initial_value]:
        number_line_initial_value += 1

    # Check if two or on-line dsfinal / dsin
    if "#" in content[number_line_initial_value]:
        step_size = 1
    else:
        step_size = 2

    # trim content:
    ini_val_list = content[number_line_initial_value: (number_line_initial_value +
                                                       step_size * size_initial_names)]
    # Alter list to create one-lined list.
    if step_size == 2:
        ini_val_list_one_line = []
        for i in range(0, len(ini_val_list), 2):
            # Concat two line into one line
            ini_val_list_one_line.append(ini_val_list[i] + ini_val_list[i + 1])
    else:
        ini_val_list_one_line = ini_val_list

    # Convert to DataFrame. Use a csv-method as it is much faster.
    out = StringIO()
    # csv_writer = writer(out)
    out.write("1;2;3;4;5;6;initialName\n")
    # df = pd.DataFrame({1:[], 2:[], 3:[], 4:[], 5:[], 6:[], "initialName":[]})
    for line in ini_val_list_one_line:
        # Get only the string of the initialName
        ini_name = line.split("#")[-1].replace(" ", "").replace("\n", "")
        vals = line.split("#")[0].split()
        vals.append(ini_name)
        out.write(";".join(vals) + "\n")
    out.seek(0)
    df = pd.read_csv(out, header=0, sep=";", dtype=object)
    df = df.set_index("initialName")
    return df


def eliminate_parameters_from_ds_file(filename, savepath, exclude_paras, del_aux_paras=True):
    """
    Create a new dsfinal file out of the given dsfinal.txt
    All parameters except those listed in exclude_paras
    will be eliminated from the dsfinal file.
    Used for continuing of simulation in calibration problems.

    :param str,os.path.normpath filename:
        Filepath to the dsfinal or dsin file
    :param str,os.path.normpath savepath:
        .txt-file for storing output of this function
    :param list exclude_paras:
        List of parameters to exclude.
    :param bool del_aux_paras:
        Whether to delete auxiliary parameters or not.
        Default value is True.
    """

    # Check types
    if not savepath.endswith(".txt"):
        raise TypeError('File %s is not of type .txt' % savepath)
    if not isinstance(exclude_paras, list):
        raise TypeError(f"Given exclude_paras is of type {type(exclude_paras).__name__} "
                        f"but should be of type list")

    df = convert_ds_file_to_dataframe(filename)

    # Manipulate DataFrame
    if del_aux_paras:
        # Delete all rows with a parameter or an auxiliary value
        df = df[(df["5"] != "1") & (df["5"] != "6") | [idx in exclude_paras for idx in df.index]]
    else:
        df = df[(df["5"] != "1") | [idx in exclude_paras for idx in df.index]]

    # Generate string out of trimmed DataFrame
    longest_name = len(max(df.index, key=len))
    char_initial_name = "char initialName(%s,%s)" % (len(df.index), longest_name)
    char_initial_name += "\n" + "\n".join(df.index)
    double_initial_value = "double initialValue(%s,6)" % len(df.index)  # Always 6
    for index, row in df.iterrows():
        double_initial_value += "\n" + " ".join(row) + " # " + index

    # Create resulting dsFinal string
    string_new_ds_final = char_initial_name + "\n\n" + double_initial_value

    # Reuses the experiment, tuning parameters etc. settings
    number_line_initial_name = 104  # Line where the string char initialName(,) is always stored
    with open(filename, "r") as file:
        content = file.read().split("\n")

    new_content = "\n".join(content[:number_line_initial_name])
    new_content += "\n" + string_new_ds_final
    # Save new content to given savepath
    with open(savepath, "a+") as file:
        file.seek(0)
        file.truncate()  # Delete all content of the given file
        file.write(new_content)
