"""
This package aims to help manipulate simulation files (dsfinal.txt or dsin.txt)
or to load simulation result files (.mat) efficiently into a pandas.DataFrame
"""
import re
from typing import Union, List


def get_expressions(filepath_model: str,
                    get_protected: bool = False,
                    modelica_type: Union[str, List] = "parameters",
                    excludes: List = None):
    """
    This function extracts specific expressions out of modelica models.

    :param str,os.path.normpath filepath_model:
        Full path of modelica model on the given os
        e.g. path_model = "C://MyLibrary//TestModel.mo"
    :param str,list modelica_type:
        Type you want to have matched. "parameters" and "variables"
        have a special regex pattern.
        For other models, you can parse a string like:
        "replaceable package Medium" and it will yield all
        afflicted lines. You can also give a list of strings if
        multiple strings are relevant to you.
        Special cases:
        parameters:
        - include: ["parameter"]
        - excludes: ["final", "in", "of", "replaceable"]
        variables: Note: The case for already imported SIUnits is not considered here.
        - include: ["Modelica.SIunits", "Real", "Boolean", "Integer"]
        - excludes: ["parameter", "import", "constant"]
    :param list excludes:
        List of strings to exclude from expression. Default is None.
    :param Boolean get_protected:
        Whether to extract protected parameters or not. Default is false

    :return: list matches
        List with all lines matching the given expression.
    """
    if excludes is None:
        excludes = []
    if modelica_type == "parameters":
        _includes = ["parameter"]
        _excludes = ["final", "in", "of", "replaceable"] + excludes
    elif modelica_type == "variables":
        _includes = ["Modelica.SIunits", "Real", "Boolean", "Integer"]
        _excludes = ["parameter", "import", "constant"] + excludes
    else:
        _includes = [modelica_type]
        _excludes = excludes
    if _excludes:
        _exclude_str = r"(?<!" + r"\s)(?<!".join(_excludes) + r"\s)"
    else:  # Case if list is empty
        _exclude_str = ""
    _pattern = r'((?:\s.+)?{}({})(.|\n)*?;)'.format(_exclude_str,
                                                    "|".join(_includes))

    # Open the file
    with open(filepath_model, "r") as file:
        file.seek(0)
        script = file.read()

    # Find desired expression in modelica script
    expr = re.findall(_pattern, script, re.MULTILINE)

    expr_filtered = [" ".join(expr_unfiltered[0].split()) for expr_unfiltered in expr]

    if not get_protected:
        return expr_filtered

    # Get position of expressions
    pos_expr = []
    for match in re.finditer(_pattern, script, re.MULTILINE):
        pos_expr.append(match.span()[0])

    # Check if parameter are protected
    expr_unprotected = []
    expr_protected = []

    # Get position of "protected"-expression
    protected = re.search(r'protected', script)

    if protected:
        pos_protected = protected.span()[0]
        for i, temp_expr in enumerate(expr):
            # If expressions is before 'proteceted' --> refer to expr_unprotected
            if pos_expr[i] < pos_protected:
                expr_unprotected.append(temp_expr)
            # If expressions is after 'proteceted' --> refer to expr_protected
            else:
                expr_protected.append(temp_expr)
    else:
        expr_unprotected = expr

    return expr_unprotected, expr_protected


def get_names_and_values_of_lines(lines: List[str]) -> dict:
    """
    All unnecessary code is deleted (annotations, doc).
    Only the name of the variable and the value is extracted.

    :param List[str] lines:
        List of strings with lines from a modelica file.

    :return:
        dict: Containing the names as key and values as value.

    Example:

    >>> lines = ['parameter Boolean my_boolean=true "Some description"',
    >>>          'parameter Real my_real=12.0 "Some description" annotation("Some annotation")']
    >>> output = get_names_and_values_of_lines(lines=lines)
    >>> print(output)
    {'my_boolean': True, 'my_real': 12.0}
    """
    res = {}
    for line in lines:
        line = line.replace(";", "")

        # Check if line is a commented line and if so, skip the line:
        if line.startswith("//"):
            continue

        # Remove part behind possible annotation:
        loc = line.find("annotation")
        if loc >= 0:
            line = line[:loc]
        # Remove possible brackets, like "param(min=0, start=5)
        line = re.sub(r'[\(\[].*?[\)\]]', '', line)
        # And now any quotes / doc / strings
        line = re.sub(r'".*"', '', line)
        # If a value is present (e.g. for parameters, one = sign is still present (always)
        if line.find("=") >= 0:
            name_str, val_str = line.split("=")
            name_str = name_str.strip()
            name = name_str.split(" ")[-1].replace(" ", "")
            val_str_stripped = val_str.replace(" ", "")
            if val_str_stripped in ["true", "false"]:
                value = val_str_stripped == "true"
            else:
                try:
                    value = float(val_str_stripped)
                except ValueError:
                    # Neither float, integer nor boolean, hence None
                    value = None
        # else no value is stored in the line
        else:
            line = line.strip()
            name = line.split(" ")[-1].replace(" ", "")
            value = None
        res.update({name: value})

    return res
