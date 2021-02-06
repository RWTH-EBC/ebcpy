"""
Example file for the dymola_api module and class. The usage of the
dymola_api should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
from ebcpy.simulationapi import dymola_api


def setup_dymola_api(cd=None, show_window=True):
    """
    Function to show how to setup the DymolaAPI.
    As stated in the DymolaAPI-documentation, you need to
    pass a current working directory, the name of the model
    you want to work with and the necessary packages.

    :param str,os.path.normpath cd:
        Default is the current python working directory.
        A testzone-folder is created to keep everything ordered.
        Pass another directory to work where you want.
    :param bool show_window:
        True if you want to see the Dymola instance on
        your machine. You can see what commands in the
        interface.
    :return: The DymolaAPI created in the function
    :rtype: dymola_api.DymolaAPI

    Example:
    --------
    >>> DYM_API = setup_dymola_api(show_window=True)
    >>> DYM_API.set_sim_setup({"startTime": 100,
    >>>                        "stopTime": 200})
    >>> DYM_API.simulate()
    >>> DYM_API.close()
    """
    # Define path in which you want ot work:
    if cd is None:
        cd = os.path.normpath(os.path.join(os.getcwd(), "testzone"))

    # Define the name of your model and the packages needed for import
    # and setup the simulation api of choice
    model_name = "AixCalTest_TestModel"
    packages = [os.path.normpath(os.path.join(os.path.dirname(__file__),
                                              "Modelica",
                                              "TestModel.mo"))]
    # Setup the dymola api
    dym_api = dymola_api.DymolaAPI(cd,
                                   model_name,
                                   packages,
                                   show_window=show_window)
    return dym_api


if __name__ == "__main__":
    # Setup the dymola-api:
    DYM_API = setup_dymola_api()
    # Run example:
    DYM_API.get_all_parameters()
