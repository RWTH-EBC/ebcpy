"""
Example file for the dymola_api module and class. The usage of the
dymola_api should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
from ebcpy.simulationapi import dymola_api


def example_dymola_api(dym_api):
    """
    Function to show the usage of the function
    get_all_tuner_parameters() of the DymolAPI.

    :param dymola_api.DymolaAPI dym_api:
        DymolaAPI that can be generated using :meth:` this function<setup_dymola_api>`
    :return: tuner parameters which can be used for other examples
    :rtype: ebcpy.data_types.TunerParas
    """
    tuner_paras = dym_api.get_all_tuner_parameters()
    return tuner_paras


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
    #>>> DYM_API.close()
    """
    # Define path in which you want ot work:
    if cd is None:
        cd = os.path.normpath(os.path.join(os.getcwd(), "testzone"))

    # Define the name of your model and the packages needed for import
    # and setup the simulation api of choice
    model_name = "AixCalTest.TestModel"
    packages = [os.path.normpath(os.path.join(os.path.dirname(__file__),
                                              "Modelica",
                                              "AixCalTest",
                                              "package.mo"))]
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
    example_dymola_api(DYM_API)
