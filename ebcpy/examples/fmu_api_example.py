"""
Example file for the dymola_api module and class. The usage of the
dymola_api should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
from ebcpy.simulationapi import py_fmi


def setup_fmu_api(cd=None):
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
    >>> FMU_API = setup_fmu_api()
    >>> FMU_API.set_sim_setup({"startTime": 100,
    >>>                        "stopTime": 200})
    >>> FMU_API.simulate()
    >>> FMU_API.close()
    """
    # Define path in which you want ot work:
    if cd is None:
        cd = os.path.normpath(os.path.join(os.getcwd(), "testzone"))

    # Define the name of your model and setup the simulation api of choice
    model_name = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                              "Modelica",
                                              "AixCalTest_TestModel.fmu"))
    # Setup the dymola api
    dym_api = py_fmi.FMU_API(cd, model_name)
    return dym_api


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Setup the dymola-api:
    FMU_API = setup_fmu_api()
    FMU_API.set_sim_setup({"stopTime": 86400,})
    res = FMU_API.simulate()
    plt.plot(res["Demand.thermalZoneOneElement.volAir.T"])
    plt.show()
