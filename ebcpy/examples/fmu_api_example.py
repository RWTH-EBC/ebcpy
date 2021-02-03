"""
Example file for the dymola_api module and class. The usage of the
dymola_api should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
from ebcpy.simulationapi import fmu


def setup_fmu_api(cd=None):
    """
    Function to show how to setup the FMU_API.

    :param str,os.path.normpath cd:
        Default is the current python working directory.
        A testzone-folder is created to keep everything ordered.
        Pass another directory to work where you want.
    :return: The FMU_API created in the function
    :rtype: fmu.FMU_API
    """
    # Define path in which you want ot work:
    if cd is None:
        cd = os.path.normpath(os.path.join(os.getcwd(), "testzone"))

    # Define the name of your model and setup the simulation api of choice
    model_name = os.path.join(os.path.dirname(__file__),
                              "Modelica",
                              "TestModel.fmu")
    # Setup the dymola api
    fmu_api = fmu.FMU_API(cd, model_name)
    return fmu_api


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Setup the dymola-api:
    FMU_API = setup_fmu_api()
    FMU_API.set_sim_setup({"stopTime": 3600,
                           "resultNames": ["heater1.heatPorts[1].T"]})
    res = FMU_API.simulate()
    plt.plot(res["heater1.heatPorts[1].T"])
    # Close the api to remove the created files:
    FMU_API.close()
    plt.show()
