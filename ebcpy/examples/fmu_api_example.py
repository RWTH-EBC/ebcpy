"""
Example file for the dymola_api module and class. The usage of the
dymola_api should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
from ebcpy.simulationapi import py_fmi


def setup_fmu_api(cd=None):
    """
    Todo
    """
    # Define path in which you want ot work:
    if cd is None:
        cd = os.path.normpath(os.path.join(os.getcwd(), "testzone"))

    # Define the name of your model and setup the simulation api of choice
    model_name = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                              "Modelica",
                                              "AixCalTest_TestModel.fmu"))
    # Setup the dymola api
    fmu_api = py_fmi.FMU_API(cd, model_name)
    return fmu_api


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Setup the dymola-api:
    FMU_API = setup_fmu_api()
    FMU_API.set_sim_setup({"stopTime": 86400,
                           "resultNames": ["heater1.heatPorts[1].T"]})
    res = FMU_API.simulate()
    plt.plot(res["heater1.heatPorts[1].T"])
    plt.show()
