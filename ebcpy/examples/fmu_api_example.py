"""
Example file for the dymola_api module and class. The usage of the
dymola_api should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
from ebcpy.simulationapi import fmu


def setup_fmu_api(cd=None, n_cpu=1):
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
                              "PumpAndValve.fmu")
    # Setup the dymola api
    fmu_api = fmu.FMU_API(cd=cd,
                          model_name=model_name,
                          n_cpu=n_cpu)
    return fmu_api


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Setup the dymola-api:
    FMU_API = setup_fmu_api(n_cpu=3)
    FMU_API_2 = setup_fmu_api(n_cpu=3, cd=os.path.join(os.getcwd(), "testzone_2"))
    FMU_API.sim_setup = {"stopTime": 2,
                         "outputInterval": 0.001,
                         "resultNames": ["heatCapacitor.T"],
                         "initialNames": ["speedRamp.duration"]}
    FMU_API_2.sim_setup = {"stopTime": 2,
                         "outputInterval": 0.001,
                         "resultNames": ["heatCapacitor.T"],
                         "initialNames": ["speedRamp.duration"]}
    sim_setups = [{"initialValues": [0.1 + 0.1 * i]} for i in range(12)]
    res = FMU_API.simulate(sim_setup=sim_setups)
    res2 = FMU_API_2.simulate(sim_setup=sim_setups)
    for idx, _res in enumerate(res):
        plt.plot(_res["heatCapacitor.T"], label=idx)
    # Close the api to remove the created files:
    FMU_API.close()
    plt.legend()
    plt.show()
