"""
Example file for the py_fmi module and class. The usage of the
fmu_api should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
from ebcpy.simulationapi import py_fmi
from ebcpy.examples.dymola_api_example import setup_dymola_api


def setup_fmu_api(cd=None):
    """
    Function to show how to setup the FMU_API.
    As stated in the DymolaAPI-documentation, you need to
    pass a current working directory, the name of the model
    you want to work with and the necessary packages.

    :param str,os.path.normpath cd:
        Default is the current python working directory.
        A testzone-folder is created to keep everything ordered.
        Pass another directory to work where you want.
    :return: The FMU_API created in the function
    :rtype: dymola_api.DymolaAPI

    Example:
    --------
    >>> FMU_API = setup_fmu_api(show_window=True)
    >>> FMU_API.set_sim_setup({"startTime": 100,
    >>>                        "stopTime": 200})
    >>> FMU_API.simulate()
    #>>> DYM_API.close()
    """
    # Define path in which you want ot work:
    if cd is None:
        cd = os.path.normpath(os.path.join(os.getcwd(), "testzone"))

    # Setup the dymola api to extract the fmu:
    dym_api = setup_dymola_api(cd=cd)

    fmu_p = dym_api.dymola.translateModelFMU(dym_api.model_name,
                                             storeResult=False,
                                             modelName=dym_api.model_name.split(".")[-1],
                                             fmiVersion='2',
                                             fmiType='me',  # Model Exchange only
                                             includeSource=False,
                                             includeImage=0)

    fmu_api = py_fmi.FMU_API(cd=cd, model_name=os.path.join(cd, f"{fmu_p}.fmu"))
    return fmu_api


if __name__ == "__main__":
    # Setup the dymola-api:
    FMU_API = setup_fmu_api()
    FMU_API.set_sim_setup({"startTime": 100,
                           "stopTime": 200})
    df = FMU_API.simulate()
    print(df)