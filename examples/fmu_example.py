"""
Goals of this part of the examples:
1. Learn how to use the FMU_API
2. Understand model variables
3. Learn how to change variables to store (result_names)
4. Learn how to change parameters
5. Learn how to change inputs of the simulation
6. Learn how to run the simulations in parallel
"""
# Start by importing all relevant packages
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Imports from ebcpy
from ebcpy import FMU_API


def main(
        cd=None,
        n_cpu=1,
        log_fmu=True,
        n_sim=5
):
    """
    Arguments of this example:
    :param str cd:
        Path in which to store the output.
        Default is the examples\results folder
    :param int n_cpu:
        Number of processes to use
    :param bool log_fmu:
        Whether to get the FMU log output
    :param int n_sim:
        Number of simulations to run
    """

    # Path
    if cd is None:
        cd = pathlib.Path(__file__).parent.joinpath("results")

    # %% Setup the FMU-API:
    fmu_api = FMU_API(model_name="data//HeatPumpSystemWithInput.fmu",
                      cd=cd,
                      n_cpu=n_cpu,
                      log_fmu=log_fmu)
    print("Number of variables:", len(fmu_api.variables))
    print("Number of outputs:", len(fmu_api.outputs))
    print("Number of inputs:", len(fmu_api.inputs))
    print("Number of parameters:", len(fmu_api.parameters))
    print("Number of states:", len(fmu_api.states))
    print("Variables to store when simulating:", fmu_api.result_names)
    print("Outputs of the fmu", fmu_api.outputs)

    df_inputs = pd.DataFrame({"TDryBul": [265.15, 273.15]}, index=[0, 86400])
    # Change the simulation settings:
    # Which settings can I change?
    print("Supported setup options:", fmu_api.get_simulation_setup_fields())
    fmu_api.sim_setup.start_time = 0
    fmu_api.sim_setup.stop_time = 3600
    # Or pass a dictionary. This makes using configs (toml, json) much easier
    simulation_setup = {"start_time": 0,
                        "stop_time": 3600,
                        "output_interval": 100}
    fmu_api.set_sim_setup(sim_setup=simulation_setup)
    # Study on influence of heat capacity in the model:
    fmu_api.result_names = ['Pel', "heatCap.T"]
    # Oops, `heatCap.T` is not part of the model. We warn you about such typos.
    # This way, you can easier debug your simulations if something goes wrong.
    # Set the correct names:
    fmu_api.result_names = ['Pel', "heaCap.T"]
    print(fmu_api.result_names)
    # Let's get some parameter to change, e.g. the capacity of the thermal mass:
    print(fmu_api.parameters['heaCap.C'])
    hea_cap_c = fmu_api.parameters['heaCap.C'].value
    # Let's alter it from 10% to 1000 % in n_sim simulations:
    sizings = np.linspace(0.1, 10, n_sim)
    parameters = []
    for sizing in sizings:
        parameters.append({"heaCap.C": hea_cap_c * sizing})
    # Pass the created list to the simulate function
    results = fmu_api.simulate(parameters=parameters,
                               inputs=df_inputs)
    # Close the fmu
    fmu_api.close()
    # Plot the result
    for res, sizing in zip(results, sizings):
        plt.plot(res['heaCap.T'], label=sizing)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main(
        n_cpu=1,
        log_fmu=True,
        n_sim=5
    )
