"""
Goals of this part of the examples:

1. Learn how to use the `FMU_API`
2. Understand model variables
3. Learn how to change variables to store (`result_names`)
4. Learn how to change parameters of a simulation
5. Learn how to change inputs of a simulation
6. Learn how to run simulations in parallel
"""
# Start by importing all relevant packages
import pathlib
import numpy as np
import matplotlib.pyplot as plt
# Imports from ebcpy
from ebcpy import FMU_API, TimeSeriesData


def main(
        cd=None,
        n_cpu=1,
        log_fmu=True,
        n_sim=5,
        output_interval=100
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
    :param int output_interval:
        Output interval / step size of the simulation
    """

    # General settings
    if cd is None:
        cd = pathlib.Path(__file__).parent.joinpath("results")

    # ######################### Simulation API Instantiation ##########################
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

    # ######################### Simulation Setup Part ##########################
    # Change the simulation settings:
    # Which settings can I change?
    print("Supported setup options:", fmu_api.get_simulation_setup_fields())
    fmu_api.sim_setup.start_time = 0
    fmu_api.sim_setup.stop_time = 3600
    fmu_api.sim_setup.output_interval = output_interval
    # Or pass a dictionary. This makes using configs (toml, json) much easier
    simulation_setup = {"start_time": 0,
                        "stop_time": 3600,
                        "output_interval": output_interval}
    fmu_api.set_sim_setup(sim_setup=simulation_setup)

    # ######################### Parameters ##########################
    # Let's get some parameter to change, e.g. the capacity of the thermal mass:
    print(fmu_api.parameters['heaCap.C'])
    hea_cap_c = fmu_api.parameters['heaCap.C'].value
    # Let's alter it from 10% to 1000 % in n_sim simulations:
    sizings = np.linspace(0.1, 10, n_sim)
    parameters = []
    for sizing in sizings:
        parameters.append({"heaCap.C": hea_cap_c * sizing})

    # ######################### Inputs ##########################
    # Let's also change the input of the simulation:
    print("Inputs names are:", fmu_api.inputs)
    # We only have TDryBul (outdoor air temperature) as an input.
    # Start with the setup of a time-index that matches our simulation setup
    # Feel free to play around with the settings to see what happens if your time_index is malformed.
    time_index = np.arange(
        fmu_api.sim_setup.start_time,
        fmu_api.sim_setup.stop_time,
        fmu_api.sim_setup.output_interval
    )
    # Apply some sinus function for the outdoor air temperature
    t_dry_bulb = np.sin(time_index/3600*np.pi) * 10 + 263.15
    df_inputs = TimeSeriesData({"TDryBul": t_dry_bulb}, index=time_index)
    # Warning: If you enable the following line you will trigger an error.
    # It only goes to show that inputs to the simulation must contain clear
    # tags.
    # df_inputs[('TDryBul', 'constant_0_degC')] = 275.15

    # ######################### Results to store ##########################
    # As we vary the heating capacity,
    # let's plot the influence on the temperature of said capacity:
    # Per default, all outputs will be stored:
    print("Results that will be stored", fmu_api.result_names)
    # In our case, we are not interested in Pel but in other states:
    # First, the temperature
    # Second, the input outdoor air temperature to see if our input is correctly used.
    fmu_api.result_names = ["heatCap.T", "TDryBul"]
    # Oops, `heatCap.T` is not part of the model. We warn you about such typos.
    # This way, you can easier debug your simulations if something goes wrong.
    # Set the correct names:
    fmu_api.result_names = ["heaCap.T", "TDryBul"]
    print("Results that will be stored", fmu_api.result_names)

    # ######################### Execute simulation ##########################
    # Pass the created list to the simulate function
    results = fmu_api.simulate(parameters=parameters,
                               inputs=df_inputs)

    # ######################### Closing ##########################
    # Close the fmu. If you forget to do so,
    # we call this function at the exit of your script.
    # It deleted all extracted FMU files.
    fmu_api.close()

    # ######################### Visualization ##########################
    # Plot the result
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].set_ylabel("TDryBul in K")
    ax[1].set_ylabel("T_Cap in K")
    ax[1].set_xlabel("Time in s")
    ax[0].plot(df_inputs, label="Inputs", linestyle="--")
    for res, sizing in zip(results, sizings):
        ax[0].plot(res['TDryBul'])
        ax[1].plot(res['heaCap.T'], label=sizing)
    for _ax in ax:
        _ax.legend(bbox_to_anchor=(1, 1.05), loc="upper left")
    plt.show()


if __name__ == '__main__':
    main(
        n_cpu=1,
        log_fmu=False,
        n_sim=5,
        output_interval=100
    )
