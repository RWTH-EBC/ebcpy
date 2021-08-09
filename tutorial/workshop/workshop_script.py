import pathlib
import pandas as pd
import matplotlib.pyplot as plt


def workshop_time_series_data():
    pass

def workshop_dymola_api():
    """
    Goals of this part of the workshop:
    1. Learn how to use the DymolaAPI
    2. Learn the different result options of the simulation
    3. Get a first idea of the usage of TimeSeriesData
    """
    from ebcpy import DymolaAPI
    cd = pathlib.Path(__file__).parent.joinpath("results")
    dym_api = DymolaAPI(
        model_name="HeatPumpSystemWithInput.mo",
        cd=cd,
        packages=[r"path_to_aixlib\AixLib\package.mo"],
        show_window=True,
        n_restart=-1,
        equidistant_output=True,
        get_structural_parameters=True
        # Only necessary if you need a specific dymola version
        #dymola_path=None,
        #dymola_interface_path=None
    )
    simulation_setup = {"start_time": 0,
                        "stop_time": 3600,
                        "output_interval": 100}
    dym_api.set_sim_setup(sim_setup=simulation_setup)
    dym_api.result_names = ["Pel", "heaCap.T"]
    result_ts = dym_api.simulate(return_option="time_series")
    print(type(result_ts))
    print(result_ts)
    result_lp = dym_api.simulate(return_option="last_point")
    print(type(result_lp))
    print(result_lp)
    result_sp = dym_api.simulate(return_option="savepath")
    print(result_sp)
    # Or change the savepath by using two keyword arguments.
    result_sp_2 = dym_api.simulate(return_option="savepath",
                                   savepath=r"D:\00_temp",
                                   result_file_name="anotherResultFile")
    print(result_sp_2)
    # Now let's load the TimeSeriesData
    from ebcpy import TimeSeriesData
    tsd = TimeSeriesData(result_sp_2)
    print("TimeSeriesData inherits from", TimeSeriesData.__base__)
    print(tsd)


def workshop_fmu_api():
    """
    Goals of this part of the workshop:
    1. Learn how to use the FMU_API
    2. Understand model variables
    3. Learn how to change variables to store (result_names)
    4. Learn how to change parameters
    5. Learn how to change inputs of the simulation
    6. Learn how to run the simulations in parallel
    """
    from ebcpy import FMU_API
    # %% Define global settings for this tutorial
    cd = pathlib.Path(__file__).parent.joinpath("results")
    n_cpu = 10  # Number of processes to use
    log_fmu = False  # Whether to get the FMU log output
    n_sim = 100  # Number of simulations to run in the tutorial

    # %% Setup the FMU-API:
    fmu_api = FMU_API(model_name="HeatPumpSystemWithInput.fmu",
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

    df = pd.DataFrame({"TDryBul": [265.15, 273.15]}, index=[0, 86400])
    # Change the simulation settings:
    fmu_api.sim_setup.start_time = 0
    fmu_api.sim_setup.stop_time = 3600
    # Or pass a dictionary. This makes usings configs (toml, json) much easier
    simulation_setup = {"start_time": 0,
                        "stop_time": 3600,
                        "output_interval": 100}
    fmu_api.set_sim_setup(sim_setup=simulation_setup)
    # Study on influence of heat capacity in the model:
    fmu_api.result_names = ['Pel', "heatCap.T"]
    fmu_api.result_names = ['Pel', "heaCap.T"]
    print(fmu_api.result_names)
    print(fmu_api.parameters['heaCap.C'])
    default = fmu_api.parameters['heaCap.C'].value
    # Let's alter it from 10% to 1000 % in n_sim simulations:
    import numpy as np
    sizings = np.linspace(0.1, 10, n_sim)
    parameters = []
    for sizing in sizings:
        parameters.append({"heaCap.C": default * sizing})
    # Pass the created list to the simulate function
    results = fmu_api.simulate(parameters=parameters)
    # Close the fmu
    fmu_api.close()
    # Plot the result
    for res, sizing in zip(results, sizings):
        plt.plot(res['heaCap.T'], label=sizing)
    plt.legend()
    plt.show()


def time_series_data_workshop():
    pass


if __name__ == "__main__":
    # Before starting the tutorial:
    # 1. Create a clean environment of python 3.7 or 3.8
    # 2. Activate the environment in your terminal
    # 3. Clone the repository by running `git clone TODO`
    # 4. Clone the AixLib in order to use the models:
    # `git clone AixLib`
    # 5. Install the library using `pip install -e ebcpy`
    # 6. Install jupyter notebook using `pip install jupyter-notebook
    # 7. Run the jupyter notebook by executing `jupyter notebook ebcpy\tutorial\tutorial.ipynb`
