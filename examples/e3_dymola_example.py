"""
Goals of this part of the examples:
1. Learn how to use the `DymolaAPI`
2. Learn the different result options of the simulation
3. Learn how to convert inputs into the Dymola format
"""
# Start by importing all relevant packages
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt
# Imports from ebcpy
from ebcpy import DymolaAPI, TimeSeriesData, FMU_API
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt


def main(
        aixlib_mo,
        cd=None,
        n_cpu=1,
        with_plot=True
):
    """
    Arguments of this example:
    :param str aixlib_mo:
        Path to the package.mo of the AixLib.
        This example was tested for AixLib version 1.0.0.
    :param str cd:
        Path in which to store the output.
        Default is the examples\results folder
    :param int n_cpu:
        Number of processes to use
    :param bool with_plot:
        Show the plot at the end of the script. Default is True.
    """

    # General settings
    if cd is None:
        cd = pathlib.Path(__file__).parent.joinpath("results")

    # ######################### Simulation API Instantiation ##########################
    # %% Setup the Dymola-API:
    # organize settings in configuration dict
    config_dict = {
                  'model_name': 'AixLib.Systems.HeatPumpSystems.Examples.HeatPumpSystem',
                  'cd': cd,
                  'packages': [aixlib_mo]
                  }
    dym_api = DymolaAPI(
        config_dict,
        n_cpu=n_cpu,
        show_window=True,
        n_restart=-1,
        equidistant_output=False
        # Only necessary if you need a specific dymola version
        #dymola_path=None,
        #dymola_version=None
    )
    print("Number of variables:", len(dym_api.variables))
    print("Number of outputs:", len(dym_api.outputs))
    print("Number of inputs:", len(dym_api.inputs))
    print("Number of parameters:", len(dym_api.parameters))
    print("Number of states:", len(dym_api.states))

    # ######################### Settings ##########################
    # To understand what is happening here we refer to fmu_example.py
    # As both simulation_apis are based on the same class, most interfaces are equal.
    # Only difference is the simulation setup:
    print("Fields of DymolaAPISetup", DymolaAPI.get_simulation_setup_fields())
    print("Fields of FMU_APISetup", FMU_API.get_simulation_setup_fields())

    simulation_setup = {"start_time": 0,
                        "stop_time": 3600,
                        "output_interval": 100}
    dym_api.set_sim_setup(sim_setup=simulation_setup)
    p_el_name = "heatPumpSystem.heatPump.sigBus.PelMea"
    dym_api.result_names = [p_el_name]

    # ######################### Inputs ##########################
    # Sadly, setting inputs directly is not supported in Dymola.
    # Hence, you have to use the model `Modelica.Blocks.Sources.CombiTimeTable`.
    # In the model "AixLib.Systems.HeatPumpSystems.Examples.HeatPumpSystem" we
    # already use this model to simulate heat gains into the room
    # We called the instance of the model `timTab`.
    # To get the output of the table, let's add it to the result names:
    dym_api.result_names = [p_el_name, 'timTab.y[1]']
    # In order to change the inputs, you have to change the model in Dymola by:
    # 1. Double click on timTab
    # 2. Set tableOnFile = true
    # 3. Set tableName = "myCustomInput" (or any other nice string)
    table_name = "myCustomInput"
    # 4. Enter the fileName where you want to store your input. This can be any filepath.
    # For this tutorial to work, set
    # 5. Last, add a parameter in the model to ensure the simulation works without tuning any parameter.
    # Sadly, this is a requirement. Models with parameters do not require this feature. As this model has
    # no parameters, it's required. Go into the text-section and add:
    # 'parameter Real n=1;'.

    # fileName=Modelica.Utilities.Files.loadResource("modelica://AixLib/Resources/my_custom_input.txt")
    file_name = pathlib.Path(aixlib_mo).parent.joinpath("Resources", "my_custom_input.txt")
    # This input generate is re-used from the fmu_example.py file.
    time_index = np.arange(
        dym_api.sim_setup.start_time,
        dym_api.sim_setup.stop_time,
        dym_api.sim_setup.output_interval
    )
    # Apply some sinus function for the outdoor air temperature
    internal_gains = np.sin(time_index/3600*np.pi) * 1000
    tsd_input = TimeSeriesData({"InternalGains": internal_gains}, index=time_index)
    # To generate the input in the correct format, use the convert_tsd_to_modelica_txt function:
    filepath = convert_tsd_to_modelica_txt(
        tsd=tsd_input,
        table_name=table_name,
        save_path_file=file_name
    )
    print("Successfully created Dymola input file at", filepath)

    # ######################### Simulation options ##########################
    # Look at the doc of simulate() in the website
    # Besides parameters (explained in fmu_example), return_option is important
    result_time_series = dym_api.simulate(
        return_option="time_series",
        # Info: You would not need these following keyword-arguments,
        # as we've already created our file above.
        # However, you can also pass the arguments
        # from above directly into the function call:
        inputs=tsd_input,
        table_name=table_name,
        file_name=file_name
    )
    print(type(result_time_series))
    print(result_time_series)
    result_last_point = dym_api.simulate(
        return_option="last_point"
    )
    print(type(result_last_point))
    print(result_last_point)
    result_sp = dym_api.simulate(
        return_option="savepath"
    )
    print(result_sp)
    # Or change the savepath by using two keyword arguments.
    result_sp_2 = dym_api.simulate(
        return_option="savepath",
        savepath=r"D:\00_temp",
        result_file_name="anotherResultFile"
    )
    print(result_sp_2)

    # ######################### Simulation analysis ##########################
    # Now let's load the TimeSeriesData
    tsd_1 = TimeSeriesData(result_sp)
    tsd_2 = TimeSeriesData(result_sp_2)
    print("Both .mat's are equal:", all(tsd_1 == tsd_2))
    # Let's look at both results. The .mat-file contains more indexes as events are stored as well.
    # The return_option 'time_series' omits these events (as does fmpy). Thus, it's less accurate.
    # But, it's much faster!
    print("Number of points for return option 'time_series':", len(result_time_series.index))
    print("Number of points for return option 'savepath':", len(tsd_1.index))
    plt.plot(tsd_1[p_el_name], color="blue", label="savepath", marker="^")
    plt.plot(result_time_series[p_el_name], color="red", label="time_series", marker="^")
    plt.scatter(result_last_point["Time"], result_last_point[p_el_name],
                color="black", label="last_point", marker="^")
    plt.legend()
    plt.title("Difference in output for different return_options")
    plt.figure()
    plt.plot(tsd_1['timTab.y[1]'], color="blue")
    plt.title("Input of CombiTimeTable 'timTab'")
    if with_plot:
        plt.show()

    # Save the data for later reproduction  # todo: activate once merged
    file = dym_api.save_for_reproduction(
        title="MyDymolaStudy",
        files=[result_sp, result_sp_2]
    )
    print("ZIP-File to reproduce all this:", file)


if __name__ == '__main__':
    # TODO-User: Change the AixLib path!
    main(
        aixlib_mo=r"D:\02_workshop\AixLib\AixLib\package.mo",
        n_cpu=5
    )
