"""
Goals of this part of the examples:
1. Learn how to use the `DymolaAPI`
2. Learn how to dynamically modify parameters in the model
"""
# Start by importing all relevant packages
import pathlib
# Imports from ebcpy
from ebcpy import DymolaAPI, TimeSeriesData, FMU_API


def main(
        aixlib_mo,
        modify_parameters,
        working_directory=None,
        n_cpu=1,
        with_plot=True
):
    """
    Arguments of this example:
    :param str aixlib_mo:
        Path to the package.mo of the AixLib.
        This example was tested for AixLib version 1.0.0.
    :param list modify_parameters:
        List of parameters and values which should be modified in the model.
    :param str working_directory:
        Path in which to store the output.
        Default is the examples\results folder
    :param int n_cpu:
        Number of processes to use
    :param bool with_plot:
        Show the plot at the end of the script. Default is True.
    """

    # General settings
    if working_directory is None:
        working_directory = pathlib.Path(__file__).parent.joinpath("results")

    # ######################### Simulation API Instantiation ##########################
    # %% Setup the Dymola-API:
    dym_api = DymolaAPI(
        model_name="AixLib.Systems.HeatPumpSystems.Examples.HeatPumpSystem",
        working_directory=working_directory,
        n_cpu=n_cpu,
        packages=[aixlib_mo],
        show_window=True,
        n_restart=-1,
        equidistant_output=False,
        # Only necessary if you need a specific dymola version
        dymola_path=r"C:\Program Files\Dymola 2023x",
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
    room_vol = "vol.V"
    dym_api.result_names = [p_el_name, room_vol]

    # ######################### Inputs ##########################
    # Modified parameters are defined in the main function and handed over in modifiy_paramters.
    # Parameters will be set when dym_api.simulation() is called.

    # ######################### Simulation options ##########################
    # Look at the doc of simulate() in the website
    # Besides parameters (explained in fmu_example), return_option is important

    # Or change the savepath by using two keyword arguments.
    result_sp_2 = dym_api.simulate(
        return_option="time_series",
        savepath=r"D:\00_temp",
        result_file_name="anotherResultFile",
        parameters=modify_parameters
    )
    print(result_sp_2)

    # ######################### Closing ##########################
    # Close Dymola. If you forget to do so,
    # we call this function at the exit of your script.
    dym_api.close()

    # ######################### Simulation analysis ##########################


if __name__ == '__main__':
    # TODO-User: Change the AixLib path!

    # Create array including parameters to modify.
    # In this example will be the Volume 'V' of object 'vol' set to 20 (m^3).
    param = {}
    param["vol.V"] = 20.0  # default = 40.0 m^3

    # call function main
    main(
        aixlib_mo=r"D:\900_repository\000_general\AixLib\AixLib\package.mo",
        modify_parameters=param
    )
