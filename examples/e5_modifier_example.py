"""
Goals of this part of the examples:
1. Learn how to use the `DymolaAPI`
2. Learn how to dynamically modify parameters in the model
"""
# Start by importing all relevant packages
import pathlib
# Imports from ebcpy
from ebcpy import DymolaAPI


def main(
        besmod_startup_mos,
        ext_model_name,
        working_directory=None,
        n_cpu=1
):
    """
    Arguments of this example:

    :param str besmod_startup_mos:
        Path to the startup.mos of the BESMod.
        This example was tested for BESMod version 0.4.0.
        This example was tested for IBSPA version 3.0.0.
        This example was tested for AixLib version 1.3.2.
    :param list ext_model_name:
        Executable model name with redeclared subsystems and modifiers.
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
        model_name=ext_model_name,
        working_directory=working_directory,
        n_cpu=n_cpu,
        mos_script_pre=besmod_startup_mos,
        show_window=False,
        # Only necessary if you need a specific dymola version
        dymola_path=r"C:\Program Files\Dymola 2023x",
    )
    print("Number of variables:", len(dym_api.variables))
    print("Number of outputs:", len(dym_api.outputs))
    print("Number of inputs:", len(dym_api.inputs))
    print("Number of parameters:", len(dym_api.parameters))
    print("Number of states:", len(dym_api.states))

    # ######################### Settings ##########################
    simulation_setup = {"start_time": 0,
                        "stop_time": 3600,
                        "output_interval": 100}
    dym_api.set_sim_setup(sim_setup=simulation_setup)

    # ######################### Simulation options ##########################
    # Look at the doc of simulate() in the website or previous examples
    result_sp_2 = dym_api.simulate(
        return_option="time_series"
    )
    print(result_sp_2)

    # You can also simulate a list of different `model_names` (or modified versions of the same model)
    # by passing a list to the `simulate` function in `DymolaAPI`:
    model_names_to_simulate = [
        "BESMod.Examples.DesignOptimization.BES",
        "BESMod.Examples.GasBoilerBuildingOnly(redeclare BESMod.Systems.Control.DHWSuperheating control(dTDHW=10))",
        "BESMod.Examples.GasBoilerBuildingOnly(redeclare BESMod.Systems.Control.DHWSuperheating control(dTDHW=5))",
    ]
    results = dym_api.simulate(
        return_option="time_series",
        model_names=model_names_to_simulate
    )
    print(results)
    dym_api.save_for_reproduction(
        title="FMUTest",
        log_message="This is just an example."
    )
    # ######################### Closing ##########################
    # Close Dymola. If you forget to do so,
    # we call this function at the exit of your script.
    dym_api.close()


if __name__ == '__main__':
    # TODO-User: Change the BESMod path!
    # call function main
    # - External libraries AixLib, IBSPA and BESMod will be loaded
    # - Model ext_model_name will be called. Subsystem for controller will be exchanged from NoControl to DHWSuperheating.
    #   Additional to the new subsystem, the parameter dTDHW will be set from 5 K to 10 K.
    # Furthermore, inside the main function, a method for simulating multiple models at one call is shown.
    main(
        besmod_startup_mos=r"D:\04_git\BESMod\startup.mos",
        ext_model_name='BESMod.Examples.GasBoilerBuildingOnly(redeclare BESMod.Systems.Control.DHWSuperheating control(dTDHW=10))'
    )
