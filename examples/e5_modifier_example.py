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
        ibpsa_mo,
        besmod_mo,
        ext_model_name,
        working_directory=None,
        n_cpu=1,
        with_plot=True
):
    """
    Arguments of this example:
    :param str aixlib_mo:
        Path to the package.mo of the AixLib.
        This example was tested for AixLib version 1.3.2.
    :param str ibpsa_mo:
        Path to the package.mo of the IBSPA.
        This example was tested for IBSPA version 3.0.0.
    :param str besmod_mo:
        Path to the package.mo of the BESMod.
        This example was tested for BESMod version 0.4.0.
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
        packages=[aixlib_mo, ibpsa_mo, besmod_mo],
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
    simulation_setup = {"start_time": 0,
                        "stop_time": 3600,
                        "output_interval": 100}
    dym_api.set_sim_setup(sim_setup=simulation_setup)

    # ######################### Simulation options ##########################
    # Look at the doc of simulate() in the website
    # Besides parameters (explained in fmu_example), return_option is important

    # Or change the savepath by using two keyword arguments.
    result_sp_2 = dym_api.simulate(
        return_option="time_series"
    )
    print(result_sp_2)

    # ######################### Closing ##########################
    # Close Dymola. If you forget to do so,
    # we call this function at the exit of your script.
    dym_api.close()

    # ######################### Simulation analysis ##########################


if __name__ == '__main__':
    # TODO-User: Change the AixLib and BESMod path!

    # call function main
    # - External libraries AixLib, IBSPA and BESMod will be loaded
    # - Model name will be called. Subsystem for controller will be exchanged from NoControl to DHWSuperheating.
    #   Additional to the new subsystem, the parameter dTDHW will be set from 5 K to 10 K.
    main(
        aixlib_mo=r"D:\900_repository\000_general\AixLib-1.3.2\AixLib\package.mo",
        ibpsa_mo=r"D:\900_repository\000_general\modelica-ibpsa-master\IBPSA\package.mo",
        besmod_mo=r"D:\900_repository\000_general\BESMod\BESMod\package.mo",
        ext_model_name='BESMod.Examples.GasBoilerBuildingOnly(redeclare BESMod.Systems.Control.DHWSuperheating control(dTDHW=10))'
    )
