import pathlib
import pandas as pd
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    workshop_fmu_api()
