"""
Goals of this part of the examples:
1. Learn a basic workflow for Dymola simulation studies
"""
import datetime
import os
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd

from ebcpy import DymolaAPI, load_time_series_data

def convert_to_datetime_and_csv(mat_result_file, first_day_of_year, variable_names):
    df = load_time_series_data(mat_result_file, variable_names=variable_names)
    df.tsd.to_datetime_index(origin=first_day_of_year)
    df_path = Path(mat_result_file).with_suffix(".parquet")
    df.tsd.save(df_path)

    os.remove(mat_result_file)
    return df_path


def empty_postprocessing(mat_result, **_kwargs):
    return mat_result


def simple_dymola_sim_study(
        model_names: list[str],
        mos_script_pre: Union[str, Path],
        parameters: Union[dict, List[dict]] = None,
        working_directory: Union[str, Path] = None,
        n_cpu: int = 2,
        use_postprocessing: bool = True,
        use_parameter_study: bool = True,
        model_result_file_names: List[str] = None,
        save_path: Union[str, Path] = None,
):
    # General settings
    if working_directory is None:
        working_directory = Path(__file__).parent.joinpath("results", "working_directory")
    if save_path is None:
        save_path = Path(__file__).parent.joinpath("results", "SimResults")

    # keywords practical for debugging
    dymola_debugging_kwargs = {
        "show_window": True,  # opens
        "debug": True,
    }

    # keywords to increase dymola stability
    dymola_stability_kwargs = {
        "time_delay_between_starts": 0, # recommended for large models where each model run needs to be translated
    }


    first_day_of_year = datetime.datetime(2015, 1, 1, 0, 0)

    simulation_setup = {"start_time": 0,
                        "stop_time": 3600 * 24 * 30,
                        "output_interval": 100}

    if use_postprocessing:
        postprocess_mat_result = convert_to_datetime_and_csv
        kwargs_postprocessing = dict(
            variable_names=["*outputs*"],
            first_day_of_year=first_day_of_year
        )
    else:
        postprocess_mat_result = empty_postprocessing
        kwargs_postprocessing = {}


    if use_parameter_study:
        all_result_paths = {}
        for model_name, result_file_name in zip(model_names, model_result_file_names):
            result_file_names = [f"{result_file_name}_{str(param_dict['parameterStudy.VPerQFlow']).replace(".","_")}" for param_dict in parameters]
            dym_api = DymolaAPI(
                mos_script_pre=mos_script_pre,
                model_name=model_name,
                working_directory=working_directory,
                n_cpu=n_cpu,
                n_restart=-1,
                equidistant_output=True,
                show_window=False,
                debug=False,
            )


            dym_api.set_sim_setup(sim_setup=simulation_setup)

            result_paths = dym_api.simulate(
                parameters=parameters,
                return_option="savepath",
                savepath=save_path,
                result_file_name=result_file_names,
                postprocess_mat_result=postprocess_mat_result,
                kwargs_postprocessing=kwargs_postprocessing,
            )
            all_result_paths[model_name] = result_paths
            dym_api.close()
        return all_result_paths
    else:
        dym_api = DymolaAPI(
            mos_script_pre=mos_script_pre,
            # model_name=model_names[0],
            working_directory=working_directory,
            n_cpu=n_cpu,
            show_window=False,
            n_restart=-1,
            equidistant_output=True,
            debug=True,
        )
        dym_api.set_sim_setup(sim_setup=simulation_setup)
        result_paths = dym_api.simulate(
            model_names=model_names,
            parameters=parameters,
            return_option="savepath",
            savepath=save_path,
            result_file_name=model_result_file_names,
            postprocess_mat_result=postprocess_mat_result,
            kwargs_postprocessing=kwargs_postprocessing,
        )
        dym_api.close()
        return result_paths

if __name__ == "__main__":
    base_model_name = "BESMod.Examples.DesignOptimization.BES"

    storage_layers = np.arange(1,5,1,dtype=int)
    model_names_to_simulate = [
        f"{base_model_name}(hydraulic.distribution.parStoBuf(nLayer={n}))"
        for n in storage_layers
    ]
    model_result_names = [f"BufSto_nLayer{n}" for n in storage_layers]

    print(model_names_to_simulate)

    parameter_study_params = [{"parameterStudy.VPerQFlow": np.round(v, decimals=1)} for v in np.linspace(5, 100, 8)]
    print(parameter_study_params)
    print("study 1")
    simple_dymola_sim_study(
        model_names=model_names_to_simulate,
        mos_script_pre=r"D:\01_git\BESMod\startup.mos",
        parameters=parameter_study_params,
        use_parameter_study=True,
        model_result_file_names=model_result_names,
        save_path=Path(__file__).parent.joinpath("results", "SimResults_1")
    )
    
    print("study 2")
    model_study_params = {"parameterStudy.VPerQFlow": 50, "parameterStudy.TBiv": 273.15-5}
    simple_dymola_sim_study(
        model_names=model_names_to_simulate,
        mos_script_pre=r"D:\01_git\BESMod\startup.mos",
        parameters=model_study_params,
        use_parameter_study=False,
        model_result_file_names=model_result_names,
        save_path=Path(__file__).parent.joinpath("results", "SimResults_2")
    )
    print("study 3")
    rng = np.random.default_rng(42)
    random_example_values = rng.integers(5, 100, size=len(model_names_to_simulate))
    model_study_div_params = [{"parameterStudy.VPerQFlow": val} for val in random_example_values]
    simple_dymola_sim_study(
        model_names=model_names_to_simulate,
        mos_script_pre=r"D:\01_git\BESMod\startup.mos",
        parameters=model_study_div_params,
        use_parameter_study=False,
        model_result_file_names=model_result_names,
        save_path=Path(__file__).parent.joinpath("results", "SimResults_3")
    )
    print("finished")

