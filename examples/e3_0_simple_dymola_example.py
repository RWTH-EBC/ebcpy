# # Example: Basic Dymola Simulation Workflow
#
# Goals of this example:
# 1. Learn a common workflow for Dymola simulation studies using ebcpy
# 2. Understand how to run parameter studies across multiple model variants
# 3. Learn how to use model name modifiers
# 4. Learn how to post-process simulation results into usable formats
#
# This example demonstrates common simulation patterns:
# - **Study 0:** Single model — verify your setup works
# - **Study 1:** Parameter study — multiple models × multiple parameter sets
# - **Study 2:** Model comparison — multiple models with the same parameters
# - **Study 3:** Model comparison — multiple models with individual parameters
#
# **Prerequisites:**
# This example requires a Dymola installation and the BESMod library with AixLib.
# Adjust the ``MOS_SCRIPT`` path to your local BESMod startup script.

import datetime
import os
from pathlib import Path

import numpy as np

from ebcpy import simple_dymola_sim_study, load_time_series_data
from ebcpy.utils import get_names


def custom_postprocessing(mat_result_file, first_day_of_year, variable_names):
    """
    Post-process a Dymola .mat result file into a datetime-indexed parquet file.

    This function is passed to ``DymolaAPI.simulate()`` via the
    ``postprocess_mat_result`` keyword. It is called automatically
    for each simulation result. Adapt this function to your study.

    :param str mat_result_file:
        Path to the .mat file produced by Dymola.
        Mandatory parameter passed by ``DymolaAPI.simulate()``.
    :param datetime.datetime first_day_of_year:
        Reference datetime for converting the float index (in seconds)
        to a DatetimeIndex.
    :param list variable_names:
        Variable names or wildcard patterns to extract from the .mat file.
        E.g. ``["*outputs*"]`` to extract all output variables.
    :return: Path to the saved parquet file.
    :rtype: Path
    """
    df = load_time_series_data(mat_result_file, variable_names=variable_names)
    df.tsd.to_datetime_index(origin=first_day_of_year)

    # --- Adapt this section to your study ---
    # Example: compute mean storage temperature across all layers
    variables = df.columns.to_list()
    layer_temp_vars = get_names(variables, "hydraulic.distribution.stoBuf.layer[*].T")
    if layer_temp_vars:
        df["stoBuf.mean_T"] = df[layer_temp_vars].mean(axis=1)

    # save the partial result dataframe in a different format
    # Here we use parquet because it is fast and results in small file sizes
    # If you also save variables with a lot of constant segments
    # you can use further compressions like ".parquet.snappy"
    # for small and short studies you could also use csv but this is not recommended
    df_path = Path(mat_result_file).with_suffix(".parquet")
    df.tsd.save(df_path)
    # remove the old mat file
    os.remove(mat_result_file)
    return df_path


if __name__ == "__main__":
    # TODO: Adjust this path to your local BESMod startup script
    MOS_SCRIPT = r"D:\01_git\BESMod\startup.mos"

    # ## Simulation setup
    # Adapt these values to your study.
    SIMULATION_SETUP = {
        "start_time": 0,
        "stop_time": 3600 * 24 * 30,  # 30 days
        "output_interval": 900         # one data point every 900 s (15 min)
    }

    # ## Define model variants
    # We use model name modifiers to vary the number of storage layers.
    # This is the recommended approach for structural parameters which
    # need a retranslation — write the modifier directly in the model name
    # string. You can also redeclare models here.
    BASE_MODEL = "BESMod.Examples.DesignOptimization.BES"
    storage_layers = np.arange(1, 5, 1, dtype=int)
    model_names_to_simulate = [
        f"{BASE_MODEL}(hydraulic.distribution.parStoBuf(nLayer={n}))"
        for n in storage_layers
    ]
    # Base names for result files — one per model variant
    model_result_names = [f"BufSto_nLayer{n}" for n in storage_layers]
    print("Model variants:", model_names_to_simulate)

    # ## Post-processing configuration
    # Adapt the variable_names patterns to select which variables
    # you want to keep from the .mat files.
    FIRST_DAY_OF_YEAR = datetime.datetime(2015, 1, 1, 0, 0)
    KWARGS_POSTPROCESSING = dict(
        variable_names=[
            "outputs*",  # stars as wildcards are supported
            "hydraulic.distribution.stoBuf.layer[*].T",
            "hydraulic.distribution.stoBuf.port_a_consumer.m_flow",
            "hydraulic.distribution.stoBuf.port_a_heatGenerator.m_flow",
            "hydraulic.distribution.sigBusDistr.*",
            "hydraulic.generation.sigBusGen.*",
        ],
        first_day_of_year=FIRST_DAY_OF_YEAR,
    )

    # ## Study 0: Single model, default parameters
    # The simplest possible simulation — verify your setup works.
    print("\n--- Study 0: Single model ---")
    result_paths_study_0 = simple_dymola_sim_study(
        model_names=[model_names_to_simulate[0]],
        mos_script_pre=MOS_SCRIPT,
        simulation_setup=SIMULATION_SETUP,
        # save path and working directory should not be the same folder
        # and if you are in a git repository add the working_directory folder to .gitignore
        save_path=Path(__file__).parent.joinpath("results", "SimResults_0"),
        working_directory=Path(__file__).parent.joinpath("results", "working_directory"),
        model_result_file_names=[model_result_names[0]],
    )

    # ## Study 1: Parameter study
    # Each model variant is simulated with all parameter sets.
    # Each parameter set can have multiple and different parameters
    # Total simulations: len(model_names) × len(parameter_sets) = 4 × 8 = 32
    print("\n--- Study 1: Parameter study ---")
    parameter_study_params = [
        {"parameterStudy.VPerQFlow": np.round(v, decimals=1)}
        for v in np.linspace(5, 100, 8)
    ]
    print("Parameter sets:", parameter_study_params)
    result_paths_study_1 = simple_dymola_sim_study(
        model_names=model_names_to_simulate,
        mos_script_pre=MOS_SCRIPT,
        simulation_setup=SIMULATION_SETUP,
        working_directory=Path(__file__).parent.joinpath("results", "working_directory"),
        save_path=Path(__file__).parent.joinpath("results", "SimResults_1"),
        parameters=parameter_study_params,
        use_parameter_study=True,
        model_result_file_names=model_result_names,
        postprocess_mat_result=custom_postprocessing,
        kwargs_postprocessing=KWARGS_POSTPROCESSING,
    )

    # ## Study 2: Model comparison with shared parameters
    # All model variants are simulated with the same parameter set. You do not need to specify any parameters.
    # Total simulations: len(model_names) = 4
    print("\n--- Study 2: Model comparison (shared parameters) ---")
    model_study_params = {
        "parameterStudy.VPerQFlow": 50,
        "parameterStudy.TBiv": 273.15 - 5,
    }
    result_paths_study_2 = simple_dymola_sim_study(
        model_names=model_names_to_simulate,
        mos_script_pre=MOS_SCRIPT,
        simulation_setup=SIMULATION_SETUP,
        working_directory=Path(__file__).parent.joinpath("results", "working_directory"),
        save_path=Path(__file__).parent.joinpath("results", "SimResults_2"),
        parameters=model_study_params,
        model_result_file_names=model_result_names,
        postprocess_mat_result=custom_postprocessing,
        kwargs_postprocessing=KWARGS_POSTPROCESSING,
    )

    # ## Study 3: Model comparison with individual parameters
    # Each model variant gets its own parameter set.
    # If one model variant does not have a parameter set you have to set an empty dictionary {}
    # Total simulations: len(model_names) = 4
    print("\n--- Study 3: Model comparison (individual parameters) ---")
    rng = np.random.default_rng(42)
    random_example_values = rng.integers(5, 100, size=len(model_names_to_simulate))
    model_study_div_params = [
        {"parameterStudy.VPerQFlow": int(val)}
        for val in random_example_values
    ]
    result_paths_study_3 = simple_dymola_sim_study(
        model_names=model_names_to_simulate,
        mos_script_pre=MOS_SCRIPT,
        simulation_setup=SIMULATION_SETUP,
        working_directory=Path(__file__).parent.joinpath("results", "working_directory"),
        save_path=Path(__file__).parent.joinpath("results", "SimResults_3"),
        parameters=model_study_div_params,
        model_result_file_names=model_result_names,
        postprocess_mat_result=custom_postprocessing,
        kwargs_postprocessing=KWARGS_POSTPROCESSING,
    )

    print("\n--- All studies finished ---")
    print("You could now load the data again in a different script with `ebcpy.load_time_series_data()` "
          "for plotting and analysis.")