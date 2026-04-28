# # Example: Basic Dymola Simulation Workflow
#
# Goals of this example:
# 1. Learn the recommended workflow for Dymola simulation studies using ebcpy
# 2. Understand how to run parameter studies across multiple model variants
# 3. Learn how to use model name modifiers
# 4. Learn how to post-process simulation results into usable formats
#
# This example demonstrates three common simulation patterns:
# - **Study 1:** Parameter study — multiple models × multiple parameter sets
# - **Study 2:** Model comparison — multiple models with the same parameters
# - **Study 3:** Model comparison — multiple models with individual parameters
#
# **Prerequisites:**
# This example requires a Dymola installation and the BESMod library.
# Adjust the `mos_script_pre` path to your local BESMod startup script.

import datetime
import os
from pathlib import Path
from typing import Union, List

import numpy as np

from ebcpy import DymolaAPI, load_time_series_data


def filter_strings(strings, required_substrings, forbidden_substrings=None):
    """
    Filters and returns strings that:
      - Contain all substrings from required_substrings.
      - Do not contain any substring from forbidden_substrings.

    Parameters:
        strings (list of str): The list of strings to search within.
        required_substrings (list of str): The substrings that each string must contain.
        forbidden_substrings (list of str, optional): The substrings that must not be present in any string.
            Defaults to an empty list if not provided.

    Returns:
        list of str: A list of strings that meet both criteria.
    """
    if forbidden_substrings is None:
        forbidden_substrings = []

    return [
        s for s in strings
        if all(req in s for req in required_substrings) and all(forb not in s for forb in forbidden_substrings)
    ]


def custom_postprocessing(mat_result_file, first_day_of_year, variable_names):
    """
    Post-process a Dymola .mat result file into a datetime-indexed parquet file.

    This function is passed to ``DymolaAPI.simulate()`` via the
    ``postprocess_mat_result`` keyword. It is called automatically
    for each simulation result. You can write your own custom post-processing functions

    :param str mat_result_file:
        Path to the .mat file produced by Dymola.
        Mandatory parameter passed over by ``DymolaAPI.simulate()``
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

    # adapt this to your study
    variables = df.columns.to_list()
    var_names_layer_temp = filter_strings(variables, ["layer", "T"])
    df["stoBuf.mean_T"] = df[var_names_layer_temp].mean(axis=1)

    df_path = Path(mat_result_file).with_suffix(".parquet")
    df.tsd.save(df_path)
    os.remove(mat_result_file)
    return df_path


def empty_postprocessing(mat_result, **_kwargs):
    """
    No-op post-processing function. Returns the .mat file path unchanged.
    Use this when you want to keep the original .mat result files.
    """
    return mat_result


def simple_dymola_sim_study(
        model_names: List[str],
        mos_script_pre: Union[str, Path],
        simulation_setup,
        parameters: Union[dict, List[dict]] = None,
        working_directory: Union[str, Path] = None,
        n_cpu: int = 2,
        use_postprocessing: bool = True,
        use_parameter_study: bool = False,
        model_result_file_names: List[str] = None,
        save_path: Union[str, Path] = None,
        kwargs_postprocessing: dict = None,
        **kwargs
):
    """
    Run a Dymola simulation study with multiple models and/or parameter variations.

    This function demonstrates two simulation modes:

    **Parameter study** (``use_parameter_study=True``):
        Each model is simulated separately with all parameter sets.
        Useful when you want the full cross-product of models × parameters.
        Each model gets its own DymolaAPI instance, which translates the model
        once and then runs all parameter variations.

    **Model comparison** (``use_parameter_study=False``):
        All models are simulated in a single ``simulate()`` call using the
        ``model_names`` keyword. Each model can receive the same parameters
        (pass a single dict) or individual parameters (pass a list of dicts
        matching the length of ``model_names``). Each model run will be translated

    Both modes use model name modifiers to change structural parameters.
    This is the recommended approach — write the modifier directly in the
    model name string.

    :param list[str] model_names:
        List of Dymola model names, optionally with modifiers.
        E.g. ``["MyModel(nLayer=1)", "MyModel(nLayer=2)"]``
    :param str,Path mos_script_pre:
        Path to a .mos script executed before loading packages.
        Typically the startup script of your Modelica library.
    :param dict,list[dict] parameters:
        Parameter values for the simulation. For parameter studies, pass a list
        of dicts. For model comparison, pass a single dict (applied to all models)
        or a list of dicts (one per model).
    :param str,Path working_directory:
        Dymola working directory. Default is ``./results/working_directory``.
    :param int n_cpu:
        Number of parallel Dymola processes. Default is 2.
    :param bool use_postprocessing:
        If True, .mat files are converted to datetime-indexed parquet files.
        If False, raw .mat files are kept.
    :param bool use_parameter_study:
        If True, runs each model with all parameter sets (cross-product).
        If False, runs all models in a single call.
    :param list[str] model_result_file_names:
        Base names for the result files, one per model.
    :param str,Path save_path:
        Directory for saving simulation results. Default is ``./results/SimResults``.
    :return: Result file paths. For parameter studies, a dict mapping model names
        to lists of paths. For model comparison, a list of paths.
    :rtype: dict or list
    """
    # ## Default paths
    if working_directory is None:
        working_directory = Path(__file__).parent.joinpath("results", "working_directory")
    if save_path is None:
        save_path = Path(__file__).parent.joinpath("results", "SimResults")

    # ## Simulation setup
    # Define the time range and output resolution for all simulations.
    # The output_interval determines the time step of the result data.

    additional_packages = kwargs.pop("additional_packages", [])  # use this for custom packages which are note in the mos_script_pre

    # ## Post-processing setup
    # Dymola produces .mat files by default. These are large and use a float
    # index (seconds). The post-processing function converts them to
    # datetime-indexed parquet files containing only the variables we need.
    # If you want to keep the raw .mat files, set use_postprocessing=False.
    if use_postprocessing:
        postprocess_mat_result = custom_postprocessing
        if kwargs_postprocessing is None:
            raise Exception
    else:
        postprocess_mat_result = empty_postprocessing
        kwargs_postprocessing = {}

    # ## Run simulations
    if use_parameter_study:
        # ### Parameter study mode
        # Iterate over each model variant. For each model, a separate DymolaAPI
        # instance is created, the model is translated once, and all parameter
        # sets are simulated. This is efficient because translation (the slow part)
        # happens only once per model.
        all_result_paths = {}
        for model_name, result_file_name in zip(model_names, model_result_file_names):
            # Create unique result file names by encoding the varied parameter values
            result_file_names = []
            for param_dict in parameters:
                name = result_file_name
                for key, val in param_dict.items():
                    name += f"_{key.split(".")[-1]}{val.replace('.', '_')}"
                result_file_names.append(name)

            dym_api = DymolaAPI(
                mos_script_pre=mos_script_pre,
                model_name=model_name,
                working_directory=working_directory,
                n_cpu=n_cpu,
                equidistant_output=True,
                packages=additional_packages
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
        # ### Model comparison mode
        # All models are simulated in a single DymolaAPI call using the
        # model_names keyword. Dymola translates each model on the fly.
        # This is convenient for comparing different model configurations
        # with the same or individual parameter sets.
        dym_api = DymolaAPI(
            mos_script_pre=mos_script_pre,
            working_directory=working_directory,
            n_cpu=n_cpu,
            equidistant_output=True,
            packages=additional_packages
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
    # TODO: Adjust this path to your local BESMod startup script
    MOS_SCRIPT = r"D:\01_git\BESMod\startup.mos"

    # adapt this to your study
    simulation_setup = {
        "start_time": 0,
        "stop_time": 3600 * 24 * 30,  # 30 days
        "output_interval": 900  # one data point every 900 s (15 min)
    }

    # ## Define model variants
    # We use model name modifiers to vary the number of storage layers.
    # This is the recommended approach for structural parameters which need a retranslation —
    # write the modifier directly in the model name string. You can also redeclare models here.
    base_model_name = "BESMod.Examples.DesignOptimization.BES"
    storage_layers = np.arange(1, 5, 1, dtype=int)
    model_names_to_simulate = [
        f"{base_model_name}(hydraulic.distribution.parStoBuf(nLayer={n}))"
        for n in storage_layers
    ]
    model_result_names = [f"BufSto_nLayer{n}" for n in storage_layers]

    print("Model variants:", model_names_to_simulate)

    # ## Study 0: Single model, default parameters
    # The simplest possible simulation — verify your setup works.
    print("\n--- Study 0: Single model ---")
    result_paths_study_0 = simple_dymola_sim_study(
        model_names=[model_names_to_simulate[0]],
        mos_script_pre=MOS_SCRIPT,
        simulation_setup=simulation_setup,
        use_parameter_study=False,
        use_postprocessing=False,
        model_result_file_names=[model_result_names[0]],
        save_path=Path(__file__).parent.joinpath("results", "SimResults_0"),
    )
    # adapt this to your study
    first_day_of_year = datetime.datetime(2015, 1, 1, 0, 0)
    kwargs_postprocessing = dict(
        variable_names=["outputs*",
                        "hydraulic.distribution.stoBuf.layer[*].T",
                        "hydraulic.distribution.stoBuf.port_a_consumer.m_flow",
                        "hydraulic.distribution.stoBuf.port_a_heatGenerator.m_flow",
                        "hydraulic.distribution.sigBusDis.*"
                        "hydraulic.generation.sigBusGen.*"],
        first_day_of_year=first_day_of_year
    )

    # ## Study 1: Parameter study
    # ## Define parameter variations
    parameter_study_params = [
        {"parameterStudy.VPerQFlow": np.round(v, decimals=1)}
        for v in np.linspace(5, 100, 8)
    ]
    print("Parameter sets:", parameter_study_params)

    # Each model variant is simulated with all parameter sets.
    # Total simulations: len(model_names) × len(parameter_sets) = 4 × 8 = 32
    print("\n--- Study 1: Parameter study ---")
    result_paths_study_1 = simple_dymola_sim_study(
        model_names=model_names_to_simulate,
        mos_script_pre=MOS_SCRIPT,
        simulation_setup=simulation_setup,
        parameters=parameter_study_params,
        use_parameter_study=True,
        model_result_file_names=model_result_names,
        save_path=Path(__file__).parent.joinpath("results", "SimResults_1"),
        kwargs_postprocessing=kwargs_postprocessing,
    )

    # ## Study 2: Model comparison with shared parameters
    # All model variants are simulated with the same parameter set.
    # Total simulations: len(model_names) = 4
    print("\n--- Study 2: Model comparison (shared parameters) ---")
    model_study_params = {
        "parameterStudy.VPerQFlow": 50,
        "parameterStudy.TBiv": 273.15 - 5
    }
    result_paths_study_2 = simple_dymola_sim_study(
        model_names=model_names_to_simulate,
        mos_script_pre=MOS_SCRIPT,
        simulation_setup=simulation_setup,
        parameters=model_study_params,
        use_parameter_study=False,
        model_result_file_names=model_result_names,
        save_path=Path(__file__).parent.joinpath("results", "SimResults_2"),
        kwargs_postprocessing=kwargs_postprocessing,
    )

    # ## Study 3: Model comparison with individual parameters
    # Each model variant gets its own parameter set.
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
        simulation_setup=simulation_setup,
        parameters=model_study_div_params,
        use_parameter_study=False,
        model_result_file_names=model_result_names,
        save_path=Path(__file__).parent.joinpath("results", "SimResults_3"),
        kwargs_postprocessing=kwargs_postprocessing,
    )

    print("\n--- All studies finished ---")