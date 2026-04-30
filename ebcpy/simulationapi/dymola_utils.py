from pathlib import Path
from typing import Union, List

from ebcpy import DymolaAPI, load_time_series_data


def _default_result_file_names(result_file_name, parameters):
    """Generate unique result file names from parameter values."""
    result_file_names = []
    for idx, param_dict in enumerate(parameters):
        name = result_file_name
        for key, val in param_dict.items():
            short_key = key.split(".")[-1]
            name += f"_{short_key}{str(val).replace('.', '_')}"
        result_file_names.append(name)
    return result_file_names


def simple_dymola_sim_study(
        model_names: List[str],
        simulation_setup: dict,
        working_directory: Union[str, Path],
        save_path: Union[str, Path],
        model_result_file_names: List[str],
        parameters: Union[dict, List[dict]] = None,
        n_cpu: int = 4,
        use_parameter_study: bool = False,
        result_file_name_func=None,
        kwargs_postprocessing: dict = None,
        postprocess_mat_result=None,
        mos_script_pre: Union[str, Path] = None,
        packages: List[Union[str, Path]] = None,
        **kwargs
):
    """
    Run a Dymola simulation study with multiple models and/or parameter variations.

    This function supports two simulation modes:

    **Parameter study** (``use_parameter_study=True``):
        Each model is simulated separately with all parameter sets.
        Useful when you want the full cross-product of models × parameters.
        Each model gets its own DymolaAPI instance, which translates the model
        once and then runs all parameter variations.

    **Model comparison** (``use_parameter_study=False``):
        All models are simulated in a single ``simulate()`` call using the
        ``model_names`` keyword. Each model can receive the same parameters
        (pass a single dict) or individual parameters (pass a list of dicts
        matching the length of ``model_names``). Each model is translated
        individually.

    Both modes use model name modifiers to change structural parameters.
    This is the recommended approach — write the modifier directly in the
    model name string.

    :param list[str] model_names:
        List of Dymola model names, optionally with modifiers.
        E.g. ``["MyModel(nLayer=1)", "MyModel(nLayer=2)"]``
    :param dict simulation_setup:
        Simulation settings with keys ``start_time``, ``stop_time``,
        and ``output_interval``.
    :param str,Path working_directory:
        Dymola working directory.
    :param str,Path save_path:
        Directory for saving simulation results.
    :param list[str] model_result_file_names:
        Base names for the result files, one per model.
    :param dict,list[dict] parameters:
        Parameter values for the simulation. For parameter studies, pass a list
        of dicts. For model comparison, pass a single dict (applied to all models)
        or a list of dicts (one per model).
    :param int n_cpu:
        Number of parallel Dymola processes. Default is 4.
    :param bool use_parameter_study:
        If True, runs each model with all parameter sets (cross-product).
        If False, runs all models in a single call.
    :param callable result_file_name_func:
        Function to generate unique result file names for parameter studies.
        Signature: ``func(result_file_name, parameters) -> list[str]``
        Default generates names by appending parameter key-value pairs.
    :param dict kwargs_postprocessing:
        Keyword arguments passed to the post-processing function.
        Required if ``postprocess_mat_result`` is provided.
    :param callable postprocess_mat_result:
        Custom post-processing function. If None (default), .mat files
        are kept unchanged. Signature: ``func(mat_result_file, **kwargs_postprocessing)``
    :param str,Path mos_script_pre:
        Path to a .mos script executed before loading packages.
        Typically, the startup script of your Modelica library.
    :param list packages:
        Additional Modelica packages not loaded by ``mos_script_pre``.
    :param kwargs:
        Additional keyword arguments forwarded to ``DymolaAPI`` constructor
        (e.g. ``show_window``, ``debug``, ``n_restart``, ``dymola_version``)
        and to ``DymolaAPI.simulate()`` (e.g. ``fail_on_error``).
    :return: Result file paths. For parameter studies, a dict mapping model names
        to lists of paths. For model comparison, a list of paths.
    :rtype: dict or list
    """
    # ## Default paths
    if working_directory is None:
        working_directory = Path(__file__).parent.joinpath("results", "working_directory")
    if save_path is None:
        save_path = Path(__file__).parent.joinpath("results", "SimResults")
    if packages is None:
        packages = []

    if len(model_result_file_names) != len(model_names):
        raise ValueError(
            f"model_result_file_names has length {len(model_result_file_names)} "
            f"but model_names has length {len(model_names)}. They must match."
        )
    if use_parameter_study and not isinstance(parameters, list):
        raise TypeError(
            "For parameter studies, parameters must be a list of dicts."
        )

    # ## Post-processing setup
    # Dymola produces .mat files by default. These are large and use a float
    # index (seconds). The post-processing function converts them to a more
    # usable format (e.g. datetime-indexed parquet) containing only the
    # variables you need.
    if postprocess_mat_result is not None and kwargs_postprocessing is None:
        raise ValueError(
            "kwargs_postprocessing is required when postprocess_mat_result is provided. "
            "Pass a dict with the keyword arguments for your post-processing function."
        )
        # Build the simulate kwargs for postprocessing
    postprocessing_kwargs = {}
    if postprocess_mat_result is not None:
        postprocessing_kwargs["postprocess_mat_result"] = postprocess_mat_result
        postprocessing_kwargs["kwargs_postprocessing"] = kwargs_postprocessing

    # ## Separate kwargs for DymolaAPI constructor and simulate()
    # Known simulate() kwargs are forwarded there, everything else goes to DymolaAPI.
    simulate_kwarg_keys = {"inputs", "table_name", "file_name", "fail_on_error", "show_eventlog", "squeeze"}
    simulate_kwargs = {k: kwargs.pop(k) for k in simulate_kwarg_keys if k in kwargs}

    # ## Run simulations
    if use_parameter_study:
        # ### Parameter study mode
        # Iterate over each model variant. For each model, a separate DymolaAPI
        # instance is created, the model is translated once, and all parameter
        # sets are simulated. This is efficient because translation (the slow part)
        # happens only once per model.
        if result_file_name_func is None:
            result_file_name_func = _default_result_file_names
        all_result_paths = {}
        for model_name, result_file_name in zip(model_names, model_result_file_names):
            # Create unique result file names by encoding the varied parameter values.
            # Adapt this naming scheme to your parameter study.
            result_file_names = result_file_name_func(result_file_name, parameters)

            dym_api = DymolaAPI(
                mos_script_pre=mos_script_pre,
                model_name=model_name,
                working_directory=working_directory,
                n_cpu=n_cpu,
                packages=packages,
                **kwargs
            )
            dym_api.set_sim_setup(sim_setup=simulation_setup)

            result_paths = dym_api.simulate(
                parameters=parameters,
                return_option="savepath",
                savepath=save_path,
                result_file_name=result_file_names,
                **postprocessing_kwargs,
                **simulate_kwargs
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
            packages=packages,
            **kwargs
        )
        dym_api.set_sim_setup(sim_setup=simulation_setup)

        result_paths = dym_api.simulate(
            model_names=model_names,
            parameters=parameters,
            return_option="savepath",
            savepath=save_path,
            result_file_name=model_result_file_names,
            **postprocessing_kwargs,
            **simulate_kwargs
        )
        dym_api.close()

        return result_paths