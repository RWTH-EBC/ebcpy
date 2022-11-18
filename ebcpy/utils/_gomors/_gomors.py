"""
This module contains functions to implement to GOMORS-Solver
for "Multi objective optimization of computationally
expensive multi-modal functions with RBF surrogates
and multi-rule selection"
Source:
https://link.springer.com/article/10.1007/s10898-015-0270-y
"""

import os
from scipy.interpolate import Rbf
import numpy as np
from ebcpy.utils._gomors._hypervolume import HyperVolume

try:
    import lhsmdu
    import matplotlib.pyplot as plt
except ImportError as err:
    raise ImportError("To use GOMORS, you need to install: lhsmdu") from err


def generate_inital_points(num_variables, num_samples, experiment_method="latin-hypercube"):
    """
    This part of step 1 of the algorithm in the paper.
    Generate a uniform set of initial values to be evaluated
    by the optimizer (uniform: \in (0, 1))
    Evaluate the first set of parameters via a choosen
    method of experimental design, e.g. latin hypercube

    :param Integer num_variables:
        Number of variables for design
    :param Integer num_samples:
        Number of samples to simulate
    :param str experiment_method:
        Method for experimental design. Default is 'latin-hypercube'
        Supported are:
            - 'latin-hypercube'

    Thoughts on OED for RBF Interpolation
        - The measurements are the simulation
        - We have no model error --> sigma^2 is zero
        - Maximal information is maximum distance in objective space ??
        - The superstructure

    :return:
    """
    _supp_methods = ["latin-hypercube"]
    if experiment_method not in _supp_methods:
        raise ValueError(f"Given experiment_method {experiment_method} not supported. "
                         f"\nSupported methods: {', '.join(_supp_methods)}")

    if experiment_method == "latin-hypercube":
        samples = lhsmdu.sample(numDimensions=num_variables,
                                numSamples=num_samples)
        # Reshape from (num_variables, num_samples) to (num_variables, num_samples)
        return np.transpose(samples)


def optimize(objective, bounds, max_func_evals, gap_radius, ref_point, is_integer=None, **kwargs):
    """

    :param func objective:
        Objective function to be optimized. Should have the form:
        def foo(x):
            ...
        where x is the array of optimization variables
    :param np.array bounds:
        Boundaries used to scale the problem to a uniform space (between 0 and 1).
        Further used to design the initial evaluation.
    :param Integer max_func_evals:
        Maximum number of expensive function evaluations
    :param float gap_radius:
        The Gap radius parameter used in Step 2.3
    :param Integer initial_func_evals:
        Number of initial points to generate first response surface model.
    :param list is_integer:
        Out of the boundaries given, specify which entry (idx) is
        an integer (decision) variable. Example:
        [False, False, True, False] for 4 optimization variables with the third
        one being an integer.
        Default is None -> All False
    :return:
    """
    # Check input:
    if is_integer is None:
        is_integer = [False for _ in bounds]

    # Generate general settings:
    _n_opt_vars = len(bounds)

    # Unpack kwargs:
    n_random = kwargs.get("n_random", 1)
    n_hv = kwargs.get("n_hv", 1)
    n_domain = kwargs.get("n_domain", 1)
    n_obj_space = kwargs.get("n_obj_space", 1)
    n_hv_gap = kwargs.get("n_hv_gap", 1)
    ideal_point = kwargs.get("ideal_point", None)
    initial_func_evals = kwargs.get("initial_func_evals", _n_opt_vars * 10)
    show_plot3D = kwargs.get("show_plot3D", False)
    function = kwargs.get("function", "multiquadric")
    verbose = kwargs.get("verbose", False)
    save_log = kwargs.get("save_log", True)
    epsilon = kwargs.get("epsilon", None)
    smooth = kwargs.get("smooth", 0.0)

    # Perform step 1:
    inital_points = generate_inital_points(num_variables=_n_opt_vars,
                                           num_samples=initial_func_evals)

    # Descale and eval:
    s_m = np.array(inital_points)
    s_m_descaled = scale_to_normal_bounds(s_m, bounds)
    # Constraints:
    s_m = scale_to_unit_bounds(s_m_descaled, bounds)
    sF_m = objective(s_m_descaled)

    # Convert to numpy array for easier processing:
    sF_m = np.array(sF_m)
    # TODO-Future: https://cs.lbl.gov/assets/CSSSP-Slides/20200625-Mueller.pdf
    #  Handle NaNs better --> Equation for probability of failure
    _nan_mask = np.isnan(sF_m)
    # Convert so it works for s_m and objectives (different dims):
    _nan_mask = np.any(_nan_mask, axis=1)
    sF_m = sF_m[~_nan_mask]
    s_m = s_m[~_nan_mask]
    is_pareto_efficient = get_non_dominated_points(sF_m)
    p_m = s_m[is_pareto_efficient]
    pF_m = sF_m[is_pareto_efficient]

    if verbose:
        plt.plot(pF_m[:, 0], pF_m[:, 1], "ro")
        plt.draw()
        plt.pause(1e-5)

    if show_plot3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(s_m[:, 0], s_m[:, 1], sF_m[:, 0], color="r")
        plt.draw()
        plt.pause(1e-5)

    m = len(sF_m)
    while m <= max_func_evals:
        # Step 2.1: Fit Model:
        # radial basis function interpolator instance with shape for each objective
        rbf_models = []
        for obj_idx in range(sF_m.shape[1]):
            args = [s_m[:, i] for i in range(s_m.shape[1])] + [sF_m[:, obj_idx]]
            rbf_models.append(Rbf(*args, function=function, epsilon=epsilon, smooth=smooth))
            if show_plot3D:
                x = np.linspace(0, 1, 50)
                y = np.linspace(0, 1, 50)
                X, Y = np.meshgrid(x, y)
                Z = rbf_models[0](*[X, Y])
                ax.clear()
                ax.plot_surface(X, Y, Z)
                plt.draw()
                plt.pause(1e-5)

        # Step 2.2 - 2.3 are based pymoo
        rbf_pA_m, rbf_pAF_m = global_search(rbf_models, bounds, is_integer)
        x_crowded = get_least_crowded(p_m)
        rbf_pB_m, rbf_pBF_m = local_search(rbf_models, x_crowded, gap_radius, is_integer)

        # Step 2.4: Select Points for Expensive Function Evaluations
        s_curr = multi_rule_selection(sF_m, s_m, pF_m, rbf_pA_m, rbf_pB_m, rbf_pAF_m,
                                      rbf_pBF_m, ref_point, n_random=n_random, n_hv=n_hv,
                                      n_domain=n_domain, n_obj_space=n_obj_space, n_hv_gap=n_hv_gap,
                                      ideal_point=ideal_point)

        # Step 2.5 - Do expensive function evaluations and update Non-dominated solution set
        s_curr_descaled = scale_to_normal_bounds(s_m=s_curr, bounds=bounds)
        sF_m_curr = objective(s_curr_descaled)

        sF_m_curr = np.array(sF_m_curr)
        # TODO-Future: https://cs.lbl.gov/assets/CSSSP-Slides/20200625-Mueller.pdf
        #  Handle NaNs better --> Equation for probability of failure
        _nan_mask = np.isnan(sF_m_curr)
        # Convert so it works for s_m and objectives (different dims):
        _nan_mask = np.any(_nan_mask, axis=1)
        sF_m_curr = sF_m_curr[~_nan_mask]
        s_curr = s_curr[~_nan_mask]
        if show_plot3D:
            ax.scatter(s_curr[:, 0], s_curr[:, 1], sF_m_curr[:, 0], color="r")
            plt.draw()
            plt.pause(1e-5)

        # Build the intersection of the sets s_m and s_curr, update m:
        s_m = np.append(s_m, s_curr, axis=0)
        sF_m = np.append(sF_m, sF_m_curr, axis=0)
        # Update p_m:
        is_pareto_efficient = get_non_dominated_points(sF_m)
        p_m = s_m[is_pareto_efficient]
        pF_m = sF_m[is_pareto_efficient]
        if verbose:
            plt.plot(pF_m[:, 0], pF_m[:, 1], "ro")
            plt.draw()
            plt.pause(1e-5)
        m += len(s_curr)
        if save_log:
            with open(os.path.join(os.getcwd(), "gomors_log.csv"), "a+") as file:
                file.write(f"{m}_pF_m: {pF_m}\n")
                file.write(f"{m}_rbf_pAF_m: {rbf_pAF_m}\n")
                file.write(f"{m}_rbf_pBF_m: {rbf_pBF_m}\n")
        print(f"GOMORS evaluated {m} / {max_func_evals} expensive simulations")

    return p_m, pF_m


def get_non_dominated_points(obj_values):
    """
    Find the pareto-efficient points
    :param obj_values: An (n_points, n_costs) array
    :return: An array of indices of pareto-efficient points.
        This will be an (n_points, ) boolean array
    """
    is_efficient = np.arange(obj_values.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(obj_values):
        nondominated_point_mask = np.any(obj_values < obj_values[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        obj_values = obj_values[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    return is_efficient


def multi_rule_selection(sF_m, s_m, pF_m, rbf_pA_m, rbf_pB_m, rbf_pAF_m,
                         rbf_pBF_m, ref_point, n_random=1, n_hv=1, n_domain=1,
                         n_obj_space=1, n_hv_gap=1, ideal_point=None):
    """
    This algorithm is based on the details section of Step 2.4
    We have 5 rules:
    Rule 0: Random sampling
    Rule 1: Hypervolume Improvement
    Rule 2: Maximize Minimum Domain Euclidean Distance
    Rule 3: Maximize Minimum Objective Space Euclidean Distance
    Rule 4: Hypervolume Improvement in “Gap optimization” candidates (Temp disabled)
    """
    # If not enough new points can be selected, generate new random ones.
    _n_rdm_additional = 0

    _n_obj_vars = s_m.shape[1]

    _temp_s = []  # Used to avoid singular matrix

    # Rule 1:
    hypervolume_improvement = {}
    for idx, x_j in enumerate(rbf_pA_m):
        # Get HV-Improvement
        _hv_i = get_hypervolume(np.vstack((pF_m, rbf_pAF_m[idx, :])), ref_point=ref_point, ideal_point=ideal_point) - \
                get_hypervolume(pF_m, ref_point=ref_point, ideal_point=ideal_point)
        hypervolume_improvement[_hv_i] = x_j
    # Select n arg max values:
    for i in range(n_hv):
        if not hypervolume_improvement:
            _n_rdm_additional += n_hv - i
            break
        argmax_hv_i = hypervolume_improvement.get(max(hypervolume_improvement.keys()))
        _temp_s.append(argmax_hv_i)
        hypervolume_improvement.pop(max(hypervolume_improvement.keys()))

    # Rule 2:
    max_min_dom_euc_dis = {}
    for x_iA in rbf_pA_m:
        _min_dom_euc_dis = min([np.linalg.norm(x_iA - x_j) for x_j in s_m])
        max_min_dom_euc_dis[_min_dom_euc_dis] = x_iA
    # Select n min max domain values
    for i in range(n_domain):
        if not max_min_dom_euc_dis:
            _n_rdm_additional += n_domain - i
            break
        argmax_min_dom_euc_dis = max_min_dom_euc_dis.get(max(max_min_dom_euc_dis.keys()))
        _temp_s.append(argmax_min_dom_euc_dis)
        max_min_dom_euc_dis.pop(max(max_min_dom_euc_dis.keys()))

    # Rule 3:
    max_min_obj_euc_dis = {}
    for idx_pAm, x_iA in enumerate(rbf_pA_m):
        _min_obj_euc_dis = min([np.linalg.norm(rbf_pAF_m[idx_pAm, :] - sF_m[idx_sm])
                                for idx_sm, x_j in enumerate(s_m)])
        max_min_obj_euc_dis[_min_obj_euc_dis] = x_iA
    for i in range(n_obj_space):
        if not max_min_obj_euc_dis:
            _n_rdm_additional += n_obj_space - i
            break
        argmax_min_obj_euc_dis = max_min_obj_euc_dis.get(max(max_min_obj_euc_dis.keys()))
        _temp_s.append(argmax_min_obj_euc_dis)
        max_min_obj_euc_dis.pop(max(max_min_obj_euc_dis.keys()))

    # Rule 4: Temporarly disabled, as this maybe makes no sense in our case.
    hypervolume_improvement_gap = {}
    for idx, x_j in enumerate(rbf_pB_m):
        # Get HV-Improvment
        _hv_i_gap = get_hypervolume(np.vstack((pF_m, rbf_pBF_m[idx, :])),
                                    ref_point=ref_point,
                                    ideal_point=ideal_point) - \
                    get_hypervolume(pF_m, ref_point=ref_point,
                                    ideal_point=ideal_point)
        hypervolume_improvement_gap[_hv_i_gap] = x_j
    # Select arg max:
    for i in range(n_hv_gap):
        if not hypervolume_improvement_gap:
            _n_rdm_additional += n_hv_gap - i
            break
        argmax_hv_i_gap = hypervolume_improvement_gap.get(max(hypervolume_improvement_gap.keys()))
        _temp_s.append(argmax_hv_i_gap)
        hypervolume_improvement_gap.pop(max(hypervolume_improvement_gap.keys()))

    # Further check double entries and add more random values
    _n_rdm_additional += len(np.unique(_temp_s, axis=0))

    # Rule 0:
    for _ in range(n_random + _n_rdm_additional):
        _temp_s.append(np.random.rand(_n_obj_vars))

    # Remove possible doubled selected parameters
    s_curr = np.unique(_temp_s, axis=0)
    # Return only wanted set:
    return s_curr


def get_hypervolume(pF_m, ref_point, ideal_point=None):
    """
    Return the hypervolume of the given pareto-front
    """
    hv = HyperVolume(ref_point)
    vol = hv.compute(pF_m)
    if ideal_point:
        vol_ideal = hv.compute(ideal_point)
        return vol_ideal - vol
    else:
        return vol


def get_least_crowded(p_m):
    """
    Return least crowded parameter in p_m
    """
    cd = get_crowding_distances(p_m)

    return p_m[cd.argmin()]


def local_search(rbf_models, x_crowded, gap_radius, is_integer):
    """
    Function to perform the local search with the NSGA-II of cho's thesis
    :param rbf_models:
    :param x_crowded:
    :param gap_radius:
    :param normal_bounds:
    :param is_integer:
    :return:
    """
    # Create gap radius based bounds and descale them
    _bounds = [[max(0, x_c_scalar - gap_radius),
                min(x_c_scalar + gap_radius, 1)]
               for x_c_scalar in x_crowded]
    settings = {"pop_size": 20,
                "sampling": "real_random",  # Notice that changing Hyper-Parameters may change pop size.
                "selection": "random",
                "crossover": "real_sbx",
                "mutation": "real_pm",
                "eliminate_duplicates": True,
                "n_offsprings": None}
    return _ga_solver(
        rbf_models=rbf_models,
        settings=settings,
        bounds=_bounds,
        is_integer=is_integer
    )


def global_search(rbf_models, bounds, is_integer):
    settings = {"pop_size": 20,
                "sampling": "real_random",  # Notice that changing Hyper-Parameters may change pop size.
                "selection": "random",
                "crossover": "real_sbx",
                "mutation": "real_pm",
                "eliminate_duplicates": True,
                "n_offsprings": None}
    return _ga_solver(
        rbf_models=rbf_models,
        settings=settings,
        bounds=[(0, 1) for _ in bounds],
        is_integer=is_integer
    )


def _ga_solver(rbf_models, settings, bounds, is_integer):
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.problems.single import Problem
    from pymoo import factory
    from ebcpy.optimization import Optimizer
    kwargs = Optimizer.get_default_config(framework="pymoo")
    kwargs.update(settings)

    # GA:
    pop_size = settings["pop_size"]
    sampling = factory.get_sampling(name=kwargs["sampling"])
    selection = factory.get_selection(name=kwargs["selection"])
    crossover = factory.get_crossover(name=kwargs["crossover"])
    mutation = factory.get_mutation(name=kwargs["mutation"])

    eliminate_duplicates = kwargs["eliminate_duplicates"]
    n_offsprings = kwargs["n_offsprings"]
    algorithm = NSGA2(pop_size=pop_size,
                      sampling=sampling,
                      selection=selection,
                      crossover=crossover,
                      mutation=mutation,
                      eliminate_duplicates=eliminate_duplicates,
                      n_offsprings=n_offsprings
                      )

    class GOMORSGAProblem(Problem):
        """Construct wrapper problem class."""

        def __init__(self, _rbf_models: list, **kwargs):
            self._rbf_models = _rbf_models
            super().__init__(n_obj=len(_rbf_models), **kwargs)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = np.array([[rbf_model(*_x) for rbf_model in self._rbf_models] for _x in x])

    termination = kwargs.pop("termination")
    if termination is None:
        termination = ("n_gen", kwargs.pop("n_gen"))
    seed = kwargs.pop("seed")
    verbose = kwargs.pop("verbose")
    save_history = kwargs.pop("save_history")
    copy_algorithm = kwargs.pop("copy_algorithm")
    copy_termination = kwargs.pop("copy_termination")

    res = minimize(
        problem=GOMORSGAProblem(
            _rbf_models=rbf_models, bounds=bounds, n_constr=0,
            xl=np.array([b[0] for b in bounds]),
            xu=np.array([b[1] for b in bounds]),
            n_var=len(bounds)
        ),
        algorithm=algorithm,
        termination=termination,
        seed=seed,
        verbose=verbose,
        display=None,
        callback=None,
        save_history=save_history,
        copy_algorithm=copy_algorithm,
        copy_termination=copy_termination,
    )

    return res.X, res.F


def scale_to_unit_bounds(s_m_descaled, bounds):
    """
    Scale normal bounds (e.g 2kW to 20 kW) to
    unit box constraints (0, 1)

    :param np.array s_m_descaled:
        Numpy array in normal bounds
    :param np.array bounds:
        Bounds of opt_variables
    :return: np.array s_m:
        Numpy array in unit bounds
    """
    s_m = np.divide((s_m_descaled - bounds[:, 0]), (bounds[:, 1] - bounds[:, 0]))
    return s_m


def scale_to_normal_bounds(s_m, bounds):
    """
    Scale unit box constraints (0, 1) to
    normal bounds (e.g 2kW to 20 kW).

    :param np.array s_m:
        Numpy array in unit bounds
    :param np.array bounds:
        Bounds of opt_variables
    :return: np.array s_m_descaled:
        Numpy array in normal bounds
    """
    s_m_descaled = np.multiply(s_m, (bounds[:, 1] - bounds[:, 0])) + bounds[:, 0]
    return s_m_descaled


def get_crowding_distances(scores):
    """
    Crowding is based on a vector for each individual
    All dimension is normalised between low and high. For any one dimension, all
    solutions are sorted in order low to high. Crowding for chromsome x
    for that score is the difference between the next highest and next
    lowest score. Total crowding value sums all crowding for all scores
    Taken from: https://pythonhealthcare.org/2018/10/06/95-when-too-many-multi-objective-solutions-exist-selecting-solutions-based-on-crowding-distances/

    :param np.array scores:
        Pareto optimal solutions to calculate crowding distance

    :returns
        Crowding distances
    """

    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    normed_scores = (scores - scores.min(0)) / scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])

        sorted_scores_index = np.argsort(
            normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1:population_size - 1] = \
            (sorted_scores[2:population_size] -
             sorted_scores[0:population_size - 2])

        # resort to orginal order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances


if __name__ == "__main__":
    def foo(x_mp):
        """Dummy function to test the solver"""
        return [np.array([x[0] * x[1] ** 3 * 0.12738 + np.sin(x[1]) * 1.12738 * x[1] * x[0] ** 4, x[0]]) for x in x_mp]
    def foo(x_mp):
        """Dummy function to test the solver"""
        return [np.array([x[0] ** 3 - 2 * x[0] * x[1], x[0] * x[1] - x[1] ** 2]) for x in x_mp]


    x, F = optimize(objective=foo,
                    bounds=np.array([(10, 20), (20, 40)]),
                    max_func_evals=100,
                    gap_radius=0.1,
                    ref_point=[100, 100],
                    initial_func_evals=10,
                    show_plot3D=False,
                    verbose=True,
                    n_random=2,
                    n_hv=2,
                    n_domain=2,
                    n_hv_gap=2,
                    n_obj_space=2)
    plt.plot(F[:, 0], F[:, 1], "ro")
    plt.show()
