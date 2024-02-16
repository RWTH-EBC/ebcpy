"""Base-module for the whole optimization pacakge.
Used to define Base-Classes such as Optimizer and
Calibrator."""

import os
from pathlib import Path
import warnings
from typing import List, Tuple, Union
from collections import namedtuple
from abc import abstractmethod
import numpy as np
from ebcpy.utils import setup_logger
# pylint: disable=import-outside-toplevel
# pylint: disable=broad-except


class Optimizer:
    """
    Base class for optimization in ebcpy. All classes
    performing optimization tasks must inherit from this
    class.
    The main feature of this class is the common interface
    for different available solvers in python. This makes the
    testing of different solvers and methods more easy.
    For available frameworks/solvers, check the function
    self.optimize().


    :param str,Path working_directory:
        Directory for storing all output of optimization via a logger.
    :keyword list bounds:
        The boundaries for the optimization variables.
    """

    # Used to display number of obj-function-calls
    _counter = 0
    # Used to access the current parameter set if an optimization-step fails
    _current_iterate = np.array([])
    # Used to access the best iterate if an optimization step fails
    _current_best_iterate = {"Objective": np.inf}
    # List storing every objective value for plotting and logging.
    # Can be used, but will enlarge runtime
    _obj_his = []

    def __init__(self, working_directory: Union[Path, str] = None, **kwargs):
        """Instantiate class parameters"""
        if working_directory is None and "cd" in kwargs:
            warnings.warn("cd was renamed to working_directory in all classes. Use working_directory instead.", category=DeprecationWarning)
            self.working_directory = kwargs["cd"]
        elif working_directory is None:
            self._working_directory = None
        else:
            self.working_directory = working_directory

        self.logger = setup_logger(working_directory=self.working_directory, name=self.__class__.__name__)
        # Set kwargs
        self.bounds = kwargs.get("bounds", None)

    @abstractmethod
    def obj(self, xk, *args):
        """
        Base objective function. Overload this function and create your own
        objective function. Make sure that the return value is a scalar.
        Furthermore, the parameter vector xk is always a numpy array.

        :param np.array xk:
            Array with parameters for optimization

        :return: float result:
            A scalar (float/ 1d) value for the optimization framework.
        """
        raise NotImplementedError(f'{self.__class__.__name__}.obj function is not defined')

    @abstractmethod
    def mp_obj(self, x, *args):
        """
        Objective function for Multiprocessing.

        :param np.array x:
            Array with parameters for optimization.
            Shape of the array is (number_of_evaluations x number_of_variables).
            For instance, optimizating 10 variables and evaluating
            900 objectives in parallel, the shape would be 900 x 10.
        :param int n_cpu:
            Number of logical Processors to run optimization on.
        """
        raise NotImplementedError(f'{self.__class__.__name__}.obj function is not defined')

    @property
    def supported_frameworks(self):
        """
        List with all frameworks supported by this
        wrapper class.
        """
        return ["scipy_minimize",
                "scipy_differential_evolution",
                "dlib_minimize",
                "pymoo"]

    @property
    def working_directory(self) -> Path:
        """The current working directory"""
        return self._working_directory

    @working_directory.setter
    def working_directory(self, working_directory: Union[Path, str]):
        """Set current working directory"""
        if isinstance(working_directory, str):
            working_directory = Path(working_directory)
        os.makedirs(working_directory, exist_ok=True)
        self._working_directory = working_directory

    @property
    def cd(self) -> Path:
        warnings.warn("cd was renamed to working_directory in all classes. Use working_directory instead instead.", category=DeprecationWarning)
        return self.working_directory

    @cd.setter
    def cd(self, cd: Union[Path, str]):
        warnings.warn("cd was renamed to working_directory in all classes. Use working_directory instead instead.", category=DeprecationWarning)
        self.working_directory = cd

    @property
    def bounds(self) -> List[Union[Tuple, List]]:
        """The boundaries of the optimization problem."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """Set the boundaries to the optimization variables"""
        self._bounds = bounds

    def optimize(self, framework, method=None, n_cpu=1, **kwargs):
        """
        Perform the optimization based on the given method and framework.

        :param str framework:
            The framework (python module) you want to use to perform the optimization.
            Currently, "scipy_minimize", "dlib_minimize" and "scipy_differential_evolution"
            are supported options. To further inform yourself about these frameworks, please see:
            - `dlib <http://dlib.net/python/index.html>`_
            - `scipy minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
            - `scipy differential evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`_
            - 'pymoo' <https://pymoo.org/index.html>
        :param str method:
            The method you pass depends on the methods available in the framework
            you chose when setting up the class. Some frameworks don't require a
            method, as only one exists. This is the case for dlib. For any framework
            with different methods, you must provide one.
            For the scipy.differential_evolution function, method is equal to the
            strategy.
            For the pymoo function, method is equal to the
            algorithm.
        :param int n_cpu:
            Number of parallel processes used for the evaluation.
            Ignored if the framework-method combination does not
            support multi-processing.

        Keyword arguments:
            Depending on the framework an method you use, you can fine-tune the
            optimization tool using extra arguments. We refer to the documentation of
            each framework for a listing of what parameters are supported and how
            to set them.
            E.g. For scipy.optimize.minimize one could
            add "tol=1e-3" as a kwarg.

        :return: res
            Optimization result.
        """
        # Choose the framework
        minimize_func, requires_method = self._choose_framework(framework)
        if method is None and requires_method:
            raise ValueError(f"{framework} requires a method, but None is "
                             f"provided. Please choose one.")
        # Perform minimization
        res = minimize_func(method=method, n_cpu=n_cpu, **kwargs)
        return res

    def _choose_framework(self, framework):
        """
        Function to select the functions for optimization
        and for executing said functions.

        :param str framework:
            String for selection of the relevant function. Supported options are:
            - scipy_minimize
            - dlib_minimize
            - scipy_differential_evolution
            - pymoo
        """
        if framework.lower() == "scipy_minimize":
            return self._scipy_minimize, True
        if framework.lower() == "dlib_minimize":
            return self._dlib_minimize, False
        if framework.lower() == "scipy_differential_evolution":
            return self._scipy_differential_evolution, True
        if framework.lower() == "pymoo":
            return self._pymoo, True
        raise TypeError(f"Given framework {framework} is currently not supported.")

    def _scipy_minimize(self, method, n_cpu=1, **kwargs):
        """
        Possible kwargs for the scipy minimize function with default values:

        x0: Required
        tol = None
        options = None
        constraints = {}
        jac = None
        hess = None
        hessp = None
        """
        default_kwargs = self.get_default_config(framework="scipy_minimize")
        default_kwargs.update(kwargs)
        try:
            import scipy.optimize as opt
        except ImportError as error:
            raise ImportError("Please install scipy to use "
                              "the minimize_scipy function.") from error

        try:
            if "x0" not in kwargs:
                raise KeyError("An initial guess (x0) is required "
                               "for scipy.minimize. You passed None")
            res = opt.minimize(
                fun=self.obj,
                x0=kwargs["x0"],
                method=method,
                jac=default_kwargs["jac"],
                hess=default_kwargs["hess"],
                hessp=default_kwargs["hessp"],
                bounds=self.bounds,
                constraints=default_kwargs["constraints"],
                tol=default_kwargs["tol"],
                options=default_kwargs["options"]
            )
            return res
        except (KeyboardInterrupt, Exception) as error:
            # pylint: disable=inconsistent-return-statements
            self._handle_error(error)

    def _dlib_minimize(self, method=None, n_cpu=1, **kwargs):
        """
        Possible kwargs for the dlib minimize function with default values:

        is_integer_variable = None
        solver_epsilon = 0
        num_function_calls = int(1e9)
        """
        default_kwargs = self.get_default_config(framework="dlib_minimize")
        default_kwargs.update(kwargs)
        try:
            import dlib
        except ImportError as error:
            raise ImportError("Please install dlib to use the minimize_dlib function.") from error
        try:
            _bounds_2d = np.array(self.bounds)
            _bound_min = list(_bounds_2d[:, 0])
            _bound_max = list(_bounds_2d[:, 1])
            if "is_integer_variable" not in kwargs:
                is_integer_variable = list(np.zeros(len(_bound_max)))
            else:
                is_integer_variable = kwargs["is_integer_variable"]

            # This check is only necessary as the error-messages from dlib are quite indirect.
            # Any new user would not get that these parameters cause the error.
            for key in ["solver_epsilon", "num_function_calls"]:
                value = kwargs.get(key)
                if value is not None:
                    if not isinstance(value, (float, int)):
                        raise TypeError(
                            f"Given {key} is of type {type(value).__name__} but "
                            f"should be type float or int"
                        )

            x_res, f_res = dlib.find_min_global(
                f=self._dlib_obj,
                bound1=_bound_min,
                bound2=_bound_max,
                is_integer_variable=is_integer_variable,
                num_function_calls=int(default_kwargs["num_function_calls"]),
                solver_epsilon=float(default_kwargs["solver_epsilon"])
            )
            res_tuple = namedtuple("res_tuple", "x fun")
            res = res_tuple(x=x_res, fun=f_res)
            return res
        except (KeyboardInterrupt, Exception) as error:
            # pylint: disable=inconsistent-return-statements
            self._handle_error(error)

    def _scipy_differential_evolution(self, method="best1bin", n_cpu=1, **kwargs):
        """
        Possible kwargs for the dlib minimize function with default values:

        maxiter = 1000
        popsize = 15
        tol = None
        mutation = (0.5, 1)
        recombination = 0.7
        seed = None
        polish = True
        init = 'latinhypercube'
        atol = 0
        """
        default_kwargs = self.get_default_config(framework="scipy_differential_evolution")
        default_kwargs.update(kwargs)
        try:
            import scipy.optimize as opt
        except ImportError as error:
            raise ImportError("Please install scipy to use the minimize_scipy function.") from error

        try:
            if self.bounds is None:
                raise ValueError("For the differential evolution approach, you need to specify "
                                 "boundaries. Currently, no bounds are specified.")

            res = opt.differential_evolution(
                func=self.obj,
                bounds=self.bounds,
                strategy=method,
                maxiter=default_kwargs["maxiter"],
                popsize=default_kwargs["popsize"],
                tol=default_kwargs["tol"],
                mutation=default_kwargs["mutation"],
                recombination=default_kwargs["recombination"],
                seed=default_kwargs["seed"],
                disp=False,  # We have our own logging
                polish=default_kwargs["polish"],
                init=default_kwargs["init"],
                atol=default_kwargs["atol"]
            )
            return res
        except (KeyboardInterrupt, Exception) as error:
            # pylint: disable=inconsistent-return-statements
            self._handle_error(error)

    def _pymoo(self, method="NSGA2", n_cpu=1, **kwargs):
        """
        Possible kwargs for the dlib minimize function with default values:

        algorithm=NGSA2
        termination=None
        seed=None
        verbose=False
        display=None
        callback=None
        save_history=False
        copy_algorithm=False
        copy_termination=False
        """
        default_kwargs = self.get_default_config(framework="pymoo")

        try:
            from pymoo.optimize import minimize
            from pymoo.problems.single import Problem
            from pymoo.factory import get_algorithm, get_sampling, get_mutation, get_crossover, get_selection
        except ImportError as error:
            raise ImportError("Please install pymoo to use this function.") from error

        class EBCPYProblem(Problem):
            """Construct wrapper problem class."""
            def __init__(self,
                         ebcpy_class: Optimizer
                         ):
                self.ebcpy_class = ebcpy_class
                super().__init__(n_var=len(ebcpy_class.bounds),
                                 n_obj=1,
                                 n_constr=0,
                                 xl=np.array([bound[0] for bound in ebcpy_class.bounds]),
                                 xu=np.array([bound[1] for bound in ebcpy_class.bounds])
                                 )

            def _evaluate(self, x, out, *args, **kwargs):
                if n_cpu > 1:
                    out["F"] = self.ebcpy_class.mp_obj(x, n_cpu, *args)
                else:
                    out["F"] = np.array([self.ebcpy_class.obj(xk=_x, *args) for _x in x])

        try:
            if self.bounds is None:
                raise ValueError("For pymoo, you need to specify "
                                 "boundaries. Currently, no bounds are specified.")

            termination = default_kwargs.pop("termination")
            if termination is None:
                termination = ("n_gen", default_kwargs.pop("n_gen"))
            seed = default_kwargs.pop("seed")
            verbose = default_kwargs.pop("verbose")
            save_history = default_kwargs.pop("save_history")
            copy_algorithm = default_kwargs.pop("copy_algorithm")
            copy_termination = default_kwargs.pop("copy_termination")

            # Init algorithm
            if method.lower() == "ga":
                from pymoo.algorithms.soo.nonconvex.ga import GA
                # GA:
                pop_size = kwargs["pop_size"]
                sampling = get_sampling(name=kwargs["sampling"])
                selection = get_selection(name=kwargs["selection"])
                crossover = get_crossover(name=kwargs["crossover"])
                mutation = get_mutation(name=kwargs["mutation"])
                eliminate_duplicates = kwargs["eliminate_duplicates"]
                n_offsprings = kwargs["n_offsprings"]
                algorithm = GA(pop_size=pop_size,
                               sampling=sampling,
                               selection=selection,
                               crossover=crossover,
                               mutation=mutation,
                               eliminate_duplicates=eliminate_duplicates,
                               n_offsprings=n_offsprings
                               )
            else:
                default_kwargs.update(kwargs)
                algorithm = get_algorithm(name=method.lower(),
                                          **default_kwargs)

            res = minimize(
                problem=EBCPYProblem(ebcpy_class=self),
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
            res_tuple = namedtuple("res_tuple", "x fun")
            res = res_tuple(x=res.X, fun=res.F[0])
            return res
        except (KeyboardInterrupt, Exception) as error:
            # pylint: disable=inconsistent-return-statements
            self._handle_error(error)

    def _dlib_obj(self, *args):
        """
        This function is needed as the signature for the dlib-obj
        is different than the standard signature. dlib will parse a number of
        parameters
        """
        return self.obj(np.array(args))

    def _handle_error(self, error):
        """
        Function to handle the case when an optimization step fails (e.g. simulation-fail).
        The parameter set which caused the failure and the best iterate until this point
        are of interest for the user in such case.
        :param error:
            Any Exception that may occur
        """
        self.logger.error(f"Parameter set which caused the failure: {self._current_iterate}")
        self.logger.error("Current best objective and parameter set:")
        self.logger.error("\n".join([f"{key}: {value}"
                                     for key, value in self._current_best_iterate.items()]))
        raise error

    @staticmethod
    def get_default_config(framework: str) -> dict:
        """
        Return the default config or kwargs for the
        given framework.

        The default values are extracted of the corresponding
        framework directly.
        """
        if framework.lower() == "scipy_minimize":
            return {"tol": None,
                    "options": None,
                    "constraints": None,
                    "jac": None,
                    "hess": None,
                    "hessp": None}
        if framework.lower() == "dlib_minimize":
            return {"num_function_calls": int(1e9),
                    "solver_epsilon": 0}
        if framework.lower() == "scipy_differential_evolution":
            return {"maxiter": 1000,
                    "popsize": 15,
                    "tol": 0.01,
                    "mutation": (0.5, 1),
                    "recombination": 0.7,
                    "seed": None,
                    "polish": True,
                    "init": 'latinhypercube',
                    "atol": 0
                    }
        if framework.lower() == "pymoo":
            return {"n_gen": 1000,
                    "termination": None,
                    "seed": 1,
                    "verbose": False,
                    "display": None,
                    "callback": None,
                    "save_history": False,
                    "copy_algorithm": False,
                    "copy_termination": False
                    }
        return {}
