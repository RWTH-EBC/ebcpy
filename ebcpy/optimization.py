"""Base-module for the whole optimization pacakge.
Used to define Base-Classes such as Optimizer and
Calibrator."""

from collections import namedtuple
from abc import abstractmethod
import numpy as np
from ebcpy.utils import visualizer


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


    :param str,os.path.normpath cd:
        Directory for storing all output of optimization via a logger.
    :param dict kwargs:
        Keyword arguments can be used to further tune the optimization to your need.
        All keywords used in different optimization frameworks will be passed automatically
        to the functions when calling them, E.g. For scipy.optimize.minimize one could
        add "tol=1e-3" as a kwarg.
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
    # Dummy variable for selected optimization function
    _minimize_func = None
    # Initial-Value parameter
    x0 = np.array([])
    # Bounds for every parameter
    bounds = None
    _bound_max = None
    _bound_min = None
    # Instantiate framework parameter:
    framework = None
    method = None
    _framework_requires_method = True

    # Handle the kwargs
    # Scipy-minimize:
    tol = None
    options = None
    constraints = None
    jac = None
    hess = None
    hessp = None
    # dlib
    is_integer_variable = None
    # The maximal number of function evaluations in dlib is 1e9.
    solver_epsilon = 0
    num_function_calls = int(1e9)
    # scipy differential evolution
    maxiter = 1000
    popsize = 15
    mutation = (0.5, 1)
    recombination = 0.7
    seed = None
    polish = True
    init = 'latinhypercube'
    atol = 0
    # Define the list of supported kwargs:
    _dlib_kwargs = ["solver_epsilon", "num_function_calls"]
    _supported_kwargs = ["tol", "options", "constraints", "jac", "hess",
                         "hessp", "is_integer_variable", "method", "maxiter",
                         "popsize", "mutation", "recombination", "seed",
                         "polish", "init", "atol"] + _dlib_kwargs

    def __init__(self, cd, **kwargs):
        """Instantiate class parameters"""
        self.cd = cd
        self.logger = visualizer.Logger(self.cd, "Optimization")

        # Update kwargs with regard to what kwargs are supported.
        _not_supported = set(kwargs.keys()).difference(self._supported_kwargs)
        if _not_supported:
            raise KeyError("The following keyword-arguments are not "
                           "supported: \n{}".format(", ".join(list(_not_supported))))

        # By know only supported kwargs are in the dictionary.
        self.__dict__.update(kwargs)

        # This check if only necessary as the error-messages from dlib are quite indirect.
        # Any new user would not get that these parameters cause the error.
        for key in self._dlib_kwargs:
            value = self.__getattribute__(key)
            if not isinstance(value, (float, int)):
                raise TypeError("Given {} is of type {} but should be type float or "
                                "int".format(key, type(value).__name__))

    @abstractmethod
    def obj(self, xk, *args):
        """
        Base objective function. Overload this function and create your own
        objective function. Make sure that the return value is a scalar.
        Furthermore, the parameter vector xk is always a numpy array.

        :param np.array xk:
            Array with parameters for optimization
        :returns float result
            A scalar (float/ 1d) value for the optimization framework.
        """
        raise NotImplementedError('{}.obj function is not defined'.format(self.__class__.__name__))

    def optimize(self, framework, method=None):
        """
        Perform the optimization based on the given method and framework.

    :param str framework:
        The framework (python module) you want to use to perform the optimization.
        Currently, "scipy_minimize", "dlib_minimize" and "scipy_differential_evolution"
        are supported options. To further inform yourself about these frameworks, please see:
            - `dlib <http://dlib.net/python/index.html>`_
            - `scipy minimize <https://docs.scipy.org/doc/scipy/
            reference/generated/scipy.optimize.minimize.html>`_
            - `scipy differential evolution <https://docs.scipy.org/doc/scipy/
            reference/generated/scipy.optimize.differential_evolution.html>`_
        :param str method:
            The method you pass depends on the methods available in the framework
            you chose when setting up the class. Some frameworks don't require a
            method, as only one exists. This is the case for dlib. For any framework
            with different methods, you must provide one.
            For the scipy.differential_evolution function, method is equal to the
            strategy.
        :return: res
            Optimization result.
        """
        # Chosse the framework
        self._choose_framework(framework)
        if method:
            self.method = method
        if self.method is None and self._framework_requires_method:
            raise ValueError(f"{framework} requires a method, but None is "
                             f"provided. Please choose one.")
        # Perform minimization
        res = self._minimize_func(self.method)
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
        """
        if framework.lower() == "scipy_minimize":
            self._minimize_func = self._scipy_minimize
            self._framework_requires_method = True
        elif framework.lower() == "dlib_minimize":
            self._minimize_func = self._dlib_minimize
            self._framework_requires_method = False
        elif framework.lower() == "scipy_differential_evolution":
            self._minimize_func = self._scipy_differential_evolution
            self._framework_requires_method = True
        else:
            raise TypeError("Given framework {} is currently not supported.".format(framework))

    def _scipy_minimize(self, method):
        try:
            import scipy.optimize as opt
        except ImportError:
            raise ImportError("Please install scipy to use the minimize_scipy function.")

        try:
            res = opt.minimize(fun=self.obj,
                               x0=self.x0,
                               method=method,
                               jac=self.jac,
                               hess=self.hess,
                               hessp=self.hessp,
                               bounds=self.bounds,
                               constraints=self.constraints,
                               tol=self.tol,
                               options=self.options)
            return res
        except (KeyboardInterrupt, Exception) as error:
            self._handle_error(error)

    def _dlib_minimize(self, _):
        try:
            import dlib
        except ImportError:
            raise ImportError("Please install dlib to use the minimize_dlib function.")
        try:
            _bounds_2d = np.array(self.bounds)
            self._bound_min = list(_bounds_2d[:, 0])
            self._bound_max = list(_bounds_2d[:, 1])
            self.is_integer_variable = list(np.zeros(len(self._bound_max)))
            x_res, f_res = dlib.find_min_global(f=self._dlib_obj,
                                                bound1=self._bound_min,
                                                bound2=self._bound_max,
                                                is_integer_variable=self.is_integer_variable,
                                                num_function_calls=int(self.num_function_calls),
                                                solver_epsilon=float(self.solver_epsilon))
            res_tuple = namedtuple("res_tuple", "x fun")
            res = res_tuple(x=x_res, fun=f_res)
            return res
        except (KeyboardInterrupt, Exception) as error:
            self._handle_error(error)

    def _scipy_differential_evolution(self, method="best1bin"):
        try:
            import scipy.optimize as opt
        except ImportError:
            raise ImportError("Please install scipy to use the minimize_scipy function.")

        try:
            if self.bounds is None:
                raise ValueError("For the differential evolution approach, you need to specify "
                                 "boundaries. Currently, no bounds are specified.")
            if self.tol is None:
                # Default value. tol kwarg for scipy_minimize is None,
                # therefore this adjustment is necessary
                self.tol = 0.01

            # Anmerkung: Beim initialisieren wird self.obj, welche abstrakte methode in Klasse Calibrator ist,
            # welche wiederum Methode in ModelicaCalibrator Klasse ist, als callable class definiert.
            res = opt.differential_evolution(func=self.obj,         # callable: calls def obj() in calibrator class
                                             bounds=self.bounds,
                                             strategy=method,
                                             maxiter=self.maxiter,
                                             popsize=self.popsize,
                                             tol=self.tol,
                                             mutation=self.mutation,
                                             recombination=self.recombination,
                                             seed=self.seed,
                                             disp=False,  # We have our own logging.
                                             polish=self.polish,
                                             init=self.init,
                                             atol=self.atol)
            return res
        except (KeyboardInterrupt, Exception) as error:
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
        self.logger.log("Parameter set which caused the failure:")
        self.logger.log(str(self._current_iterate))
        self.logger.log("Current best objective and parameter set:")
        self.logger.log("\n".join(["{}: {}".format(key, value)
                                   for key, value in self._current_best_iterate.items()]))
        raise error
