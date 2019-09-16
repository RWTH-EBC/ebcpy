"""Base-module for the whole optimization pacakge.
Used to define Base-Classes such as Optimizer and
Calibrator."""

from collections import namedtuple
from abc import abstractmethod
from ebcpy.utils import visualizer
import numpy as np


class Optimizer:
    """Base class for optimization in ebcpy. All classes
    performing optimization tasks must inherit from this
    class.

    :param str framework:
        The framework (python module) you want to use to perform the optimization.
        Currently, scipy and dlib are supported options. The further inform yourself
        about these frameworks, please see:
            - `dlib <http://dlib.net/python/index.html>`_
            - `scipy <https://docs.scipy.org/doc/scipy/
            reference/generated/scipy.optimize.minimize.html>`_
    :param str,os.path.normpath cd:
        Directory for storing all output of optimization.
    :param dict kwargs:
        Keyword arguments can be passed to this class. All given keywords
        will be an object of this class. The ones used in different optimization
        frameworks will be passed automatically to the functions when calling them.
        E.g. For scipy.optimize.minimize one could add "tol=1e-3" as a kwarg.

    """

    # Used to display number of obj-function-calls
    _counter = 0
    # Used to access the current parameter set if a optimization-step fails
    _current_iterate = np.array([])
    # List storing every objective value for plotting and logging.
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
    # Scipy:
    tol = None
    options = None
    constraints = ()
    jac = None
    hess = None
    hessp = None
    # dlib
    is_integer_variable = None
    # The maximal number of function evaluations in dlib is 1e9.
    solver_epsilon = 0
    num_function_calls = int(1e9)
    show_plot = True

    # Define the list of supported kwargs:
    _supported_kwargs = ["tol", "options", "constraints", "jac", "hess",
                         "hessp", "is_integer_variable", "solver_epsilon",
                         "num_function_calls", "show_plot", "method"]
    _dlib_kwargs = ["solver_epsilon", "num_function_calls"]

    def __init__(self, framework, cd, **kwargs):
        """Instantiate class parameters"""
        self.cd = cd
        self.logger = visualizer.Logger(self.cd, "Optimization")
        # Select the framework to work with while optimizing.
        self._choose_framework(framework)

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

    def optimize(self, method=None, framework=None):
        """
        Perform the optimization based on the given method and framework.

        :param str method:
            The method you pass depends on the methods available in the framework
            you choosed when setting up the class. Some frameworks don't require a
            method, as only one exists. This is the case for dlib. For any framework
            with different methods, you must provide one.
        :param str framework:
            If different you want to alter the frameworks within the same script,
            pass one of the supported frameworks as an optional argument here.
        :return: res
            Optimization result.
        """
        if framework:
            self._choose_framework(framework)
        if method:
            self.method = method
        if self.method is None and self._framework_requires_method:
            raise ValueError(f"{self.framework} requires a method, but None is "
                             f"provided. Please choose one.")
        # Perform minimization
        res = self._minimize_func(self.method)
        return res

    def _choose_framework(self, framework):
        """
        Function to select the functions for optimization
        and for executing said functions.

        :param str framework:
            String for selection of the relevant function. Currently,
            scipy and dlib are supported frameworks.
        """
        if framework.lower() == "scipy":
            self._minimize_func = self._minimize_scipy
            self._framework_requires_method = True
        elif framework.lower() == "dlib":
            self._minimize_func = self._minimize_dlib
            self._framework_requires_method = False
        else:
            raise TypeError("Given framework {} is currently not supported.".format(framework))
        # Update the class-parameter
        self.framework = framework.lower()

    def _minimize_scipy(self, method):
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
        except Exception as error:
            self.logger.log("Parameter set which caused the failure:")
            self.logger.log(str(self._current_iterate))
            raise error

    def _minimize_dlib(self, _):
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
        except Exception as error:
            self.logger.log("Parameter set which caused the failure:")
            self.logger.log(self._current_iterate)
            raise error

    def _dlib_obj(self, *args):
        """
        This function is needed as the signature for the dlib-obj
        is different than the standard signature. dlib will parse a number of
        parameters
        """
        return self.obj(np.array(args))
