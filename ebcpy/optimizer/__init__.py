"""Base-module for the whole optimizer pacakge.
Used to define Base-Classes such as Optimizer and
Calibrator."""

import os
from collections import namedtuple
from abc import abstractmethod
from ebcpy.utils import visualizer
from ebcpy import data_types
import numpy as np


class Optimizer:
    """Base class for optimization in ebcpy. All classes
    performing optimization tasks must inherit from this
    class.

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
    # The maximal number of function evaluations in dlib is 1e9.
    solver_epsilon = 0
    num_function_calls = int(1e9)
    show_plot = True

    # Define the list of supported kwargs:
    _supported_kwargs = ["tol", "options", "constraints", "jac", "hess",
                         "hessp", "is_integer_variable", "solver_epsilon",
                         "num_function_calls", "show_plot"]
    _dlib_kwargs = ["solver_epsilon", "num_function_calls"]

    def __init__(self, cd, **kwargs):
        """Instantiate class parameters"""
        # Check if given directory exists. If not, create it.
        if not os.path.isdir(cd):
            os.mkdir(cd)
        self.cd = cd
        self.logger = visualizer.Logger(cd, "optimization")

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
        Base objective function.

        :param np.array xk:
            Array with parameters for optimization
        """
        raise NotImplementedError('{}.obj function is not defined'.format(self.__class__.__name__))

    @abstractmethod
    def run(self, method, framework):
        """
        Function to select the functions for optimization
        and for executing said functions. This function has to be
        overloaded, only the selection of said functions takes place here.

        :param str method:
            Method for optimization
        :param str framework:
            String for selection of the relevant function
        """
        if framework.lower() == "scipy":
            self._minimize_func = self._minimize_scipy
        elif framework.lower() == "dlib":
            self._minimize_func = self._minimize_dlib
        else:
            raise TypeError("Given framework {} is currently not supported.".format(framework))

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


class Calibrator(Optimizer):
    """Base class for calibration in ebcpy. All classes
    performing calibration tasks must inherit from this
    class.
    """

    tuner_paras = data_types.TunerParas
    goals = data_types.Goals

    def __init__(self, cd, sim_api, statistical_measure, **kwargs):
        super().__init__(cd, **kwargs)
        self.sim_api = sim_api
        self.statistical_measure = statistical_measure

    @abstractmethod
    def obj(self, xk, *args):
        raise NotImplementedError('{}.obj function is not defined'.format(self.__class__.__name__))

    @abstractmethod
    def run(self, method, framework):
        super().run(method, framework)

    @abstractmethod
    def validate(self, goals):
        """
        Function to use different measurement data and run the objective function
        again to validate the calibration. The final parameter vector of the
        calibration is used.

        :param data_types.Goals goals:
            Goals with data to be validated
        """
        raise NotImplementedError('{}.validate function is not'
                                  ' defined'.format(self.__class__.__name__))
