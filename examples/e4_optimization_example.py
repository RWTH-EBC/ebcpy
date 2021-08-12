"""
Goals of this part of the examples:
1. Learn how to create a custom `Optimizer` class
2. Learn the different optimizer frameworks
3. Learn the usage of `StatisticsAnalyzer`
4. Understand the motivation behing `AixCaliBuHA`
"""
# Start by importing all relevant packages
import matplotlib.pyplot as plt
import numpy as np
# Imports from ebcpy
from ebcpy.optimization import Optimizer
from ebcpy.utils.statistics_analyzer import StatisticsAnalyzer


def main(statistical_measure="MAE", with_plot=True):
    """
    Arguments of this example:
    :param str statistical_measure:
        The measure to use for regression analysis. Default is MAE.
        We refer to the documentation of the `StatisticsAnalyzer`
        class for other options
    :param bool with_plot:
        Show the plot at the end of the script. Default is True.
    """

    # ######################### Class definition ##########################
    # To create a custom optimizer, one needs to inherit from the Optimizer
    class PolynomalFitOptimizer(Optimizer):
        """
        Define a custom Optimizer by inheriting.
        This Optimizer finds the value a, b anc c for the function:
        f(x) = a * x ** 2 + b * x + c
        """

        def __init__(self, goal, data, stat_anaylzer, **kwargs):
            """
            In the init, add any data you want to access during optimization.
            You could also use global variables and don't overwrite the init,
            but as we all now: Don't use global variables.
            """
            super().__init__(**kwargs)
            # Set your custom data
            self.goal = goal
            self.data = data
            self.stat_anaylzer = stat_anaylzer

        def obj(self, xk, *args):
            """
            The only function you have to overwrite is the Optimizer.obj
            Here you have to calculate, based on the given current optimization variables xk,
            the objective value to minimize.
            This has to be a scalar value!!
            """
            # Calculate the quadratic formula:
            a, b, c = xk
            f_x = a * self.data ** 2 + b * self.data + c
            # Return the choosen statistical measure
            return self.stat_anaylzer.calc(self.goal, f_x)

    # Generate an array between 0 and pi
    my_data = np.linspace(0, np.pi, 100)
    my_goal = np.sin(my_data)
    stat_analyzer = StatisticsAnalyzer(statistical_measure)

    mco = PolynomalFitOptimizer(
        goal=my_goal,
        data=my_data,
        stat_anaylzer=stat_analyzer,
        bounds=[(-100, 100), (-100, 100), (-100, 100)]  # Specify bounds to the optimization
    )

    framework_methods = {
        "scipy_differential_evolution": ("best1bin", {}),
        "scipy_minimize": ("L-BFGS-B", {"x0": [0, 0, 0]}),
        "dlib_minimize": (None, {"num_function_calls": 1000}),
        "pymoo": ("NSGA2", {})
    }

    for framework, method_kwargs in framework_methods.items():
        method, kwargs = method_kwargs
        mco.logger.info("Optimizing framework %s with method %s and %s",
                        framework, method, statistical_measure)
        try:
            res = mco.optimize(framework=framework, method=method, **kwargs)
        except ImportError as err:
            mco.logger.error("Could not optimize due to import error %s", err)
        plt.figure()
        plt.plot(my_data, my_goal, "r", label="Reference")
        plt.plot(my_data, res.x[0] * my_data ** 2 + res.x[1] * my_data + res.x[2],
                 "b.", label="Regression")
        plt.legend(loc="upper left")
        plt.title(f"{framework}: {method}")
    if with_plot:
        plt.show()


if __name__ == '__main__':
    main(statistical_measure="R2")
