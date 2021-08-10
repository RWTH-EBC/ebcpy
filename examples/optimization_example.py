"""
Goals of this part of the workshop:
1. Learn how to create a custom Optimizer class
2. Learn the different optimizer frameworks
"""
# Start by importing all relevant packages
import matplotlib.pyplot as plt
import numpy as np
# Imports from ebcpy
from ebcpy.optimization import Optimizer


def main():

    class MyCustomOptimizer(Optimizer):

        def __init__(self, goal, data, **kwargs):
            super().__init__(**kwargs)
            self.goal = goal
            self.data = data

        def obj(self, xk, *args):
            # Calculate the quadratic formula:
            quadratic_func = xk[0] * self.data ** 2 \
                             + xk[1] * self.data \
                             + xk[2]
            # Return the MAE of the quadratic function.
            return np.sum(np.abs(self.goal - quadratic_func))

    # Generate an array between 0 and pi
    my_data = np.linspace(0, np.pi, 100)
    my_goal = np.sin(my_data)

    mco = MyCustomOptimizer(goal=my_goal,
                            data=my_data,
                            bounds=[(-100, 100), (-100, 100), (-100, 100)]  # Specify bounds to the optimization
                            )

    framework_methods = {
        "scipy_differential_evolution": ("best1bin", {}),
        "scipy_minimize": ("L-BFGS-B", {"x0": [0, 0, 0]}),
        "dlib_minimize": (None, {"num_function_calls": 1000}),
        "pymoo": ("DE", {})
    }

    for framework, method_kwargs in framework_methods.items():
        method, kwargs = method_kwargs
        mco.logger.info("Optimizing framework %s with method %s", framework, method)
        res = mco.optimize(framework=framework, method=method, **kwargs)
        plt.figure()
        plt.plot(my_data, my_goal, "r", label="Reference")
        plt.plot(my_data, res.x[0] * my_data ** 2 + res.x[1] * my_data + res.x[2],
                 "b.", label="Regression")
        plt.legend(loc="upper left")
        plt.title(f"{framework}: {method}")

    plt.show()


if __name__ == '__main__':
    main()
