"""Test-module for all classes inside
ebcpy.optimization."""

import unittest
import os
import shutil
from ebcpy.optimization import Optimizer
import numpy as np


class TestOptimizer(unittest.TestCase):
    """Test-class for optimization class"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(os.path.join(self.framework_dir,
                                                         "examples",
                                                         "data"))
        self.example_opt_dir = os.path.normpath(os.path.join(self.example_dir,
                                                             "test_optimization"))
        self.supported_frameworks = ["scipy", "dlib"]

    def test_optimizer_choose_function(self):
        """Test-case for the base-class for optimization."""
        opt = Optimizer("scipy", self.example_opt_dir)
        for _framework in self.supported_frameworks:
            if _framework == "scipy":
                reference_function = opt._minimize_scipy
            elif _framework == "dlib":
                reference_function = opt._minimize_dlib
            opt._choose_framework(_framework)
            self.assertEqual(opt._minimize_func, reference_function)
        with self.assertRaises(TypeError):
            opt._choose_framework("not_supported_framework")

    def test_custom_optimizer(self):
        """Test-case for the customization of the optimization-base-class."""
        # Define the customized class:
        class CustomOptimizer(Optimizer):
            """Dummy class"""

            x_goal = np.random.rand(3)
            x0 = np.array([0, 0, 0])

            def obj(self, xk, *args):
                x_data = np.linspace(-2, 2, 100)
                param_1, param_2, param_3 = xk[0], xk[1], xk[2]
                _param_1, _param_2, _param_3 = self.x_goal[0], self.x_goal[1], self.x_goal[2]
                quadratic_func_should = _param_1*x_data**2 + _param_2*x_data + _param_3
                quadratic_func_is = param_1*x_data**2 + param_2*x_data + param_3
                # Return the MAE of the quadratic function.
                return np.sum(np.abs(quadratic_func_should - quadratic_func_is))

        my_custom_optimizer = CustomOptimizer("scipy", self.example_opt_dir)
        res = my_custom_optimizer.optimize("L-BFGS-B")
        delta_solution = np.sum(res.x - my_custom_optimizer.x_goal)
        self.assertEqual(0.0, np.round(delta_solution, 3))
        with self.assertRaises(ValueError):
            my_custom_optimizer.optimize()

    def tearDown(self):
        """Remove all created folders while optimizing."""
        try:
            shutil.rmtree(self.example_opt_dir)
        except (FileNotFoundError, PermissionError):
            pass


if __name__ == "__main__":
    unittest.main()
