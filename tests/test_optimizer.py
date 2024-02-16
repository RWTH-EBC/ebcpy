"""Test-module for all classes inside
ebcpy.optimization."""
import sys
import unittest
import os
import shutil
import pathlib
import logging
import numpy as np
from ebcpy.optimization import Optimizer


logger = logging.getLogger(__name__)


class TestOptimizer(unittest.TestCase):
    """Test-class for optimization class"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.example_opt_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                             "test_optimization"))

    def test_optimizer_choose_function(self):
        """Test-case for the base-class for optimization."""
        # pylint: disable=protected-access
        opt = Optimizer()
        for _framework in opt.supported_frameworks:
            if _framework == "scipy_minimize":
                reference_function = opt._scipy_minimize
            elif _framework == "dlib_minimize":
                reference_function = opt._dlib_minimize
            elif _framework == "scipy_differential_evolution":
                reference_function = opt._scipy_differential_evolution
            elif _framework == "pymoo":
                reference_function = opt._pymoo
            _minimize_func, required_method = opt._choose_framework(_framework)
            self.assertEqual(_minimize_func, reference_function)
        with self.assertRaises(TypeError):
            opt._choose_framework("not_supported_framework")

    def test_set_and_delete_working_directory(self):
        """Test the working_directory and delete functions"""
        example_dir_as_pathlib_path = pathlib.Path(self.example_opt_dir)
        opt = Optimizer()
        self.assertIsNone(opt.working_directory)
        opt = Optimizer(working_directory=self.example_opt_dir)
        self.assertEqual(opt.working_directory, example_dir_as_pathlib_path)
        opt = Optimizer(working_directory=example_dir_as_pathlib_path)
        self.assertEqual(opt.working_directory, example_dir_as_pathlib_path)
        shutil.rmtree(opt.working_directory)

    def test_custom_optimizer(self):
        """Test-case for the customization of the optimization-base-class."""
        # Define the customized class:
        class CustomOptimizer(Optimizer):
            """Dummy class"""

            x_goal = np.random.rand(3)

            def obj(self, xk, *args):
                x_data = np.linspace(-2, 2, 100)
                param_1, param_2, param_3 = xk[0], xk[1], xk[2]
                _param_1, _param_2, _param_3 = self.x_goal[0], self.x_goal[1], self.x_goal[2]
                quadratic_func_should = _param_1*x_data**2 + _param_2*x_data + _param_3
                quadratic_func_is = param_1*x_data**2 + param_2*x_data + param_3
                # Return the MAE of the quadratic function.
                return np.sum(np.abs(quadratic_func_should - quadratic_func_is))

        my_custom_optimizer = CustomOptimizer()
        # Test value error if no method is supplied
        with self.assertRaises(ValueError):
            my_custom_optimizer.optimize(framework="scipy_minimize")
        # Test without x0
        with self.assertRaises(KeyError):
            my_custom_optimizer.optimize(framework="scipy_minimize",
                                         method="L-BFGS-B")
        # Test scipy minimize
        res_min = my_custom_optimizer.optimize(framework="scipy_minimize",
                                               method="L-BFGS-B",
                                               x0=np.array([0, 0, 0]))
        delta_solution = np.sum(res_min.x - my_custom_optimizer.x_goal)
        self.assertEqual(0.0, np.round(delta_solution, 3))
        # test wrong bounds in pymoo and sp_dif_evo
        my_custom_optimizer.bounds = None
        with self.assertRaises(ValueError):
            my_custom_optimizer.optimize(framework="scipy_differential_evolution",
                                         method="best2bin")
        with self.assertRaises(ValueError):
            my_custom_optimizer.optimize(framework="pymoo",
                                         method="NSGA2")

        # Test scipy differential evolution
        # Bounds are necessary (here, 1 and 0 are sufficient,
        #  as the goal values are element of [0,1]
        my_custom_optimizer.bounds = [(0, 1) for _ in range(3)]
        res_de = my_custom_optimizer.optimize(framework="scipy_differential_evolution",
                                              method="best2bin")
        delta_solution = np.sum(res_de.x - my_custom_optimizer.x_goal)
        self.assertEqual(0.0, np.round(delta_solution, 3))
        # Skip dlib test as problems in ci occur.
        if sys.version_info.minor >= 10:
            self.skipTest("pymoo is not yet supported in python 3.10")
        res_de = my_custom_optimizer.optimize(framework="pymoo",
                                              method="NSGA2")
        delta_solution = np.sum(res_de.x - my_custom_optimizer.x_goal)
        self.assertEqual(0.0, np.round(delta_solution, 3))

    def test_error_handler(self):
        """Test if error handling works for each framework"""
        opt = Optimizer()
        try:
            raise KeyError("My own error")
        except KeyError as error:
            with self.assertRaises(KeyError):
                opt._handle_error(error)

    def test_get_cfg(self):
        """Test get the default config"""
        opt = Optimizer()
        for framework in opt.supported_frameworks:
            cfg = opt.get_default_config(framework=framework)
            self.assertGreater(len(cfg), 0)
        self.assertEqual(opt.get_default_config("not supported"),
                         {})

    def tearDown(self):
        """Remove all created folders while optimizing."""
        try:
            shutil.rmtree(self.example_opt_dir)
        except (FileNotFoundError, PermissionError) as err:
            logger.error("Could not delete files %s", err)


if __name__ == "__main__":
    unittest.main()
