"""Test-module for all examples"""
import os
import sys
import unittest
import pathlib
import importlib


class TestExample(unittest.TestCase):

    def setUp(self) -> None:
        self.timeout = 10  # Seconds which the script is allowed to run

    def _run_example(self, file: str, func_name: str, **kwargs):
        module_name = f'examples_test'
        file = pathlib.Path(__file__).parents[1].joinpath('examples', file)
        # Custom file import
        spec = importlib.util.spec_from_file_location(module_name, file)
        custom_module = importlib.util.module_from_spec(spec)
        sys.modules.update({module_name: custom_module})
        spec.loader.exec_module(custom_module)
        assert func_name in custom_module.__dict__.keys(), \
            f"Given filepath {file} does not contain the specified function {func_name}"

        test_func = custom_module.__dict__.get(func_name)
        test_func(**kwargs)

    def test_tst_example(self):
        """Execute e1_time_series_data_example.py"""
        self._run_example(file="e1_time_series_data_example.py",
                          func_name='main',
                          with_plot=False)

    def test_fmu_example(self):
        """Execute e2_fmu_example.py"""
        if "linux" in sys.platform:
            self.skipTest("Not supported in CI")
        self._run_example(file="e2_fmu_example.py",
                          func_name='main',
                          with_plot=False,
                          log_fmu=False,
                          n_cpu=1)

    def test_opt_example(self):
        """Execute e4_optimization_example.py"""
        self._run_example(file="e4_optimization_example.py",
                          func_name='main',
                          with_plot=False)
