"""Test-module for all classes inside
ebcpy.utils."""

import unittest
import os
import numpy as np
from ebcpy.utils import visualizer
from ebcpy.utils import statistics_analyzer


class TestStatisticsAnalyzer(unittest.TestCase):
    """Test-class for the StatisticsAnalyzer-Class"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples//data")
        self.meas_ex = np.random.rand(1000)
        self.sim_ex = np.random.rand(1000)*10

    def test_calc(self):
        """Test class StatisticsAnalyzer"""
        sup_methods = ["mae", "r2", "mse", "rmse", "cvrmse", "nrmse"]
        for method in sup_methods:
            stat_analyzer = statistics_analyzer.StatisticsAnalyzer(method)
            self.assertIsInstance(stat_analyzer.calc(self.meas_ex, self.sim_ex),
                                  float)
        with self.assertRaises(ValueError):
            stat_analyzer = statistics_analyzer.StatisticsAnalyzer("not_supported_method")

    def test_calc_rmse(self):
        """Test static function calc_rmse"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_rmse(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_mse(self):
        """Test static function calc_mse"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_mse(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_mae(self):
        """Test static function calc_mae"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_mae(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_nrmse(self):
        """Test static function calc_nrmse"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_nrmse(self.meas_ex, self.sim_ex),
                              float)
        with self.assertRaises(ValueError):
            custom_meas = self.meas_ex/self.meas_ex
            stat_analyzer.calc_nrmse(custom_meas, self.sim_ex)

    def test_calc_cvrmse(self):
        """Test static function calc_cvrmse"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_cvrmse(self.meas_ex, self.sim_ex),
                              float)
        with self.assertRaises(ValueError):
            custom_meas = self.meas_ex - self.meas_ex
            stat_analyzer.calc_cvrmse(custom_meas, self.sim_ex)

    def test_calc_r2(self):
        """Test static function calc_r2"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_r2(self.meas_ex, self.sim_ex),
                              float)


class TestVisualizer(unittest.TestCase):
    """Test-class for the visualizer module."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples//data")
        self.logger = visualizer.Logger(self.example_dir, "test_logger")

    def test_logging(self):
        """Test if logging works."""
        example_str = "This is a test"
        self.logger.log(example_str)
        with open(self.logger.filepath_log, "r") as logfile:
            logfile.seek(0)
            self.assertEqual(logfile.read()[-len(example_str):], example_str)

    def tearDown(self):
        """Remove created files."""
        os.remove(self.logger.filepath_log)


if __name__ == "__main__":
    unittest.main()
