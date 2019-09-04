"""Test-module for all classes inside
ebcpy.data_types."""

import os
import unittest
from ebcpy import data_types
import pandas as pd
import numpy as np


class TestDataTypes(unittest.TestCase):
    """Test-class for the data_types module of ebcpy."""

    def setUp(self):
        """Called before every test.
        Define example paths and parameters used in all test-functions.
        """
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples//data")
        self.example_data_hdf_path = os.path.normpath(self.example_dir +
                                                      "//example_data.hdf")
        self.example_data_csv_path = os.path.normpath(self.example_dir +
                                                      "//example_data.CSV")
        self.example_data_mat_path = os.path.normpath(self.example_dir +
                                                      "//example_data.mat")

    def test_time_series_data(self):
        """Test the class TimeSeriesData"""
        # Test if wrong input leads to FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            data_types.TimeSeriesData("Z:\\this_will_never_be_a_file_path.hdf")
        # Test if wrong file-ending leads to TypeError
        with self.assertRaises(TypeError):
            a_python_file = __file__
            data_types.TimeSeriesData(a_python_file)
        # If no key is provided, a KeyError has to be raised
        with self.assertRaises(KeyError):
            data_types.TimeSeriesData(self.example_data_hdf_path)
        with self.assertRaises(KeyError):
            data_types.TimeSeriesData(self.example_data_hdf_path, key="wrong_key")
        # Correctly load the .hdf:
        time_series_data = data_types.TimeSeriesData(self.example_data_hdf_path, key="parameters")
        self.assertIsInstance(
            time_series_data.df,
            type(pd.DataFrame()))
        # Correctly load the .csv:
        time_series_data = data_types.TimeSeriesData(self.example_data_csv_path, sep=",")
        self.assertIsInstance(
            time_series_data.df,
            type(pd.DataFrame()))
        # Correctly load the .mat:
        time_series_data = data_types.TimeSeriesData(self.example_data_mat_path)
        self.assertIsInstance(
            time_series_data.df,
            type(pd.DataFrame()))
        # Test load and set df functions:
        df = time_series_data.get_df()
        self.assertIsInstance(
            df,
            type(pd.DataFrame()))
        with self.assertRaises(TypeError):
            time_series_data.set_df("not a df")
        dummy_df = pd.DataFrame()
        time_series_data.set_df(dummy_df)
        self.assertTrue(time_series_data.df.empty)

    def test_meas_target_data(self):
        """Test the class MeasTargetData.
        For a detailed test of this class, see base-class test_time_series_data()"""
        meas_target_data = data_types.MeasTargetData(self.example_data_hdf_path, key="trajectories")
        self.assertEqual(meas_target_data.data_type, "MeasTargetData")

    def test_meas_input_data(self):
        """Test the class MeasInputData.
        For a detailed test of this class, see base-class test_time_series_data()"""
        meas_input_data = data_types.MeasInputData(self.example_data_hdf_path, key="trajectories")
        self.assertEqual(meas_input_data.data_type, "MeasInputData")

    def test_sim_target_data(self):
        """Test the class SimTargetData.
        For a detailed test of this class, see base-class test_time_series_data()"""
        sim_target_data = data_types.SimTargetData(self.example_data_hdf_path, key="trajectories")
        self.assertEqual(sim_target_data.data_type, "SimTargetData")

    def test_tuner_paras(self):
        """Test the class TunerParas"""
        dim = np.random.randint(1, 100)
        names = ["test_%s" % i for i in range(dim)]
        initial_values = np.random.rand(dim) * 10  # Values between 0 and 10.
        # Values between -100 and 110
        bounds = [(float(np.random.rand(1))*-100,
                   float(np.random.rand(1))*100 + 10) for i in range(dim)]
        # Check for false input
        with self.assertRaises(ValueError):
            wrong_bounds = [(0, 100),
                            (100, 0)]
            tuner_paras = data_types.TunerParas(names,
                                                initial_values,
                                                wrong_bounds)
        with self.assertRaises(ValueError):
            wrong_bounds = [(0, 100) for i in range(dim+1)]
            tuner_paras = data_types.TunerParas(names,
                                                initial_values,
                                                wrong_bounds)
        with self.assertRaises(ValueError):
            wrong_initial_values = np.random.rand(100)
            tuner_paras = data_types.TunerParas(names,
                                                wrong_initial_values)
        with self.assertRaises(TypeError):
            wrong_names = ["test_0", 123]
            tuner_paras = data_types.TunerParas(wrong_names,
                                                initial_values)
        with self.assertRaises(TypeError):
            wrong_initial_values = ["not an int", 123, 123]
            tuner_paras = data_types.TunerParas(names,
                                                wrong_initial_values)

        # Check return values of functions:
        tuner_paras = data_types.TunerParas(names,
                                            initial_values,
                                            bounds)
        scaled = np.random.rand(dim)  # between 0 and 1
        # Descale and scale again to check if the output is the almost (numeric precision) same
        descaled = tuner_paras.descale(scaled)
        scaled_return = tuner_paras.scale(descaled)
        np.testing.assert_almost_equal(scaled, scaled_return)
        self.assertEqual(names, tuner_paras.get_names())
        np.testing.assert_equal(tuner_paras.get_initial_values(),
                                initial_values)

        tuner_paras.get_bounds()
        val = tuner_paras.get_value("test_0", "min")
        tuner_paras.set_value("test_0", "min", val)
        with self.assertRaises(ValueError):
            tuner_paras.set_value("test_0", "min", 10000)
        with self.assertRaises(ValueError):
            tuner_paras.set_value("test_0", "min", "not_an_int_or_float")
        with self.assertRaises(KeyError):
            tuner_paras.set_value("test_0", "not_a_key", val)
        # Delete a name and check if the name is really gone.
        tuner_paras.remove_names(["test_0"])
        with self.assertRaises(KeyError):
            tuner_paras.get_value("test_0", "min")

    def test_goals(self):
        """Test the class Goals"""
        # Define some data.
        sim_target_data = data_types.SimTargetData(self.example_data_hdf_path, key="parameters")
        meas_target_data = data_types.MeasTargetData(self.example_data_hdf_path, key="parameters")
        meas_columns = ["sine.amplitude / ", "sine.phase / rad"]
        sim_columns = ["sine.freqHz / Hz", "sine.startTime / s"]
        # Setup the goals class:
        goals = data_types.Goals(meas_target_data,
                                 sim_target_data,
                                 meas_columns=meas_columns,
                                 sim_columns=sim_columns)
        # Check different formats of setting up:
        # First check if no columns are specified
        with self.assertRaises(TypeError):
            goals = data_types.Goals(meas_target_data,
                                     sim_target_data)
        # Now if passing of strings works:
        goals = data_types.Goals(meas_target_data,
                                 sim_target_data,
                                 sim_columns=sim_columns[0],
                                 meas_columns=meas_columns[1])
        self.assertIsInstance(goals, data_types.Goals)
        # Check the eval_difference function:
        self.assertIsInstance(goals.eval_difference("RMSE"), float)
        # Try to alter the sim_target_data object with something wrong
        with self.assertRaises(TypeError):
            goals.set_sim_target_data(meas_target_data)
        # Play around with wrong weightings:
        with self.assertRaises(ValueError):
            weightings = [1, 2]
            goals = data_types.Goals(meas_target_data,
                                     sim_target_data,
                                     meas_columns=meas_columns,
                                     sim_columns=sim_columns,
                                     weightings=weightings)
        with self.assertRaises(IndexError):
            weightings = np.ones(100)/100
            goals = data_types.Goals(meas_target_data,
                                     sim_target_data,
                                     meas_columns=meas_columns,
                                     sim_columns=sim_columns,
                                     weightings=weightings)

    def test_calibration_class(self):
        """Test the class CalibrationClass"""
        with self.assertRaises(ValueError):
            # Test if start-time higher than stop-time raises an error.
            data_types.CalibrationClass("dummy", 100, 50)
        with self.assertRaises(TypeError):
            # Test if a given name not equal to string raises an error.
            not_a_string = 1
            data_types.CalibrationClass(not_a_string, 0, 10)

        # Test set_functions for goals and tuner parameters
        dummy_tuner_para = "not TunerParas-Class"
        dummy_goal = "not Goals-Class"
        dummy_cal_class = data_types.CalibrationClass("dummy", 0, 10)
        with self.assertRaises(TypeError):
            dummy_cal_class.set_tuner_paras(dummy_tuner_para)
        with self.assertRaises(TypeError):
            dummy_cal_class.set_goals(dummy_goal)

    def test_get_keys_of_hdf_file(self):
        """Test the function get_keys_of_hdf_file.
        Check the keys of the file with e.g. the SDFEditor and
        use those keys as a reference list.
        """
        reference_list = ['parameters', 'trajectories']
        return_val = data_types.get_keys_of_hdf_file(self.example_data_hdf_path)
        self.assertListEqual(return_val, reference_list)


if __name__ == "__main__":
    unittest.main()
