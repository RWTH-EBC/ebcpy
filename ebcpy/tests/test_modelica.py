"""Test-module for all classes inside
ebcpy.optimization."""

import unittest
import os
from modelicares import SimRes
from ebcpy.modelica import manipulate_ds
import ebcpy.modelica.simres as sr_ebc
import pandas as pd


class TestSimRes(unittest.TestCase):
    """Test-class for simres module"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples//data")
        self.sim = SimRes(os.path.normpath(self.example_dir + '//ChuaCircuit.mat'))

    def test_to_pandas(self):
        """Test function for the function to_pandas"""
        df = sr_ebc.to_pandas(self.sim)
        first_col_name = df.columns[0]
        self.assertIsInstance(df, pd.DataFrame)
        df = sr_ebc.to_pandas(self.sim, with_unit=False)
        first_col_name_without_unit = df.columns[0]
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(first_col_name.startswith(first_col_name_without_unit))

    def test_get_trajectories(self):
        """Test function for the function get_trajectories"""
        trajectories = sr_ebc.get_trajectories(self.sim)
        self.assertEqual(39, len(trajectories))


class TestManipulateDS(unittest.TestCase):
    """Test-class for manipulate_ds module."""

    def setUp(self):
        """Called before every test.
            Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples//data")
        self.ds_path = os.path.normpath(self.example_dir + "//example_dsfinal.txt")

    def test_convert_ds_file_to_dataframe(self):
        """Test function for the function convert_ds_file_to_dataframe"""
        df = manipulate_ds.convert_ds_file_to_dataframe(self.ds_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_eliminate_parameters_from_ds_file(self):
        """Test function for the function eliminate_parameters_from_ds_file."""
        manipulate_ds.eliminate_parameters_from_ds_file(self.ds_path,
                                                        "dummy_dsout.txt",
                                                        [])
        self.assertTrue(os.path.isfile("dummy_dsout.txt"))
        os.remove("dummy_dsout.txt")


if __name__ == "__main__":
    unittest.main()
