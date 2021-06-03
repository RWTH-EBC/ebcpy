"""Test-module for all classes inside
ebcpy.optimization."""

import unittest
import os
from pathlib import Path
import pandas as pd
from ebcpy.modelica import manipulate_ds
from ebcpy.modelica.simres import SimRes


class TestSimRes(unittest.TestCase):
    """Test-class for simres module"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.example_dir = Path(__file__).parent.joinpath("data")
        self.sim = SimRes(self.example_dir.joinpath("example_data.mat"))

    def test_to_pandas(self):
        """Test function for the function to_pandas"""
        df = self.sim.to_pandas()
        first_col_name = df.columns[0]
        self.assertIsInstance(df, pd.DataFrame)
        df = self.sim.to_pandas(with_unit=False)
        first_col_name_without_unit = df.columns[0]
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(first_col_name.startswith(first_col_name_without_unit))

    def test_get_trajectories(self):
        """Test function for the function get_trajectories"""
        trajectories = self.sim.get_trajectories()
        self.assertEqual(9, len(trajectories))


class TestManipulateDS(unittest.TestCase):
    """Test-class for manipulate_ds module."""

    def setUp(self):
        """Called before every test.
            Used to setup relevant paths and APIs etc."""
        self.ds_path = Path(__file__).parent.joinpath("data",
                                                      "example_dsfinal.txt")

    def test_convert_ds_file_to_dataframe(self):
        """Test function for the function convert_ds_file_to_dataframe"""
        df = manipulate_ds.convert_ds_file_to_dataframe(self.ds_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_eliminate_parameters_from_ds_file(self):
        """Test function for the function eliminate_parameters_from_ds_file."""
        manipulate_ds.eliminate_parameters_from_ds_file(self.ds_path,
                                                        "dummy_dsout.txt",
                                                        [],
                                                        del_aux_paras=True)
        self.assertTrue(os.path.isfile("dummy_dsout.txt"))
        os.remove("dummy_dsout.txt")
        # Test dont remove aux
        manipulate_ds.eliminate_parameters_from_ds_file(self.ds_path,
                                                        "dummy_dsout.txt",
                                                        [],
                                                        del_aux_paras=False)
        self.assertTrue(os.path.isfile("dummy_dsout.txt"))
        os.remove("dummy_dsout.txt")
        # Test wring input:
        with self.assertRaises(TypeError):
            manipulate_ds.eliminate_parameters_from_ds_file(
                filename=self.ds_path,
                savepath="dummy_dsout.kdk",
                exclude_paras=[]
            )
        with self.assertRaises(TypeError):
            manipulate_ds.eliminate_parameters_from_ds_file(
                filename=self.ds_path,
                savepath="dummy_dsout.kdk",
                exclude_paras={"Not a": "list"}
            )


if __name__ == "__main__":
    unittest.main()
