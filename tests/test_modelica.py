"""Test-module for all classes inside
ebcpy.optimization."""

import unittest
import os
from pathlib import Path
import pandas as pd
from ebcpy.modelica import manipulate_ds, \
    get_expressions, \
    get_names_and_values_of_lines
from ebcpy.modelica.simres import mat_to_pandas


class TestToPandas(unittest.TestCase):
    """Test-class for simres module"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        data_dir = Path(__file__).parent.joinpath("data")
        self.example_mat_dir = data_dir.joinpath("example_mat_data.mat")
        self.example_mo_dir = data_dir.joinpath("HeatPumpSystem.mo")

    def test_mat_to_pandas(self):
        """Test function for the function to_pandas"""
        df = mat_to_pandas(fname=self.example_mat_dir)
        first_col_name = df.columns[0]
        self.assertIsInstance(df, pd.DataFrame)
        df = mat_to_pandas(fname=self.example_mat_dir, with_unit=False)
        first_col_name_without_unit = df.columns[0]
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(first_col_name.startswith(first_col_name_without_unit))
        df = mat_to_pandas(fname=self.example_mat_dir,
                           with_unit=False,
                           names=['combiTimeTable.y[6]'])
        self.assertEqual(len(df.columns), 1)

    def test_get_variable_code(self):
        """Test function get variable code"""
        exp = get_expressions(filepath_model=self.example_mo_dir)
        self.assertEqual(len(exp), 24)
        exp = get_expressions(filepath_model=self.example_mo_dir, modelica_type="replaceable model")
        self.assertEqual(len(exp), 2)
        exp = get_expressions(filepath_model=self.example_mo_dir, modelica_type="variables")
        self.assertEqual(len(exp), 0)
        exp, exp_pr = get_expressions(filepath_model=self.example_mo_dir, get_protected=True)
        self.assertEqual(len(exp), 16)
        self.assertEqual(len(exp_pr), 8)

    def test_get_variable_values(self):
        """Test get variable names and values"""
        exp = get_expressions(filepath_model=self.example_mo_dir)
        for var_name, var_value in get_names_and_values_of_lines(exp).items():
            self.assertTrue(" " not in var_name)
            if var_value is not None:
                self.assertIsInstance(var_value, (float, int, bool))
        # Test doctest
        lines = ['parameter Boolean my_boolean=true "Some description"',
                 'parameter Real my_real=12.0 "Some description" annotation("Some annotation")']
        self.assertEqual(get_names_and_values_of_lines(lines=lines),
                         {'my_boolean': True, 'my_real': 12.0})
        lines = ['parameter Boolean my_boolean=true "Some description"',
                 '//parameter Real my_real=12.0 "Some description" annotation("Some annotation")']
        self.assertEqual(get_names_and_values_of_lines(lines=lines),
                         {'my_boolean': True})


class TestManipulateDS(unittest.TestCase):
    """Test-class for manipulate_ds module."""

    def setUp(self):
        """Called before every test.
            Used to setup relevant paths and APIs etc."""
        self.ds_paths = []
        for file in ["dsfinal_old.txt", "dsin_2023.txt", "dsfinal_2023.txt"]:
            self.ds_paths.append(Path(__file__).parent.joinpath("data", "ds_files", file))

    def test_convert_ds_file_to_dataframe(self):
        """Test function for the function convert_ds_file_to_dataframe"""
        for ds_path in self.ds_paths:
            df = manipulate_ds.convert_ds_file_to_dataframe(ds_path)
            self.assertIsInstance(df, pd.DataFrame)

    def test_eliminate_parameters_from_ds_file(self):
        """Test function for the function eliminate_parameters_from_ds_file."""
        for ds_path in self.ds_paths:
            self._single_file_eliminate_parameters_from_ds_file(ds_path)

    def _single_file_eliminate_parameters_from_ds_file(self, ds_path):
        """Test function for the function eliminate_parameters_from_ds_file."""
        manipulate_ds.eliminate_parameters_from_ds_file(ds_path,
                                                        "dummy_dsout.txt",
                                                        [],
                                                        del_aux_paras=True)
        self.assertTrue(os.path.isfile("dummy_dsout.txt"))
        os.remove("dummy_dsout.txt")
        # Test dont remove aux
        manipulate_ds.eliminate_parameters_from_ds_file(ds_path,
                                                        "dummy_dsout.txt",
                                                        [],
                                                        del_aux_paras=False)
        self.assertTrue(os.path.isfile("dummy_dsout.txt"))
        os.remove("dummy_dsout.txt")
        # Test wring input:
        with self.assertRaises(TypeError):
            manipulate_ds.eliminate_parameters_from_ds_file(
                filename=ds_path,
                savepath="dummy_dsout.kdk",
                exclude_paras=[]
            )
        with self.assertRaises(TypeError):
            manipulate_ds.eliminate_parameters_from_ds_file(
                filename=ds_path,
                savepath="dummy_dsout.txt",
                exclude_paras={"Not a": "list"}
            )


if __name__ == "__main__":
    unittest.main()
