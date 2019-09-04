"""Test-module for all classes inside
ebcpy.optimization."""

import unittest
import os
import shutil
from ebcpy.modelica import manipulate_ds, simres
import numpy as np


class TestSimRes(unittest.TestCase):
    """Test-class for simres module"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        pass

    def test_to_pandas(self):
        """Test function for the function to_pandas"""
        pass

    def test_get_trajectories(self):
        """Test function for the function get_trajectories"""
        pass
    #TODO implement tests

class TestManipulateDS(unittest.TestCase):
    """Test-class for manipulate_ds module."""

    def setUp(self):
        """Called before every test.
            Used to setup relevant paths and APIs etc."""
        pass

    def test_convert_ds_file_to_dataframe(self):
        """Test function for the function convert_ds_file_to_dataframe"""
        pass

    def test_eliminate_parameters_from_ds_file(self):
        """Test function for the function eliminate_parameters_from_ds_file."""
        pass


if __name__ == "__main__":
    unittest.main()
