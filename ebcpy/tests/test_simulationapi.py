"""Test-module for all classes inside
ebcpy.simulationapi."""

import unittest
import os
import shutil
import pandas as pd
from ebcpy.simulationapi import dymola_api, fmu


class TestDymolaAPI(unittest.TestCase):
    """Test-Class for the DymolaAPI class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(os.path.join(framework_dir, "examples", "data"))
        self.example_sim_dir = os.path.join(self.example_dir, "testzone")
        if not os.path.exists(self.example_sim_dir):
            os.mkdir(self.example_sim_dir)
        ebcpy_test_package_dir = os.path.normpath(os.path.join(framework_dir,
                                                               "examples",
                                                               "Modelica",
                                                               "TestModel.mo"))
        packages = [ebcpy_test_package_dir]
        model_name = "AixCalTest_TestModel"
        self.initial_names = ["C",
                              "heatConv_b",
                              "heatConv_a"]
        self.initial_values = [2000, 5, 5]
        try:
            self.dym_api = dymola_api.DymolaAPI(self.example_sim_dir,
                                                model_name,
                                                packages)
        except (FileNotFoundError, ImportError, ConnectionError) as e:
            self.skipTest(f"Could not load the dymola interface on this machine. Error message: {e}")

    def test_close(self):
        """Test close functionality of dymola api"""
        self.dym_api.close()
        self.assertIsNone(self.dym_api.dymola)

    def test_simulate(self):
        """Test simulate functionality of dymola api"""
        self.dym_api.set_sim_setup({"startTime": 0.0,
                                    "stopTime": 10.0})
        res = self.dym_api.simulate()
        self.assertIsInstance(res, pd.DataFrame)

    def test_set_cd(self):
        """Test set_cd functionality of dymola api"""
        # Test the setting of the function
        self.dym_api.set_cd(self.example_dir)
        self.assertEqual(self.example_dir, self.dym_api.cd)

    def test_set_sim_setup(self):
        """Test set_sim_setup functionality of dymola api"""
        new_sim_setup = {'initialNames': self.initial_names,
                         'initialValues': self.initial_values}
        self.dym_api.set_sim_setup(new_sim_setup)
        self.assertEqual(self.dym_api.sim_setup['initialNames'],
                         new_sim_setup['initialNames'])
        self.assertEqual(self.dym_api.sim_setup['initialValues'],
                         new_sim_setup['initialValues'])
        with self.assertRaises(KeyError):
            self.dym_api.set_sim_setup({"NotAValidKey": None})
        with self.assertRaises(TypeError):
            self.dym_api.set_sim_setup({"stopTime": "not_a_float_or_int"})

    def tearDown(self):
        """Delete all files created while testing"""

        try:
            self.dym_api.close()
        except AttributeError:
            pass
        try:
            shutil.rmtree(self.example_sim_dir)
        except (FileNotFoundError, PermissionError):
            pass


class TestFMUAPI(unittest.TestCase):
    """Test-Class for the FMUAPI class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(os.path.join(framework_dir, "examples"))
        self.example_sim_dir = os.path.join(self.example_dir, "testzone")
        if not os.path.exists(self.example_sim_dir):
            os.mkdir(self.example_sim_dir)

        model_name = os.path.normpath(os.path.join(framework_dir,
                                                   "examples",
                                                   "Modelica",
                                                   "TestModel.fmu"))
        self.initial_names = ["C",
                              "heatConv_b",
                              "heatConv_a"]
        self.initial_values = [2000, 5, 5]
        try:
            self.fmu_api = fmu.FMU_API(self.example_sim_dir,
                                       model_name)
        except Exception as e:
            self.skipTest(f"Could not instantiate the fmu. Error message: {e}")

    def test_close(self):
        """Test close functionality of fmu api"""
        self.fmu_api.close()
        self.assertIsNone(self.fmu_api._unzip_dir)

    def test_simulate(self):
        """Test simulate functionality of fmu api"""
        self.fmu_api.set_sim_setup({"startTime": 0.0,
                                    "stopTime": 10.0})
        res = self.fmu_api.simulate()
        self.assertIsInstance(res, pd.DataFrame)

    def test_set_cd(self):
        """Test set_cd functionality of fmu api"""
        # Test the setting of the function
        self.fmu_api.set_cd(self.example_dir)
        self.assertEqual(self.example_dir, self.fmu_api.cd)

    def test_set_sim_setup(self):
        """Test set_sim_setup functionality of fmu api"""
        new_sim_setup = {'initialNames': self.initial_names,
                         'initialValues': self.initial_values}
        self.fmu_api.set_sim_setup(new_sim_setup)
        self.assertEqual(self.fmu_api.sim_setup['initialNames'],
                         new_sim_setup['initialNames'])
        self.assertEqual(self.fmu_api.sim_setup['initialValues'],
                         new_sim_setup['initialValues'])
        with self.assertRaises(KeyError):
            self.fmu_api.set_sim_setup({"NotAValidKey": None})
        with self.assertRaises(TypeError):
            self.fmu_api.set_sim_setup({"stopTime": "not_a_float_or_int"})

    def tearDown(self):
        """Delete all files created while testing"""

        try:
            self.fmu_api.close()
        except TypeError:
            pass
        try:
            shutil.rmtree(self.example_sim_dir)
        except (FileNotFoundError, PermissionError):
            pass


if __name__ == "__main__":
    unittest.main()
