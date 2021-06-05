"""Test-module for all classes inside
ebcpy.simulationapi."""

import unittest
import sys
import os
from pathlib import Path
import shutil
from ebcpy.simulationapi import dymola_api, fmu
from ebcpy import TimeSeriesData


class TestDymolaAPI(unittest.TestCase):
    """Test-Class for the DymolaAPI class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.example_dir = Path(__file__).parent
        self.example_sim_dir = self.example_dir.joinpath("testzone")
        if not os.path.exists(self.example_sim_dir):
            os.mkdir(self.example_sim_dir)
        ebcpy_test_package_dir = self.example_dir.joinpath("data", "TestModel.mo")
        packages = [ebcpy_test_package_dir]
        model_name = "AixCalTest_TestModel"
        self.initial_names = ["C",
                              "heatConv_b",
                              "heatConv_a"]
        self.initial_values = [2000, 5, 5]
        # Just for tests in the ci:
        if "linux" in sys.platform:
            dymola_path = "/usr/local/bin/dymola"
        else:
            dymola_path = None
        try:
            self.dym_api = dymola_api.DymolaAPI(cd=self.example_sim_dir,
                                                model_name=model_name,
                                                packages=packages,
                                                dymola_path=dymola_path)
        except (FileNotFoundError, ImportError, ConnectionError) as error:
            self.skipTest(f"Could not load the dymola interface "
                          f"on this machine. Error message: {error}")

    def test_close(self):
        """Test close functionality of dymola api"""
        self.dym_api.close()
        self.assertIsNone(self.dym_api.dymola)

    def test_simulate(self):
        """Test simulate functionality of dymola api"""
        self.dym_api.set_sim_setup({"startTime": 0.0,
                                    "stopTime": 10.0})
        res = self.dym_api.simulate()
        if len(self.dym_api.sim_setup["resultNames"]) > 1:
            self.assertIsInstance(res, TimeSeriesData)
        else:
            self.assertEqual(res, [])

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
        self.data_dir = Path(__file__).parent.joinpath("data")
        self.example_sim_dir = os.path.join(self.data_dir, "testzone")
        if not os.path.exists(self.example_sim_dir):
            os.mkdir(self.example_sim_dir)

        self.initial_names = ["C",
                              "heatConv_b",
                              "heatConv_a"]
        self.initial_values = [2000, 5, 5]
        if "win" in sys.platform:
            model_name = self.data_dir.joinpath("PumpAndValve_windows.fmu")
        else:
            model_name = self.data_dir.joinpath("PumpAndValve_linux.fmu")

        self.fmu_api = fmu.FMU_API(cd=self.example_sim_dir,
                                   model_name=model_name)

    def test_close(self):
        """Test close functionality of fmu api"""
        # pylint: disable=protected-access
        self.fmu_api.close()
        self.assertIsNone(self.fmu_api._unzip_dir)

    def test_simulate(self):
        """Test simulate functionality of fmu api"""
        self.fmu_api.set_sim_setup({"startTime": 0.0,
                                    "stopTime": 10.0})
        res = self.fmu_api.simulate()
        self.assertIsInstance(res, TimeSeriesData)

    def test_set_cd(self):
        """Test set_cd functionality of fmu api"""
        # Test the setting of the function
        self.fmu_api.set_cd(self.data_dir)
        self.assertEqual(self.data_dir, self.fmu_api.cd)

    def test_set_sim_setup(self):
        """Test set_sim_setup functionality of fmu api"""
        new_sim_setup = {'initialNames': self.initial_names,
                         'initialValues': self.initial_values}
        self.fmu_api.sim_setup = new_sim_setup
        self.assertEqual(self.fmu_api.sim_setup['initialNames'],
                         new_sim_setup['initialNames'])
        self.assertEqual(self.fmu_api.sim_setup['initialValues'],
                         new_sim_setup['initialValues'])
        with self.assertRaises(KeyError):
            self.fmu_api.sim_setup = {"NotAValidKey": None}
        with self.assertRaises(TypeError):
            self.fmu_api.sim_setup = {"stopTime": "not_a_float_or_int"}

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
