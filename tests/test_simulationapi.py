"""Test-module for all classes inside
ebcpy.simulationapi."""

import unittest
import sys
import os
from pathlib import Path
import shutil
from pydantic import ValidationError
from ebcpy.simulationapi import dymola_api, fmu
from ebcpy import TimeSeriesData


class PartialTestDymolaAPI(unittest.TestCase):

    def setUp(self) -> None:
        self.sim_api = None
        self.parameters = {}
        self.new_sim_setup = {}
        self.data_dir = Path(__file__).parent.joinpath("data")
        self.example_sim_dir = os.path.join(self.data_dir, "testzone")
        if not os.path.exists(self.example_sim_dir):
            os.mkdir(self.example_sim_dir)
        if self.__class__ == PartialTestDymolaAPI:
            self.skipTest("Just a partial class")

    def test_simulate(self):
        """Test simulate functionality of dymola api"""
        self.sim_api.set_sim_setup({"start_time": 0.0,
                                    "stop_time": 10.0})
        result_names = list(self.sim_api.states.keys())[:5]
        self.sim_api.result_names = result_names
        res = self.sim_api.simulate()  # Test with no parameters
        self.assertIsInstance(res, TimeSeriesData)
        self.assertEqual(len(res.columns), len(result_names))
        res = self.sim_api.simulate(return_option='last_point')
        self.assertIsInstance(res, dict)
        res = self.sim_api.simulate(parameters=self.parameters,
                                    return_option='savepath')
        self.assertTrue(os.path.isfile(res))
        self.assertIsInstance(res, str)
        res = self.sim_api.simulate(parameters=self.parameters,
                                    return_option='savepath',
                                    savepath=self.example_sim_dir,
                                    result_file_name="my_other_name")
        self.assertTrue(os.path.isfile(res))
        self.assertIsInstance(res, str)
        # Test non-existing parameter
        self.parameters.update({"C2": 10})
        if isinstance(self.sim_api, dymola_api.DymolaAPI):
            with self.assertRaises(KeyError):
                self.sim_api.simulate(parameters=self.parameters,
                                      return_option='savepath')
            # Does not raise anything
            self.sim_api.simulate(parameters=self.parameters)
            # Model with no parameters:
            with self.assertRaises(ValueError):
                self.sim_api.parameters = {}
                self.sim_api.simulate()  # Test with no parameters

    def test_set_cd(self):
        """Test set_cd functionality of dymola api"""
        # Test the setting of the function
        self.sim_api.set_cd(self.data_dir)
        self.assertEqual(self.data_dir, self.sim_api.cd)

    def test_set_sim_setup(self):
        """Test set_sim_setup functionality of fmu api"""
        self.sim_api.set_sim_setup(sim_setup=self.new_sim_setup)
        for key, value in self.new_sim_setup.items():
            self.assertEqual(self.sim_api.sim_setup.dict()[key],
                             value)
        with self.assertRaises(ValidationError):
            self.sim_api.set_sim_setup(sim_setup={"NotAValidKey": None})
        with self.assertRaises(ValidationError):
            self.sim_api.set_sim_setup(sim_setup={"stop_time": "not_a_float_or_int"})

    def tearDown(self):
        """Delete all files created while testing"""

        try:
            self.sim_api.close()
        except AttributeError:
            pass
        try:
            shutil.rmtree(self.example_sim_dir)
        except (FileNotFoundError, PermissionError):
            pass


class TestDymolaAPI(PartialTestDymolaAPI):
    """Test-Class for the DymolaAPI class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        super().setUp()
        ebcpy_test_package_dir = self.data_dir.joinpath("TestModel.mo")
        packages = [ebcpy_test_package_dir]
        model_name = "AixCalTest_TestModel"
        self.parameters = {"C": 2000,
                           "heatConv_a": 5,
                           "heatConv_b": 5,
                           }
        self.new_sim_setup = {
            "solver": "Dassl",
            "tolerance": 0.001
        }
        # Just for tests in the ci:
        if "linux" in sys.platform:
            dymola_path = "/usr/local/bin/dymola"
        else:
            dymola_path = None
        try:
            self.sim_api = dymola_api.DymolaAPI(cd=self.example_sim_dir,
                                                model_name=model_name,
                                                packages=packages,
                                                dymola_path=dymola_path)
        except (FileNotFoundError, ImportError, ConnectionError) as error:
            self.skipTest(f"Could not load the dymola interface "
                          f"on this machine. Error message: {error}")

    def test_close(self):
        """Test close functionality of dymola api"""
        self.sim_api.close()
        self.assertIsNone(self.sim_api.dymola)


class TestFMUAPI(PartialTestDymolaAPI):
    """Test-Class for the FMUAPI class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        super().setUp()
        if "win" in sys.platform:
            model_name = self.data_dir.joinpath("PumpAndValve_windows.fmu")
        else:
            model_name = self.data_dir.joinpath("PumpAndValve_linux.fmu")

        self.sim_api = fmu.FMU_API(cd=self.example_sim_dir,
                                   model_name=model_name)

    def test_close(self):
        """Test close functionality of fmu api"""
        # pylint: disable=protected-access
        self.sim_api.close()
        self.assertEqual(self.sim_api._unzip_dirs, {})


if __name__ == "__main__":
    unittest.main()
