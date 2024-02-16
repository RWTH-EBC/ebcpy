"""Test-module for all classes inside
ebcpy.simulationapi."""

import unittest
import sys
import os
from pathlib import Path
import shutil
import numpy as np
from pydantic import ValidationError
from ebcpy.simulationapi import dymola_api, fmu, Variable
from ebcpy import TimeSeriesData


class TestVariable(unittest.TestCase):

    def test_min_max(self):
        """Test the boundaries for variables"""
        for _type in [float, int, bool]:
            for _value in [1, 1.0, "1", True]:
                _var = Variable(
                    value=_value,
                    type=_type
                )
                self.assertIsInstance(Variable(
                    value=_value, type=_type,
                    min=_var.value - 1
                ).min, _type)
                self.assertIsInstance(Variable(
                    value=_value, type=_type,
                    max=_var.value
                ).max, _type)
        with self.assertRaises(ValidationError):
            Variable(value=1, max="1c", type=int)
        with self.assertRaises(ValidationError):
            Variable(value=1, min="1c", type=int)
        self.assertIsNone(Variable(value="s", type=str, max=0).max)
        self.assertIsNone(Variable(value="s", type=str, min=0).min)

    def test_value(self):
        """Test value conversion"""
        for _type in [float, int, bool]:
            for _value in [1, 1.0, "1", True]:
                self.assertIsInstance(
                    Variable(value=_value, type=_type).value,
                    _type
                )
        with self.assertRaises(TypeError):
            Variable(value="10c", type="int")
        self.assertIsInstance(Variable(value="Some String", type=str).value, str)


class PartialTestSimAPI(unittest.TestCase):

    def setUp(self) -> None:
        self.sim_api = None
        self.parameters = {}
        self.new_sim_setup = {}
        self.data_dir = Path(__file__).parent.joinpath("data")
        self.example_sim_dir = os.path.join(self.data_dir, "testzone")
        if not os.path.exists(self.example_sim_dir):
            os.mkdir(self.example_sim_dir)
        if self.__class__ == PartialTestSimAPI:
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

    def test_savepath_handling(self):
        """Test correct errors for wrong savepath allocation"""
        self.sim_api.set_sim_setup({"start_time": 0.0,
                                    "stop_time": 10.0})
        result_names = list(self.sim_api.states.keys())[:5]
        self.sim_api.result_names = result_names
        _some_par = list(self.sim_api.parameters.keys())[0]
        pars = {_some_par: self.sim_api.parameters[_some_par].value}
        parameters = [pars for i in range(2)]
        with self.assertRaises(ValueError):
            res = self.sim_api.simulate(parameters=parameters,
                                        return_option='savepath')
        with self.assertRaises(ValueError):
            res = self.sim_api.simulate(parameters=parameters,
                                        return_option='savepath',
                                        result_file_name=["t", "t"],
                                        savepath=[self.example_sim_dir, self.example_sim_dir])

        # Test multiple result_file_names
        _saves = [os.path.join(self.example_sim_dir, f"test_{i}") for i in range(len(parameters))]
        _save_tests = [
            self.example_sim_dir,
            _saves,
            _saves
        ]
        _names = [f"test_{i}" for i in range(len(parameters))]
        _name_tests = [
            _names,
            "test",
            _names
        ]
        for _save, _name in zip(_save_tests, _name_tests):
            res = self.sim_api.simulate(
                parameters=parameters,
                return_option="savepath",
                savepath=_save,
                result_file_name=_name
                )
            for r in res:
                self.assertTrue(os.path.isfile(r))
                self.assertIsInstance(r, str)

    def test_set_working_directory(self):
        """Test set_working_directory functionality of dymola api"""
        # Test the setting of the function
        self.sim_api.set_working_directory(self.data_dir)
        self.assertEqual(self.data_dir, self.sim_api.working_directory)
        # Test setting a str:
        self.sim_api.set_working_directory(str(self.data_dir))
        self.assertEqual(self.data_dir, self.sim_api.working_directory)

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


class PartialTestDymolaAPI(PartialTestSimAPI):

    n_cpu = None

    def setUp(self) -> None:
        super().setUp()
        if self.__class__ == PartialTestDymolaAPI:
            self.skipTest("Just a partial class")
        ebcpy_test_package_dir = self.data_dir.joinpath("TestModelVariables.mo")
        packages = [ebcpy_test_package_dir]
        model_name = "TestModelVariables"
        self.parameters = {"test_real": 10.0,
                           "test_int": 5,
                           "test_bool": 0,
                           "test_enum": 2
                           }
        self.new_sim_setup = {
            "solver": "Dassl",
            "tolerance": 0.001
        }
        # Mos script
        mos_script = self.data_dir.joinpath("mos_script_test.mos")

        # Just for tests in the gitlab-ci:
        if "linux" in sys.platform:
            dymola_exe_path = "/usr/local/bin/dymola"
        else:
            dymola_exe_path = None
        try:
            self.sim_api = dymola_api.DymolaAPI(
                working_directory=self.example_sim_dir,
                model_name=model_name,
                packages=packages,
                dymola_exe_path=dymola_exe_path,
                n_cpu=self.n_cpu,
                mos_script_pre=mos_script,
                mos_script_post=mos_script
            )
        except (FileNotFoundError, ImportError, ConnectionError) as error:
            self.skipTest(f"Could not load the dymola interface "
                          f"on this machine. Error message: {error}")

    def test_close(self):
        """Test close functionality of dymola api"""
        self.sim_api.close()
        self.assertIsNone(self.sim_api.dymola)

    def test_parameters(self):
        """Test non-existing parameter"""
        self.sim_api.result_names.extend(list(self.parameters.keys()))
        self.sim_api.result_names.extend(["test_out", "test_local"])
        res = self.sim_api.simulate(parameters=self.parameters,
                                    return_option="last_point")
        for k, v in res.items():
            if k in self.parameters:
                self.assertEqual(v, self.parameters[k])
        self.assertEqual(res["test_local"], 0)
        self.assertEqual(res["test_out"], self.parameters["test_int"])
        # Check boolean conversion:
        res_2 = self.sim_api.simulate(parameters={
            "test_bool": True,
        }, return_option="last_point")
        res_1 = self.sim_api.simulate(parameters={
            "test_bool": 1,
        }, return_option="last_point")
        self.assertEqual(res_1, res_2)
        # Wrong types
        with self.assertRaises(TypeError):
            self.sim_api.simulate(parameters={"test_bool": "True"})
        # Wrong parameter
        with self.assertRaises(KeyError):
            self.sim_api.simulate(
                parameters={"C2": 10},
                return_option='savepath'
            )
        # Model with no parameters:
        with self.assertRaises(ValueError):
            self.sim_api.parameters = {}
            self.sim_api.simulate()  # Test with no parameters

    def test_structural_parameters(self):
        """Test structural parameters"""
        some_val = np.random.rand()
        self.sim_api.result_names = ["test_local"]
        res = self.sim_api.simulate(
            parameters={"test_real_eval": some_val},
            return_option="last_point",
            structural_parameters=["test_real_eval"]
        )
        self.assertEqual(res["test_local"], some_val)
        res = self.sim_api.simulate(
            parameters={"test_real_eval": some_val},
            return_option="last_point"
        )
        self.assertEqual(res["test_local"], some_val)


class TestDymolaAPIMultiCore(PartialTestDymolaAPI):
    """Test-Class for the DymolaAPI class on single core."""

    n_cpu = 2


class TestDymolaAPISingleCore(PartialTestDymolaAPI):
    """Test-Class for the DymolaAPI class on multi core."""

    n_cpu = 1


class TestFMUAPI(PartialTestSimAPI):
    """Test-Class for the FMUAPI class."""

    n_cpu = None

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        super().setUp()
        if self.__class__ == PartialTestDymolaAPI:
            self.skipTest("Just a partial class")
        if "win" in sys.platform:
            model_name = self.data_dir.joinpath("PumpAndValve_windows.fmu")
        else:
            model_name = self.data_dir.joinpath("PumpAndValve_linux.fmu")

        self.sim_api = fmu.FMU_API(working_directory=self.example_sim_dir,
                                   model_name=model_name)

    def test_close(self):
        """Test close functionality of fmu api"""
        # pylint: disable=protected-access
        self.sim_api.close()
        self.assertIsNone(self.sim_api._unzip_dir)


class TestFMUAPISingleCore(TestFMUAPI):
    """Test-Class for the FMU_API class on single core"""

    n_cpu = 1


class TestFMUAPIMultiCore(TestFMUAPI):
    """Test-Class for the FMU_API class on multi core"""

    n_cpu = 2


if __name__ == "__main__":
    unittest.main()
