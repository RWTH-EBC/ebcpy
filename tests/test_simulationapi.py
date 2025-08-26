"""Test-module for all classes inside
ebcpy.simulationapi."""

import unittest
import sys
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from pydantic import ValidationError
from ebcpy.simulationapi import dymola_api, fmu, Variable
from ebcpy import load_time_series_data


def postprocess_mat_result(mat_result_file, variable_names, n):
    """
    Dummy function to test postprocessing of mat results.
    Loads only given variable names and returns the last n values.

    Must be defined globally to allow multiprocessing.
    """
    return load_time_series_data(mat_result_file, variable_names=variable_names).iloc[-n:]


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
        self.example_sim_dir = self.data_dir.joinpath("testzone")
        if not os.path.exists(self.example_sim_dir):
            os.mkdir(self.example_sim_dir)
        if self.__class__ == PartialTestSimAPI:
            self.skipTest("Just a partial class")

    def start_api(self, save_logs: bool, **kwargs):
        raise NotImplementedError

    def test_simulate(self):
        """Test simulate functionality of dymola api"""
        self.sim_api.set_sim_setup({"start_time": 0.0,
                                    "stop_time": 10.0})
        result_names = list(self.sim_api.states.keys())[:5]
        self.sim_api.result_names = result_names
        res = self.sim_api.simulate()  # Test with no parameters
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(len(res.columns), len(result_names))
        res = self.sim_api.simulate(return_option='last_point')
        self.assertIsInstance(res, dict)
        res = self.sim_api.simulate(parameters=self.parameters,
                                    return_option='savepath')
        self.assertTrue(os.path.isfile(res))
        self.assertIsInstance(res, str)
        res = self.sim_api.simulate(parameters=self.parameters,
                                    return_option='savepath',
                                    savepath=os.path.join(self.example_sim_dir, "my_new_folder"),
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
            os.path.join(self.example_sim_dir, "my_save_folder"),
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
            self.assertEqual(self.sim_api.sim_setup.model_dump()[key],
                             value)
        with self.assertRaises(ValidationError):
            self.sim_api.set_sim_setup(sim_setup={"NotAValidKey": None})
        with self.assertRaises(ValidationError):
            self.sim_api.set_sim_setup(sim_setup={"stop_time": "not_a_float_or_int"})

    def test_no_log(self):
        self.sim_api.close()
        import logging
        for handler in self.sim_api.logger.handlers:
            handler.flush()
            handler.close()
        logger = logging.getLogger(self.sim_api.__class__.__name__)
        while len(logger.handlers) > 0:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        log_file = self.example_sim_dir.joinpath(f"{self.sim_api.__class__.__name__}.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        self.start_api(save_logs=False, mos_script=self.data_dir.joinpath("mos_script_test.mos"))
        self.sim_api.logger.error("This log should not be saved")
        self.assertFalse(os.path.exists(log_file))

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
        self.packages = [ebcpy_test_package_dir]
        self.parameters = {"test_real": 10.0,
                           "test_int": 5,
                           "test_bool": 0,
                           "test_enum": 2
                           }
        self.new_sim_setup = {
            "solver": "Dassl",
            "tolerance": 0.001
        }
        self.start_api()

    def start_api(self, save_logs: bool = True, **kwargs):
        # Just for tests in the gitlab-ci:
        if "linux" in sys.platform:
            dymola_exe_path = "/usr/local/bin/dymola"
        else:
            dymola_exe_path = None
        mos_script = kwargs.get("mos_script", self.data_dir.joinpath("mos_script_test.mos"))
        model_name = kwargs.get("model_name", "TestModelVariables")
        extract_variables = kwargs.get("extract_variables", True)

        try:
            self.sim_api = dymola_api.DymolaAPI(
                working_directory=self.example_sim_dir,
                model_name=model_name,
                packages=self.packages,
                dymola_exe_path=dymola_exe_path,
                n_cpu=self.n_cpu,
                mos_script_pre=mos_script,
                mos_script_post=mos_script,
                save_logs=save_logs,
                extract_variables=extract_variables
            )
        except (FileNotFoundError, ImportError, ConnectionError) as error:
            self.skipTest(f"Could not load the dymola interface "
                          f"on this machine. Error message: {error}")

    def test_no_model_none(self):
        self.sim_api.close()
        self.start_api(
            mos_script=None, model_name=None,
        )
        with self.assertRaises(ValueError):
            self.sim_api.simulate()
        self.sim_api.simulate(model_names=["TestModelVariables"], parameters=self.parameters)

    def test_no_extract_model_vars(self):
        self.sim_api.close()
        self.start_api(
            mos_script=None, extract_variables=False
        )
        self.assertEqual(self.sim_api.variables, [])

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

    def test_variables_to_save(self):
        all_false = dymola_api.ExperimentSetupOutput(
            states=False,
            derivatives=False,
            inputs=False,
            outputs=False,
            auxiliaries=False,
        )
        parameters = list(self.sim_api.parameters.keys())
        cases_to_test = [
            {
                "experiment_setup": all_false.model_copy(update={"outputs": True}),
                "variables": list(self.sim_api.outputs.keys()) + parameters
            },
            {
                "experiment_setup": all_false.model_copy(update={"inputs": True}),
                "variables": list(self.sim_api.inputs.keys()) + parameters
            },
            {
                "experiment_setup": all_false.model_copy(
                    update={"states": True, "derivatives": True, "auxiliaries": True}
                ),
                "variables": list(self.sim_api.states.keys()) + parameters
            },
        ]
        self.sim_api.set_sim_setup({"start_time": 0.0,
                                    "stop_time": 10.0})
        for case in cases_to_test:
            variables = case["variables"]
            self.sim_api.update_experiment_setup_output(case["experiment_setup"])
            res = self.sim_api.simulate(
                parameters=self.parameters,
                return_option='savepath'
            )
            df = load_time_series_data(res, variables_names=variables)
            self.assertEqual(sorted(variables), sorted(df.columns))

    def test_postprocessing_injection(self):
        """Test injection of postprocessing function for mats"""
        self.sim_api.set_sim_setup({"start_time": 0.0,
                                    "stop_time": 10.0})
        result_names = list(self.sim_api.states.keys())[:5]
        self.sim_api.result_names = result_names
        n_values_to_return = np.random.randint(1, 4)
        kwargs_postprocessing = {
            "variable_names": result_names[:2],
            "n": n_values_to_return
        }
        res = self.sim_api.simulate(
            parameters=self.parameters,
            return_option='savepath',
            postprocess_mat_result=postprocess_mat_result,
            kwargs_postprocessing=kwargs_postprocessing
        )
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(len(res.index), n_values_to_return)
        self.assertEqual(len(res.columns), 2)


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
        if self.__class__ == TestFMUAPI:
            self.skipTest("Just a partial class")
        self.start_api()

    def _get_model_name(self):
        if "win" in sys.platform:
            return self.data_dir.joinpath("PumpAndValve_windows.fmu")
        return self.data_dir.joinpath("PumpAndValve_linux.fmu")

    def start_api(self, save_logs: bool = True, **kwargs):
        self.sim_api = fmu.FMU_API(
            working_directory=self.example_sim_dir,
            model_name=self._get_model_name(),
            save_logs=save_logs
        )

    def test_relative_working_directory(self):
        self.sim_api.close()
        self.sim_api = fmu.FMU_API(
            # Complex solution to enable tests from any cwd
            working_directory=Path(__file__).parent.relative_to(Path().absolute()).joinpath("data", "testzone"),
            model_name=self._get_model_name()
        )
        self.assertEqual(self.sim_api.working_directory, self.example_sim_dir)

    def test_no_working_directory(self):
        self.sim_api.close()
        self.sim_api = fmu.FMU_API(model_name=self._get_model_name())
        self.assertEqual(self.sim_api.working_directory, self.data_dir)

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
