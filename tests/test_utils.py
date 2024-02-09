"""Test-module for all classes inside
ebcpy.utils."""
import shutil
import sys
import unittest
import os
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import scipy.io as spio
from ebcpy import TimeSeriesData
from ebcpy.utils import setup_logger, conversion, statistics_analyzer, reproduction


class TestConversion(unittest.TestCase):
    """Test-class for preprocessing."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.example_dir = Path(__file__).parent.joinpath("data")
        self.example_data_hdf_path = self.example_dir.joinpath("example_data.csv")
        self.tsd = TimeSeriesData(self.example_data_hdf_path, sep=";")
        self.columns = ["sine.freqHz / Hz"]

    def test_conversion_tsd_to_mat(self):
        """Test function conversion.convert_tsd_to_modelica_mat().
        For an example, see the doctest in the function."""
        # First convert the file
        save_path = self.example_dir.joinpath("example_data_converted.mat")
        # Test both conversion with specification of columns and without passing the names.
        for col in [self.columns, None]:
            filepath_mat = conversion.convert_tsd_to_modelica_mat(
                tsd=self.tsd,
                save_path_file=save_path,
                columns=col)
            # Check if converted file exists
            self.assertTrue(os.path.isfile(filepath_mat))
            # Check if converted filepath is provided filepath
            self.assertEqual(filepath_mat, str(save_path))
            # Now check if the created mat-file can be used.
            self.assertIsInstance(spio.loadmat(save_path), dict)
            # Remove converted file again
            os.remove(save_path)

        with self.assertRaises(ValueError):
            conversion.convert_tsd_to_modelica_mat(
                tsd=self.tsd,
                save_path_file="not_a_mat_file.txt",
                columns=col)

    def test_conversion_tsd_to_modelica_txt(self):
        """Test function conversion.convert_tsd_to_modelica_txt().
        For an example, see the doctest in the function."""
        for col, with_tag in zip([self.columns, None], [True, False]):
            # Check if successfully converted
            filepath_txt = conversion.convert_tsd_to_modelica_txt(
                tsd=self.tsd,
                save_path_file=Path("some_text_data.txt"),
                table_name="dummy",
                columns=col,
                with_tag=with_tag
            )
            # Check if converted file exists
            self.assertTrue(os.path.isfile(filepath_txt))
            # Check if converted filepath is provided filepath
            self.assertTrue(filepath_txt.endswith(".txt"))
            # Remove converted file again
            os.remove(filepath_txt)
        with self.assertRaises(ValueError):
            conversion.convert_tsd_to_modelica_txt(
                tsd=self.tsd,
                save_path_file="not_a_txt.mat",
                table_name="dummy",
                columns=self.columns[0])

    def test_conversion_tsd_to_clustering_txt(self):
        """Test function conversion.convert_tsd_to_clustering_txt().
        For an example, see the doctest in the function."""
        # First convert the file
        save_path = os.path.normpath(os.path.join(self.example_dir, "example_data_converted.txt"))
        # Test both conversion with specification of columns and without passing the names.
        for col in [self.columns, None]:
            filepath_txt = conversion.convert_tsd_to_clustering_txt(
                tsd=self.tsd,
                save_path_file=save_path,
                columns=col)
            # Check if converted file exists
            self.assertTrue(os.path.isfile(filepath_txt))
            # Check if converted filepath is provided filepath
            self.assertEqual(filepath_txt, save_path)
            # Remove converted file again
            os.remove(save_path)

    def test_convert_subset(self):
        """Test _convert_to_df_subset function"""
        df = pd.DataFrame({"val": np.random.rand(100)})
        df, headers = conversion._convert_to_subset(
            df=df,
            columns=[],
            offset=0)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(headers), 2)
        # Try with NaN
        df = pd.DataFrame({"val": np.random.rand(100)})
        df.loc[2, "val"] = np.NAN
        with self.assertRaises(ValueError):
            conversion._convert_to_subset(
                df=df,
                columns=[],
                offset=0)
        with self.assertRaises(IndexError):
            df = pd.DataFrame({"val": 5}, index=["string"])
            conversion._convert_to_subset(
                df=df,
                columns=[], offset=0
            )


class TestStatisticsAnalyzer(unittest.TestCase):
    """Test-class for the StatisticsAnalyzer-Class"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.example_dir = Path(__file__).parent.joinpath("data")
        self.meas_ex = np.random.rand(1000)
        self.sim_ex = np.random.rand(1000) * 10

    def test_calc(self):
        """Test class StatisticsAnalyzer"""
        sup_methods = ["mae", "r2", "mse", "rmse", "cvrmse", "nrmse"]
        for method in sup_methods:
            stat_analyzer = statistics_analyzer.StatisticsAnalyzer(method)
            self.assertIsInstance(stat_analyzer.calc(self.meas_ex, self.sim_ex),
                                  float)
        with self.assertRaises(ValueError):
            stat_analyzer = statistics_analyzer.StatisticsAnalyzer("not_supported_method")
        # Test factor:
        stat_analyzer_min = statistics_analyzer.StatisticsAnalyzer("R2")
        stat_analyzer_max = statistics_analyzer.StatisticsAnalyzer("R2", for_minimization=False)
        self.assertEqual(-1 * stat_analyzer_min.calc(self.meas_ex, self.sim_ex),
                         stat_analyzer_max.calc(self.meas_ex, self.sim_ex))

    def test_calc_rmse(self):
        """Test static function calc_rmse"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_rmse(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_mse(self):
        """Test static function calc_mse"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_mse(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_mae(self):
        """Test static function calc_mae"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_mae(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_nrmse(self):
        """Test static function calc_nrmse"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_nrmse(self.meas_ex, self.sim_ex),
                              float)
        with self.assertRaises(ValueError):
            custom_meas = self.meas_ex / self.meas_ex
            stat_analyzer.calc_nrmse(custom_meas, self.sim_ex)

    def test_calc_cvrmse(self):
        """Test static function calc_cvrmse"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_cvrmse(self.meas_ex, self.sim_ex),
                              float)
        with self.assertRaises(ValueError):
            custom_meas = self.meas_ex - self.meas_ex
            stat_analyzer.calc_cvrmse(custom_meas, self.sim_ex)

    def test_calc_r2(self):
        """Test static function calc_r2"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_r2(self.meas_ex, self.sim_ex),
                              float)

    def test_custom_func(self):
        """Test custom function calc_r2"""

        def my_func(x, y):
            return x - y

        stat_meas = statistics_analyzer.StatisticsAnalyzer(method=my_func)
        self.assertEqual(stat_meas.calc(10, 10), 0)
        self.assertEqual(stat_meas.calc(20, 10), 10)
        with self.assertRaises(TypeError):
            stat_meas.calc(1, 2, 3)


class TestLogger(unittest.TestCase):
    """Test-class for the logger function."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.example_dir = Path(__file__).parent.joinpath("test_logger")
        self.logger = setup_logger(working_directory=self.example_dir,
                                   name="test_logger")

    def test_logging(self):
        """Test if logging works."""
        example_str = "This is a test"
        self.logger.info(example_str)
        with open(self.logger.handlers[1].baseFilename, "r") as logfile:
            logfile.seek(0)
            self.assertTrue(example_str in logfile.read())

    def tearDown(self):
        """Remove created files."""
        self.logger.handlers[0].close()
        try:
            os.remove(self.logger.handlers[1].baseFilename)
        except PermissionError:
            pass


class TestReproduction(unittest.TestCase):
    """Test-class for the reproduction functions"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.data_dir = Path(__file__).parent.joinpath("data")
        self.save_dir = self.data_dir.joinpath('testzone', 'reproduction_tests')
        self.working_directory_files = []

    def test_save_reproduction_archive(self):
        os.getlogin = lambda: "test_login"
        # test no input
        reproduction.input = lambda _: ""
        zip_file = reproduction.save_reproduction_archive()
        # for tearDown of the files which will be saved in working_directory when no path is given
        self.working_directory_files.append(zip_file)
        file = Path(sys.modules['__main__'].__file__).absolute().name.replace(".py", "")
        logger_name = f"Study_log_{file}.txt"
        self.working_directory_files.append(logger_name)
        self.assertTrue(zipfile.is_zipfile(zip_file))
        # create a file to remove with CopyFile but leave it open for executing except block
        f = open(self.data_dir.joinpath('remove.txt'), 'w')
        f.write('test file to remove')
        f.close()
        # test different correct inputs
        test_files = [
            'not a file string',
            str(self.data_dir.joinpath('example_data.csv')),
            reproduction.CopyFile(
                filename='example_data_copy_file.csv',
                sourcepath=self.data_dir.joinpath('example_data.csv'),
                remove=False),
            reproduction.CopyFile(
                filename="test_remove_file.txt",
                sourcepath=self.data_dir.joinpath('remove.txt'),
                remove=True
            )
        ]
        zip_file = reproduction.save_reproduction_archive(
            title="Test_correct_inputs",
            path=self.save_dir,
            log_message="Test study correct inputs",
            files=test_files
        )
        self.assertTrue(zipfile.is_zipfile(zip_file))
        # test false inputs
        false_test_files = [1]
        with self.assertRaises(TypeError):
            reproduction.save_reproduction_archive(
                title="Test_false_inputs",
                path=self.save_dir,
                log_message="Test study false inputs",
                files=false_test_files
            )

    def test_creat_copy_files_from_dir(self):
        files = reproduction.creat_copy_files_from_dir(
            foldername='test_dir',
            sourcepath=self.data_dir,
            remove=False
        )
        for copy_file in files:
            self.assertIsInstance(copy_file, reproduction.CopyFile)

    def tearDown(self) -> None:
        """Delete saved files"""
        try:
            shutil.rmtree(self.save_dir, ignore_errors=True)
            for file in self.working_directory_files:
                os.remove(file)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
