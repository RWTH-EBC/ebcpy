"""Test-module for all classes inside
ebcpy.data_types."""
import os
import sys
import shutil
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from ebcpy import data_types


class TestDataTypes(unittest.TestCase):
    """Test-class for the data_types module of ebcpy."""

    def setUp(self):
        """Called before every test.
        Define example paths and parameters used in all test-functions.
        """
        self.example_dir = Path(__file__).parent.joinpath("data")
        self.example_data_hdf_path = self.example_dir.joinpath("example_data.hdf")
        self.example_data_csv_path = self.example_dir.joinpath("example_data.csv")
        self.example_data_mat_path = self.example_dir.joinpath("example_mat_data.mat")
        self.example_data_xls_path = self.example_dir.joinpath("example_data.xls")
        self.savedir = self.example_dir.joinpath("test_save")
        os.makedirs(self.savedir, exist_ok=True)

    def test_default_tag(self):
        """Test the default_tag property"""
        tsd = data_types.TimeSeriesData(self.example_data_csv_path,
                                        sep=";")
        self.assertIsInstance(tsd.default_tag, str)
        with self.assertRaises(KeyError):
            tsd.default_tag = "Not in current keys"
        tsd.default_tag = tsd.default_tag

    def test_multilevel(self):
        """Test the two-level format of tsd"""
        df = pd.DataFrame({"col_1": [5]})
        df.columns = pd.MultiIndex.from_product(
            [["first"], ["second"], ["third"]])
        with self.assertRaises(TypeError):
            data_types.TimeSeriesData(df)
        df.columns = pd.MultiIndex.from_product(
            [["first"], ["second"]],
            names=["Not Variables", "not a tag"])
        with self.assertRaises(TypeError):
            data_types.TimeSeriesData(df)

    def test_save(self):
        """Test fp property"""
        tsd = data_types.TimeSeriesData(self.example_data_csv_path,
                                        sep=";")

        filepath = self.savedir.joinpath("test_hdf.hdf")
        with self.assertRaises(KeyError):
            tsd.save(filepath=filepath)
        try:
            tsd.save(filepath=filepath, key="test")
            self.assertTrue(os.path.isfile(filepath))
        except ImportError:
            pass   # Skip the optional part
        # Test csv and the setter.
        filepath = self.savedir.joinpath("test_hdf.csv")
        tsd.filepath = filepath
        tsd.save()
        self.assertTrue(os.path.isfile(filepath))

        with self.assertRaises(TypeError):
            filepath = self.savedir.joinpath("test_mat.mat")
            tsd.save(filepath=filepath)

    def test_load_save_csv(self):
        """Test correct loading and saving"""
        tsd_ref = data_types.TimeSeriesData(self.example_data_csv_path,
                                        sep=";")
        filepath = self.savedir.joinpath("test_hdf.csv")
        tsd_ref.save(filepath=filepath)
        tsd = data_types.TimeSeriesData(filepath)
        self.assertTrue(tsd.equals(tsd_ref))
        # test wrong and multi-index
        with self.assertRaises(IndexError):
            data_types.TimeSeriesData(filepath, index_col=[0, 1])
        # Test wrong and single index headers
        with self.assertRaises(IndexError):
            data_types.TimeSeriesData(filepath, header=0)

    def _load_save_parquet(self, engine):
        """Test correct loading and saving for all parquet options"""
        tsd_ref = data_types.TimeSeriesData(self.example_data_csv_path,
                                            sep=";")
        parquet_formats = ['parquet', 'parquet.snappy', 'parquet.gzip', 'parquet.brotli']
        # Test parquet engine pyarrow
        for suffix in parquet_formats:
            filepath = self.savedir.joinpath(f"test_parquet.{suffix}")
            tsd_ref.save(filepath=filepath, engine=engine)
            self.assertTrue(os.path.isfile(filepath))
            tsd = data_types.TimeSeriesData(filepath, engine=engine)
            self.assertTrue(tsd.equals(tsd_ref))

    def test_load_save_parquet_pyarrow(self):
        """Test correct loading and saving for parquet pyarrow options"""
        self._load_save_parquet(engine="pyarrow")

    @unittest.skipIf(sys.version_info.minor < 9 and sys.version_info.major == 3,
                     reason="Not supported for py<3.9")
    def test_load_save_parquet_fastparquet(self):
        """Test correct loading and saving for parquet fastparquet options"""
        try:
            self._load_save_parquet(engine="fastparquet")
        except KeyError:
            self.skipTest("fastparquet error which is currently not handled")

    def test_time_series_data(self):
        """Test the class TimeSeriesData"""
        # Test if wrong input leads to FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            data_types.TimeSeriesData("Z:\\this_will_never_be_a_file_path.hdf")
        # Test if wrong file-ending leads to TypeError
        with self.assertRaises(TypeError):
            a_python_file = __file__
            data_types.TimeSeriesData(a_python_file)
        # If no key is provided, a KeyError has to be raised
        try:
            with self.assertRaises(KeyError):
                data_types.TimeSeriesData(self.example_data_hdf_path)
            with self.assertRaises(KeyError):
                data_types.TimeSeriesData(self.example_data_hdf_path,
                                          key="wrong_key")
            with self.assertRaises(KeyError):
                data_types.TimeSeriesData(self.example_data_hdf_path,
                                          key="")
            with self.assertRaises(KeyError):
                data_types.TimeSeriesData(self.example_data_xls_path)
            # Correctly load the .hdf:
            time_series_data = data_types.TimeSeriesData(self.example_data_hdf_path,
                                                         key="parameters")
            self.assertIsInstance(
                time_series_data,
                type(pd.DataFrame()))
        except ImportError:
            pass   # Skip the optional part
        # Correctly load the .csv:
        time_series_data = data_types.TimeSeriesData(self.example_data_csv_path,
                                                     sep=";")
        self.assertIsInstance(
            time_series_data,
            type(pd.DataFrame()))
        # Correctly load the .csv:
        time_series_data = data_types.TimeSeriesData(self.example_data_xls_path,
                                                     sheet_name="example_data")
        self.assertIsInstance(
            time_series_data,
            type(pd.DataFrame()))
        # Correctly load the .mat:
        time_series_data = data_types.TimeSeriesData(self.example_data_mat_path)
        self.assertIsInstance(
            time_series_data,
            type(pd.DataFrame()))
        # Load with variable names:
        variable_names = ["combiTimeTable.y[6]"]
        time_series_data = data_types.TimeSeriesData(
            self.example_data_mat_path, variable_names=variable_names
        )
        self.assertIsInstance(
            time_series_data,
            type(pd.DataFrame()))
        self.assertEqual(len(time_series_data.columns), 1)
        self.assertEqual(time_series_data.to_df().columns[0], variable_names[0])
        # Test load and set df functions:
        df = time_series_data
        self.assertIsInstance(
            df,
            type(pd.DataFrame()))
        # Test converters:
        tsd = data_types.TimeSeriesData(self.example_data_csv_path,
                                        sep=";")
        tsd.to_datetime_index()
        self.assertIsInstance(tsd.index, pd.DatetimeIndex)
        tsd.to_datetime_index()
        self.assertIsInstance(tsd.index, pd.DatetimeIndex)
        tsd.to_float_index()
        self.assertIsInstance(tsd.index, type(pd.Index([], dtype="float64")))
        tsd.to_float_index()
        self.assertIsInstance(tsd.index, type(pd.Index([], dtype="float64")))
        tsd.to_datetime_index()
        self.assertIsInstance(tsd.index, pd.DatetimeIndex)
        tsd.clean_and_space_equally(desired_freq="1s")
        self.assertIsInstance(tsd.index, pd.DatetimeIndex)

    def test_time_series_tagging(self):
        """Test tagging functions"""
        with self.assertRaises(TypeError):
            data_types.TimeSeriesData(self.example_data_mat_path,
                                      default_tag=10)
        tsd1 = data_types.TimeSeriesData(self.example_data_mat_path,
                                         default_tag='other_default')
        with self.assertRaises(TypeError):
            tsd1.default_tag = 10
        self.assertEqual(tsd1.get_columns_by_tag('other_default').size,
                         tsd1.size)
        with self.assertRaises(KeyError):
            tsd1.get_columns_by_tag('this_is_never_a:tag')
        tsd2 = data_types.TimeSeriesData(self.example_data_mat_path,
                                         default_tag='new_data')
        tsd3 = pd.concat([tsd1, tsd2], axis=1)
        with self.assertRaises(KeyError):
            tsd3.default_tag = 'new_default_tag'
        self.assertEqual(tsd3.size,
                         tsd1.size + tsd2.size)
        self.assertEqual(tsd3.get_columns_by_tag('new_data').size,
                         tsd2.size)

    def test_get_cols_by_tag(self):
        time_series_data = data_types.TimeSeriesData(self.example_data_mat_path)
        tsd2 = time_series_data.get_columns_by_tag(tag="raw")
        self.assertTrue(np.all(tsd2 == time_series_data))
        tsd2 = time_series_data.get_columns_by_tag(tag="raw",
                                                   variables=["combiTimeTable.smoothness"])
        self.assertEqual(len(tsd2.columns), 1)
        with self.assertRaises(TypeError):
            time_series_data.get_columns_by_tag(tag="raw",
                                                return_type="not_supported")
        tsd = time_series_data.get_columns_by_tag(tag="raw",
                                                  return_type="np")
        self.assertIsInstance(tsd, np.ndarray)
        tsd = time_series_data.get_columns_by_tag(tag="raw",
                                                  return_type="control")
        self.assertIsInstance(tsd, np.ndarray)

    def test_time_series_utils(self):
        """Test the utils for time series"""
        tsd = data_types.TimeSeriesData(self.example_data_mat_path)
        self.assertEqual(len(tsd.get_variable_names()), tsd.shape[1])
        self.assertIsNotNone(tsd.get_tags())
        self.assertLessEqual(len(tsd.get_variable_names()), tsd.shape[1])

    def test_get_keys_of_hdf_file(self):
        """Test the function get_keys_of_hdf_file.
        Check the keys of the file with e.g. the SDFEditor and
        use those keys as a reference list.
        """
        # pylint: disable=import-outside-toplevel
        # pylint: disable=unused-import
        try:
            import h5py
        except ImportError:
            self.skipTest("Test only makes sense if h5py is installed")
        reference_list = ['parameters', 'trajectories']
        return_val = data_types.get_keys_of_hdf_file(self.example_data_hdf_path)
        self.assertListEqual(return_val, reference_list)

    def test_to_df(self):
        """Test the to_df function"""
        df = pd.DataFrame({"my_variable": np.random.rand(5),
                           "my_variable1": np.random.rand(5)})
        tsd = data_types.TimeSeriesData(df.copy())
        self.assertEqual(tsd.to_df().columns.values.tolist(),
                         df.columns.values.tolist())
        self.assertEqual(tsd.to_df().columns.nlevels, 1)
        tsd.loc[:, ("my_variable", "new_tag")] = 5
        self.assertIsInstance(tsd.copy().to_df(), pd.DataFrame)
        self.assertEqual(tsd.columns.nlevels, 2)
        # Test force index:
        with self.assertRaises(IndexError):
            tsd.to_df(force_single_index=True)

    def test_frequ(self):
        """Test frequency property"""
        df = pd.DataFrame({"my_variable": np.random.rand(5),
                           "my_variable1": np.random.rand(5)})
        tsd = data_types.TimeSeriesData(df.copy())
        fre, std = tsd.frequency
        self.assertEqual(fre, 1)
        self.assertEqual(std, 0)
        tsd.to_datetime_index()
        fre, std = tsd.frequency
        self.assertEqual(fre, 1)
        self.assertEqual(std, 0)

    def test_preprocessing_api(self):
        """Test function accessed in preprocessing"""
        tsd = data_types.TimeSeriesData(self.example_data_csv_path,
                                        sep=";")
        # number_lines_totally_na
        self.assertEqual(tsd.number_lines_totally_na(), 0)
        tsd.moving_average(window=2, variable="sine.startTime / s")
        tsd.moving_average(window=5, variable="sine.startTime / s",
                           tag="raw", new_tag="some_new_tag")
        tsd.low_pass_filter(crit_freq=0.1, filter_order=2,
                            variable="sine.amplitude / ")
        tsd.low_pass_filter(crit_freq=0.1, filter_order=2,
                            variable="sine.startTime / s",
                            tag="raw", new_tag="some_new_tag")

    def test_time_series(self):
        """Test the time series object"""
        time_series = data_types.TimeSeries(np.random.rand(100))
        self.assertIsInstance(time_series.to_frame(),
                              data_types.TimeSeriesData)

    def tearDown(self) -> None:
        """Delete saved files"""
        try:
            shutil.rmtree(self.savedir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
