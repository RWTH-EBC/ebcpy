"""Test-module for all classes inside
enstats.segmentizer."""

from ebcpy.tsa import DecisionTreeClassifier
import unittest
import os
import shutil
from ebcpy import data_types
import sklearn.tree as sktree
import numpy as np


class TestClassifier(unittest.TestCase):
    """
    Test the segmentizer class of enstats
    """

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples//data")
        self.example_xlsx_input = os.path.normpath(self.example_dir +
                                                   "//HeatPumpClassification.xlsx")
        # Load timeSeriesData
        self.tsd = data_types.TimeSeriesData(self.example_xlsx_input, sheet_name='Sheet1')
        self.example_clas_dir = os.path.normpath(self.example_dir + "//test_classifier")
        # List with column names that should be part of the classifier analysis
        self.variable_names = ['VDot', 'T_RL', 'T_VL', 'T_Amb', 'MassFlow', 'TempDiff']
        self.class_names = 'class'  # Column name where classes are listed

    def test_DecisionTreeClassifier(self):
        """Test class DecisionTreeClassifier and all it's main methods"""
        # Use a random sample-size between 0.1 and 0.9 to ensure all values will work
        _test_size = np.random.uniform(low=0.01, high=0.99)
        # Instantiate class
        dtc = DecisionTreeClassifier(self.example_clas_dir,
                                     time_series_data=self.tsd,
                                     variable_list=self.variable_names,
                                     class_list=self.class_names,
                                     test_size=_test_size)
        dtree = dtc.create_decision_tree()
        self.assertIsInstance(dtree, sktree.DecisionTreeClassifier)
        dtc.validate_decision_tree()
        self.assertTrue(os.path.isfile(self.example_clas_dir + "//pairplot.png"))
        dtc.export_decision_tree_to_pickle()
        variable_names_multi_index = list(map(tuple, [[var_name, 'raw'] for var_name in self.variable_names]))
        classes = dtc.classify(self.tsd[variable_names_multi_index])
        for temp_class in classes:
            self.assertIsInstance(temp_class, dict)

        # Test-save of dtree:
        filepath = dtc.export_decision_tree_to_pickle(dtree=dtree)
        self.assertTrue(os.path.isfile(filepath))
        # Test-load of dtree:
        dtree_loaded, _ = dtc.load_decision_tree_from_pickle(filepath)
        # Create new classifier-object with loaded tree:
        dtc = DecisionTreeClassifier(self.example_clas_dir,
                                     dtree=dtree_loaded)
        # Classify again with loaded tree and check if the output is still the same.
        classes_loaded = dtc.classify(self.tsd[variable_names_multi_index])

        for (i, temp_class) in enumerate(classes):
            self.assertEqual(temp_class["name"], classes_loaded[i]["name"])
            self.assertEqual(temp_class["stop_time"], classes_loaded[i]["stop_time"])
            self.assertEqual(temp_class["start_time"], classes_loaded[i]["start_time"])

    def tearDown(self):
        """Delete all files created while testing"""
        try:
            shutil.rmtree(self.example_clas_dir)
        except (FileNotFoundError, PermissionError):
            pass


if __name__ == "__main__":
    unittest.main()
